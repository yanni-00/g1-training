"""[OMA] Shared cyclic human reference, driven by explicit physical commands.

This plans target states only. Actual velocity is measured separately in play.
"""
import torch
from .cyclic_reference_clock import CyclicSupportPhaseClock
from .velocity_reference import VelocityReferenceRoot


def quat_multiply(a,b):
    aw,av,bw,bv=a[...,:1],a[...,1:],b[...,:1],b[...,1:]
    return torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.linalg.cross(av,bv)),dim=-1)


def rotate_yaw(vector,yaw):
    while yaw.ndim < vector.ndim-1:
        yaw=yaw.unsqueeze(-1)
    c,s=torch.cos(yaw),torch.sin(yaw)
    return torch.stack((c*vector[...,0]-s*vector[...,1],s*vector[...,0]+c*vector[...,1],vector[...,2]),dim=-1)


class VelocityGaitReference:
    def __init__(self,arrays,count,limits,device='cpu',acceleration=2.0):
        self.device=torch.device(device)
        self.arrays={key:torch.as_tensor(arrays[key],dtype=torch.float32,device=device) for key in
                     ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
        self.fps=float(torch.as_tensor(arrays['fps']).reshape(-1)[0])
        n=int(torch.as_tensor(arrays['cycle_frames']).reshape(-1)[0])
        self.nominal_speed=float(torch.as_tensor(arrays['nominal_reference_speed_mps']).reshape(-1)[0])
        if self.nominal_speed <= 0 or not torch.isfinite(torch.tensor(self.nominal_speed)):
            raise ValueError('Invalid nominal gait speed')
        if any(len(v)!=n or not torch.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError('Invalid periodic gait tensors')
        self.clock=CyclicSupportPhaseClock(count,self.fps,n,arrays['support_frames'],device,acceleration)
        self.root=VelocityReferenceRoot(count,limits,device)
        self.raw_command=torch.zeros(count,3,device=device)
        self.bounded_command=torch.zeros_like(self.raw_command)
        self.effective_command=torch.zeros_like(self.raw_command)
        self.enabled=torch.zeros(count,dtype=torch.bool,device=device)
        self.saturated=torch.zeros(count,3,dtype=torch.bool,device=device)
        self.unit_twist=torch.zeros_like(self.raw_command)

    def reset(self,ids,support_indices,position,yaw):
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.device)
        self.clock.reset(ids,support_indices)
        self.root.reset(ids,position,yaw)
        for value in [self.raw_command,self.bounded_command,self.effective_command,self.unit_twist]:
            value[ids]=0
        self.enabled[ids]=False
        self.saturated[ids]=False

    def request(self,command,walk_enabled):
        # Reuse the exact command-boundary validation; root receives a separate
        # effective/braked twist when step() integrates the reference.
        packet=torch.as_tensor(command,dtype=torch.float32,device=self.device)
        enabled=torch.as_tensor(walk_enabled,dtype=torch.bool,device=self.device)
        if packet.shape != self.raw_command.shape or enabled.shape != self.enabled.shape:
            raise ValueError('Command packet shape mismatch')
        if not torch.isfinite(packet).all():
            raise ValueError('Non-finite command')
        if (packet[:,0] < 0).any() or (packet[:,1] != 0).any():
            raise ValueError('F1 first version excludes backward and lateral walking')
        bounded=torch.minimum(torch.maximum(packet,self.root.limits[:,0]),self.root.limits[:,1])
        if (enabled & (bounded[:,:2].norm(dim=-1)==0) & (bounded[:,2]!=0)).any():
            raise ValueError('In-place turning requires a separately validated gait')
        self.root.request(packet,enabled)
        self.raw_command.copy_(self.root.raw_command)
        self.bounded_command.copy_(self.root.command)
        self.enabled.copy_(self.root.walk_enabled)
        self.saturated.copy_(self.root.saturated)
        self.root.command.copy_(self.effective_command)

    def initialize_rate(self,ids):
        """Only for genuine moving episode initialization, never command changes."""
        rate=self.bounded_command[:,:2].norm(dim=-1)/self.nominal_speed
        self.clock.rate[ids]=rate[ids]
        self.clock.was_go[ids]=rate[ids]>0
        active=rate>0
        self.unit_twist[active]=self.bounded_command[active]/rate[active,None]
        self.effective_command[ids]=self.unit_twist[ids]*rate[ids,None]
        self.root.command[ids]=self.effective_command[ids]

    def step(self,dt):
        rate=self.bounded_command[:,:2].norm(dim=-1)/self.nominal_speed
        active=rate>0
        if ((~active)&(self.bounded_command[:,2]!=0)).any():
            raise ValueError('In-place turning requires a separately validated gait; not supported by this template')
        self.unit_twist[active]=self.bounded_command[active]/rate[active,None]
        old_phase=self.clock.phase.clone()
        old_cycles=self.clock.cycles.clone()
        self.clock.step(rate,dt)
        # Integrate trapezoidal phase advance using the SAME actual phase
        # increment, including the final support snap, rather than extrapolate.
        advance=(self.clock.cycles-old_cycles)*self.clock.period+self.clock.phase-old_phase
        mean_rate=advance/(self.fps*dt)
        # Final support approach can cap phase advance, so get exact distance
        # from phase state instead of imposing a lower bound on root velocity.
        effective=self.unit_twist*self.clock.rate[:,None]
        self.root.command.copy_(self.unit_twist*mean_rate[:,None])
        self.root.step(dt)
        self.effective_command.copy_(effective)
        self.root.command.copy_(effective)

    def sample(self,key,quaternion=False):
        data=self.arrays[key]
        phase=self.clock.phase
        lo=phase.floor().long()
        hi=(lo+1)%self.clock.period
        weight=(phase-lo).reshape((len(phase),)+(1,)*(data.ndim-1))
        left,right=data[lo],data[hi]
        if quaternion:
            right=torch.where((left*right).sum(-1,keepdim=True)<0,-right,right)
        value=torch.lerp(left,right,weight)
        return value/value.norm(dim=-1,keepdim=True).clamp_min(1e-8) if quaternion else value

    @property
    def joint_pos(self):
        return self.sample('joint_pos')

    @property
    def joint_vel(self):
        return self.sample('joint_vel')*self.clock.rate[:,None]

    @property
    def body_pos_w(self):
        local=rotate_yaw(self.sample('body_pos_w'),self.root.yaw)
        offset=torch.nn.functional.pad(self.root.position,(0,1))
        return local+offset[:,None,:]

    @property
    def body_quat_w(self):
        yaw=self.root.yaw*.5
        rotation=torch.stack((torch.cos(yaw),torch.zeros_like(yaw),torch.zeros_like(yaw),torch.sin(yaw)),dim=-1)
        local=self.sample('body_quat_w',quaternion=True)
        return quat_multiply(rotation[:,None,:].expand_as(local),local)

    @property
    def body_lin_vel_w(self):
        local=rotate_yaw(self.sample('body_pos_w'),self.root.yaw)
        velocity=rotate_yaw(self.sample('body_lin_vel_w'),self.root.yaw)*self.clock.rate[:,None,None]
        root_v=torch.nn.functional.pad(self.root.velocity_world,(0,1))
        omega=torch.zeros_like(root_v);omega[:,2]=self.effective_command[:,2]
        return velocity+root_v[:,None,:]+torch.linalg.cross(omega[:,None,:].expand_as(local),local)

    @property
    def body_ang_vel_w(self):
        velocity=rotate_yaw(self.sample('body_ang_vel_w'),self.root.yaw)*self.clock.rate[:,None,None]
        omega=torch.zeros_like(self.effective_command);omega[:,2]=self.effective_command[:,2]
        return velocity+omega[:,None,:]
