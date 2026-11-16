"""[OMA] Command-root/cyclic human-reference extension of native BeyondMimic."""
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, sample_uniform, yaw_quat
from .commands import MotionCommand, MotionCommandCfg
from f1_beyondmimic.utils.velocity_gait_reference import VelocityGaitReference


class VelocityGaitCommand(MotionCommand):
    def __init__(self,cfg,env):
        # [OMA] F1: dimension equality cannot establish joint/body correspondence.
        from f1_beyondmimic.contract import validate_motion, reorder_motion_arrays
        prepared = validate_motion(cfg.motion_file)
        robot = env.scene[cfg.asset_name]
        reorder_motion_arrays(prepared, robot.joint_names, robot.body_names)
        if not (0 <= cfg.initial_go_probability <= 1 and 0 < cfg.speed_scale_range[0] <= cfg.speed_scale_range[1]):
            raise ValueError('Invalid command curriculum settings')
        if not (0 <= cfg.straight_probability <= 1 and
                np.isfinite(cfg.yaw_rate_range).all() and cfg.yaw_rate_range[0] <= 0 <= cfg.yaw_rate_range[1]):
            raise ValueError('Invalid walking-turn curriculum')
        super().__init__(cfg,env)
        with np.load(cfg.motion_file,allow_pickle=False) as data:
            if str(data['reference_space']) not in {'cyclic_pelvis_xy_local_v1', 'cyclic_heading_stride_local_v1'}:
                raise ValueError('Requires freshly recomputed native cyclic local FK')
            arrays={key:data[key].copy() for key in data.files}
        arrays = reorder_motion_arrays(arrays, robot.joint_names, robot.body_names)
        nominal=float(arrays['nominal_reference_speed_mps'][0])
        self.command_limits=((0.0,nominal*cfg.speed_scale_range[1]),(0.0,0.0),tuple(cfg.yaw_rate_range))
        self.reference=VelocityGaitReference(arrays,self.num_envs,self.command_limits,self.device,cfg.phase_acceleration)
        self.clock=self.reference.clock
        if abs(self.reference.fps*env.cfg.sim.dt*env.cfg.decimation-1) > 1e-5:
            raise ValueError('Reference FPS must equal policy rate')
        self.user_command=torch.zeros(self.num_envs,3,device=self.device)
        self.walk_enabled=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.desired_go=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.mode_time=torch.zeros(self.num_envs,device=self.device)
        self.manual_command=False
        for key in ['requested_vx','requested_vy','requested_yaw_rate','effective_vx',
                    'reference_rate','reference_cycles','actual_planar_speed','command_speed_error',
                    'effective_yaw_rate','actual_yaw_rate','command_yaw_rate_error']:
            self.metrics[key]=torch.zeros(self.num_envs,device=self.device)

    def request(self,velocity,walk_enabled=True):
        """Physical body vx/vy/yaw_rate; unsupported axes explicitly rejected."""
        self.manual_command=True
        value=torch.as_tensor(velocity,dtype=torch.float32,device=self.device)
        if value.shape == (3,):
            value=value.expand(self.num_envs,3)
        enabled=torch.as_tensor(walk_enabled,dtype=torch.bool,device=self.device).expand(self.num_envs)
        self.reference.request(value,enabled)
        self.user_command.copy_(value)
        self.walk_enabled.copy_(enabled)
        self.desired_go.copy_(enabled & (value[:,:2].norm(dim=-1)>0))

    @property
    def joint_pos(self): return self.reference.joint_pos
    @property
    def joint_vel(self): return self.reference.joint_vel
    @property
    def body_pos_w(self): return self.reference.body_pos_w[:,self.body_indexes]
    @property
    def body_quat_w(self): return self.reference.body_quat_w[:,self.body_indexes]
    @property
    def body_lin_vel_w(self): return self.reference.body_lin_vel_w[:,self.body_indexes]
    @property
    def body_ang_vel_w(self): return self.reference.body_ang_vel_w[:,self.body_indexes]
    @property
    def anchor_pos_w(self): return self.body_pos_w[:,self.motion_anchor_body_index]
    @property
    def anchor_quat_w(self): return self.body_quat_w[:,self.motion_anchor_body_index]
    @property
    def anchor_lin_vel_w(self): return self.body_lin_vel_w[:,self.motion_anchor_body_index]
    @property
    def anchor_ang_vel_w(self): return self.body_ang_vel_w[:,self.motion_anchor_body_index]

    def _sample_mode(self,ids):
        moving=self.desired_go[ids]
        scales=sample_uniform(*self.cfg.speed_scale_range,(len(ids),),self.device)
        self.user_command[ids]=0
        self.user_command[ids,0]=moving*scales*self.reference.nominal_speed
        if self.cfg.yaw_rate_range != (0.0,0.0) and any(self.cfg.yaw_rate_range):
            yaw=sample_uniform(*self.cfg.yaw_rate_range,(len(ids),),self.device)
            straight=torch.rand(len(ids),device=self.device)<self.cfg.straight_probability
            self.user_command[ids,2]=torch.where(moving & ~straight,yaw,torch.zeros_like(yaw))
        go_time=sample_uniform(*self.cfg.go_duration_range,(len(ids),),self.device)
        stop_time=sample_uniform(*self.cfg.stop_duration_range,(len(ids),),self.device)
        self.mode_time[ids]=torch.where(moving,go_time,stop_time)

    def _adaptive_sampling(self,env_ids):
        # The inherited resampler writes robot state ONLY on genuine episode reset.
        ids=torch.as_tensor(env_ids,dtype=torch.long,device=self.device)
        support=torch.randint(len(self.clock.support),(len(ids),),device=self.device)
        if self.cfg.evaluation_start_support is not None:
            support[:]=self.cfg.evaluation_start_support
        self.reference.reset(ids,support,self._env.scene.env_origins[ids,:2],torch.zeros(len(ids),device=self.device))
        self.time_steps[ids]=self.clock.phase[ids].long()
        moving=torch.rand(len(ids),device=self.device)<self.cfg.initial_go_probability
        if self.cfg.evaluation_start_support is not None:
            moving.zero_()
        self.desired_go[ids]=moving
        self.walk_enabled[ids]=True
        self._sample_mode(ids)
        self.reference.request(self.user_command,self.walk_enabled)
        self.reference.initialize_rate(ids)
        self.metrics['sampling_entropy'][:]=1
        self.metrics['sampling_top1_prob'][:]=1/len(self.clock.support)
        self.metrics['sampling_top1_bin'][ids]=support.float()/len(self.clock.support)

    def _update_command(self):
        dt=self._env.cfg.sim.dt*self._env.cfg.decimation
        if not self.manual_command:
            self.mode_time-=dt
            ids=torch.where(self.mode_time<=0)[0]
            if len(ids):
                self.desired_go[ids]=~self.desired_go[ids]
                self._sample_mode(ids)
        self.reference.request(self.user_command,self.walk_enabled)
        self.reference.step(dt)
        self.time_steps.copy_(self.clock.phase.long())
        # Preserve native relative-body alignment, without any cyclic resampling
        # or simulator-state write at command changes or gait wrap.
        anchor_p=self.anchor_pos_w[:,None,:].expand(-1,len(self.cfg.body_names),-1)
        anchor_q=self.anchor_quat_w[:,None,:].expand(-1,len(self.cfg.body_names),-1)
        robot_p=self.robot_anchor_pos_w[:,None,:].repeat(1,len(self.cfg.body_names),1)
        robot_q=self.robot_anchor_quat_w[:,None,:].expand_as(anchor_q)
        robot_p[...,2]=anchor_p[...,2]
        rotation=yaw_quat(quat_mul(robot_q,quat_inv(anchor_q)))
        self.body_quat_relative_w=quat_mul(rotation,self.body_quat_w)
        self.body_pos_relative_w=robot_p+quat_apply(rotation,self.body_pos_w-anchor_p)

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics['requested_vx']=self.user_command[:,0].clone()
        self.metrics['requested_vy']=self.user_command[:,1].clone()
        self.metrics['requested_yaw_rate']=self.user_command[:,2].clone()
        self.metrics['effective_vx']=self.reference.effective_command[:,0].clone()
        self.metrics['reference_rate']=self.clock.rate.clone()
        self.metrics['reference_cycles']=self.clock.cycles.float()
        velocity=quat_apply(quat_inv(self.robot.data.root_quat_w),self.robot.data.root_lin_vel_w)
        self.metrics['actual_planar_speed']=velocity[:,:2].norm(dim=-1)
        # Raw user target is distinct from the support-braking reference target.
        target=torch.where(self.walk_enabled[:,None],self.user_command,0)
        self.metrics['command_speed_error']=(velocity[:,:2]-target[:,:2]).norm(dim=-1)
        yaw_rate=self.robot.data.root_ang_vel_w[:,2]
        self.metrics['effective_yaw_rate']=self.reference.effective_command[:,2].clone()
        self.metrics['actual_yaw_rate']=yaw_rate.clone()
        self.metrics['command_yaw_rate_error']=(yaw_rate-target[:,2]).abs()


@configclass
class VelocityGaitCommandCfg(MotionCommandCfg):
    class_type:type=VelocityGaitCommand
    evaluation_start_support:int|None=None
    initial_go_probability:float=.5
    phase_acceleration:float=2.0
    # Engineering curriculum, fractions of the selected human template speed;
    # not measured hardware limits. Forward-only first integration is partial.
    speed_scale_range:tuple[float,float]=(.5,1.0)
    yaw_rate_range:tuple[float,float]=(0.0,0.0)
    straight_probability:float=1.0
    go_duration_range:tuple[float,float]=(2.0,4.0)
    stop_duration_range:tuple[float,float]=(2.0,4.0)
