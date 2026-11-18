"""Passive contact-point and exact collision-support telemetry; never steps physics."""
import math
import numpy as np
import mujoco


def rotation(q):
    out=np.empty(9)
    mujoco.mju_quat2Mat(out,np.asarray(q,dtype=np.float64))
    return out.reshape(3,3)


def lowest_collision_z(model,gids,body_position,body_rotation):
    values=[]
    for g in gids:
        pos=body_position+body_rotation@model.geom_pos[g]
        rot=body_rotation@rotation(model.geom_quat[g])
        size=model.geom_size[g];kind=model.geom_type[g]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
            z=pos[2]+np.min(model.mesh_vert[start:start+count]@rot[2])
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:z=pos[2]-np.abs(rot[2])@size
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:z=pos[2]-size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:z=pos[2]-size[0]-size[1]*abs(rot[2,2])
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            z=pos[2]-size[0]*math.hypot(rot[2,0],rot[2,1])-size[1]*abs(rot[2,2])
        elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:z=pos[2]-np.linalg.norm(size*rot[2])
        else:raise ValueError('Unsupported exact collision support type: '+str(kind))
        values.append(float(z))
    if not values:raise ValueError('Foot has no collision geometry; refuse invented clearance')
    return min(values)


class ContactTelemetry:
    def __init__(self,model,feet,body_names):
        self.model=model;self.floor=model.geom('floor').id
        self.items=[]
        for side,name in zip(('left','right'),feet):
            bid=model.body(name).id
            gids=[g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and
                  (model.geom_contype[g] or model.geom_conaffinity[g])]
            if not gids:raise ValueError('Missing foot collision geometry: '+name)
            self.items.append((side,bid,body_names.index(name),gids))

    def sample(self,data,reference):
        m=self.model;out={}
        for side,bid,rid,gids in self.items:
            values=[]
            for i in range(data.ncon):
                c=data.contact[i]
                if not ((c.geom1==self.floor and m.geom_bodyid[c.geom2]==bid) or
                        (c.geom2==self.floor and m.geom_bodyid[c.geom1]==bid)):continue
                force=np.zeros(6);mujoco.mj_contactForce(m,data,i,force)
                jac=np.zeros((3,m.nv));jr=np.zeros_like(jac)
                mujoco.mj_jac(m,data,jac,jr,c.pos,bid)
                velocity=jac@data.qvel;normal=c.frame.reshape(3,3)[0]
                tangent=velocity-normal*np.dot(velocity,normal)
                values.append((abs(float(force[0])),velocity,tangent))
            weight=sum(v[0] for v in values)
            avg=sum((w*v for w,v,t in values),np.zeros(3))/weight if weight>0 else np.zeros(3)
            rms=math.sqrt(sum(w*np.dot(t,t) for w,v,t in values)/weight) if weight>0 else 0.
            out.update({side+':contact_point_normal_force_N':weight,
                side+':contact_point_tangent_rms_mps':rms,
                side+':contact_point_tangent_max_mps':max((float(np.linalg.norm(t)) for w,v,t in values if w>0),default=0.),
                side+':collision_lowest_z_m':lowest_collision_z(m,gids,data.xpos[bid],data.xmat[bid].reshape(3,3)),
                side+':reference_collision_lowest_z_m':lowest_collision_z(m,gids,reference.body_pos_w[0,rid].numpy(),rotation(reference.body_quat_w[0,rid].numpy()))})
            for axis,value in zip('xyz',avg):out[side+':contact_point_v'+axis+'_mps']=float(value)
            for axis,value in zip('xyz',reference.body_lin_vel_w[0,rid].numpy()):out[side+':reference_body_v'+axis+'_mps']=float(value)
        return out
