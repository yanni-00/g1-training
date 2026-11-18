"""Read-only URDF/inertia/initial-state audit. No integration or policy updates."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from mujoco_replay_diagnostic import (ROOT, load_checkpoint, load_contract, effective_control,
    load_json, sha256, matrix, VelocityGaitReference, torch, mujoco)
from mujoco_model import rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve previous audit')
    saved = load_checkpoint(a.checkpoint)
    runtime = saved['infos']['f1_runtime']
    cfg, controls = load_contract(), effective_control()
    prior = load_json(a.scene.parent / 'runtime_audit.json')
    if sha256(a.scene) != prior['scene_sha256']: raise ValueError('Scene changed')
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    protocol = load_json(a.protocol)
    ref = VelocityGaitReference(runtime['motion'], 1, runtime['command_limits'],
                                acceleration=runtime['phase_acceleration'])
    ref.reset(torch.tensor([0]), torch.tensor([protocol['initial_support_index']]), torch.zeros(1,2), torch.zeros(1))
    names = runtime['joint_names']
    joints = np.array([model.joint(n).id for n in names])
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    root_id = runtime['body_names'].index('base_link')
    data.qpos[:3] = ref.body_pos_w[0,root_id].numpy()
    data.qpos[3:7] = ref.body_quat_w[0,root_id].numpy()
    data.qpos[qids] = ref.joint_pos[0].numpy()
    data.qvel[:] = 0
    actuators = np.array([model.actuator(n).id for n in names])
    data.ctrl[actuators] = ref.joint_pos[0].numpy()
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    try: mujoco.mj_fullM(model, data, mass)
    except TypeError: mujoco.mj_fullM(model, mass, data.qM)
    inverse = np.linalg.inv(mass)
    source = ET.parse(ROOT / cfg['urdf']).getroot()
    bodies = []
    for link in source.findall('link'):
        inertial = link.find('inertial')
        if inertial is None: continue
        bid = model.body(link.get('name')).id
        inertia = inertial.find('inertia')
        tensor = np.array([[float(inertia.get(k)) for k in row] for row in
            (('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz'))])
        origin = inertial.find('origin')
        r = rotation([float(x) for x in origin.get('rpy','0 0 0').split()])
        tensor = r @ tensor @ r.T
        ri = matrix(model.body_iquat[bid])
        compiled = ri @ np.diag(model.body_inertia[bid]) @ ri.T
        eig = np.linalg.eigvalsh(tensor)
        bodies.append({'name':link.get('name'), 'source_mass':float(inertial.find('mass').get('value')),
            'compiled_mass':float(model.body_mass[bid]), 'source_tensor':tensor.tolist(),
            'compiled_tensor':compiled.tolist(), 'source_principal_inertia':eig.tolist(),
            'triangle_inequality_margin':float(eig[0]+eig[1]-eig[2]),
            'tensor_max_abs_error':float(np.max(np.abs(tensor-compiled)))})
    drives = []
    for n,j,q,v in zip(names,joints,qids,vids):
        act = model.actuator(n).id
        drives.append({'joint':n, 'source_velocity_limit':controls[n]['velocity_limit'],
            'source_effort_limit':controls[n]['effort_limit'], 'compiled_effort_range':model.actuator_forcerange[act].tolist(),
            'kp':controls[n]['stiffness'], 'pd_damping':controls[n]['pd_damping'],
            'passive_damping':float(model.dof_damping[v]), 'armature':float(model.dof_armature[v]),
            'initial_configuration_inverse_mass_diagonal':float(inverse[v,v]),
            'initial_configuration_free_effective_inertia':float(1/inverse[v,v]),
            'initial_qacc_with_reference_target':float(data.qacc[v]),
            'compiled_position_range':model.jnt_range[j].tolist()})
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        contacts.append({'geom1':model.geom(c.geom1).name, 'geom2':model.geom(c.geom2).name,
            'body1':model.body(model.geom_bodyid[c.geom1]).name,
            'body2':model.body(model.geom_bodyid[c.geom2]).name, 'distance':float(c.dist)})
    report = {'format':'f1_mujoco_readonly_dynamics_audit_v1', 'mujoco_version':mujoco.__version__,
        'checkpoint_sha256':sha256(a.checkpoint), 'urdf_sha256':cfg['urdf_sha256'],
        'scene_sha256':sha256(a.scene), 'script_sha256':sha256(Path(__file__)),
        'policy_dt':runtime['policy_dt'], 'physics_dt':float(model.opt.timestep),
        'integration_steps':0, 'optimizer_updates':0, 'bodies':bodies, 'joints':drives, 'contacts':contacts,
        'limits':{'velocity_limit_sim_mapping':'absent in original MuJoCo affine drive',
                  'effective_inertia':'Computed from this simulator mass matrix at the exact initial support pose; not hardware identification',
                  'initial_target':'Reference pose only; audit does not evaluate the actor or advance physics'}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'bodies':len(bodies),'contacts':len(contacts),
        'mass_max_abs_error':max(abs(b['source_mass']-b['compiled_mass']) for b in bodies),
        'inertia_max_abs_error':max(b['tensor_max_abs_error'] for b in bodies),
        'inertia_triangle_violations':[b['name'] for b in bodies if b['triangle_inequality_margin']<0]}))

if __name__ == '__main__': main()
