"""Native full-clip FK export, collision audit and periodic walking reference."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from build_ssot_retarget import TRAINING, native_model, sha, write_json
from motion_kinematics import F1Kinematics, quat_rotation, rot_z
from refine_ssot_motion import runs
from repair_walking_motion import audit, load_motion
from optimize_periodic_motion import optimize_cycle


def native_fk(model, q, names):
    data = mujoco.MjData(model)
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    addresses = np.array([model.joint(name).qposadr[0] for name in names])
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    position, quaternion, collisions = [], [], {}
    for frame, row in enumerate(q):
        data.qpos[:7] = row[:7]
        data.qpos[addresses] = row[7:]
        mujoco.mj_forward(model, data)
        position.append(data.xpos[1:].copy())
        quaternion.append(data.xquat[1:].copy())
        for contact in data.contact[:data.ncon]:
            if contact.dist >= -1e-5:
                continue
            bodies = [int(model.geom_bodyid[contact.geom[i]]) for i in (0, 1)]
            if 0 in bodies:
                continue
            key = ' / '.join(sorted(model.body(i).name for i in bodies))
            entry = collisions.setdefault(key, {'frames': set(), 'minimum_distance_m': 0.})
            entry['frames'].add(frame)
            entry['minimum_distance_m'] = min(entry['minimum_distance_m'], float(contact.dist))
    collisions = {key: {'frames_0based': sorted(v['frames']), 'minimum_distance_m': v['minimum_distance_m']}
                  for key, v in collisions.items()}
    return {'joint_names': np.array(joint_names), 'body_names': np.array(body_names),
            'joint_pos': q[:, 7:][:, [names.index(name) for name in joint_names]],
            'body_pos_w': np.array(position), 'body_quat_w': np.array(quaternion)}, collisions


def derivatives(a, fps, periodic=False):
    if periodic:
        return (np.roll(a, -1, axis=0)-np.roll(a, 1, axis=0))*(fps/2)
    return np.gradient(a, 1/fps, axis=0, edge_order=2)


def angular_velocity(quaternion, fps, periodic=False):
    before = np.roll(quaternion, 1, axis=0)
    after = np.roll(quaternion, -1, axis=0)
    if not periodic:
        before[0], after[-1] = quaternion[0], quaternion[-1]
    delta = Rotation.from_quat(after.reshape(-1, 4)[:, [1, 2, 3, 0]]) * Rotation.from_quat(before.reshape(-1, 4)[:, [1, 2, 3, 0]]).inv()
    result = delta.as_rotvec().reshape(quaternion.shape[:-1]+(3,))*(fps/2)
    if not periodic:
        result[[0, -1]] *= 2
    return result


def add_velocities(arrays, fps, periodic=False):
    arrays['joint_vel'] = derivatives(arrays['joint_pos'], fps, periodic)
    arrays['body_lin_vel_w'] = derivatives(arrays['body_pos_w'], fps, periodic)
    arrays['body_ang_vel_w'] = angular_velocity(arrays['body_quat_w'], fps, periodic)


def native_audit(model, kin, q, names, fps, contact, nominal=0., periodic=False):
    values, collisions = native_fk(model, q, names)
    add_velocities(values, fps, periodic)
    basic, detail = audit(kin, q, names, fps)
    feet = detail['feet']
    foot_speed = {}
    for side in ('left', 'right'):
        speed = derivatives(feet[side]['center'], fps, periodic)
        speed[:, 0] += nominal
        mask = contact[side]
        interior = mask & np.roll(mask, 1) & np.roll(mask, -1)
        if not periodic:
            interior[[0, -1]] = False
        speed = np.linalg.norm(speed[:, :2], axis=1)
        foot_speed[side] = {'maximum_mps': float(speed[mask].max()) if mask.any() else None,
                            'median_mps': float(np.median(speed[mask])) if mask.any() else None,
                            'maximum_interior_mps': float(speed[interior].max()) if interior.any() else None}
    return values, {'kinematics': basic, 'native_self_collision_pairs': collisions,
                    'stance_foot_center_planar_speed': foot_speed,
                    'contact_label_is_measured_force': False,
                    'collision_representation': 'MuJoCo convex collision meshes from SSOT collision geometry; not exact concave triangle intersections'}


def save_arrays(path, values):
    clean = {key: value.astype(np.float32) if isinstance(value, np.ndarray) and value.dtype.kind == 'f' else value
             for key, value in values.items()}
    np.savez_compressed(path, **clean)


def cycle_candidates(q, names, contacts, fps):
    starts = [a for a, _ in runs(contacts['left'])]
    candidates = []
    for a, b in zip(starts[:-1], starts[1:]):
        if not contacts['right'][a:b].any():
            continue
        double = contacts['left'][a:b] & contacts['right'][a:b]
        if len(runs(double)) < 2:
            continue
        score = float(np.linalg.norm(q[a, 7:]-q[b, 7:]))
        candidates.append({'start': a, 'end': b, 'source_period_s': (b-a)/fps,
                           'joint_endpoint_norm_rad': score})
    return sorted(candidates, key=lambda c: c['joint_endpoint_norm_rad'])


def periodic_pose(q, start, end, source_fps, target_fps):
    clip = q[start:end+1].copy()
    displacement = clip[-1, :2]-clip[0, :2]
    stride = float(np.linalg.norm(displacement))
    heading = float(np.arctan2(displacement[1], displacement[0]))
    rotation = rot_z(np.array([-heading]))[0]
    phase = np.linspace(0., 1., len(clip))
    local = (clip[:, :3]-np.r_[clip[0, :2], 0.]) @ rotation.T
    # Keep periodic pelvis sway/speed variation around a constant command path.
    local[:, 0] -= phase*stride
    root_r = rotation @ quat_rotation(clip[:, 3:7])
    vectors = Rotation.from_matrix(root_r).as_rotvec()
    state = np.column_stack([local, vectors, clip[:, 7:]])
    mean = (state[0]+state[-1])/2
    before = state.copy()
    blend_frames = max(2, (len(state)-1)//5)
    weight = np.linspace(1., 0., blend_frames)
    weight = weight*weight*(3-2*weight)
    state[:blend_frames] = (1-weight[:, None])*state[:blend_frames]+weight[:, None]*mean
    state[-blend_frames:] = (1-weight[::-1, None])*state[-blend_frames:]+weight[::-1, None]*mean
    state[-1] = state[0]
    n = int(round((end-start)/source_fps*target_fps))
    periodic_phase = np.r_[phase[:-1]-1, phase[:-1], phase+1]
    periodic_state = np.concatenate([state[:-1], state[:-1], state], axis=0)
    sampled = PchipInterpolator(periodic_phase, periodic_state, axis=0)(np.arange(n)/n)
    xyzw = Rotation.from_rotvec(sampled[:, 3:6]).as_quat()
    output = np.column_stack([sampled[:, :3], xyzw[:, [3, 0, 1, 2]], sampled[:, 6:]])
    return output, stride/(n/target_fps), {'unique_frames': n, 'source_heading_rad': heading,
        'measured_stride_m': stride, 'resampled_period_s': n/target_fps,
        'maximum_joint_closure_adjustment_rad': float(np.abs(state[:, 6:]-before[:, 6:]).max()),
        'root_space': 'heading-aligned; subtract constant forward stride, retain periodic pelvis XY residual'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cycle', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Use a new export directory')
    out.mkdir(parents=True)
    arrays, q, names, fps = load_motion(args.input)
    if not bool(arrays.get('kinematic_checks_passed', False)):
        raise ValueError('Refinement must pass before native export')
    contract = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    kin = F1Kinematics(TRAINING/contract['urdf'])
    model = native_model(args.model.resolve())
    contacts = {s: arrays['contact_'+s].astype(bool) for s in ('left', 'right')}
    full, full_audit = native_audit(model, kin, q, names, fps, contacts)
    full.update(fps=np.array([fps]), qpos=q, robot_joint_names=np.array(names),
                reference_space=np.array('world_clip_v1'), urdf_sha256=np.array(contract['urdf_sha256']),
                source_npz_sha256=np.array(sha(args.input)), contact_left=contacts['left'], contact_right=contacts['right'],
                training_ready=np.array(False))
    save_arrays(out/'full_clip.npz', full)
    report = {'source': str(args.input.resolve()), 'source_sha256': sha(args.input),
              'native_model_sha256': sha(args.model), 'urdf_sha256': contract['urdf_sha256'],
              'full_clip': full_audit, 'training_ready': False, 'native_engine': 'MuJoCo '+mujoco.__version__,
              'isaac_order_verified': False, 'dynamic_feasibility_verified': False}
    if args.cycle:
        settings = json.loads((TRAINING/'configs/ssot_motion_processing.json').read_text(encoding='utf-8'))
        stable = json.loads((TRAINING/'configs/reference/stable_runtime_settings.json').read_text(encoding='utf-8'))
        target_fps = 1/stable['policy_dt']
        candidates = cycle_candidates(q, names, contacts, fps)
        if not candidates:
            raise ValueError('No complete cycle containing both support intervals')
        reports = []
        best = None
        for candidate in candidates:
            a, b = candidate['start'], candidate['end']
            periodic, nominal, transform = periodic_pose(q, a, b, fps, target_fps)
            n = len(periodic)
            selection = a+(np.arange(n)*(b-a)/n).round().astype(int)
            contact = {s: contacts[s][selection] for s in ('left', 'right')}
            periodic, contact, optimization = optimize_cycle(kin, periodic, names, contact, target_fps, nominal, settings)
            # A common periodic root lift removes tiny spline corner penetration.
            feet = kin.world_feet(periodic, names)
            extents = kin.mesh_projections(feet, np.tile([0., 0., 1.], (n, 1)))
            minimum = np.minimum(extents['left'][0], extents['right'][0])
            lift = np.maximum(0., -minimum+settings['numeric_sole_clearance_m'])
            smooth = gaussian_filter1d(lift, 1., mode='wrap')
            smooth += max(0., float((lift-smooth).max()))
            periodic[:, 2] += smooth
            values, checks = native_audit(model, kin, periodic, names, target_fps, contact, nominal, True)
            source_support = contact['left'] & contact['right']
            feet = kin.world_feet(periodic, names)
            planar = []
            for side in ('left', 'right'):
                v = derivatives(feet[side]['center'], target_fps, True)
                v[:, 0] += nominal
                planar.append(np.linalg.norm(v[:, :2], axis=1))
            support_speed = np.maximum(*planar)
            support_mask = source_support & (support_speed <= settings['cycle_stance_planar_speed_tolerance_mps'])
            support_list = []
            for start in np.flatnonzero(source_support & ~np.roll(source_support, 1)):
                interval = [int(start)]
                while len(interval) < n and source_support[(interval[-1]+1) % n]:
                    interval.append((interval[-1]+1) % n)
                eligible = [i for i in interval if support_mask[i]]
                if eligible:
                    support_list.append(min(eligible, key=lambda i: support_speed[i]))
            support = np.array(sorted(support_list), dtype=np.int64)
            checks['source_support_frames'] = support.tolist()
            checks['periodic_root_lift_max_m'] = float(smooth.max())
            checks['source_interval'] = candidate
            checks['transform'] = transform
            checks['periodic_optimization'] = optimization
            bounds = checks['kinematics']
            valid = (len(support) >= 2 and not bounds['joint_position_limit_violations'] and not bounds['joint_velocity_limit_violations']
                     and min(bounds['sole_min_world_z_m'][s]['min'] for s in ('left', 'right')) >= -1e-6)
            speed = max(checks['stance_foot_center_planar_speed'][s]['maximum_mps'] or 0. for s in ('left', 'right'))
            score = (not valid, len(bounds['joint_velocity_limit_violations']), speed)
            checks['kinematic_format_checks_passed'] = valid
            reports.append(checks)
            if best is None or score < best[0]:
                best = (score, periodic, nominal, values, support, checks, contact)
            if valid and max(checks['stance_foot_center_planar_speed'][s]['maximum_interior_mps'] or 0. for s in ('left', 'right')) <= settings['cycle_stance_planar_speed_tolerance_mps']:
                break
        _, periodic, nominal, values, support, checks, contact = best
        values.update(fps=np.array([target_fps]), qpos=periodic, robot_joint_names=np.array(names),
            cycle_frames=np.array([len(periodic)], dtype=np.int64), support_frames=support,
            nominal_reference_speed_mps=np.array([nominal]), urdf_sha256=np.array(contract['urdf_sha256']),
            reference_space=np.array('cyclic_heading_stride_local_v1'),
            contact_left=contact['left'], contact_right=contact['right'],
            source_npz_sha256=np.array(sha(args.input)), training_ready=np.array(False))
        save_arrays(out/'walking_cycle.npz', values)
        report.update(cycle_candidates=reports, selected_cycle=checks,
                      cycle_reference_sha256=sha(out/'walking_cycle.npz'),
                      reference_space='cyclic_heading_stride_local_v1')
    write_json(out/'native_audit.json', report)
    print(json.dumps({'output': str(out), 'full_self_collision_pair_count': len(full_audit['native_self_collision_pairs']),
                      'selected_cycle': report.get('selected_cycle', {}).get('kinematic_format_checks_passed')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
