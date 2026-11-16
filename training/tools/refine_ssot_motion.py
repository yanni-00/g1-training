"""Contact-aware pose refinement of newly regenerated SSOT UMR trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation, Slerp

from build_ssot_retarget import TRAINING, sha, write_json
from motion_kinematics import F1Kinematics, SIDES, apply, orientation_angle, rot_z
from repair_walking_motion import audit, heading, load_motion, smooth_non_leg_velocity, wrap


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def source_contacts(joints, fps, settings):
    height = joints[:, [10, 11], 2]
    difference = gaussian_filter1d(height[:, 0]-height[:, 1], 2)
    peaks = find_peaks(difference, prominence=np.ptp(difference)*settings["source_peak_prominence_fraction"])[0]
    if len(peaks) < 2:
        # A stop/start clip may contain too few steps; use the whole actual clip
        # for local-floor inference instead of inventing a gait frequency.
        period = len(joints)
    else:
        period = int(np.median(np.diff(peaks)))
    window = max(3, period//2*2+1)
    floor = percentile_filter(height.min(axis=1), settings["source_floor_percentile"], size=window, mode="nearest")
    floor = gaussian_filter1d(floor, max(1., period/8))
    contacts = {}
    for side, index in (("left", 10), ("right", 11)):
        h = joints[:, index, 2]
        offset = float(np.percentile(h-floor, 5))
        speed = np.linalg.norm(np.gradient(joints[:, index, :2], 1/fps, axis=0, edge_order=2), axis=1)
        mask = (h-floor-offset <= settings["contact_height_above_local_floor_m"]) & (speed <= settings["contact_source_planar_speed_mps"])
        for start, end in runs(~mask):
            if start > 0 and end < len(mask) and (end-start)/fps <= settings["contact_merge_gap_s"]:
                mask[start:end] = True
        for start, end in runs(mask):
            if (end-start)/fps < settings["contact_min_duration_s"]:
                mask[start:end] = False
        contacts[side] = mask
    return contacts, floor-floor[0], {"source_left_swing_peaks_0based": peaks.tolist(),
        "inferred_period_frames": period, "floor_window_frames": window,
        "source_ground_drift_min_m": float((floor-floor[0]).min()),
        "source_ground_drift_max_m": float((floor-floor[0]).max()),
        "contact_runs_exclusive_end": {s: [[int(a), int(b)] for a, b in runs(contacts[s])] for s in SIDES},
        "contact_label_basis": "Source SMPL-X foot-joint height above local floor and planar speed; no measured contact forces"}


def foot_targets(model, q, names, contacts, settings):
    _, yaw, lateral = heading(q)
    feet = model.world_feet(q, names)
    reference_yaw = float(np.median(np.unwrap(yaw)))
    direction = np.array([-np.sin(reference_yaw), np.cos(reference_yaw), 0.])
    cosine = lateral @ direction
    if np.any(cosine <= 0):
        raise ValueError("Split large-turn clips before constant lateral widening")
    span = np.sum((feet['left']['center']-feet['right']['center'])*lateral, axis=1)
    low, high = 0., model.neutral_center_span
    for _ in range(60):
        mid = (low+high)/2
        if np.median(span+mid*cosine) < model.neutral_center_span:
            low = mid
        else:
            high = mid
    shift = 0. if np.median(span) >= model.neutral_center_span else high
    result = {}
    for side, sign in (("left", 1), ("right", -1)):
        foot = feet[side]
        centers = foot['center']+sign*shift/2*direction
        old_yaw = np.arctan2(foot['r'][:, 1, 0], foot['r'][:, 0, 0])
        rotations = rot_z(wrap(yaw-old_yaw)) @ foot['r']
        original_centers, original_rotations = centers.copy(), rotations.copy()
        contact_runs = runs(contacts[side])
        if not contact_runs:
            raise ValueError("No source contact interval for "+side)
        for a, b in contact_runs:
            anchor = np.median(centers[a:b], axis=0)
            anchor_r = rot_z(np.array([float(np.median(np.unwrap(yaw[a:b])))]))[0]
            relative = (model.vertices[side]-model.centers[side]) @ anchor_r.T
            anchor[2] = -relative[:, 2].min()+settings['numeric_sole_clearance_m']
            centers[a:b] = anchor
            rotations[a:b] = anchor_r
        # Carry measured source swing shape between fixed stance anchors.
        segments = [(0, contact_runs[0][0]+1)]
        segments += [(contact_runs[i][1]-1, contact_runs[i+1][0]+1) for i in range(len(contact_runs)-1)]
        segments += [(contact_runs[-1][1]-1, len(q))]
        for a, b in segments:
            if b-a < 2:
                continue
            if not contacts[side][a]:
                left = right = contact_runs[0][0]
            elif not contacts[side][b-1]:
                left = right = contact_runs[-1][1]-1
            else:
                left, right = a, b-1
            dl = centers[left]-original_centers[left]
            dr = centers[right]-original_centers[right]
            t = np.linspace(0., 1., b-a)
            blend = t*t*(3-2*t)
            centers[a:b] = original_centers[a:b]+(1-blend[:, None])*dl+blend[:, None]*dr
            delta_left = rotations[left] @ original_rotations[left].T
            delta_right = rotations[right] @ original_rotations[right].T
            interpolator = Slerp([0., .5, 1.], Rotation.from_matrix(np.stack([delta_left, np.eye(3), delta_right])))
            rotations[a:b] = interpolator(blend).as_matrix() @ original_rotations[a:b]
        # Keep every actual collision vertex at/above the chosen flat ground.
        relative = model.vertices[side]-model.centers[side]
        minimum = np.min(np.einsum('tij,vj->tvi', rotations, relative)[:, :, 2], axis=1)
        centers[:, 2] = np.maximum(centers[:, 2], -minimum+settings['numeric_sole_clearance_m'])
        result[side] = {'p': centers-apply(rotations, model.centers[side]), 'r': rotations, 'center': centers}
    return result, shift


def solve_targets(model, source, names, targets, settings, ik, fps):
    result = source.copy()
    base_r, _, _ = heading(source)
    ids = {s: np.array(model.leg_indices(s, names))+7 for s in SIDES}
    nominal = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    nominal_map = dict(zip(nominal['model_joint_names'], nominal['default_joint_pos']))
    nominal_legs = np.r_[[nominal_map[j['name']] for j in model.legs['left']],
                         [nominal_map[j['name']] for j in model.legs['right']]]
    source_x = np.column_stack([source[:, :3], source[:, ids['left']], source[:, ids['right']]])
    regularization = np.r_[np.full(3, settings['root_pose_regularization']),
                           np.full(12, settings['leg_pose_regularization'])]
    temporal = settings['temporal_regularization']
    previous = None
    step = np.r_[[joint['limit']['velocity'] for joint in model.legs['left']],
                 [joint['limit']['velocity'] for joint in model.legs['right']]]/fps*.9999
    iterations = []
    for frame in range(len(source)):
        prior = source_x[frame]
        lower = np.r_[prior[:3]-model.leg_length, model.lower['left'], model.lower['right']]
        upper = np.r_[prior[:3]+model.leg_length, model.upper['left'], model.upper['right']]
        if previous is None:
            initial = np.r_[prior[:3], nominal_legs]
            temporal_prior = prior
        else:
            temporal_prior = previous+(prior-source_x[frame-1])
            initial = temporal_prior.copy()
            lower[3:] = np.maximum(lower[3:], previous[3:]-step)
            upper[3:] = np.minimum(upper[3:], previous[3:]+step)

        def evaluate(x, jac=False):
            values = np.empty(12)
            derivative = np.zeros((12, 15)) if jac else None
            for side, start, row in (('left', 3, 0), ('right', 9, 6)):
                output = model.leg_fk(side, x[None, start:start+6], jacobian=jac)
                p, r = output[:2]
                world_p = x[:3]+base_r[frame] @ p[0]
                world_r = base_r[frame] @ r[0]
                phi = Rotation.from_matrix(targets[side]['r'][frame] @ world_r.T).as_rotvec()
                values[row:row+3] = world_p-targets[side]['p'][frame]
                values[row+3:row+6] = model.leg_length*phi
                if jac:
                    jp, jr = output[2:]
                    px, py, pz = phi
                    skew = np.array([[0., -pz, py], [pz, 0., -px], [-py, px, 0.]])
                    theta = float(np.linalg.norm(phi))
                    coefficient = 1/12 if theta < 1e-5 else (1-(theta/2)/np.tan(theta/2))/(theta*theta)
                    right_inverse = np.eye(3)+.5*skew+coefficient*(skew@skew)
                    derivative[row:row+3, :3] = np.eye(3)
                    derivative[row:row+3, start:start+6] = base_r[frame] @ jp[0]
                    derivative[row+3:row+6, start:start+6] = -model.leg_length*right_inverse @ base_r[frame] @ jr[0]
            if jac:
                return np.vstack([derivative, np.diag(regularization), np.eye(15)*temporal])
            return np.r_[values, (x-prior)*regularization, (x-temporal_prior)*temporal]

        solution = least_squares(evaluate, np.clip(initial, lower+1e-10, upper-1e-10),
            jac=lambda x: evaluate(x, True), bounds=(lower, upper),
            max_nfev=settings['joint_root_fallback_max_nfev'], ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if np.linalg.norm(evaluate(solution.x)[:12]) > ik['position_tolerance_m']:
            alternate_seed = np.r_[prior[:3], nominal_legs]
            alternate = least_squares(evaluate, np.clip(alternate_seed, lower+1e-10, upper-1e-10),
                jac=lambda x: evaluate(x, True), bounds=(lower, upper),
                max_nfev=settings['joint_root_fallback_max_nfev'], ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if np.linalg.norm(evaluate(alternate.x)[:12]) < np.linalg.norm(evaluate(solution.x)[:12]):
                solution = alternate
        previous = solution.x
        result[frame, :3] = previous[:3]
        result[frame, ids['left']] = previous[3:9]
        result[frame, ids['right']] = previous[9:15]
        iterations.append(int(solution.nfev))
        if frame % 100 == 0:
            print('[SSOT refine] joint/root frame', frame, '/', len(source),
                  'task residual', float(np.linalg.norm(evaluate(previous)[:12])), flush=True)
    actual = model.world_feet(result, names)
    failed = np.zeros(len(source), dtype=bool)
    states = {}
    for side in SIDES:
        pe = np.linalg.norm(actual[side]['p']-targets[side]['p'], axis=1)
        re = orientation_angle(actual[side]['r'], targets[side]['r'])
        failed |= (pe > settings['delivered_ik_position_tolerance_m']) | (re > ik['orientation_tolerance_rad'])
        states[side] = {'maximum_position_error_m': float(pe.max()),
                        'maximum_orientation_error_rad': float(re.max())}
    change = result[:, :3]-source[:, :3]
    return result, {'joint_root_optimization': 'joint root XYZ and both legs, with source-pose and temporal priors',
        'maximum_nfev': max(iterations),
        'root_height_change_min_m': float(change[:, 2].min()),
        'root_height_change_max_m': float(change[:, 2].max()),
        'maximum_root_xy_change_m': float(np.linalg.norm(change[:, :2], axis=1).max()),
        'position_check_numerical_tolerance_m': settings['delivered_ik_position_tolerance_m'],
        'failed_frames_0based': np.flatnonzero(failed).tolist(), 'ik': states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite motion')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings_path = TRAINING/'configs/ssot_motion_processing.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    repair_settings = json.loads((TRAINING/'configs/motion_repair.json').read_text(encoding='utf-8'))
    contract = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    model = F1Kinematics(TRAINING/contract['urdf'])
    arrays, q, names, fps = load_motion(args.input)
    with np.load(args.input.with_suffix('.source_ground_joints.npz')) as z:
        joints = z['joints_world']
    if len(joints) != len(q):
        raise ValueError('Source contact trajectory frame count mismatch')
    contacts, drift, source_info = source_contacts(joints, fps, settings)
    base = q.copy()
    base[:, 2] -= drift*float(arrays['smpl_scale'].reshape(-1)[0])
    targets, widening = foot_targets(model, base, names, contacts, settings)
    corrected, solver = solve_targets(model, base, names, targets, settings, repair_settings['ik'], fps)
    smoothing = smooth_non_leg_velocity(model, corrected, names, fps, repair_settings['non_leg_velocity_filter'])
    corrected = corrected.astype(np.float32)
    before, _ = audit(model, q, names, fps)
    after, detail = audit(model, corrected.astype(float), names, fps)
    stance = {}
    for side in SIDES:
        distances = []
        for a, b in runs(contacts[side]):
            center = detail['feet'][side]['center'][a:b]
            distances.extend(np.linalg.norm(center[:, :2]-np.median(center[:, :2], axis=0), axis=1).tolist())
        stance[side] = {'source_contact_frames': int(contacts[side].sum()),
                        'maximum_anchor_xy_deviation_m': max(distances, default=0.)}
    passed = (not solver['failed_frames_0based'] and not after['joint_position_limit_violations']
              and not after['joint_velocity_limit_violations']
              and after['lateral_projected_mesh_edge_clearance_m']['min'] >= 0
              and min(after['sole_min_world_z_m'][s]['min'] for s in SIDES) >= -repair_settings['numeric_only_clearance_tolerance_m'])
    report = {'source': str(args.input.resolve()), 'source_sha256': sha(args.input), 'urdf_sha256': contract['urdf_sha256'],
        'settings': settings, 'settings_sha256': sha(settings_path), 'source_contact': source_info,
        'total_constant_world_lateral_offset_m': widening, 'target_median_spacing_reference_m': model.neutral_center_span,
        'source_floor_correction_applied': True, 'solver': solver, 'non_leg_velocity_smoothing': smoothing,
        'before': before, 'after': after, 'stance': stance, 'kinematic_checks_passed': passed,
        'native_collision_verified': False, 'training_ready': False, 'tool_sha256': sha(Path(__file__))}
    output_arrays = {k: v.astype(str) if v.dtype.hasobject else v for k, v in arrays.items()}
    output_arrays.update(qpos=corrected, target_urdf_sha256=np.array(contract['urdf_sha256']),
                         source_npz_sha256=np.array(sha(args.input)), training_ready=np.array(False),
                         kinematic_checks_passed=np.array(passed), repair_info_json=np.array(json.dumps(report)),
                         contact_left=contacts['left'], contact_right=contacts['right'],
                         source_ground_drift_m=drift.astype(np.float32),
                         contact_label_basis=np.array(source_info['contact_label_basis']))
    np.savez_compressed(args.output, **output_arrays)
    report['output'] = str(args.output.resolve())
    report['output_sha256'] = sha(args.output)
    write_json(args.output.with_suffix('.audit.json'), report)
    print(json.dumps({'output': str(args.output), 'kinematic_passed': passed, 'solver': solver,
                      'position_limit_violations': len(after['joint_position_limit_violations']),
                      'velocity_limit_violations': after['joint_velocity_limit_violations'], 'stance': stance}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
