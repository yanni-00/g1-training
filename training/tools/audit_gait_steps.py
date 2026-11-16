"""Measure actual periodic foot motion in world and camera-following coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np

from build_ssot_retarget import TRAINING, sha, write_json
from motion_kinematics import F1Kinematics
from repair_walking_motion import load_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior audit')
    arrays, q, names, fps = load_motion(args.input)
    if 'cycle_frames' not in arrays:
        raise ValueError('This diagnostic requires a periodic reference')
    if str(arrays.get('reference_space', '')) != 'cyclic_heading_stride_local_v1':
        raise ValueError('This diagnostic requires the constant-stride local coordinate marker')
    contract = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    settings = json.loads((TRAINING/'configs/ssot_motion_processing.json').read_text(encoding='utf-8'))
    kin = F1Kinematics(TRAINING/contract['urdf'])
    feet = kin.world_feet(q, names)
    n = len(q)
    nominal = float(arrays['nominal_reference_speed_mps'][0])
    time = np.arange(n)/fps
    progress = np.column_stack([nominal*time, np.zeros((n, 2))])
    world_root = q[:, :3]+progress
    tiled_root = np.tile(q[:, :3], (3, 1))
    tiled_root[:, 0] += nominal*np.arange(3*n)/fps
    root_velocity = np.gradient(tiled_root, 1/fps, axis=0, edge_order=2)[n:2*n]
    vertical = np.tile([0., 0., 1.], (n, 1))
    heights = kin.mesh_projections(feet, vertical)
    report = {'input': str(args.input.resolve()), 'input_sha256': sha(args.input),
        'auditor_sha256': sha(Path(__file__)),
        'source': 'qpos/fps/contact_left/contact_right/nominal_reference_speed_mps and SSOT foot collision vertices',
        'coordinate_conversion': 'world = cyclic_heading_stride_local_v1 + nominal_speed * time along X',
        'urdf_sha256': contract['urdf_sha256'], 'fps': fps, 'cycle_unique_frames': n,
        'cycle_period_s': n/fps, 'root_forward_advance_per_cycle_m': nominal*n/fps,
        'nominal_speed_mps': nominal, 'feet': {},
        'contact_labels_are_measured_force': False, 'dynamic_feasibility_verified': False}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout='constrained')
    axes[0].plot(time, world_root[:, 0], color='gray', label='pelvis world X')
    for side, color in (('left', '#2878b8'), ('right', '#df7c26')):
        local = feet[side]['center']
        world = local+progress
        contact = arrays['contact_'+side].astype(bool)
        tiled = np.tile(local, (3, 1))
        tiled[:, 0] += nominal*np.arange(3*n)/fps
        world_velocity = np.gradient(tiled, 1/fps, axis=0, edge_order=2)[n:2*n]
        speed = np.linalg.norm(world_velocity[:, :2], axis=1)
        minimum = heights[side][0]
        unwrapped = np.tile(contact, 3)
        starts = np.flatnonzero(unwrapped & ~np.r_[False, unwrapped[:-1]])
        segments = []
        for start in starts:
            if not n <= start < 2*n:
                continue
            end = int(start)+1
            while end < 3*n and unwrapped[end]:
                end += 1
            points = tiled[start:end]
            segments.append({'start_frame_mod_cycle': int(start % n), 'end_frame_exclusive_unwrapped': int(end-n),
                'sample_count': end-int(start),
                'max_center_planar_displacement_from_first_sample_m': float(np.linalg.norm(points[:, :2]-points[0, :2], axis=1).max())})
        near_ground = (~contact) & (minimum <= settings['cycle_contact_height_tolerance_m'])
        scuff = near_ground & (speed > settings['cycle_stance_planar_speed_tolerance_mps'])
        report['feet'][side] = {
            'contact_frame_count': int(contact.sum()), 'swing_frame_count': int((~contact).sum()),
            'maximum_sole_lowest_point_height_m': float(minimum.max()),
            'swing_sole_lowest_point_median_height_m': float(np.median(minimum[~contact])),
            'world_stance_center_planar_speed_max_mps': float(speed[contact].max()),
            'world_stance_center_planar_speed_median_mps': float(np.median(speed[contact])),
            'camera_following_stance_center_speed_median_mps': float(np.median(np.linalg.norm((world_velocity-root_velocity)[:, :2], axis=1)[contact])),
            'swing_near_ground_moving_frame_ids': np.flatnonzero(scuff).tolist(),
            'threshold_source': 'ssot_motion_processing.json: cycle_contact_height_tolerance_m / cycle_stance_planar_speed_tolerance_mps',
            'planted_segments': segments,
            'per_frame': {'sole_min_world_z_m': minimum.tolist(), 'world_center_x_m': world[:, 0].tolist(),
                'world_center_planar_speed_mps': speed.tolist(), 'contact': contact.tolist()}}
        axes[0].plot(time, world[:, 0], color=color, label=side+' foot world X')
        axes[0].scatter(time[contact], world[contact, 0], color=color, s=7)
        axes[1].plot(time, minimum*100, color=color, label=side+' lowest sole point')
        axes[2].plot(time, speed, color=color, label=side+' world foot speed')
        axes[2].scatter(time[contact], speed[contact], color=color, s=7)
    axes[0].set_ylabel('Forward position (m)')
    axes[1].set_ylabel('Sole clearance (cm)')
    axes[2].set_ylabel('Planar speed (m/s)')
    axes[2].set_xlabel('Source time within cycle (s)')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(loc='upper left', fontsize=8)
    fig.suptitle('Actual saved cycle / world coordinates; dots = contact-labelled frames')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'feet'}))
    print(json.dumps({s: {k: v for k, v in data.items() if k != 'per_frame'} for s, data in report['feet'].items()}))


if __name__ == '__main__':
    main()
