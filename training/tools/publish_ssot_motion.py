"""Independently verify delivered float32 reference bytes, then publish a copy."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from build_ssot_retarget import TRAINING, native_model, sha, write_json
from export_ssot_motion import native_audit, derivatives
from motion_kinematics import F1Kinematics

sys.path.insert(0, str(TRAINING/'source/f1_beyondmimic'))
from f1_beyondmimic.contract import validate_motion, reorder_motion_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--qpos', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to replace a released candidate')
    path = args.export/'walking_cycle.npz'
    validated = validate_motion(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    names = arrays['robot_joint_names'].tolist()
    q, fps = arrays['qpos'].astype(float), float(arrays['fps'][0])
    nominal = float(arrays['nominal_reference_speed_mps'][0])
    contact = {s: arrays['contact_'+s].astype(bool) for s in ('left', 'right')}
    contract = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    settings = json.loads((TRAINING/'configs/ssot_motion_processing.json').read_text(encoding='utf-8'))
    kin = F1Kinematics(TRAINING/contract['urdf'])
    model = native_model(args.model.resolve())
    actual, checks = native_audit(model, kin, q, names, fps, contact, nominal, True)
    if actual['joint_names'].tolist() != arrays['joint_names'].tolist() or actual['body_names'].tolist() != arrays['body_names'].tolist():
        raise ValueError('Export/native identities differ')
    error = {key: float(np.max(np.abs(actual[key]-arrays[key]))) for key in
             ('joint_pos', 'joint_vel', 'body_pos_w', 'body_lin_vel_w', 'body_ang_vel_w')}
    error['body_quaternion_dot_error'] = float(np.max(1-np.abs(np.sum(actual['body_quat_w']*arrays['body_quat_w'], axis=-1))))
    velocity_limits = np.array([contract['joint_properties'][name]['limit']['velocity'] for name in names])
    interval_speed = np.abs(np.roll(q[:, 7:], -1, axis=0)-q[:, 7:])*fps
    velocity_ratio = float((interval_speed/velocity_limits).max())
    feet = kin.world_feet(q, names)
    support = arrays['support_frames'].astype(int)
    support_speed, support_height, stance_heading = {}, {}, {}
    extents = kin.mesh_projections(feet, np.tile([0., 0., 1.], (len(q), 1)))
    for side in ('left', 'right'):
        v = derivatives(feet[side]['center'], fps, True)
        v[:, 0] += nominal
        support_speed[side] = float(np.linalg.norm(v[support, :2], axis=1).max())
        support_height[side] = float(np.abs(extents[side][0][support]).max())
        yaw = np.arctan2(feet[side]['r'][:, 1, 0], feet[side]['r'][:, 0, 0])
        stance_heading[side] = float(np.abs(yaw[contact[side]]).max())
    passed = (not checks['kinematics']['joint_position_limit_violations']
        and not checks['kinematics']['joint_velocity_limit_violations'] and velocity_ratio <= 1.00001
        and not checks['native_self_collision_pairs'] and len(support) >= 2
        and min(checks['kinematics']['sole_min_world_z_m'][s]['min'] for s in ('left', 'right')) >= -2e-6
        and max(support_speed.values()) <= settings['cycle_stance_planar_speed_tolerance_mps']
        and max(support_height.values()) <= settings['cycle_contact_height_tolerance_m']
        and max(stance_heading.values()) <= 1e-5
        and max(error.values()) <= 1e-4)
    # Prove the importer can map a different engine order without changing named states.
    remapped = reorder_motion_arrays(validated, arrays['joint_names'][::-1].tolist(), arrays['body_names'][::-1].tolist())
    roundtrip = reorder_motion_arrays(remapped, arrays['joint_names'].tolist(), arrays['body_names'].tolist())
    for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        if not np.array_equal(roundtrip[key], validated[key]):
            raise ValueError('Named reference remapping roundtrip failed: '+key)
    report = {'scope': 'Actual float32 cycle bytes; MuJoCo native FK/contact/collision and explicit name mapping',
        'reference': str(path.resolve()), 'reference_sha256': sha(path), 'urdf_sha256': contract['urdf_sha256'],
        'native_recomputed_errors': error, 'maximum_adjacent_speed_over_urdf_limit_ratio': velocity_ratio,
        'support_frames': support.tolist(), 'support_planar_speed_max_mps': support_speed,
        'support_sole_height_abs_max_m': support_height, 'native_checks': checks,
        'maximum_absolute_stance_foot_heading_rad': stance_heading,
        'name_remapping_roundtrip_passed': True, 'passed': passed,
        'isaaclab_loaded': False, 'dynamic_tracking_verified': False, 'owner_visual_acceptance': False,
        'training_authorized': False, 'training_ready': False}
    args.output.mkdir(parents=True)
    write_json(args.output/'delivery_validation.json', report)
    if not passed:
        raise ValueError('Delivered reference failed independent native verification; see delivery_validation.json')
    files = {'walking_cycle_50hz.npz': path, 'B3_ssot_walking_qpos.npz': args.qpos,
             'B3_ssot_walking_full.npz': args.export/'full_clip.npz', 'native_audit.json': args.export/'native_audit.json'}
    for name, source in files.items():
        shutil.copy2(source, args.output/name)
    manifest = {'format': 'f1_ssot_regenerated_walking_dataset_v1', 'urdf_sha256': contract['urdf_sha256'],
                'files': {name: sha(args.output/name) for name in files},
                'delivery_validation_sha256': sha(args.output/'delivery_validation.json'),
                'kinematic_native_data_checks_passed': True, 'training_ready': False,
                'reference_space': str(arrays['reference_space']),
                'native_joint_names': arrays['joint_names'].tolist(), 'native_body_names': arrays['body_names'].tolist(),
                'source_window_selection': str(args.qpos.with_suffix('.audit.json').resolve()),
                'source_window_selection_sha256': sha(args.qpos.with_suffix('.audit.json')),
                'model_audit_sha256': sha(args.model.parent/'model_audit.json'),
                'processing_settings': settings, 'processing_settings_sha256': sha(TRAINING/'configs/ssot_motion_processing.json')}
    write_json(args.output/'dataset_manifest.json', manifest)
    print(json.dumps({'published': str(args.output), 'passed': passed, 'support_speeds': support_speed,
                      'body_fk_max_error_m': error['body_pos_w'], 'training_ready': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
