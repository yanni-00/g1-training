"""Adopt an explicitly approved F1 simulation baseline without inventing identification.

Preserves delivered motion and source SSOT. Rebinds a separate training reference
only after proving dynamics-only changes and recomputing all current-URDF body FK.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from model_geometry import forward_kinematics
from motion_kinematics import quat_rotation

TRAINING = Path(__file__).resolve().parents[1]
LOOP = TRAINING.parent
PROJECT = LOOP.parents[1]
sys.path.insert(0, str(TRAINING/'source/f1_beyondmimic'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(element):
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or '').strip(),
            tuple(signature(child) for child in element if child.tag != 'dynamics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-training', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_training.resolve()
    work = LOOP/'.oma/work/training-start-20261116'
    work.mkdir(parents=True, exist_ok=True)
    contract_path = TRAINING/'configs/robot_contract.json'
    contract = read(contract_path)
    source_contract = read(source/'configs/robot_contract.json')
    source_urdf = source/source_contract['urdf']
    ssot = PROJECT/'f1_simulation_ssot/01_SPEC_规范与接口/SYS-SPEC-0002-V1.1-F1整机模型29DOF/urdf'/source_urdf.name
    original, replacement = ET.parse(ssot).getroot(), ET.parse(source_urdf).getroot()
    if signature(original) != signature(replacement):
        raise ValueError('URDF changed outside joint dynamics; regenerate/review motion rather than rebind')
    if contract['model_joint_names'] != source_contract['model_joint_names']:
        raise ValueError('Joint identity/order changed')
    owner_source = source/'configs/owner_training_profile.json'
    owner = read(owner_source)
    if set(owner['joints']) != set(contract['model_joint_names']):
        raise ValueError('Owner PD joint coverage mismatch')
    for joint in replacement.findall('joint'):
        if joint.get('type') == 'fixed':
            continue
        if joint.find('dynamics').attrib != {'damping': '0.05', 'friction': '0'}:
            raise ValueError('Actual URDF does not match the approved owner dynamics')
    for path in (contract_path, LOOP/'.oma/impl/source_manifest.json', TRAINING/'data/motion_manifest.json'):
        backup = work/'before-control-sync'/path.relative_to(LOOP)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
    target_urdf = TRAINING/contract['urdf']
    shutil.copy2(source_urdf, target_urdf)
    contract['geometry_ssot_sha256'] = sha(ssot)
    contract['urdf_sha256'] = sha(target_urdf)
    for joint in replacement.findall('joint'):
        if joint.get('type') != 'fixed':
            contract['joint_properties'][joint.get('name')]['dynamics'] = {
                key: float(value) for key, value in joint.find('dynamics').attrib.items()}
    write(contract_path, contract)
    owner_path = TRAINING/'configs/owner_training_profile.json'
    shutil.copy2(owner_source, owner_path)
    profile = {'status': 'owner_specified_simulation_baseline', 'urdf_sha256': contract['urdf_sha256'],
        'source_status': 'User adoption 2026-11-16; not measured parameter identification',
        'sources': [{'path': 'configs/owner_training_profile.json', 'sha256': sha(owner_path),
                     'description': 'User explicitly adopted this complete F1 PD/armature/passive-dynamics baseline on 2026-11-16'}],
        'delivery_filter': None,
        'joints': {name: {**values, 'armature': owner['additional_armature']} for name, values in owner['joints'].items()}}
    write(TRAINING/'configs/control_profile.json', profile)
    identification_path = TRAINING/'configs/parameter_identification.json'
    identification = read(identification_path)
    if identification['status'] != 'pending' or identification['sources']:
        raise ValueError('Existing measured identification needs explicit review, not replacement')
    identification['urdf_sha256'] = contract['urdf_sha256']
    write(identification_path, identification)
    from f1_beyondmimic.contract import effective_control
    controls = effective_control()
    delivery = TRAINING/'data/regenerated/20261109/final'
    input_path = delivery/'walking_cycle_50hz.npz'
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if str(arrays['urdf_sha256']) != sha(ssot):
        raise ValueError('Input reference is not bound to the unchanged source geometry')
    names, body_names = arrays['robot_joint_names'].tolist(), arrays['body_names'].tolist()
    q = arrays['qpos'].astype(float)
    root_rotation = quat_rotation(q[:, 3:7])
    position_error, rotation_error = 0., 0.
    for frame, row in enumerate(q):
        links = forward_kinematics(replacement, dict(zip(names, row[7:])))
        for index, body in enumerate(body_names):
            transform = links[body]
            position = row[:3]+root_rotation[frame]@transform[:3, 3]
            rotation = root_rotation[frame]@transform[:3, :3]
            saved_rotation = quat_rotation(arrays['body_quat_w'][frame, index])
            position_error = max(position_error, float(np.abs(position-arrays['body_pos_w'][frame, index]).max()))
            rotation_error = max(rotation_error, float(np.abs(rotation-saved_rotation).max()))
    if max(position_error, rotation_error) > 1e-4:
        raise ValueError('Current-URDF full-body FK does not match the actual saved reference')
    out = TRAINING/'data/prepared/owner-pd-20261116'
    out.mkdir(parents=True, exist_ok=True)
    arrays['geometry_source_urdf_sha256'] = arrays['urdf_sha256'].copy()
    arrays['urdf_sha256'] = np.array(contract['urdf_sha256'])
    arrays['source_reference_sha256'] = np.array(sha(input_path))
    arrays['training_ready'] = np.array(False)  # Dynamic/naturalness acceptance is not implied.
    output_path = out/'walking_cycle_50hz.npz'
    if output_path.exists():
        raise FileExistsError('Prepared reference already exists; do not replace it')
    np.savez_compressed(output_path, **arrays)
    audit = {'format': 'f1_beyondmimic_native_fk_v1', 'passed': True,
        'scope': 'Training input: prior actual MuJoCo audit plus dynamics-only URDF equivalence and independently recomputed current full-body FK',
        'urdf_sha256': contract['urdf_sha256'], 'reference_sha256': sha(output_path),
        'joint_names': arrays['joint_names'].tolist(), 'body_names': body_names,
        'position_max_component_error_m': position_error, 'rotation_max_component_error': rotation_error,
        'kinematic_definitions_unchanged': True, 'isaaclab_loaded': False,
        'dynamic_tracking_verified': False, 'owner_visual_acceptance': False,
        'sources': [{'path': str(path.relative_to(TRAINING)), 'sha256': sha(path), 'description': description}
            for path, description in ((input_path, 'Unmodified delivered reference under the source SSOT'),
                (delivery/'delivery_validation.json', 'Actual MuJoCo native FK/collision/support verification of source geometry'),
                (target_urdf, 'Current approved training URDF; only joint dynamics changed'))]}
    audit_path = out/'native_audit.json'
    write(audit_path, audit)
    manifest_path = TRAINING/'data/motion_manifest.json'
    manifest = read(manifest_path)
    manifest.update(status='ready', urdf_sha256=contract['urdf_sha256'],
        prepared_path=output_path.relative_to(TRAINING).as_posix(), prepared_sha256=sha(output_path),
        native_audit_path=audit_path.relative_to(TRAINING).as_posix(), native_audit_sha256=sha(audit_path),
        blocked_reason=None, readiness_scope='Offline training input; Isaac dynamics and naturalness remain unaccepted')
    write(manifest_path, manifest)
    source_manifest_path = LOOP/'.oma/impl/source_manifest.json'
    source_manifest = read(source_manifest_path)
    for entry in source_manifest['assets']:
        if entry['path'] == contract['urdf']:
            entry['source_sha256'] = sha(ssot)
            entry['sha256'] = sha(target_urdf)
            entry['transformation'] = 'Owner-approved movable-joint friction=0/damping=0.05; all other XML definitions identical'
    generated = {row['path'] for row in source_manifest['generated_training_files']}
    generated.update('training/configs/'+name for name in ('control_profile.json', 'owner_training_profile.json'))
    source_manifest['generated_training_files'] = [{'path': path, 'sha256': sha(LOOP/path)} for path in sorted(generated)]
    write(source_manifest_path, source_manifest)
    report = {'source_training': str(source), 'owner_profile_source_sha256': sha(owner_source),
        'source_ssot_sha256': sha(ssot), 'training_urdf_sha256': sha(target_urdf),
        'kinematic_definitions_unchanged': True, 'current_urdf_reference_fk': audit,
        'identification_status': identification['status'], 'control_joint_count': len(controls),
        'training_reference': output_path.relative_to(TRAINING).as_posix(), 'training_started': False}
    write(work/'control_sync.json', report)
    config_path = LOOP/'.oma/config.json'
    config = read(config_path)
    config.update(goalRevision='F1-BEYONDMIMIC-WALK-v3', status='control_synced_preparing_formal_training')
    config['authorization'].update(training=True, cloud_tasks=True, remote_publish=True, evaluation=True,
        per_task_spend_cap_milli_credit=50000, real_robot=False)
    write(config_path, config)
    state_path = LOOP/'.oma/loop.json'
    state = read(state_path)
    state.update(goalRevision=config['goalRevision'], pausedScopes=[], stage='implement',
        status='control_synced_preparing_formal_training', pauseReason=None)
    write(state_path, state)
    print(json.dumps({'synchronized': True, 'control_joints': len(controls), 'identification': identification['status'],
        'current_reference_fk_position_error_m': position_error, 'training_started': False}))


if __name__ == '__main__':
    main()
