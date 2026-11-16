"""Select an explicitly audited continuous window; never hide rejected frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from build_ssot_retarget import TRAINING, sha, write_json
from export_ssot_motion import cycle_candidates
from motion_kinematics import F1Kinematics
from refine_ssot_motion import runs
from repair_walking_motion import audit, load_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output')
    arrays, q, names, fps = load_motion(args.input)
    report = json.loads(args.input.with_suffix('.audit.json').read_text(encoding='utf-8'))
    contract = json.loads((TRAINING/'configs/robot_contract.json').read_text(encoding='utf-8'))
    model = F1Kinematics(TRAINING/contract['urdf'])
    limits = [contract['joint_properties'][n]['limit'] for n in names]
    speed = np.array([p['velocity'] for p in limits])
    velocity = np.abs(np.gradient(q[:, 7:], 1/fps, axis=0, edge_order=2))
    valid = (velocity <= speed).all(axis=1)
    bad_intervals = np.any(np.abs(np.diff(q[:, 7:], axis=0))*fps > speed, axis=1)
    valid[:-1] &= ~bad_intervals
    valid[1:] &= ~bad_intervals
    valid[report['solver']['failed_frames_0based']] = False
    _, detail = audit(model, q, names, fps)
    valid &= detail['clearance'] >= 0
    valid &= np.minimum(detail['z_extents']['left'][0], detail['z_extents']['right'][0]) >= -2e-6
    candidates = []
    for start, end in runs(valid):
        # A new clip boundary uses one-sided derivatives, unlike the central
        # derivative at the same frame in its longer source. Recheck endpoints.
        while end-start >= 3:
            derivative = np.abs(np.gradient(q[start:end, 7:], 1/fps, axis=0, edge_order=2))
            first_bad, last_bad = bool((derivative[0] > speed).any()), bool((derivative[-1] > speed).any())
            if not first_bad and not last_bad:
                break
            start += int(first_bad)
            end -= int(last_bad)
        if end-start < 3:
            continue
        contact = {s: arrays['contact_'+s][start:end] for s in ('left', 'right')}
        cycles = cycle_candidates(q[start:end], names, contact, fps)
        if not cycles:
            continue
        selected_audit, _ = audit(model, q[start:end], names, fps)
        if selected_audit['joint_velocity_limit_violations'] or selected_audit['joint_position_limit_violations']:
            continue
        candidates.append((end-start, start, end, selected_audit, cycles))
    if not candidates:
        raise ValueError('No valid continuous complete-cycle window')
    length, start, end, selected_audit, cycles = max(candidates, key=lambda x: x[0])
    output = {key: value[start:end].copy() if value.ndim > 0 and value.shape[0] == len(q) else value.copy()
              for key, value in arrays.items()}
    output.update(kinematic_checks_passed=np.array(True), training_ready=np.array(False),
                  selected_from_refined_sha256=np.array(sha(args.input)),
                  source_window_start_end_exclusive=np.array([start, end], dtype=np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    result = {'input': str(args.input.resolve()), 'input_sha256': sha(args.input),
        'output': str(args.output.resolve()), 'output_sha256': sha(args.output), 'urdf_sha256': contract['urdf_sha256'],
        'source_window_start_end_exclusive': [start, end], 'selected_frame_count': length,
        'rejected_refined_frames_0based': np.flatnonzero(~valid).tolist(),
        'selection_policy': 'Longest continuous window passing per-frame IK/mesh/position/velocity gates and containing a full contact cycle',
        'full_source_refinement_passed': bool(arrays['kinematic_checks_passed']),
        'selected_kinematic_checks_passed': True, 'kinematics': selected_audit,
        'complete_cycle_candidates': cycles, 'training_ready': False,
        'original_full_refined_motion_preserved': True}
    write_json(args.output.with_suffix('.audit.json'), result)
    print(json.dumps({'selected_window': [start, end], 'frames': length, 'cycles': len(cycles), 'output': str(args.output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
