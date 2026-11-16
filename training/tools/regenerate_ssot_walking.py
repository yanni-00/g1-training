"""Reproducible offline SSOT walking pipeline. Never starts an RL experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
TOOLS = TRAINING/'tools'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new dataset directory')
    parser.add_argument('--clips', nargs='+', choices=('B1', 'B2', 'B3'), default=['B3', 'B1', 'B2'])
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the isolated Python 3.12 interpreter for this pipeline')
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; originals and earlier evidence are never replaced')
    out.mkdir(parents=True)
    (out/'logs').mkdir()
    sources = json.loads((TRAINING/'configs/ssot_motion_sources.json').read_text(encoding='utf-8'))
    umr = (TRAINING/sources['umr_project']).resolve()
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR=str(TRAINING/'.runtime/matplotlib'))

    def run(label, tool, *arguments):
        command = [sys.executable, str(TOOLS/tool), *map(str, arguments)]
        print('[pipeline]', label, flush=True)
        with (out/'logs'/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=TRAINING.parents[2], env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(label+' failed; inspect '+str(out/'logs'/(label+'.log')))

    run('model', 'build_ssot_retarget.py', '--output', out/'model')
    results = {}
    for clip in args.clips:
        spec = sources['clips'][clip]
        raw = out/'umr'/(clip+'.npz')
        refined = out/'refined'/(clip+'.npz')
        run(clip+'-umr', 'run_umr_ssot.py', '--config', out/'model/umr_ssot.json',
            '--data', umr/spec['data'], '--seq-key', spec['sequence_key'], '--out', raw, '--smplx-batch-size', 32)
        run(clip+'-refine', 'refine_ssot_motion.py', '--input', raw, '--output', refined)
        with np.load(refined, allow_pickle=False) as data:
            passed = bool(data['kinematic_checks_passed'])
        selected = refined
        if not passed:
            selected = out/'selected'/(clip+'.npz')
            run(clip+'-select', 'select_valid_walking_window.py', '--input', refined, '--output', selected)
        options = ['--cycle'] if spec['cycle'] else []
        run(clip+'-export', 'export_ssot_motion.py', '--input', selected, '--model', out/'model/f1_ssot.xml',
            '--output-dir', out/'exports'/clip, *options)
        results[clip] = {'selected': str(selected), 'export': str(out/'exports'/clip)}
        if spec['cycle']:
            run(clip+'-publish', 'publish_ssot_motion.py', '--export', out/'exports'/clip,
                '--qpos', selected, '--model', out/'model/f1_ssot.xml', '--output', out/'release')
            if not args.no_render:
                run(clip+'-preview', 'render_ssot_motion.py', '--input', out/'release/walking_cycle_50hz.npz',
                    '--model', out/'model/f1_ssot.xml', '--output', out/'release/walking_cycle_preview.mp4', '--repeat-cycles', 3)
    (out/'pipeline_result.json').write_text(json.dumps({'scope': 'Offline data only', 'clips': results,
        'training_started': False, 'isaac_runtime_verified': False}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('[pipeline] offline regeneration complete:', out, flush=True)


if __name__ == '__main__':
    main()
