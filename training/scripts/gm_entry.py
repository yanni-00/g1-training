"""GM entry: consume the registered formal request, restore pinned assets, train."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source/f1_beyondmimic'), str(ROOT/'scripts')]
tasks = ['Tracking-VelocityReference-F1-Proprio-v0']


if __name__ == '__main__':
    import traceback
    try:
        from f1_beyondmimic.assets_archive import ensure_assets
        ensure_assets(ROOT)
        print('[F1-BM] assets restored', flush=True)
        request = json.loads((ROOT/'configs/run_request.json').read_text(encoding='utf-8'))
        launch = json.loads((ROOT/'configs/launch.json').read_text(encoding='utf-8'))
        if (request['kind'] != 'formal_training' or request['spend_cap_milli_credit'] != 50000
                or any(request[key] != launch[key] for key in ('num_envs', 'max_iterations', 'seed'))):
            raise ValueError('GM entry and registered formal limits differ')
        os.chdir(ROOT)
        os.environ['PYTHONUNBUFFERED'] = '1'
        sys.argv = [str(ROOT/'scripts/train.py'), '--headless', '--device', 'cuda:0']
        print('[F1-BM] formal request', json.dumps({key: request[key] for key in
            ('exp_id', 'num_envs', 'max_iterations', 'seed', 'spend_cap_milli_credit')}), flush=True)
        runpy.run_path(str(ROOT/'scripts/train.py'), run_name='__main__')
    except BaseException:
        # The GM platform captures stdout but not stderr; surface the real cause.
        traceback.print_exc()
        sys.stdout.flush()
        raise
