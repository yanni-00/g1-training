"""Platform probe stage 3: run the real admission check, then heartbeat."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source/f1_beyondmimic')]

if __name__ == '__main__':
    print('[PROBE] start', flush=True)
    from f1_beyondmimic.assets_archive import ensure_assets
    ensure_assets(ROOT)
    print('[PROBE] assets restored', flush=True)
    import numpy
    print('[PROBE] numpy', numpy.__version__, flush=True)
    from f1_beyondmimic.contract import assert_can_run
    print('[PROBE] contract imported', flush=True)
    launch = assert_can_run()
    print('[PROBE] admission ok', flush=True)
    for index in range(120):
        print('[PROBE] tick %d' % index, flush=True)
        time.sleep(1)
    print('[PROBE] done', flush=True)
