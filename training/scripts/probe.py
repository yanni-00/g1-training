"""Probe stage 4: numpy loaded, hash every asset, print RSS after each file."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rss_mb():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) // 1024
    return -1


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    print('[PROBE] start', flush=True)
    import numpy
    print('[PROBE] numpy', numpy.__version__, 'rss', rss_mb(), flush=True)
    manifest = json.loads((ROOT.parent/'.oma/impl/source_manifest.json').read_text(encoding='utf-8'))
    print('[PROBE] assets', len(manifest['assets']), flush=True)
    for index, asset in enumerate(manifest['assets']):
        path = ROOT/asset['path']
        size = path.stat().st_size
        print('[PROBE] asset %d %d %d' % (index, size, rss_mb()), flush=True)
        if sha(path) != asset['sha256']:
            print('[PROBE] mismatch', asset['path'], flush=True)
    print('[PROBE] all hashed rss', rss_mb(), flush=True)
    for index in range(60):
        print('[PROBE] tick %d rss %d' % (index, rss_mb()), flush=True)
        time.sleep(1)
