"""Minimal platform probe: heartbeat only, no assets, no numpy, no simulator."""
import time

if __name__ == '__main__':
    print('[PROBE] start', flush=True)
    for index in range(120):
        print('[PROBE] tick %d' % index, flush=True)
        time.sleep(1)
    print('[PROBE] done', flush=True)
