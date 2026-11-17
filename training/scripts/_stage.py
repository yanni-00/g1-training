"""Crash-proof stage markers for the GM job.

The GM SDK captures stdout but not stderr, and reports newly created files by
name. Writing one empty file per stage makes the last reached stage visible even
when the interpreter dies without a traceback (for example a native abort).
"""
import sys
import traceback
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/'logs'/'boot'


def memory_snapshot():
    """Raw Linux counters; unavailable counters are omitted."""
    result = {}
    try:
        import resource
        import signal
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['cpu_seconds'] = [usage.ru_utime, usage.ru_stime]
        result['RLIMIT_CPU'] = resource.getrlimit(resource.RLIMIT_CPU)
        result['RLIMIT_AS'] = resource.getrlimit(resource.RLIMIT_AS)
        result['alarm_seconds'] = signal.getitimer(signal.ITIMER_REAL)
    except (ImportError, AttributeError, OSError):
        pass
    for relative in ("memory.current", "memory.max", "memory.events",
                     "memory/memory.usage_in_bytes", "memory/memory.limit_in_bytes",
                     "memory/memory.failcnt"):
        try:
            result[relative] = (Path('/sys/fs/cgroup')/relative).read_text().strip()
        except OSError:
            pass
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmPeak:')):
                key, value = line.split(':', 1)
                result[key] = value.strip()
    except OSError:
        pass
    return result


def mark(name):
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR/name).write_text('')
    except Exception:
        pass
    print('[F1-BM] stage '+name+' '+json.dumps(memory_snapshot()), flush=True)


def fail(error):
    mark('error-'+type(error).__name__)
    # The GM SDK captures stdout only; never rely on sys.stderr here.
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()