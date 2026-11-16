"""Crash-proof stage markers for the GM job.

The GM SDK captures stdout but not stderr, and reports newly created files by
name. Writing one empty file per stage makes the last reached stage visible even
when the interpreter dies without a traceback (for example a native abort).
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/'logs'/'boot'


def mark(name):
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR/name).write_text('')
    except Exception:
        pass
    print('[F1-BM] stage '+name, flush=True)


def fail(error):
    mark('error-'+type(error).__name__)
    traceback.print_exc()
    sys.stdout.flush()