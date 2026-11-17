"""Check the full runtime contract without loading NumPy into the Isaac parent."""
import sys
import traceback
from _bootstrap import ROOT


if __name__ == "__main__":
    try:
        from f1_beyondmimic.contract import assert_can_run
        assert_can_run("training", ROOT)
        print("[F1-BM] isolated full contract passed", flush=True)
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
