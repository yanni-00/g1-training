"""GM fixed-weight simulation entry. Downloads only the pinned source checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
try:
    os.dup2(1, 2)
except OSError:
    pass
ROOT = Path(__file__).resolve().parents[1]
tasks = ["Tracking-VelocityReference-F1-Proprio-v0"]

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint-url", required=True)
        parser.add_argument("--request", required=True)
        parser.add_argument("--protocol", required=True)
        parser.add_argument("--iteration", type=int, required=True)
        args = parser.parse_args()
        request = ROOT / args.request
        protocol = ROOT / args.protocol
        data = json.loads(request.read_text())
        row = next(r for r in data["checkpoints"] if r["iteration"] == args.iteration)
        logs = ROOT / "training/logs/native_replay"
        logs.mkdir(parents=True, exist_ok=True)
        checkpoint = logs / ("model_" + str(args.iteration) + ".pt")
        with urllib.request.urlopen(args.checkpoint_url, timeout=120) as src, checkpoint.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        with checkpoint.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError("Source checkpoint hash mismatch")
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.chdir(ROOT)
        argv = [sys.executable, str(ROOT / "tools/native_replay.py"), "--checkpoint", str(checkpoint),
                "--request", str(request), "--protocol", str(protocol), "--output", str(logs / "result"),
                "--bundle", str(logs / "model_native_replay.pt"), "--video", "--headless", "--device", "cuda:0"]
        print("[F1 replay] verified fixed checkpoint; launching native replay, no optimizer", flush=True)
        os.execv(sys.executable, argv)
    except BaseException:
        import traceback, time
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(15)
        raise
