"""GM SDK-compatible entry for the registered, bounded F1 request."""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))


if __name__ == "__main__":
    from f1_rl_lab.assets_archive import ensure_assets
    ensure_assets()
    launch = json.loads((ROOT / "configs/launch.json").read_text(encoding="utf-8"))
    if launch.get("authorization", {}).get("training") is not True:
        raise RuntimeError("F1 training is disabled until OMA registration/preflight")
    os.chdir(ROOT)
    # All limits are consumed from the registered request, not a second CLI default.
    request = json.loads((ROOT / "configs/run_request.json").read_text(encoding="utf-8"))
    sys.argv = [str(ROOT / "scripts/rsl_rl/train.py"), "--task", request["task"], "--headless",
                "--num_envs", str(request["num_envs"]), "--max_iterations", str(request["max_iterations"]),
                "--seed", str(request["seed"]), "--run_name", request["run_name"]]
    runpy.run_path(str(ROOT / "scripts/rsl_rl/train.py"), run_name="__main__")
