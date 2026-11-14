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
                "--run_name", request["run_name"]]
    if request.get("seed") is not None:
        sys.argv += ["--seed", str(request["seed"]) ]
    if request.get("resume"):
        import hashlib
        expected = request["resume"]["sha256"]
        roots = [ROOT.parent/"checkpoints", Path("/workspace/checkpoints"),
                 Path("/workspace/isaaclab/checkpoints"), ROOT/"checkpoints"]
        matches = sorted({p.resolve() for d in roots if d.is_dir() for p in d.rglob("*.pt")
                          if hashlib.sha256(p.read_bytes()).hexdigest() == expected})
        if not matches:
            raise RuntimeError("Exact registered parent checkpoint missing")
        sys.argv += ["--resume", "--checkpoint", str(matches[0])]
        # Do not import torch before AppLauncher: it mixes incompatible image
        # NumPy packages. Full parent contract verification runs inside the app.
        runpy.run_path(str(ROOT / "scripts/rsl_rl/_isaac_train.py"), run_name="__main__")
    else:
        runpy.run_path(str(ROOT / "scripts/rsl_rl/train.py"), run_name="__main__")
