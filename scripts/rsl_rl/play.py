"""Explicit-checkpoint Isaac play entry, guarded before simulator import."""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))


def main():
    parser = argparse.ArgumentParser(description="Replay an actual F1 checkpoint in Isaac Lab")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="F1-29dof-Velocity", choices=["F1-29dof-Velocity"])
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--video", action="store_true")
    args, _ = parser.parse_known_args()
    checkpoint = args.checkpoint.resolve()
    from f1_rl_lab.assets_archive import ensure_assets
    ensure_assets()
    from f1_rl_lab.launch import require_launch
    require_launch("evaluation", checkpoint=checkpoint)
    # Resolve before changing cwd; forward the same verified bytes.
    for i, value in enumerate(sys.argv):
        if value == "--checkpoint":
            sys.argv[i+1] = str(checkpoint)
        elif value.startswith("--checkpoint="):
            sys.argv[i] = "--checkpoint=" + str(checkpoint)
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", args.task])
    os.chdir(ROOT)
    runpy.run_path(str(Path(__file__).with_name("_isaac_play.py")), run_name="__main__")


if __name__ == "__main__":
    main()
