"""F1 native RSL-RL entry; denies launch before importing Isaac in scaffold mode."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))


def main():
    parser = argparse.ArgumentParser(description="F1 full-body velocity training (Isaac Lab; currently scaffold only)",
                                     epilog="Other authorized runtime flags are forwarded to the original G1-family RSL-RL entry.")
    parser.add_argument("--task", default="F1-29dof-Velocity", choices=["F1-29dof-Velocity"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args, _ = parser.parse_known_args()
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires an explicit F1 checkpoint path")
    if args.checkpoint is not None and not args.resume:
        parser.error("--checkpoint requires --resume")
    from f1_rl_lab.assets_archive import ensure_assets
    ensure_assets()
    from f1_rl_lab.contract import load_json
    from f1_rl_lab.launch import require_launch
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    request = load_json(ROOT / "configs/run_request.json")
    # Native continuation verification uses the archived producer and imports
    # torch only after AppLauncher has initialized the training environment.
    require_launch("training", checkpoint=None if request.get("resume") else checkpoint)
    if checkpoint is not None:
        for i, value in enumerate(sys.argv):
            if value == "--checkpoint":
                sys.argv[i+1] = str(checkpoint)
            elif value.startswith("--checkpoint="):
                sys.argv[i] = "--checkpoint=" + str(checkpoint)
    # Keep logs/resources independent of the caller's working directory.
    import os
    os.chdir(ROOT)
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", args.task])
    runpy.run_path(str(Path(__file__).with_name("_isaac_train.py")), run_name="__main__")


if __name__ == "__main__":
    main()
