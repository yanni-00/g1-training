"""GM 云端续训引导：把平台挂载的 checkpoint 归位到 Isaac Lab 日志结构后转交 train.py。

平台把 checkPointFilePath 挂到 <mount>/model_*.pt（见 checkPointMountPath），
而 Isaac Lab train.py --resume 只认 logs/rsl_rl/<experiment>/<run>/model_*.pt。
本脚本：找到挂载的 pt → 归位 → 以相同参数调用 train.py（--resume --load_run=gm_resume）。
"""
import os
import shutil
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent          # g1-training/
EXPERIMENT = os.environ.get("GM_RESUME_EXPERIMENT", "unitree_g1_29dof_velocity")
MOUNT_SEARCH = [REPO_ROOT, pathlib.Path("/workspace/isaaclab"), pathlib.Path("/personal")]


def main():
    # 1. 找平台挂载的 model_*.pt
    src = None
    for root in MOUNT_SEARCH:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("model_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            src = p
            break
        if src:
            break
    if src is None:
        print("[resume_gm] ERROR: no model_*.pt found in", [str(r) for r in MOUNT_SEARCH])
        sys.exit(1)
    print(f"[resume_gm] found checkpoint: {src}")

    # 2. 归位到 train.py 的日志结构
    run_dir = pathlib.Path.cwd() / "logs" / "rsl_rl" / EXPERIMENT / "gm_resume"
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / src.name
    shutil.copy2(src, dst)
    print(f"[resume_gm] placed at: {dst}")

    # 3. 转交 train.py
    train_args = sys.argv[1:] + ["--resume", "--load_run=gm_resume", f"--checkpoint={src.stem.split('_')[1]}"]
    print(f"[resume_gm] exec: train.py {' '.join(train_args)}")
    os.execv(sys.executable, [sys.executable, str(REPO_ROOT / "scripts" / "rsl_rl" / "train.py"), *train_args])


if __name__ == "__main__":
    main()
