"""GM 云端续训引导：把平台挂载的 checkpoint 归位到 Isaac Lab 日志结构后转交 train.py。"""
import os
import shutil
import subprocess
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "scripts" / "rsl_rl" / "train.py"
EXPERIMENT = os.environ.get("GM_RESUME_EXPERIMENT", "unitree_g1_29dof_velocity")
MOUNT_SEARCH = [REPO_ROOT, pathlib.Path("/workspace/isaaclab"), pathlib.Path("/personal")]


def main():
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
        print("[resume_gm] ERROR: no model_*.pt found", flush=True)
        sys.exit(1)
    print(f"[resume_gm] found: {src}", flush=True)

    ckpt_num = "".join(ch for ch in src.stem.split("_", 1)[1] if ch.isdigit())
    run_dir = pathlib.Path.cwd() / "logs" / "rsl_rl" / EXPERIMENT / "gm_resume"
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / src.name
    shutil.copy2(src, dst)
    assert dst.exists() and dst.stat().st_size == src.stat().st_size, "copy failed"
    print(f"[resume_gm] placed: {dst} ({dst.stat().st_size} bytes), ckpt={ckpt_num}", flush=True)

    train_args = sys.argv[1:] + ["--resume", "--load_run=gm_resume", f"--checkpoint={ckpt_num}"]
    print(f"[resume_gm] exec: {TRAIN_PY.name} {' '.join(train_args)}", flush=True)
    ret = subprocess.run([sys.executable, str(TRAIN_PY), *train_args])
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
