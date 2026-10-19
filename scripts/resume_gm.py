"""GM 云端续训引导：把平台挂载的 checkpoint 归位到 Isaac Lab 日志结构后转交 train.py。"""
import os
import shutil
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

    train_args = sys.argv[1:] + ["--resume", "--load_run=gm_resume", f"--checkpoint=.*{ckpt_num}.*"]
    print(f"[resume_gm] exec: {TRAIN_PY.name} {' '.join(train_args)}", flush=True)
    # GM SDK 只可靠扫描仓库根目录的 PT 文件（实证：挂载的 model_4999 每次都被重新上传，
    # 而 subprocess/execv 形式训练写入 logs/ 的新 checkpoint 从不被上传）——
    # 开启 train.py 内置的 checkpoint 中继线程，把里程碑+终版模型复制到仓库根供 SDK 上传。
    os.environ["GM_CHECKPOINT_RELAY"] = "1"
    # 用 execv 替换进程（保持 gm-run 直接子进程关系）。
    # 实证（TASK_20261014_031）：SDK 只对 gm-run 的"直接子进程"追踪 PT 上传——
    # subprocess 包装时 15000 个新 checkpoint 全部未上传；execv 保持同一 PID，
    # SDK 视角与直接 gm-run train.py 一致（TASK_20261013_084/088 上传正常）。
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, str(TRAIN_PY), *train_args])


if __name__ == "__main__":
    main()
