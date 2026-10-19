# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--gm_resume_url", type=str, default=None, help="GM OSS URL of checkpoint to resume from (falls back to resume_url.txt)."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import time
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # GM 云端续训自举：--resume 且未显式指定 load_run 时，从 OSS URL（CLI 参数或
        # scripts/rsl_rl/resume_url.txt）下载 checkpoint 到 logs 结构的 gm_resume 目录，
        # 并设定 load_run/checkpoint。容器可直连 OSS（SDK 上传同通道），本地网络不通也能续训。
        if agent_cfg.resume and agent_cfg.load_run is None:
            import re as _re
            import urllib.request

            gm_run_dir = os.path.join(log_root_path, "gm_resume")
            os.makedirs(gm_run_dir, exist_ok=True)
            if not glob.glob(os.path.join(gm_run_dir, "model_*.pt")):
                url = args_cli.gm_resume_url
                if url is None:
                    url_file = os.path.join(os.path.dirname(__file__), "resume_url.txt")
                    if os.path.isfile(url_file):
                        url = open(url_file, encoding="utf-8").read().strip()
                if url:
                    m = _re.search(r"model_(\d+)_", url)
                    ckpt_num = m.group(1) if m else "0"
                    dst = os.path.join(gm_run_dir, f"model_{ckpt_num}.pt")
                    print(f"[gm_resume] downloading checkpoint {ckpt_num} from OSS ...", flush=True)
                    urllib.request.urlretrieve(url, dst)
                    print(f"[gm_resume] saved: {dst} ({os.path.getsize(dst)} bytes)", flush=True)
            existing = glob.glob(os.path.join(gm_run_dir, "model_*.pt"))
            if existing and agent_cfg.load_run is None:
                ckpt_num = "".join(ch for ch in os.path.basename(existing[0]) if ch.isdigit())
                agent_cfg.load_run = "gm_resume"
                agent_cfg.load_checkpoint = f".*{ckpt_num}.*"
                os.environ["GM_CHECKPOINT_RELAY"] = "1"
                print(f"[gm_resume] will load: {existing[0]}", flush=True)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    if os.environ.get("GM_CHECKPOINT_RELAY") == "1":
        import threading

        relay = threading.Thread(target=_gm_checkpoint_relay, args=(agent_cfg.experiment_name,), daemon=True)
        relay.start()
        print("[gm_relay] background checkpoint relay started (5 min cycle)", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    if os.environ.get("GM_CHECKPOINT_RELAY") == "1":
        # 终局兜底：把最后一个 checkpoint 也中继出去（任务自然结束时 SDK 仍会短暂上传）
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            import glob

            models = sorted(
                glob.glob(os.path.join(log_root, "*", "model_*.pt")),
                key=lambda p: os.path.getmtime(p),
            )
            if models:
                dst = os.path.join(repo_root, os.path.basename(models[-1]))
                if not os.path.exists(dst):
                    shutil.copy2(models[-1], dst)
                    print(f"[gm_relay] final checkpoint relayed: {os.path.basename(models[-1])}", flush=True)
        except Exception as e:
            print(f"[gm_relay] final relay error: {e}", flush=True)

    # close the simulator
    env.close()


def _gm_relay_pass(log_root: str, repo_root: str, only_milestones: bool):
    """把 log 目录里的 checkpoint 复制到仓库根（g1-training/）——GM SDK 只可靠扫描该目录。"""
    import glob

    moved = []
    for f in sorted(glob.glob(os.path.join(log_root, "*", "model_*.pt"))):
        num = "".join(ch for ch in os.path.basename(f) if ch.isdigit())
        if only_milestones and num and int(num) % 500 != 0:
            continue
        dst = os.path.join(repo_root, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            moved.append(os.path.basename(f))
    if moved:
        print(f"[gm_relay] copied to repo root: {moved}", flush=True)
    return moved


def _gm_checkpoint_relay(experiment_name: str):
    """后台线程：每 5 分钟把里程碑 checkpoint 中继到仓库根，供 GM SDK 上传（防中途停止丢模型）。"""
    import threading

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    while True:
        time.sleep(300)
        try:
            _gm_relay_pass(log_root, repo_root, only_milestones=True)
        except Exception as e:
            print(f"[gm_relay] pass error: {e}", flush=True)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
