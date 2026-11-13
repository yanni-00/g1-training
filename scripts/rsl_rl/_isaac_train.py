from pathlib import Path as _F1Path
import sys as _F1Sys
_F1Sys.path.insert(0, str(_F1Path(__file__).resolve().parents[2] / "source/f1_rl_lab"))
from f1_rl_lab.assets_archive import ensure_assets as _f1_assets
_f1_assets()
from f1_rl_lab.launch import require_launch as _f1_gate
_f1_gate("training")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


tasks = ["F1-29dof-Velocity"]

import argparse
import sys


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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

import f1_rl_lab.tasks  # noqa: F401
from f1_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

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

    if agent_cfg.empirical_normalization or agent_cfg.clip_actions is not None:
        raise ValueError("Normalization/clipping changed: revise the F1 runtime contract first")

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
        from f1_rl_lab.checkpoint import verify_checkpoint
        if not args_cli.checkpoint:
            raise ValueError("Explicit F1 checkpoint path required; latest/run regex is disabled")
        resume_path = os.path.abspath(args_cli.checkpoint)
        verify_checkpoint(resume_path)

    from f1_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
    export_deploy_cfg(env.unwrapped, log_dir)

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
    # Runtime contract was exported and checked before runner construction.
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # Bind every actual save, including intermediate checkpoints.
    from f1_rl_lab.checkpoint import write_sidecar
    from f1_rl_lab.contract import load_json
    runtime = load_json(os.path.join(log_dir, "params", "f1_runtime.json"))
    native_save = runner.save
    def save_bound(path, *args, **kwargs):
        from pathlib import Path
        from f1_rl_lab.contract import binding, load_contract
        final = Path(path)
        staged = final.with_name("." + final.name + ".unbound")
        result = native_save(str(staged), *args, **kwargs)
        payload = torch.load(staged, map_location="cpu", weights_only=True)
        payload["f1_contract"] = {"format": "f1_fullbody_checkpoint_v1", "robot": load_contract()["robot"],
                                  "binding": binding(), "runtime": runtime}
        torch.save(payload, staged)
        os.replace(staged, final)
        write_sidecar(final, runtime)
        return result
    runner.save = save_bound

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # First-admission evidence uses the saved NEW checkpoint and actual feedback.
    from f1_rl_lab.contract import ROOT as f1_root
    request_file = f1_root / "configs/run_request.json"
    if request_file.is_file():
        request = load_json(request_file)
        if request.get("phase") == "probe":
            from f1_rl_lab.checkpoint import verify_checkpoint
            from f1_rl_lab.policy import load_actor, select_actor_observation
            from f1_rl_lab.contract import sha256
            from pathlib import Path
            import json
            final = Path(log_dir) / f"model_{runner.current_learning_iteration}.pt"
            identity = verify_checkpoint(final)
            fresh = load_actor(final, identity["runtime"]).to(env.unwrapped.device)
            # RSL-RL creates simulator caches in inference_mode; reset mutates them.
            with torch.inference_mode():
                env.reset()
                observations = env.get_observations()
            samples = []
            for feedback_step in range(request["readiness_steps"]):
                with torch.inference_mode():
                    actor_observation = select_actor_observation(observations, identity["runtime"]["observation_dim"])
                    actions = fresh(actor_observation)
                    if not torch.isfinite(actor_observation).all() or not torch.isfinite(actions).all():
                        raise ValueError("Non-finite fresh-checkpoint feedback")
                    observation_sample = actor_observation[0].detach().cpu().tolist()
                    observations, _, _, _ = env.step(actions)
                    asset = env.unwrapped.scene["robot"]
                    samples.append({"step": feedback_step, "policy_observation": observation_sample,
                        "command": env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist(),
                        "velocity": asset.data.root_lin_vel_b[0].detach().cpu().tolist(),
                        "root_position": asset.data.root_pos_w[0].detach().cpu().tolist(),
                        "q": asset.data.joint_pos[0].detach().cpu().tolist(),
                        "action": actions[0].detach().cpu().tolist()})
            evidence = {"mode": "actual_saved_checkpoint_feedback", "checkpoint_iteration": runner.current_learning_iteration,
                        "steps": len(samples), "samples": samples, "seed": env_cfg.seed,
                        "rsl_rl_version": installed_version, "runtime": identity["runtime"],
                        "scope": "admission interface/finite-state evidence; not walking or owner acceptance"}
            evidence_path = Path(log_dir) / "params/readiness_feedback.json"
            evidence_path.write_text(json.dumps(evidence, allow_nan=False), encoding="utf-8")
            payload = torch.load(final, map_location="cpu", weights_only=True)
            payload["f1_readiness"] = evidence
            staged = final.with_name("." + final.name + ".evidence")
            torch.save(payload, staged)
            os.replace(staged, final)
            write_sidecar(final, runtime)
            print("[F1_READINESS] actual new checkpoint loaded and feedback complete:", str(final), len(samples), flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
