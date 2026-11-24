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
from contextlib import nullcontext
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

    from f1_rl_lab.contract import ROOT as f1_root, load_json
    request = load_json(f1_root / "configs/run_request.json")
    if request.get("resume") and (not agent_cfg.resume or not args_cli.checkpoint):
        raise ValueError("Registered continuation requires --resume and the explicit parent checkpoint")
    continuation = None
    delta = {}
    initial_training = request.get("initial_training", False)
    recovery_method = None
    if initial_training and (request.get("resume") or agent_cfg.resume):
        raise ValueError("Initial training cannot load an actor or optimizer checkpoint")
    if request.get("resume") or initial_training:
        from f1_rl_lab.continuation import apply_delta
        delta = load_json(f1_root / "configs/training_delta.json")
        continuation = apply_delta(env_cfg, delta, initial_training=initial_training)
        recovery_method = delta.get("rollout_recovery")
        if recovery_method is not None:
            from isaaclab.managers import TerminationTermCfg
            from f1_rl_lab.rollout_recovery import METHOD, TERM_NAME, raw_action_failure

            if (recovery_method != METHOD or initial_training
                    or "signed_turn" not in delta.get("forward_exposure", {})):
                raise ValueError("Raw-action recovery requires the registered signed-turn continuation")
            setattr(env_cfg.terminations, TERM_NAME, TerminationTermCfg(func=raw_action_failure, time_out=False))
        if initial_training:
            continuation["initialization"] = "random"

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
        from f1_rl_lab.continuation import verify_resume_checkpoint
        parent_identity = verify_resume_checkpoint(resume_path, request["resume"], f1_root)
        if delta.get("action_postprocessing_override"):
            previous = parent_identity["runtime"]
            if (previous.get("action_clip") != delta["action_postprocessing_override"]["parent_action_clip"]
                    or previous["training_objective"]["reward_weights"] != delta["parent_reward_weights"]
                    or previous["training_objective"]["yaw_tracking_std"] != delta["parent_yaw_tracking_std"]):
                raise ValueError("Action adaptation parent metadata differs from registered source")
        if delta.get("reward_addition") or delta.get("command_sampling_override"):
            previous = parent_identity["runtime"]["training_objective"]
            expected_weights = (continuation["reward_weights_after"] if delta.get("forward_exposure")
                                else delta["parent_reward_weights"])
            if (previous["reward_weights"] != expected_weights
                    or previous["yaw_tracking_std"] != delta["parent_yaw_tracking_std"]):
                raise ValueError("Actual parent objective differs from registered continuation")
            if delta.get("forward_exposure") and (
                    previous.get("contact_lateral_tilt", {}).get("aggregation") != "positive_world_fz_moving"
                    or previous.get("command_sampling", {}).get("forward_exposure") is not None):
                raise ValueError("Forward exposure requires the original bilateral loaded-foot parent")

    if delta.get("action_input_projection"):
        from f1_rl_lab.action_projection import install_projection, projection_contract
        from f1_rl_lab.contract import load_contract
        spec = delta["action_input_projection"]
        contract = load_contract()
        if spec != {"parent": None, "to": projection_contract(contract)}:
            raise ValueError("Unregistered environment projection adaptation")
        if (parent_identity["runtime"].get("action_input_projection") is not None
                or not delta.get("action_postprocessing_override")
                or continuation["reward_weights_after"] != delta["parent_reward_weights"]):
            raise ValueError("Projection requires exact legacy parent and unchanged reward inventory")
        install_projection(env.unwrapped, spec["to"], contract)
        continuation["action_input_projection"] = spec

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
        from f1_rl_lab.continuation import restore_adaptive_lr
        continuation.update(restore_adaptive_lr(runner, request["resume"]))
        print("[F1_RESUME]", continuation, flush=True)
    if delta.get("forward_exposure"):
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        print("[F1_FORWARD_EXPOSURE]", command_term.audit_forward_commands(after_step=False), flush=True)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # Runtime contract was exported and checked before runner construction.
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    from f1_rl_lab.continuation import install_finite_guard
    install_finite_guard(runner, log_dir)
    if initial_training or (recovery_method is None and "signed_turn" in delta.get("forward_exposure", {})):
        from f1_rl_lab.rollout_guard import install_rollout_guard
        install_rollout_guard(runner, log_dir)

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
        if continuation is not None:
            from f1_rl_lab.continuation import optimizer_steps
            payload["f1_training"] = {**continuation,
                "optimizer_steps": optimizer_steps(payload["optimizer_state_dict"]),
                "adaptive_learning_rate": float(runner.alg.learning_rate),
                "optimizer_learning_rates": [float(g["lr"]) for g in runner.alg.optimizer.param_groups],
                "objective": runtime.get("training_objective")}
            if hasattr(env.unwrapped, "f1_action_projection_audit"):
                payload["f1_training"]["action_projection_audit"] = dict(env.unwrapped.f1_action_projection_audit)
        torch.save(payload, staged)
        os.replace(staged, final)
        write_sidecar(final, runtime)
        return result
    runner.save = save_bound

    recovery = nullcontext()
    if recovery_method is not None:
        from f1_rl_lab.rollout_recovery import recover_rollout_actions
        recovery = recover_rollout_actions(runner, log_dir)
    diagnostics = nullcontext()
    if "signed_turn" in (delta.get("forward_exposure") or {}):
        from f1_rl_lab.yaw_diagnostics import install_yaw_diagnostics
        diagnostics = install_yaw_diagnostics(
            runner, log_dir, std=float(env_cfg.rewards.track_ang_vel_z.params["std"]), every=10)
    with recovery as recovery_audit, diagnostics as diagnostics_audit:
        if recovery_audit is not None:
            continuation["rollout_recovery"] = recovery_audit
            print("[F1_ROLLOUT_RECOVERY_CONFIG]", recovery_audit, flush=True)
        if diagnostics_audit is not None:
            continuation["yaw_diagnostics"] = diagnostics_audit
            print("[F1_YAW_DIAG_CONFIG]", diagnostics_audit, flush=True)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Saved-checkpoint evidence uses the NEW actor and actual feedback after training.
    from f1_rl_lab.contract import ROOT as f1_root
    request_file = f1_root / "configs/run_request.json"
    if request_file.is_file():
        request = load_json(request_file)
        if request.get("phase") == "probe" or request.get("verify_saved_checkpoint_feedback") is True:
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
            clip_feedback = None
            if "JointPositionAction" in env.unwrapped.action_manager._terms:
                action_term = env.unwrapped.action_manager._terms["JointPositionAction"]
                spec = identity["runtime"].get("action_clip")
                lower = torch.tensor(spec["lower"], device=env.unwrapped.device) if spec else None
                upper = torch.tensor(spec["upper"], device=env.unwrapped.device) if spec else None
                defaults = torch.tensor(identity["runtime"]["default_joint_pos"], device=env.unwrapped.device)
                native_apply = action_term.apply_actions
                clip_feedback = {"calls": 0, "max_error": 0.0, "raw_target_outside_count": 0, "clipping": spec is not None, "last": None}
                def audited_apply():
                    raw = action_term.raw_actions
                    unbounded = raw * identity["runtime"]["action_scale"] + defaults
                    expected = torch.clamp(unbounded, min=lower, max=upper) if spec else unbounded
                    actual = action_term.processed_actions
                    error = float(torch.max(torch.abs(actual - expected)).detach().cpu())
                    if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                        raise ValueError("Native applied target differs from declared post-scale URDF clip")
                    clip_feedback["calls"] += 1
                    clip_feedback["max_error"] = max(clip_feedback["max_error"], error)
                    if spec is not None:
                        clip_feedback["raw_target_outside_count"] += int(((unbounded < lower) | (unbounded > upper)).sum().detach().cpu())
                    clip_feedback["last"] = {"native_raw_action": raw[0].detach().cpu().tolist(),
                                             "applied_position_target": actual[0].detach().cpu().tolist(),
                                             "all_env_max_target_error": error}
                    native_apply()
                action_term.apply_actions = audited_apply
            for feedback_step in range(request["readiness_steps"]):
                with torch.inference_mode():
                    actor_observation = select_actor_observation(observations, identity["runtime"]["observation_dim"])
                    actions = fresh(actor_observation)
                    if not torch.isfinite(actor_observation).all() or not torch.isfinite(actions).all():
                        raise ValueError("Non-finite fresh-checkpoint feedback")
                    observation_sample = actor_observation[0].detach().cpu().tolist()
                    observations, _, dones, _ = env.step(actions)
                    if identity["runtime"].get("action_input_projection") is not None:
                        next_actor = select_actor_observation(observations, identity["runtime"]["observation_dim"])
                        actual_manager = env.unwrapped.action_manager.action
                        if not torch.allclose(next_actor[:, -identity["runtime"]["action_dim"]:], actual_manager, rtol=1e-6, atol=1e-6):
                            raise ValueError("Next native last_action observation differs from actual projected/reset manager action")
                    asset = env.unwrapped.scene["robot"]
                    sample = {"step": feedback_step, "policy_observation": observation_sample,
                        "command": env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist(),
                        "velocity": asset.data.root_lin_vel_b[0].detach().cpu().tolist(),
                        "root_position": asset.data.root_pos_w[0].detach().cpu().tolist(),
                        "q": asset.data.joint_pos[0].detach().cpu().tolist(),
                        "qvel": asset.data.joint_vel[0].detach().cpu().tolist(),
                        "root_quat_w": asset.data.root_quat_w[0].detach().cpu().tolist(),
                        "root_ang_vel_b": asset.data.root_ang_vel_b[0].detach().cpu().tolist(),
                        "action": actions[0].detach().cpu().tolist(),
                        "done": bool(dones[0])}
                    if clip_feedback is not None:
                        if clip_feedback["last"] is None:
                            raise ValueError("Native clipping audit did not observe apply_actions")
                        raw = torch.tensor(clip_feedback["last"]["native_raw_action"], device=actions.device)
                        expected_environment = actions[0]
                        projection = identity["runtime"].get("action_input_projection")
                        if projection is not None:
                            expected_environment = torch.clamp(actions[0],
                                min=torch.tensor(projection["lower"], device=actions.device),
                                max=torch.tensor(projection["upper"], device=actions.device))
                        if not torch.allclose(raw, expected_environment, rtol=1e-6, atol=1e-6):
                            raise ValueError("Native environment action differs from declared projection")
                        sample["environment_action"] = expected_environment.detach().cpu().tolist()
                        sample.update(clip_feedback["last"])
                    if "yaw_rate_error_l2" in env.unwrapped.reward_manager.active_terms:
                        yaw_term = env.unwrapped.reward_manager.get_term_cfg("yaw_rate_error_l2")
                        sample["post_step_yaw_cost_diagnostic"] = float(yaw_term.func(env.unwrapped, **yaw_term.params)[0].detach().cpu())
                    if "zero_command_ankle_action_rate_l2" in env.unwrapped.reward_manager.active_terms:
                        standing_term = env.unwrapped.reward_manager.get_term_cfg("zero_command_ankle_action_rate_l2")
                        ids = standing_term.params["ankle_action_ids"]
                        commands = env.unwrapped.command_manager.get_command("base_velocity")[:, :3]
                        zero = torch.all(commands == 0, dim=1)
                        manager = env.unwrapped.action_manager
                        delta_action = (manager.action - manager.prev_action)[:, ids]
                        expected_cost = zero.to(delta_action.dtype) * delta_action.square().sum(dim=1)
                        actual_cost = standing_term.func(env.unwrapped, **standing_term.params)
                        if not torch.allclose(actual_cost, expected_cost, rtol=1e-6, atol=1e-6):
                            raise ValueError("Native standing smoothness diagnostic mismatch")
                        sample["post_step_standing_smoothness_diagnostic"] = {
                            "zero_command_envs": int(zero.sum().cpu()),
                            "all_env_max_cost_error": float(abs(actual_cost-expected_cost).max().cpu()),
                            "nonzero_command_cost_count": int((actual_cost[~zero] != 0).sum().cpu()),
                            "cost_sum": float(actual_cost.sum().cpu()),
                            "ankle_action": manager.action[0, ids].cpu().tolist(),
                            "previous_ankle_action": manager.prev_action[0, ids].cpu().tolist(),
                            "cost_env0": float(actual_cost[0].cpu())}
                    if "contact_lateral_tilt" in env.unwrapped.reward_manager.active_terms:
                        tilt_reward = env.unwrapped.reward_manager.get_term_cfg("contact_lateral_tilt")
                        ac, sc = tilt_reward.params["asset_cfg"], tilt_reward.params["sensor_cfg"]
                        sensor = env.unwrapped.scene.sensors[sc.name]
                        sample["foot_quat_w"] = asset.data.body_quat_w[0, ac.body_ids].detach().cpu().tolist()
                        sample["foot_contact_time"] = sensor.data.current_contact_time[0, sc.body_ids].detach().cpu().tolist()
                        sample["foot_net_forces_w"] = sensor.data.net_forces_w[0, sc.body_ids].detach().cpu().tolist()
                        sample["contact_lateral_tilt_unweighted"] = float(
                            tilt_reward.func(env.unwrapped, **tilt_reward.params)[0].detach().cpu())
                        sample["root_quat_w"] = asset.data.root_quat_w[0].detach().cpu().tolist()
                        sample["root_ang_vel_b"] = asset.data.root_ang_vel_b[0].detach().cpu().tolist()
                    samples.append(sample)
            evidence = {"mode": "actual_saved_checkpoint_feedback", "checkpoint_iteration": runner.current_learning_iteration,
                        "steps": len(samples), "samples": samples, "seed": env_cfg.seed,
                        "rsl_rl_version": installed_version, "runtime": identity["runtime"],
                        "scope": "admission interface/finite-state evidence; not walking or owner acceptance"}
            if hasattr(env.unwrapped, "f1_action_projection_audit"):
                evidence["environment_action_projection_audit"] = dict(env.unwrapped.f1_action_projection_audit)
            if delta.get("forward_exposure"):
                evidence["forward_command_exposure"] = command_term.audit_forward_commands()
            if clip_feedback is not None:
                evidence["position_target_audit"] = {
                    key: value for key, value in clip_feedback.items() if key != "last"}
                if clip_feedback["clipping"]:
                    evidence["processed_target_clip_audit"] = evidence["position_target_audit"]
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
