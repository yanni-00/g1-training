"""Capture a finite action excursion before it enters the rollout."""

import json
import math
from pathlib import Path

import torch

from .contract import load_contract


def full_travel_raw_action_sq(contract):
    action_scale = float(contract["action_scale"])
    limits = [contract["joint_properties"][name]["limit"] for name in contract["model_joint_names"]]
    return sum(((limit["upper"] - limit["lower"]) / action_scale) ** 2 for limit in limits)


def install_rollout_guard(runner, log_dir):
    env = runner.env.unwrapped
    contract = load_contract()
    manager = env.action_manager
    # A displacement larger than all joints' full travels combined is evidence
    # of an extreme raw action. Routine target excursions are left unchanged.
    maximum_delta_sq = full_travel_raw_action_sq(contract)
    if runner.env.clip_actions is not None or manager.action.shape[-1] != len(contract["model_joint_names"]):
        raise ValueError("Rollout guard requires the registered original action interface")
    native_act = runner.alg.act
    term = env.command_manager.get_term("base_velocity")
    profile = term.cfg.forward_exposure
    path = Path(log_dir) / "params/rollout_outlier.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(row):
        return [v if math.isfinite(v) else repr(v) for v in row.detach().flatten().cpu().tolist()]

    def act(observations):
        action = native_act(observations)
        previous = manager.action
        delta_sq = (action - previous).square().sum(dim=-1)
        bad = (~torch.isfinite(delta_sq)) | (delta_sq > maximum_delta_sq)
        if torch.any(bad):
            index = int(torch.nonzero(bad, as_tuple=False)[0, 0].item())
            report = {
                "iteration_label": runner.current_learning_iteration,
                "environment": index,
                "command_group": index % profile["group_modulus"],
                "standing": bool(term.is_standing_env[index].item()),
                "command_time_left": float(term.time_left[index].item()),
                "episode_step": int(env.episode_length_buf[index].item()),
                "command": encode(term.command[index]),
                "action": encode(action[index]),
                "previous_action": encode(previous[index]),
                "action_delta_sq": encode(delta_sq[index])[0],
                "full_travel_delta_sq": maximum_delta_sq,
                "observation_blocks": {k: encode(v[index]) for k, v in observations.items()},
            }
            reset_buf = getattr(env, "reset_buf", None)
            if reset_buf is not None:
                report["reset"] = bool(reset_buf[index].item())
            path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
            print("[F1_ROLLOUT_OUTLIER]", json.dumps(report, allow_nan=False), flush=True)
            raise FloatingPointError("Extreme raw action change before environment step and PPO update")
        return action

    runner.alg.act = act
