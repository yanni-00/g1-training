"""Isolate finite action excursions during signed-turn training rollouts."""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from .contract import load_contract, sha256
from .rollout_guard import full_travel_raw_action_sq


METHOD = "finite_raw_action_failure_v1"
TERM_NAME = "raw_action_failure"


def raw_action_failure(env):
    mask = getattr(env, "f1_raw_action_failure", None)
    if mask is None:
        return torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    return mask


def action_failure_masks(action, previous, limit_sq):
    """Catch sustained large actions as well as the original step change."""
    action_sq = action.square().sum(dim=-1)
    delta_sq = (action - previous).square().sum(dim=-1)
    finite = torch.isfinite(action_sq) & torch.isfinite(delta_sq)
    excessive = finite & ((action_sq > limit_sq) | (delta_sq > limit_sq))
    return finite, excessive, action_sq, delta_sq


def _log_values(tensor):
    return [v if math.isfinite(v) else repr(v) for v in tensor.detach().flatten().cpu().tolist()]


@contextmanager
def recover_rollout_actions(runner, log_dir):
    """Apply the failure transition in the env while PPO retains its raw sample."""
    vec_env = runner.env
    env = vec_env.unwrapped
    contract = load_contract()
    manager = env.action_manager
    if (vec_env.clip_actions is not None
            or manager.action.shape[-1] != len(contract["model_joint_names"])
            or TERM_NAME not in env.termination_manager.active_terms):
        raise ValueError("Raw-action recovery requires unclipped actions and its native failure term")
    for name, default in zip(contract["model_joint_names"], contract["default_joint_pos"], strict=True):
        limit = contract["joint_properties"][name]["limit"]
        if not limit["lower"] <= default <= limit["upper"]:
            raise ValueError(f"Unsafe default position for recovery of {name}")

    limit_sq = full_travel_raw_action_sq(contract)
    # Allow at most one isolated env-step failure per collection step on average.
    # A burst across environments must still stop the job.
    recovery_budget = runner.num_steps_per_env
    recoveries = 0
    native_step, native_update = vec_env.step, runner.alg.update
    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    command_term = env.command_manager.get_term("base_velocity")
    modulus = command_term.cfg.forward_exposure["group_modulus"]

    def abort(record, reason):
        record["reason"] = reason
        message = json.dumps(record, allow_nan=False)
        (params / "rollout_outlier.json").write_text(message, encoding="utf-8")
        print("[F1_ROLLOUT_OUTLIER]", message, flush=True)
        raise FloatingPointError(f"F1 rollout recovery aborted: {reason}")

    def guarded_step(actions):
        nonlocal recoveries
        previous = manager.action
        finite, excessive, action_sq, delta_sq = action_failure_masks(actions, previous, limit_sq)
        transition = runner.alg.transition
        ppo_finite = torch.isfinite(transition.values).all() & torch.isfinite(transition.actions_log_prob).all()
        invalid_count, count, invalid_ppo = torch.stack((
            (~finite).sum(), excessive.sum(), (~ppo_finite).to(device=actions.device, dtype=torch.long)
        )).tolist()
        if not (invalid_count or count or invalid_ppo):
            return native_step(actions)

        bad = ~finite if invalid_count else excessive
        reason = None
        if invalid_count or invalid_ppo:
            reason = "nonfinite_action_or_ppo_sample"
        elif recoveries + count > recovery_budget:
            reason = "recovery_budget_exceeded"
        record = {
            "iteration_label": runner.current_learning_iteration,
            "nonfinite_envs": invalid_count,
            "nonfinite_ppo_sample": bool(invalid_ppo),
            "finite_failure_envs": count,
            "recoveries_this_update_before_step": recoveries,
            "max_recoveries_per_update": recovery_budget,
            "full_travel_raw_action_sq": limit_sq,
        }
        if invalid_count or count:
            index = int(bad.nonzero(as_tuple=False)[0, 0].item())
            record.update(environment=index, command_group=index % modulus,
                          command=_log_values(command_term.command[index]),
                          episode_step=int(env.episode_length_buf[index].item()),
                          action=_log_values(actions[index]), previous_action=_log_values(previous[index]),
                          action_sq=_log_values(action_sq[index])[0], action_delta_sq=_log_values(delta_sq[index])[0],
                          policy_mean=_log_values(transition.action_mean[index]),
                          policy_std=_log_values(transition.action_sigma[index]),
                          observation_blocks={name: _log_values(value[index])
                                              for name, value in transition.observations.items()})
        if reason is not None:
            abort(record, reason)

        # A zero raw action maps to the verified default pose. Do not mutate
        # PPO's sampled tensor or its corresponding log probability.
        env.f1_raw_action_failure = excessive
        try:
            safe_actions = torch.where(excessive[:, None], 0.0, actions)
            observations, rewards, dones, extras = native_step(safe_actions)
            if (not torch.all(dones[excessive] != 0)
                    or not torch.all(env.termination_manager.terminated[excessive])
                    or not torch.all(env.episode_length_buf[excessive] == 0)
                    or not torch.isfinite(rewards).all()):
                abort(record, "invalid_native_reset_transition")
        finally:
            env.f1_raw_action_failure = None
        if "time_outs" in extras:
            # Isaac can report a failure and an episode timeout simultaneously.
            # Failure takes priority, so RSL-RL must not bootstrap these rows.
            extras = {**extras, "time_outs": extras["time_outs"].masked_fill(excessive, 0)}
        recoveries += count
        record["recoveries_this_update_after_step"] = recoveries
        message = json.dumps(record, allow_nan=False)
        with (params / "rollout_recovery.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print("[F1_ROLLOUT_RECOVERY]", message, flush=True)
        return observations, rewards, dones, extras

    def guarded_update():
        nonlocal recoveries
        result = native_update()
        recoveries = 0
        return result

    audit = {
        "method": METHOD, "source_sha256": sha256(Path(__file__)),
        "full_travel_raw_action_sq": limit_sq, "max_recoveries_per_update": recovery_budget,
        "failure_term": TERM_NAME, "simulator_failure_action": "zero_raw_default_pose",
        "ppo_action": "original_sample", "failure_bootstrap": False,
    }
    vec_env.step, runner.alg.update = guarded_step, guarded_update
    try:
        yield audit
    finally:
        vec_env.step, runner.alg.update = native_step, native_update
        env.f1_raw_action_failure = None
