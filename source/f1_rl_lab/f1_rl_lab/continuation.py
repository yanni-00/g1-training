"""Narrow registered objective continuation and auditable resume state."""
import json
import math
from pathlib import Path


def apply_delta(env_cfg, delta):
    if set(delta["reward_override"]) != {"gait"}:
        raise ValueError("This experiment permits only the gait reward weight delta")
    spec = delta["reward_override"]["gait"]
    before = {n: float(v.weight) for n,v in vars(env_cfg.rewards).items()
              if hasattr(v,"weight") and float(v.weight) != 0.0}
    if before["gait"] != spec["from"] or not math.isfinite(spec["to"]) or spec["to"] <= 0:
        raise ValueError("Unexpected source gait weight or invalid registered target")
    env_cfg.rewards.gait.weight = spec["to"]
    after = {n: float(v.weight) for n,v in vars(env_cfg.rewards).items()
             if hasattr(v,"weight") and float(v.weight) != 0.0}
    if any(before[n] != after[n] for n in before if n != "gait"):
        raise ValueError("Unregistered reward change")
    # [OMA] r8: keep the parent's complete weight inventory while testing only
    # the existing yaw tracking kernel width. Do not change the physical binding.
    parameter_changes = {}
    overrides = delta.get("reward_param_override", {})
    if overrides:
        if set(overrides) != {"track_ang_vel_z"} or set(overrides["track_ang_vel_z"]) != {"std"}:
            raise ValueError("Only the registered yaw tracking std parameter is allowed")
        if after != delta["parent_reward_weights"]:
            raise ValueError("Continuation weights differ from the registered actual parent")
        term = env_cfg.rewards.track_ang_vel_z
        change = overrides["track_ang_vel_z"]["std"]
        old, target = float(term.params["std"]), float(change["to"])
        if old != change["from"] or not math.isfinite(target) or target <= 0:
            raise ValueError("Unexpected source yaw std or invalid registered target")
        term.params["std"] = target
        parameter_changes = {"track_ang_vel_z": {"std": {"from": old, "to": target}}}
    # The parent does not contain per-environment episode state. Restore the
    # observed curriculum command limits explicitly, never claim a bitwise resume.
    for name,values in delta["resume_command_ranges"].items():
        if name not in ("lin_vel_x","lin_vel_y","ang_vel_z"):
            raise ValueError("Unregistered command range field")
        setattr(env_cfg.commands.base_velocity.ranges,name,tuple(values))
    additions = delta.get("reward_addition", {})
    addition_audit = {}
    if additions:
        if set(additions) != {"contact_lateral_tilt"}:
            raise ValueError("Only the registered symmetric contact tilt term can be added")
        spec = additions["contact_lateral_tilt"]
        if (float(env_cfg.rewards.contact_lateral_tilt.weight) != spec["from"]
                or spec["from"] != 0.0 or "contact_lateral_tilt" in after):
            raise ValueError("Contact tilt must start disabled in the original base")
        target = float(spec["to"])
        if not math.isfinite(target) or target >= 0:
            raise ValueError("Contact tilt cost must have a finite negative weight")
        actual_parent = {n: w for n, w in after.items() if n != "contact_lateral_tilt"}
        if actual_parent != delta["parent_reward_weights"]:
            raise ValueError("Existing parent reward inventory changed while adding contact tilt")
        env_cfg.rewards.contact_lateral_tilt.weight = target
        after["contact_lateral_tilt"] = target
        addition_audit = {"contact_lateral_tilt": {"from": 0.0, "to": target}}
    return {"reward_weights_before":before,"reward_weights_after":after,
            "reward_parameter_changes":parameter_changes,
            "reward_additions":addition_audit,
            "command_ranges":delta["resume_command_ranges"],
            "episode_state_restored":False,"rng_state_restored":False,
            "scope":"Same actor/optimizer, observed command curriculum; simulator episodes/terrain assignments reset"}


def restore_adaptive_lr(runner, parent):
    rates = {float(g["lr"]) for g in runner.alg.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError("Cannot infer one adaptive learning rate from multiple optimizer rates")
    actual = rates.pop()
    if not math.isfinite(actual) or actual <= 0 or actual != parent["optimizer_learning_rate"]:
        raise ValueError("Loaded optimizer rate differs from registered parent")
    if runner.current_learning_iteration != parent["iteration"]:
        raise ValueError("Native resume iteration differs from registered parent")
    # RSL3.0.1 loads Adam groups but does not restore PPO.learning_rate. Keep
    # the already-restored rate instead of restarting from the config default.
    runner.alg.learning_rate = actual
    steps = optimizer_steps(runner.alg.optimizer.state_dict())
    if steps != [parent["optimizerStep"]]:
        raise ValueError("Optimizer counters did not restore from the actual parent")
    return {"parent":parent,"restored_adaptive_learning_rate":actual,
            "restored_optimizer_steps":steps,"restored_iteration":runner.current_learning_iteration}


def optimizer_steps(state):
    return sorted({int(float(v["step"])) for v in state["state"].values() if "step" in v})


def install_finite_guard(runner, log_dir):
    import torch
    policy = runner.alg.policy
    def check(stage):
        invalid = [name for name,p in policy.named_parameters() if not bool(torch.isfinite(p).all())]
        if hasattr(policy,"std") and (not bool(torch.isfinite(policy.std).all()) or not bool((policy.std>0).all())):
            invalid.append("std_nonpositive_or_nonfinite")
        if invalid:
            path=Path(log_dir)/"params/numerical_failure.json"
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({"stage":stage,"iteration":runner.current_learning_iteration,
                "invalid":invalid,"action":"abort; preserve previous valid bound checkpoints"}),encoding="utf-8")
            raise FloatingPointError("F1 numerical guard: "+stage+": "+str(invalid))
    native_update = runner.alg.update
    def update():
        check("before_update")
        values = native_update()
        check("after_update")
        if any(not math.isfinite(float(v)) for v in values.values()):
            raise FloatingPointError("Non-finite native PPO loss; no success claimed")
        return values
    check("restored_parent")
    runner.alg.update = update
