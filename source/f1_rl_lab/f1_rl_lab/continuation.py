"""Narrow registered objective continuation and auditable resume state."""
import json
import math
from pathlib import Path


def contact_tilt_term(weight, *, aggregation="equal_contact"):
    """Register optional reward outside the strictly bound original base files."""
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from f1_rl_lab.foot_posture import (
        LOADED_LATERAL_TILT_FORMULAS,
        feet_contact_lateral_tilt,
        feet_loaded_lateral_tilt,
    )
    feet = ["left_ankle_roll", "right_ankle_roll"]
    params = {"asset_cfg": SceneEntityCfg("robot", body_names=feet, preserve_order=True),
              "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)}
    if aggregation == "equal_contact":
        return RewardTermCfg(func=feet_contact_lateral_tilt, weight=weight, params=params)
    if aggregation in LOADED_LATERAL_TILT_FORMULAS:
        params["command_name"] = "base_velocity"
        if aggregation == "positive_world_fz_moving_left":
            params["left_only"] = True
        return RewardTermCfg(func=feet_loaded_lateral_tilt, weight=weight, params=params)
    raise ValueError("Unregistered contact tilt aggregation")


def yaw_tracking_term(weight):
    from isaaclab.managers import RewardTermCfg
    from f1_rl_lab.yaw_tracking import yaw_rate_error_l2
    return RewardTermCfg(func=yaw_rate_error_l2, weight=weight,
                         params={"command_name": "base_velocity", "asset_name": "robot"})


def standing_smoothness_term(weight):
    from isaaclab.managers import RewardTermCfg
    from f1_rl_lab.contract import load_contract
    from f1_rl_lab.standing_smoothness import zero_command_ankle_action_rate_l2
    names = load_contract()["model_joint_names"]
    ankles = [f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll")]
    return RewardTermCfg(func=zero_command_ankle_action_rate_l2, weight=weight,
        params={"command_name": "base_velocity", "ankle_action_ids": [names.index(n) for n in ankles]})


def verify_resume_checkpoint(checkpoint, request, root):
    """Verify a source-pinned parent against its preserved producer, never waive pins."""
    from f1_rl_lab.checkpoint import verify_checkpoint
    from f1_rl_lab.contract import binding, sha256
    relative = request.get("producer_root")
    if relative is None:
        return verify_checkpoint(checkpoint, root=root)
    root = Path(root).resolve()
    producer = (root / relative).resolve()
    if not producer.is_relative_to(root) or producer == root:
        raise ValueError("Parent producer snapshot must stay inside the published source")
    if sha256(checkpoint) != request["sha256"]:
        raise ValueError("Parent checkpoint differs from registered exact bytes")
    identity = verify_checkpoint(checkpoint, root=producer)
    if identity["binding"] != binding(root):
        raise ValueError("Parent-to-child physical/observation/base source binding changed")
    runtime = identity["runtime"]
    expected = request.get("producer_runtime_format", "f1_fullbody_runtime_v4")
    if (expected not in {"f1_fullbody_runtime_v4", "f1_fullbody_runtime_v6"}
            or runtime.get("format") != expected or runtime.get("action_clip") is not None
            or runtime.get("action_input_projection") is not None):
        raise ValueError("Continuation parent must match its registered original-action runtime")
    return identity


def apply_delta(env_cfg, delta, *, initial_training=False):
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
    retained = delta.get("retained_custom_rewards", {})
    if retained:
        sampling_only = bool(delta.get("command_sampling_override")) and not delta.get("reward_addition")
        if (retained != {"yaw_rate_error_l2": -0.5}
                or (delta.get("reward_addition", {}).keys() != {"zero_command_ankle_action_rate_l2"} and not sampling_only)
                or "yaw_rate_error_l2" in after):
            raise ValueError("Only the actual12998 yaw term can be retained for standing continuation")
        env_cfg.rewards.yaw_rate_error_l2 = yaw_tracking_term(retained["yaw_rate_error_l2"])
        after["yaw_rate_error_l2"] = retained["yaw_rate_error_l2"]
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
    window = delta.get("yaw_window_override")
    if window:
        if (set(window) != {"function", "window_s", "from", "gate"}
                or window["function"] != "track_ang_vel_z_window_exp"
                or not math.isfinite(float(window["window_s"])) or float(window["window_s"]) <= 0
                or window.get("gate") != "forward_and_wz_nonzero"
                or after != delta["parent_reward_weights"]):
            raise ValueError("Only the registered cross-step yaw window override is allowed")
        from isaaclab.managers import RewardTermCfg
        from f1_rl_lab.yaw_tracking import track_ang_vel_z_window_exp
        term = env_cfg.rewards.track_ang_vel_z
        weight = float(term.weight)
        std = float(term.params["std"])
        env_cfg.rewards.track_ang_vel_z = RewardTermCfg(
            func=track_ang_vel_z_window_exp, weight=weight,
            params={"command_name": "base_velocity", "asset_name": "robot",
                    "std": std, "window_s": float(window["window_s"])})
        parameter_changes["track_ang_vel_z"]["window_s"] = {
            "from": window["from"], "to": float(window["window_s"]),
            "function": window["function"], "gate": window["gate"]}
    # The parent does not contain per-environment episode state. Restore the
    # observed curriculum command limits explicitly, never claim a bitwise resume.
    ranges = delta["initial_command_ranges" if initial_training else "resume_command_ranges"]
    for name, values in ranges.items():
        if name not in ("lin_vel_x","lin_vel_y","ang_vel_z"):
            raise ValueError("Unregistered command range field")
        setattr(env_cfg.commands.base_velocity.ranges,name,tuple(values))
    sampling_audit = {}
    sampling = delta.get("command_sampling_override", {})
    if sampling:
        if set(sampling) != {"rel_standing_envs"} or after != delta["parent_reward_weights"] or delta.get("reward_addition"):
            raise ValueError("Standing exposure permits one sampler field with exact parent rewards")
        spec = sampling["rel_standing_envs"]
        actual = env_cfg.commands.base_velocity.rel_standing_envs
        target = float(spec["to"])
        if actual != spec["from"] or not math.isfinite(target) or not 0 < target < 1:
            raise ValueError("Unexpected standing sampler baseline or invalid probability")
        env_cfg.commands.base_velocity.rel_standing_envs = target
        sampling_audit = {"rel_standing_envs": {"from": actual, "to": target}}
    additions = delta.get("reward_addition", {})
    addition_audit = {}
    if set(additions) == {"zero_command_ankle_action_rate_l2"}:
        spec = additions["zero_command_ankle_action_rate_l2"]
        if (not retained or spec["from"] != 0.0 or after != delta["parent_reward_weights"]
                or hasattr(env_cfg.rewards, "zero_command_ankle_action_rate_l2")):
            raise ValueError("Standing addition requires exact unchanged actual12998 reward inventory")
        target = float(spec["to"])
        if not math.isfinite(target) or target >= 0:
            raise ValueError("Standing action-rate cost must have a finite negative weight")
        env_cfg.rewards.zero_command_ankle_action_rate_l2 = standing_smoothness_term(target)
        after["zero_command_ankle_action_rate_l2"] = target
        addition_audit = {"zero_command_ankle_action_rate_l2": {"from": 0.0, "to": target}}
    elif set(additions) == {"yaw_rate_error_l2"}:
        spec = additions["yaw_rate_error_l2"]
        if (hasattr(env_cfg.rewards, "yaw_rate_error_l2")
                or spec["from"] != 0.0 or "yaw_rate_error_l2" in after
                or after != delta["parent_reward_weights"]):
            raise ValueError("Yaw L2 addition requires exact unchanged parent reward inventory")
        target = float(spec["to"])
        if not math.isfinite(target) or target >= 0:
            raise ValueError("Yaw error cost requires a finite negative weight")
        env_cfg.rewards.yaw_rate_error_l2 = yaw_tracking_term(target)
        after["yaw_rate_error_l2"] = target
        addition_audit = {"yaw_rate_error_l2": {"from": 0.0, "to": target}}
    elif additions:
        if set(additions) != {"contact_lateral_tilt"}:
            raise ValueError("Only the registered contact tilt term can be added")
        spec = additions["contact_lateral_tilt"]
        if (hasattr(env_cfg.rewards, "contact_lateral_tilt")
                or spec["from"] != 0.0 or "contact_lateral_tilt" in after):
            raise ValueError("Contact tilt must not exist in the original base")
        target = float(spec["to"])
        if not math.isfinite(target) or target >= 0:
            raise ValueError("Contact tilt cost must have a finite negative weight")
        actual_parent = {n: w for n, w in after.items() if n != "contact_lateral_tilt"}
        if actual_parent != delta["parent_reward_weights"]:
            raise ValueError("Existing parent reward inventory changed while adding contact tilt")
        rule = spec.get("aggregation", "equal_contact")
        env_cfg.rewards.contact_lateral_tilt = contact_tilt_term(target, aggregation=rule)
        after["contact_lateral_tilt"] = target
        addition_audit = {"contact_lateral_tilt": {"from": 0.0, "to": target}}
    forward_exposure = delta.get("forward_exposure")
    if forward_exposure is not None:
        from f1_rl_lab.command_exposure import install_forward_exposure
        if (sampling or retained or set(additions) != {"contact_lateral_tilt"}
                or additions["contact_lateral_tilt"].get("aggregation") != "positive_world_fz_moving"
                or delta.get("action_postprocessing_override") or delta.get("action_input_projection")):
            raise ValueError("Forward exposure retains the bilateral loaded-foot objective and original actions")
        install_forward_exposure(env_cfg.commands.base_velocity, forward_exposure)
        sampling_audit["forward_exposure"] = dict(forward_exposure)
    gait_method = delta.get("forward_gait")
    if gait_method is not None:
        from f1_rl_lab.forward_gait import ALTERNATING_METHOD, METHOD, SPEED_MATCH_METHOD, feet_gait_with_forward_progress
        from isaaclab.managers import SceneEntityCfg

        if (not initial_training or not forward_exposure or "route_stop" not in forward_exposure
                or set(gait_method) != {"method", "activation_speed"}
                or gait_method["method"] not in (METHOD, ALTERNATING_METHOD, SPEED_MATCH_METHOD)
                or gait_method["activation_speed"] != ranges["lin_vel_x"][1]
                or not 0 < gait_method["activation_speed"] <= forward_exposure["vx"]):
            raise ValueError("Forward gait replacement needs the registered fresh route curriculum")
        gait_term = env_cfg.rewards.gait
        if gait_term.func.__name__ != "feet_gait" or gait_term.params["command_name"] != "base_velocity":
            raise ValueError("Unexpected original gait term for forward gait replacement")
        if gait_method["method"] in (ALTERNATING_METHOD, SPEED_MATCH_METHOD):
            from f1_rl_lab.alternating_support_reward import AlternatingSupportGait

            gait_term.func = AlternatingSupportGait
            gait_term.params["match_speed"] = gait_method["method"] == SPEED_MATCH_METHOD
        else:
            gait_term.func = feet_gait_with_forward_progress
        gait_term.params["activation_speed"] = gait_method["activation_speed"]
        gait_term.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["left_ankle_roll", "right_ankle_roll"], preserve_order=True)
    action_change = None
    if delta.get("action_postprocessing_override"):
        from f1_rl_lab.contract import load_contract
        from f1_rl_lab.action_limits import position_clip_contract
        change = delta["action_postprocessing_override"]
        target = position_clip_contract(load_contract())
        if (set(change) != {"parent_action_clip", "to"} or change["parent_action_clip"] is not None
                or change["to"] != target or env_cfg.actions.JointPositionAction.clip is not None):
            raise ValueError("Only registered legacy-to-URDF processed-position clipping is allowed")
        if after != delta["parent_reward_weights"]:
            raise ValueError("Action adaptation must retain the exact parent reward inventory")
        env_cfg.actions.JointPositionAction.clip = {
            name: (lower, upper) for name, lower, upper in
            zip(target["joint_names"], target["lower"], target["upper"])}
        action_change = {"from": None, "to": target,
                         "scope": "Native processed position target clip only; raw actions/history, scale, PD and physics unchanged"}
    return {"reward_weights_before":before,"reward_weights_after":after,
            "reward_parameter_changes":parameter_changes,
            "reward_additions":addition_audit,
            **({"forward_gait": dict(gait_method)} if gait_method is not None else {}),
            "retained_custom_rewards": retained,
            "command_sampling_changes": sampling_audit,
            "action_postprocessing_change": action_change,
            "command_ranges": ranges,
            "episode_state_restored":False,"rng_state_restored":False,
            "scope": ("Random actor/optimizer initialization; registered starting curriculum"
                      if initial_training else
                      "Same actor/optimizer, observed command curriculum; simulator episodes/terrain assignments reset")}


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
        invalid_losses = {k: repr(float(v)) for k, v in values.items() if not math.isfinite(float(v))}
        if invalid_losses:
            path = Path(log_dir) / "params/numerical_failure.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"stage": "after_native_update",
                "iteration": runner.current_learning_iteration, "invalid_losses": invalid_losses}), encoding="utf-8")
            raise FloatingPointError("Non-finite native PPO loss: " + str(invalid_losses) + "; no success claimed")
        raw_env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else None
        if raw_env is not None and hasattr(raw_env, "f1_action_projection_audit"):
            path = Path(log_dir) / "params/action_projection_training.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"iteration": runner.current_learning_iteration,
                    "losses": {k: float(v) for k, v in values.items()},
                    "projection": dict(raw_env.f1_action_projection_audit)}, allow_nan=False) + "\n")
        return values
    check("restored_parent")
    runner.alg.update = update
