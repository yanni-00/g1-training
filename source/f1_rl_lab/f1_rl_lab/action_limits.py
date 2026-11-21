"""Explicit processed-position clipping contract, separate from raw policy actions."""
from pathlib import Path


def position_clip_contract(contract, projected=False):
    names = contract["model_joint_names"]
    return {
        "kind": "urdf_joint_position_limits_v1", "joint_names": names,
        "lower": [contract["joint_properties"][n]["limit"]["lower"] for n in names],
        "upper": [contract["joint_properties"][n]["limit"]["upper"] for n in names],
        "urdf_sha256": contract["urdf_sha256"], "units": "rad",
        "stage": "after_scale_and_default_offset",
        "last_action": "projected_environment_action" if projected else "raw_network_output",
    }


def validate_position_clip(spec, contract, projected=None):
    allowed = [position_clip_contract(contract, projected=p) for p in
               ([False, True] if projected is None else [projected])]
    if spec is not None and spec not in allowed:
        raise ValueError("Processed-position clipping differs from exact named URDF limits")


def action_source_binding(root, projection=False):
    from .contract import sha256
    names = ["source/f1_rl_lab/f1_rl_lab/action_limits.py",
             "source/f1_rl_lab/f1_rl_lab/interface.py",
             "source/f1_rl_lab/f1_rl_lab/checkpoint.py",
             "source/f1_rl_lab/f1_rl_lab/contract.py",
             "source/f1_rl_lab/f1_rl_lab/continuation.py",
             "source/f1_rl_lab/f1_rl_lab/utils/export_deploy_cfg.py",
             "scripts/rsl_rl/_isaac_train.py", "scripts/rsl_rl/sim2sim_play.py"]
    if projection:
        names.append("source/f1_rl_lab/f1_rl_lab/action_projection.py")
    return {name: sha256(Path(root) / name) for name in names}


def verify_runtime_action_contract(runtime, contract, root):
    if runtime.get("format") == "f1_fullbody_runtime_v6":
        from .contract import sha256
        from .foot_posture import LOADED_LATERAL_TILT_FORMULAS
        name = "source/f1_rl_lab/f1_rl_lab/foot_posture.py"
        objective = runtime.get("training_objective", {})
        tilt = objective.get("contact_lateral_tilt", {})
        aggregation = tilt.get("aggregation")
        exposure = objective.get("command_sampling", {}).get("forward_exposure")
        feet = ["left_ankle_roll", "right_ankle_roll"]
        if (runtime.get("action_clip") is not None or runtime.get("action_input_projection") is not None
                or runtime.get("action_interface_sources") != action_source_binding(root, projection=True)
                or objective.get("custom_reward_sources") != {name: sha256(Path(root) / name)}
                or aggregation not in LOADED_LATERAL_TILT_FORMULAS
                or tilt.get("formula") != LOADED_LATERAL_TILT_FORMULAS[aggregation]
                or tilt.get("function") != "feet_loaded_lateral_tilt"
                or tilt.get("command_name") != "base_velocity"
                or tilt.get("asset_body_names") != feet or tilt.get("sensor_body_names") != feet):
            raise ValueError("Loaded-foot runtime_v6 source/action/foot/force binding changed")
        if exposure is not None:
            source = "source/f1_rl_lab/f1_rl_lab/command_exposure.py"
            keys = {"source_sha256", "vx", "group_modulus"}
            if (not isinstance(exposure, dict) or set(exposure) not in (keys, keys | {"route_stop"})
                    or exposure["source_sha256"] != sha256(Path(root) / source)):
                raise ValueError("Forward command exposure source binding changed")
        forward_gait = objective.get("forward_gait")
        if forward_gait is not None:
            from .forward_gait import METHOD, SOURCE_PATH
            from .contract import load_json

            delta = load_json(Path(root) / "configs/training_delta.json")
            if (not exposure or "route_stop" not in exposure
                    or not isinstance(forward_gait, dict)
                    or set(forward_gait) != {"method", "activation_speed", "source_sha256"}
                    or forward_gait["method"] != METHOD
                    or not 0 < forward_gait["activation_speed"] <= exposure["vx"]
                    or {k: forward_gait[k] for k in ("method", "activation_speed")} != delta.get("forward_gait")
                    or forward_gait["source_sha256"] != sha256(Path(root) / SOURCE_PATH)):
                raise ValueError("Forward gait reward source or training parameters changed")
    elif runtime.get("format") == "f1_fullbody_runtime_v5":
        from .contract import sha256
        rewards = ["source/f1_rl_lab/f1_rl_lab/yaw_tracking.py",
                   "source/f1_rl_lab/f1_rl_lab/standing_smoothness.py"]
        objective = runtime.get("training_objective", {})
        ankles = [f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll")]
        standing = objective.get("zero_command_ankle_action_rate_l2", {})
        if (runtime.get("action_clip") is not None or runtime.get("action_input_projection") is not None
                or runtime.get("action_interface_sources") != action_source_binding(root, projection=True)
                or objective.get("custom_reward_sources") != {n: sha256(Path(root)/n) for n in rewards}
                or standing.get("ankle_joint_names") != ankles
                or standing.get("ankle_action_ids") != [contract["model_joint_names"].index(n) for n in ankles]
                or standing.get("gate") != "all(command_vx,command_vy,command_wz)==0"):
            raise ValueError("Standing runtime_v5 source/action/joint/gate binding changed")
    elif runtime.get("format") == "f1_fullbody_runtime_v4":
        from .contract import sha256
        expected_rewards = {"source/f1_rl_lab/f1_rl_lab/yaw_tracking.py":
                            sha256(Path(root) / "source/f1_rl_lab/f1_rl_lab/yaw_tracking.py")}
        if (runtime.get("action_clip") is not None or runtime.get("action_input_projection") is not None
                or runtime.get("action_interface_sources") != action_source_binding(root, projection=True)
                or runtime.get("training_objective", {}).get("custom_reward_sources") != expected_rewards):
            raise ValueError("Yaw-cost runtime_v4 action/reward source binding changed")
    elif runtime.get("training_objective", {}).get("custom_reward_sources"):
        raise ValueError("Custom reward sources require explicit source-pinned runtime")
    spec = runtime.get("action_clip")
    projection = runtime.get("action_input_projection")
    validate_position_clip(spec, contract, projected=projection is not None)
    from .action_projection import validate_projection
    validate_projection(projection, contract)
    if projection is not None and (spec is None or runtime.get("format") != "f1_fullbody_runtime_v3"):
        raise ValueError("Environment action projection requires clipped runtime_v3")
    if spec is not None:
        expected_format = "f1_fullbody_runtime_v3" if projection is not None else "f1_fullbody_runtime_v2"
        if runtime.get("format") != expected_format:
            raise ValueError("Clipped targets require explicit runtime_v2 metadata")
        if runtime.get("action_interface_sources") != action_source_binding(root, projection=projection is not None):
            raise ValueError("Clipped checkpoint action-interface source binding changed")
