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
    if runtime.get("format") == "f1_fullbody_runtime_v5":
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
