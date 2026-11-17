"""Explicit processed-position clipping contract, separate from raw policy actions."""
from pathlib import Path


def position_clip_contract(contract):
    names = contract["model_joint_names"]
    return {
        "kind": "urdf_joint_position_limits_v1", "joint_names": names,
        "lower": [contract["joint_properties"][n]["limit"]["lower"] for n in names],
        "upper": [contract["joint_properties"][n]["limit"]["upper"] for n in names],
        "urdf_sha256": contract["urdf_sha256"], "units": "rad",
        "stage": "after_scale_and_default_offset", "last_action": "raw_network_output",
    }


def validate_position_clip(spec, contract):
    if spec is not None and spec != position_clip_contract(contract):
        raise ValueError("Processed-position clipping differs from exact named URDF limits")


def action_source_binding(root):
    from .contract import sha256
    names = ["source/f1_rl_lab/f1_rl_lab/action_limits.py",
             "source/f1_rl_lab/f1_rl_lab/interface.py",
             "source/f1_rl_lab/f1_rl_lab/checkpoint.py",
             "source/f1_rl_lab/f1_rl_lab/contract.py",
             "source/f1_rl_lab/f1_rl_lab/continuation.py",
             "source/f1_rl_lab/f1_rl_lab/utils/export_deploy_cfg.py",
             "scripts/rsl_rl/_isaac_train.py", "scripts/rsl_rl/sim2sim_play.py"]
    return {name: sha256(Path(root) / name) for name in names}


def verify_runtime_action_contract(runtime, contract, root):
    spec = runtime.get("action_clip")
    validate_position_clip(spec, contract)
    if spec is not None:
        if runtime.get("format") != "f1_fullbody_runtime_v2":
            raise ValueError("Clipped targets require explicit runtime_v2 metadata")
        if runtime.get("action_interface_sources") != action_source_binding(root):
            raise ValueError("Clipped checkpoint action-interface source binding changed")
