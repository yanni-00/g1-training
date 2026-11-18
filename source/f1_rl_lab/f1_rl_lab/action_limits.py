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
