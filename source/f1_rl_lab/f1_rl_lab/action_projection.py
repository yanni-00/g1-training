"""Project environment actions, history and rate costs onto named URDF limits.

PPO retains its original Gaussian sample/log probability. Only the input to the
environment is projected, before ActionManager stores current/previous actions.
No guessed scalar action bound and no changes to PD, pose, scaling or rewards.
"""
import numpy as np


def projection_contract(contract):
    scale = np.asarray(contract["action_scale"], dtype=np.float64)
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Projection requires finite positive named action scaling")
    names = contract["model_joint_names"]
    offset = np.asarray(contract["default_joint_pos"], dtype=np.float64)
    lower = np.asarray([contract["joint_properties"][n]["limit"]["lower"] for n in names])
    upper = np.asarray([contract["joint_properties"][n]["limit"]["upper"] for n in names])
    return {"kind": "urdf_environment_action_projection_v1", "joint_names": names,
            "lower": ((lower - offset) / scale).tolist(),
            "upper": ((upper - offset) / scale).tolist(),
            "urdf_sha256": contract["urdf_sha256"],
            "stage": "before_action_manager_process_action",
            "last_action_and_action_rate": "projected_environment_action",
            "ppo_sample_and_log_probability": "original_unmodified_gaussian"}


def validate_projection(spec, contract):
    if spec is not None and spec != projection_contract(contract):
        raise ValueError("Environment action projection differs from named URDF-derived bounds")


def project_numpy(actions, spec, contract):
    validate_projection(spec, contract)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != contract["action_dim"] or not np.isfinite(actions).all():
        raise FloatingPointError("Non-finite or malformed policy action before projection")
    if spec is None:
        return actions.copy()
    return np.clip(actions, np.asarray(spec["lower"], dtype=actions.dtype),
                   np.asarray(spec["upper"], dtype=actions.dtype))


def install_projection(env, spec, contract):
    """Install once, before stepping/export; manager remains the native manager."""
    import torch
    validate_projection(spec, contract)
    if spec is None or hasattr(env, "f1_action_input_projection"):
        raise ValueError("Require one explicit, new projection installation")
    manager = env.action_manager
    native = manager.process_action
    lower = torch.tensor(spec["lower"], device=env.device, dtype=torch.float32)
    upper = torch.tensor(spec["upper"], device=env.device, dtype=torch.float32)
    audit = {"calls": 0, "projected_elements": 0, "raw_absmax": 0.0,
             "projected_absmax": 0.0, "nonfinite_actions": 0,
             "action_rate_l2_max": 0.0}

    def process(actions):
        if actions.shape[-1] != contract["action_dim"] or not bool(torch.isfinite(actions).all()):
            audit["nonfinite_actions"] += 1
            raise FloatingPointError("Non-finite or malformed native policy action before projection")
        projected = torch.clamp(actions, min=lower, max=upper)
        # Never mutate actions: RSL storage/log-probabilities own this tensor.
        audit["calls"] += 1
        audit["projected_elements"] += int((projected != actions).sum().item())
        audit["raw_absmax"] = max(audit["raw_absmax"], float(actions.abs().max().item()))
        audit["projected_absmax"] = max(audit["projected_absmax"], float(projected.abs().max().item()))
        result = native(projected)
        if not torch.equal(manager.action, projected):
            raise ValueError("Native ActionManager did not retain projected history action")
        rate = (manager.action - manager.prev_action).square().sum(-1)
        if not bool(torch.isfinite(rate).all()):
            raise FloatingPointError("Non-finite projected native action-rate cost")
        audit["action_rate_l2_max"] = max(audit["action_rate_l2_max"], float(rate.max().item()))
        return result

    manager.process_action = process
    env.f1_action_input_projection = spec
    env.f1_action_projection_audit = audit
