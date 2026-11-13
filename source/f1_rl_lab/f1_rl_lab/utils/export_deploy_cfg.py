"""Export the actual F1 environment contract without rounding or SDK assumptions."""
import json
from pathlib import Path

import numpy as np

from f1_rl_lab.contract import binding, effective_control, load_contract
from f1_rl_lab.interface import joint_permutation


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def export_deploy_cfg(env, log_dir):
    c = load_contract()
    asset = env.scene["robot"]
    names = list(asset.data.joint_names)
    model_to_engine = joint_permutation(names, c["model_joint_names"])
    if list(env.action_manager.active_terms) != ["JointPositionAction"]:
        raise ValueError("Unexpected action manager; update the F1 contract before training")
    action = env.action_manager._terms["JointPositionAction"]
    action_ids = list(range(len(names))) if isinstance(action._joint_ids, slice) else list(action._joint_ids)
    action_names = [names[i] for i in action_ids]
    if action_names != c["model_joint_names"]:
        raise ValueError("Actual action order differs from the F1 model contract")
    if not np.isclose(env.step_dt, c["step_dt"]):
        raise ValueError("Actual control timestep differs from the F1 contract")
    expected_terms = [t["name"] for t in c["observation_terms"]]
    if list(env.observation_manager.active_terms["policy"]) != expected_terms:
        raise ValueError("Unexpected actor inputs")
    actual_dim = int(np.prod(env.observation_manager.group_obs_dim["policy"]))
    if actual_dim != c["observation_dim"]:
        raise ValueError("Actual actor dimension differs from the contract")
    # Explicit ordering above permits PhysX to enumerate joints differently from URDF XML.
    def ordered(field):
        return _array(getattr(asset.data, field))[0, model_to_engine].tolist()
    defaults = ordered("default_joint_pos")
    if not np.allclose(defaults, c["default_joint_pos"]):
        raise ValueError("Actual default pose differs from the F1 contract")
    scale = _array(action._scale)
    offset = _array(action._offset)
    if not np.allclose(scale, c["action_scale"]) or not np.allclose(offset, c["default_joint_pos"]):
        raise ValueError("Actual action scaling/offset differs from the F1 contract")
    if action.cfg.clip is not None:
        raise ValueError("New action clipping requires an explicit contract revision")
    observations = []
    for name, cfg, spec in zip(expected_terms, env.observation_manager._group_obs_term_cfgs["policy"], c["observation_terms"]):
        if cfg.history_length != c["history_length"]:
            raise ValueError(f"History differs for {name}")
        raw = cfg.func(env, **cfg.params)
        if raw.shape[-1] != spec["dim"]:
            raise ValueError(f"Raw dimension differs for {name}")
        term_scale = 1.0 if cfg.scale is None else _array(cfg.scale)
        if not np.allclose(term_scale, spec["scale"]) or cfg.clip is not None:
            raise ValueError(f"Scaling/clip differs for {name}")
        observations.append({**spec, "history_length": cfg.history_length})
    control = effective_control()
    kp, kd = ordered("default_joint_stiffness"), ordered("default_joint_damping")
    if not np.allclose(kp, [control[n]["stiffness"] for n in c["model_joint_names"]]):
        raise ValueError("Actual stiffness differs from effective control")
    if not np.allclose(kd, [control[n]["total_damping"] for n in c["model_joint_names"]]):
        raise ValueError("Actual damping differs from effective control")
    runtime = {"format": "f1_fullbody_runtime_v1", "binding": binding(), "urdf_sha256": c["urdf_sha256"],
               "policy_joint_names": action_names, "engine_joint_names": names,
               "model_to_engine": model_to_engine, "hardware_sdk_order": None,
               "default_joint_pos": defaults, "observation_dim": actual_dim,
               "observation_terms": observations, "history_length": c["history_length"],
               "history_layout": c["history_layout"], "action_dim": action.action_dim,
               "action_scale": c["action_scale"], "action_clip": None,
               "step_dt": env.step_dt, "sim_dt": env.cfg.sim.dt, "decimation": env.cfg.decimation,
               "gravity": list(env.cfg.sim.gravity),
               "kp_policy": kp, "total_damping_policy": kd,
               "pd_damping_policy": [control[n]["pd_damping"] for n in action_names],
               "passive_damping_policy": [control[n]["passive_damping"] for n in action_names],
               "armature_policy": [control[n]["armature"] for n in action_names],
               "effort_limit_policy": [control[n]["effort_limit"] for n in action_names],
               "velocity_limit_policy": [control[n]["velocity_limit"] for n in action_names],
               "empirical_normalization": False,
               "deployment_status": "offline interface only; actual F1 hardware chain not verified"}
    target = Path(log_dir) / "params/f1_runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime
