"""Export the actual F1 environment contract without rounding or SDK assumptions."""
import json
from pathlib import Path

import numpy as np

from f1_rl_lab.contract import ROOT, binding, effective_control, load_contract
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
    action_clip = None
    if action.cfg.clip is not None:
        from f1_rl_lab.action_limits import position_clip_contract
        action_clip = position_clip_contract(c)
        expected = {n: (lo, hi) for n, lo, hi in
                    zip(action_clip["joint_names"], action_clip["lower"], action_clip["upper"])}
        if action.cfg.clip != expected:
            raise ValueError("Native processed target clip differs from named URDF limits")
        actual_clip = _array(action._clip)
        expected_clip = np.column_stack([action_clip["lower"], action_clip["upper"]])
        if not np.allclose(actual_clip, expected_clip, rtol=1e-6, atol=1e-6):
            raise ValueError("Resolved native clipping tensor differs from runtime action contract")
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
                "action_scale": c["action_scale"], "action_clip": action_clip,
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
    if action_clip is not None:
        from f1_rl_lab.action_limits import action_source_binding
        runtime["format"] = "f1_fullbody_runtime_v2"
        runtime["action_interface_sources"] = action_source_binding(ROOT)
    projection = getattr(env, "f1_action_input_projection", None)
    if projection is not None:
        from f1_rl_lab.action_projection import validate_projection
        from f1_rl_lab.action_limits import action_source_binding
        validate_projection(projection, c)
        if action_clip is None:
            raise ValueError("Projected input requires explicit processed target clipping")
        runtime["format"] = "f1_fullbody_runtime_v3"
        runtime["action_clip"] = {**action_clip, "last_action": "projected_environment_action"}
        runtime["action_input_projection"] = projection
        runtime["action_interface_sources"] = action_source_binding(ROOT, projection=True)
    delta_path = ROOT / "configs/training_delta.json"
    if delta_path.is_file():
        from f1_rl_lab.contract import sha256
        runtime["training_objective"] = {
            "yaw_tracking_std": float(env.reward_manager.get_term_cfg("track_ang_vel_z").params["std"]),
            "delta_sha256": sha256(delta_path),
            "reward_weights": {n: float(env.reward_manager.get_term_cfg(n).weight)
                               for n in env.reward_manager.active_terms},
            "scope": "Actual runtime reward inventory; effective action postprocessing is recorded separately"}
        if "contact_lateral_tilt" in env.reward_manager.active_terms:
            term = env.reward_manager.get_term_cfg("contact_lateral_tilt")
            ac, sc = term.params["asset_cfg"], term.params["sensor_cfg"]
            sensor = env.scene.sensors[sc.name]
            runtime["training_objective"]["contact_lateral_tilt"] = {
                "weight": float(term.weight), "function": term.func.__name__,
                "asset_body_names": [asset.body_names[i] for i in ac.body_ids],
                "sensor_body_names": [sensor.body_names[i] for i in sc.body_ids],
                "asset_body_ids": list(ac.body_ids), "sensor_body_ids": list(sc.body_ids),
                "quaternion_order": "wxyz", "contact": "current_contact_time>0",
                "formula": "mean over contacting feet of (2*(w*x+y*z))^2; zero if no contact",
                "scope": "Reward only; no actor inputs or action processing changed",
            }
    target = Path(log_dir) / "params/f1_runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime
