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
        yaw_term = env.reward_manager.get_term_cfg("track_ang_vel_z")
        if yaw_term.func.__name__ == "track_ang_vel_z_window_exp":
            runtime["training_objective"]["track_ang_vel_z_window"] = {
                "function": yaw_term.func.__name__,
                "weight": float(yaw_term.weight),
                "std": float(yaw_term.params["std"]),
                "window_s": float(yaw_term.params["window_s"]),
                "gate": "forward_and_wz_nonzero",
                "formula": "exp(-(cmd_wz-mean(last window_s body wz))^2/std^2); window only while forward and cmd_wz!=0; instantaneous otherwise",
                "scope": "Reward only; actor inputs, actions, sampling and weights unchanged"}
        command = env.command_manager.get_term("base_velocity")
        runtime["training_objective"]["command_sampling"] = {
            "rel_standing_envs": float(command.cfg.rel_standing_envs),
            "resampling_time_range": list(command.cfg.resampling_time_range),
            "source": "Resolved native CommandManager base_velocity term cfg"}
        if hasattr(command.cfg, "forward_exposure"):
            from f1_rl_lab.command_exposure import (
                SOURCE_PATH,
                ForwardExposureVelocityCommand,
                SignedTurnVelocityCommand,
            )

            if not isinstance(command, (ForwardExposureVelocityCommand, SignedTurnVelocityCommand)):
                raise ValueError("Forward exposure did not reach the native command manager")
            runtime["training_objective"]["command_sampling"]["forward_exposure"] = {
                **command.cfg.forward_exposure, "source_sha256": sha256(ROOT / SOURCE_PATH)}
            if "signed_turn" in command.cfg.forward_exposure and "turn_s" in command.cfg.forward_exposure["signed_turn"]:
                runtime["training_objective"]["command_sampling"]["forward_exposure"]["signed_turn_turn_window_s"] = (
                    float(command.cfg.forward_exposure["signed_turn"]["turn_s"]))
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
            if term.func.__name__ == "feet_loaded_lateral_tilt":
                from f1_rl_lab.action_limits import action_source_binding
                from f1_rl_lab.foot_posture import LOADED_LATERAL_TILT_FORMULAS
                if action_clip is not None or projection is not None or term.params.get("command_name") != "base_velocity":
                    raise ValueError("Loaded-foot repair requires original actions and base_velocity command")
                name = "source/f1_rl_lab/f1_rl_lab/foot_posture.py"
                runtime["format"] = "f1_fullbody_runtime_v6"
                runtime["action_interface_sources"] = action_source_binding(ROOT, projection=True)
                runtime["training_objective"]["custom_reward_sources"] = {name: sha256(ROOT / name)}
                aggregation = ("positive_world_fz_moving_left" if term.params.get("left_only", False)
                               else "positive_world_fz_moving")
                runtime["training_objective"]["contact_lateral_tilt"].update(
                    formula=LOADED_LATERAL_TILT_FORMULAS[aggregation], aggregation=aggregation,
                    command_name="base_velocity")
        gait = env.reward_manager.get_term_cfg("gait")
        gait_name = gait.func.__name__
        if gait_name in ("feet_gait_with_forward_progress", "AlternatingSupportGait"):
            from f1_rl_lab.forward_gait import (
                ALTERNATING_ADAPTER, ALTERNATING_METHOD, METHOD, SOURCE_PATH, SPEED_MATCH_METHOD,
            )

            sensor = gait.params["sensor_cfg"]
            body_names = env.scene.sensors[sensor.name].body_names
            if [body_names[i] for i in sensor.body_ids] != ["left_ankle_roll", "right_ankle_roll"]:
                raise ValueError("Forward gait uses named left/right contact sensors in order")
            alternating = gait_name == "AlternatingSupportGait"
            spec = {
                "method": ALTERNATING_METHOD if alternating else METHOD,
                "activation_speed": gait.params["activation_speed"],
                "source_sha256": sha256(ROOT / SOURCE_PATH),
            }
            if alternating:
                if gait.params.get("match_speed", False):
                    spec["method"] = SPEED_MATCH_METHOD
                spec.update(adapter_source_sha256=sha256(ROOT / ALTERNATING_ADAPTER),
                            swing_limit_s=gait.params["period"] * (1 - gait.params["threshold"]))
            runtime["training_objective"]["forward_gait"] = spec
    if "yaw_rate_error_l2" in env.reward_manager.active_terms:
        from f1_rl_lab.action_limits import action_source_binding
        from f1_rl_lab.contract import sha256
        if action_clip is not None or projection is not None:
            raise ValueError("Yaw-cost experiment requires the original11999 raw action mapping")
        term = env.reward_manager.get_term_cfg("yaw_rate_error_l2")
        if term.func.__name__ != "yaw_rate_error_l2" or term.params != {"command_name": "base_velocity", "asset_name": "robot"}:
            raise ValueError("Unexpected resolved yaw-cost function/parameters")
        runtime["format"] = "f1_fullbody_runtime_v4"
        runtime["action_interface_sources"] = action_source_binding(ROOT, projection=True)
        runtime["training_objective"]["custom_reward_sources"] = {
            "source/f1_rl_lab/f1_rl_lab/yaw_tracking.py": sha256(ROOT / "source/f1_rl_lab/f1_rl_lab/yaw_tracking.py")}
        runtime["training_objective"]["yaw_rate_error_l2"] = {"weight": float(term.weight),
            "formula": "(command_base_velocity_z-root_ang_vel_b_z)^2", "frame": "root_body_z",
            "command_name": "base_velocity", "asset_name": "robot"}
    if "zero_command_ankle_action_rate_l2" in env.reward_manager.active_terms:
        from f1_rl_lab.contract import sha256
        if runtime["format"] != "f1_fullbody_runtime_v4":
            raise ValueError("Standing continuation requires original-action yaw-cost runtime")
        term = env.reward_manager.get_term_cfg("zero_command_ankle_action_rate_l2")
        ankles = [f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll")]
        ids = [action_names.index(n) for n in ankles]
        if (term.func.__name__ != "zero_command_ankle_action_rate_l2"
                or term.params != {"command_name": "base_velocity", "ankle_action_ids": ids}):
            raise ValueError("Unexpected resolved standing smoothness function or ankle order")
        runtime["format"] = "f1_fullbody_runtime_v5"
        name = "source/f1_rl_lab/f1_rl_lab/standing_smoothness.py"
        runtime["training_objective"]["custom_reward_sources"][name] = sha256(ROOT / name)
        runtime["training_objective"]["zero_command_ankle_action_rate_l2"] = {
            "weight": float(term.weight), "function": term.func.__name__,
            "command_name": "base_velocity", "ankle_joint_names": ankles, "ankle_action_ids": ids,
            "gate": "all(command_vx,command_vy,command_wz)==0",
            "formula": "zero_command * sum_4ankles((raw_action-previous_raw_action)^2)",
            "scope": "Training-only reward; no observation, action filtering, target clipping or state mutation"}
    target = Path(log_dir) / "params/f1_runtime.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime
