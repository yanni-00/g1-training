"""Bind F1 checkpoint/reference/control/normalization without G1 compatibility."""
import torch
from f1_beyondmimic.contract import FORMAT, ROOT, binding, effective_control, load_contract, load_json, sha256
from .actor_observations import audit_native_observations


def audit_native_contact(env):
    """Check resolved entities/indices, not SceneEntityCfg object identity."""
    base = env.unwrapped
    c = load_contract()
    expected = load_json(ROOT / "configs/reference/stable_runtime_settings.json")["reward_parameters"]["feet_contact_planar_speed"]
    manager = base.reward_manager
    if "feet_contact_planar_speed" not in manager.active_terms:
        raise RuntimeError("F1 native bilateral contact cost is absent")
    live = manager.get_term_cfg("feet_contact_planar_speed")
    if live.weight != expected["weight"] or live.func.__module__ != "f1_beyondmimic.tasks.tracking.mdp.rewards":
        raise RuntimeError("F1 native contact cost differs from the selected stable method")
    resolved = {}
    for key, entity_name in (("asset_cfg", "robot"), ("sensor_cfg", "contact_forces")):
        selection = live.params[key]
        entity = base.scene[entity_name]
        indices = selection.body_ids
        if isinstance(indices, slice):
            indices = list(range(len(entity.body_names)))[indices]
        else:
            indices = list(indices)
        expected_ids = [entity.body_names.index(name) for name in c["feet"]]
        if (selection.name != entity_name or selection.body_names != c["feet"]
                or selection.preserve_order is not True or indices != expected_ids):
            raise RuntimeError("F1 contact cost native selector/order mismatch: " + key)
        resolved[key] = {"body_names": c["feet"], "body_ids": indices}
    if live.params["force_threshold"] != base.scene["contact_forces"].cfg.force_threshold:
        raise RuntimeError("Native contact threshold differs from actual sensor")
    value = live.func(base, **live.params)
    if value.shape != (base.num_envs,) or not torch.isfinite(value).all() or (value < 0).any():
        raise RuntimeError("F1 native contact cost execution failed")
    return {"weight": live.weight, "force_threshold": live.params["force_threshold"],
            "entities": resolved, "finite_nonnegative": True}


def collect_runtime(env, policy):
    base = env.unwrapped
    c = load_contract()
    command = base.command_manager.get_term("motion")
    robot = command.robot
    names = list(robot.joint_names)
    if set(names) != set(c["model_joint_names"]):
        raise RuntimeError("Native articulation does not contain exactly the F1 joints")
    action = base.action_manager.get_term("joint_pos")
    if list(action._joint_names) != names:
        raise RuntimeError("Native action order differs from reference/articulation")
    data = robot.data
    nominal = getattr(data, "default_joint_pos_nominal", None)
    if nominal is None:
        raise RuntimeError("Nominal pre-randomization pose was not recorded")
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    if getattr(policy, "actor_obs_normalization", False) is not True or normalizer is None:
        raise RuntimeError("Native empirical actor normalization must be active")
    actor_contract = audit_native_observations(env)
    if policy.actor[0].in_features != actor_contract["dimension"]:
        raise RuntimeError("Policy input differs from native observation manager")
    scale = action._scale
    if isinstance(scale, torch.Tensor):
        scale = scale[0].detach().cpu().clone()
    else:
        scale = torch.full((len(names),), float(scale))
    limits = data.joint_pos_limits[0]
    control = {"default_joint_pos": nominal.detach().cpu().clone(), "action_scale": scale,
               "joint_lower": limits[:, 0].detach().cpu().clone(),
               "joint_upper": limits[:, 1].detach().cpu().clone(),
               "kp": data.default_joint_stiffness[0].detach().cpu().clone(),
               "kd_total": data.default_joint_damping[0].detach().cpu().clone(),
               "armature": data.default_joint_armature[0].detach().cpu().clone()}
    if any(v.shape != (len(names),) or not torch.isfinite(v).all() for v in control.values()):
        raise RuntimeError("Invalid native F1 control tensors")
    expected_controls = effective_control()
    for key, expected in {
        "kp": [expected_controls[name]["stiffness"] for name in names],
        "kd_total": [expected_controls[name]["pd_damping"] + expected_controls[name]["passive_damping"] for name in names],
        "armature": [expected_controls[name]["armature"] for name in names],
        "action_scale": [expected_controls[name]["action_scale"] for name in names],
    }.items():
        torch.testing.assert_close(control[key], torch.tensor(expected, dtype=control[key].dtype))
    contact_audit = audit_native_contact(env)
    motion = {k: v.detach().cpu().clone() for k, v in command.reference.arrays.items()}
    motion.update(fps=[command.reference.fps], cycle_frames=[command.clock.period],
                  support_frames=command.clock.support.detach().cpu().clone(),
                  nominal_reference_speed_mps=[command.reference.nominal_speed])
    if len(robot.body_names) != motion["body_pos_w"].shape[1]:
        raise RuntimeError("Native F1 reference body ordering mismatch")
    return {"format": FORMAT, "binding": binding(), "urdf_sha256": c["urdf_sha256"],
            "actor_observation_contract": actor_contract, "actor_obs_normalization": True,
            "normalizer_epsilon": float(normalizer.eps), "joint_names": names,
            "body_names": list(robot.body_names), "tracked_body_names": c["tracked_body_names"],
            "anchor_body_name": c["anchor_body_name"], "policy_dt": float(base.step_dt),
            "phase_acceleration": float(command.cfg.phase_acceleration),
            "command_limits": [list(v) for v in command.command_limits],
            "reference_sha256": sha256(command.cfg.motion_file), "motion": motion, "control": control,
            "action_processing": "default_plus_scaled_raw_then_urdf_hard_clip_v1",
            "native_contact_audit": contact_audit,
            "training_launch": load_json(ROOT / "configs/launch.json"),
            "deployment_evidence": load_json(ROOT / "configs/deployment_contract.json")}


def load_checkpoint(path, check_binding=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    runtime = checkpoint.get("infos", {}).get("f1_runtime", {})
    if runtime.get("format") != FORMAT or runtime.get("urdf_sha256") != load_contract()["urdf_sha256"]:
        raise ValueError("An actual checkpoint from this F1 task is required; G1 weights are incompatible")
    if check_binding and runtime.get("binding") != binding():
        raise ValueError("F1 source/data/control binding changed; continuation/evaluation needs explicit review")
    return checkpoint


def position_targets(raw_action, runtime):
    """Same target equation as F1JointPositionAction; returns targets, not torques."""
    control = runtime["control"]
    raw = torch.as_tensor(raw_action, dtype=torch.float32)
    if raw.shape[-1] != len(runtime["joint_names"]) or not torch.isfinite(raw).all():
        raise ValueError("Invalid F1 raw actions")
    target = control["default_joint_pos"] + control["action_scale"] * raw
    return torch.clamp(target, control["joint_lower"], control["joint_upper"])
