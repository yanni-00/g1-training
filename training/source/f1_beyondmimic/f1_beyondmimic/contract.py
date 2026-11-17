"""F1 identities, deferred inputs and local admission. No simulator imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOOP = ROOT.parent
FORMAT = "f1_beyondmimic_runtime_v1"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        # Reuse the bounded readinto buffer used by asset restoration as well.
        # Keep all identity checks; do not materialize large mesh files or run
        # their per-block Python loop under the platform's execution tracing.
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(path, root=ROOT):
    path = Path(str(path).replace("\\", "/"))
    return path if path.is_absolute() else Path(root) / path


def load_contract(root=ROOT):
    root = Path(root)
    c = load_json(root / "configs/robot_contract.json")
    if sha256(root / c["urdf"]) != c["urdf_sha256"]:
        raise ValueError("F1 URDF identity changed; review/rebuild the robot contract")
    return c


def verify_sources(sources, root=ROOT):
    if not isinstance(sources, list) or not sources:
        raise ValueError("Actual source files with hashes and descriptions are required")
    for source in sources:
        if not source.get("description") or sha256(resolve(source["path"], root)) != source["sha256"]:
            raise ValueError("Source description or file hash mismatch")


def effective_control(root=ROOT):
    root = Path(root)
    c = load_contract(root)
    data = load_json(root / "configs/parameter_identification.json")
    status = data.get("status")
    if status not in {"pending", "partial", "validated"}:
        raise ValueError("Unknown F1 identification status")
    if data.get("urdf_sha256") != c["urdf_sha256"]:
        raise ValueError("Identification belongs to a different F1 URDF")
    if data.get("delivery_filter") is not None:
        raise ValueError("A non-null delivery filter needs an implemented and audited runtime adapter")
    baseline = None
    if status != "validated":
        profile_path = root / "configs/control_profile.json"
        if not profile_path.is_file():
            raise ValueError("F1 identification is pending; no owner-authorized control baseline")
        baseline = load_json(profile_path)
        if (baseline.get("status") != "owner_specified_simulation_baseline"
                or baseline.get("urdf_sha256") != c["urdf_sha256"]
                or baseline.get("delivery_filter") is not None):
            raise ValueError("Owner control baseline identity/semantics mismatch")
        verify_sources(baseline.get("sources"), root)
    if status != "pending":
        verify_sources(data.get("sources"), root)
    names = c["model_joint_names"]
    if set(data["joints"]) != set(names):
        raise ValueError("Identification joint coverage differs from F1")
    if baseline is not None and set(baseline["joints"]) != set(names):
        raise ValueError("Owner control baseline joint coverage differs from F1")
    result = {}
    for name in names:
        keys = ("stiffness", "pd_damping", "armature")
        values = {k: (baseline["joints"][name].get(k) if baseline else data["joints"][name].get(k)) for k in keys}
        if status == "partial":
            values.update({k: data["joints"][name][k] for k in keys if data["joints"][name].get(k) is not None})
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in values.values()) or values["stiffness"] <= 0:
            raise ValueError(f"Missing/invalid identification: {name}")
        props = c["joint_properties"][name]
        # URDF passive damping and controller D gain have different meanings.
        values.update(effort_limit=props["limit"]["effort"], velocity_limit=props["limit"]["velocity"],
                      passive_damping=props["dynamics"].get("damping", 0.0))
        values["total_damping"] = values["pd_damping"] + values["passive_damping"]
        # Same engineering action-scale formula as the frozen BeyondMimic g1.py.
        values["action_scale"] = 0.25 * values["effort_limit"] / values["stiffness"]
        result[name] = values
    return result


def validate_motion(path, root=ROOT):
    """Validate prepared data only; never repair or reinterpret raw UMR qpos."""
    import numpy as np
    c = load_contract(root)
    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                "body_ang_vel_w", "joint_names", "body_names", "reference_space", "cycle_frames",
                "support_frames", "nominal_reference_speed_mps", "urdf_sha256"}
    with np.load(path, allow_pickle=False) as z:
        if not required <= set(z.files):
            raise ValueError("Prepared F1 cyclic reference missing fields: " + str(sorted(required - set(z.files))))
        a = {key: z[key].copy() for key in required}
    if str(a["urdf_sha256"].item()) != c["urdf_sha256"]:
        raise ValueError("Motion F1 URDF hash mismatch")
    # Heading/stride-local references preserve periodic pelvis sway around the
    # command-integrated path; positions and velocities use the same transform.
    if str(a["reference_space"].item()) not in {"cyclic_pelvis_xy_local_v1", "cyclic_heading_stride_local_v1"}:
        raise ValueError("Unsupported cyclic reference space")
    names = a["joint_names"].tolist()
    bodies = a["body_names"].tolist()
    if len(names) != len(set(names)) or set(names) != set(c["model_joint_names"]):
        raise ValueError("Motion does not cover exactly the F1 joints")
    if len(bodies) != len(set(bodies)) or not set(c["tracked_body_names"]) <= set(bodies):
        raise ValueError("Motion missing/duplicate tracked F1 bodies")
    if not set(bodies) <= set(c["link_names"]):
        raise ValueError("Motion contains bodies absent from F1")
    n = a["cycle_frames"].reshape(-1)
    if n.size != 1 or not np.isfinite(n).all() or n[0] < 2 or n[0] != int(n[0]):
        raise ValueError("Invalid unique cycle frame count")
    n = int(n[0])
    for key, tail in {"joint_pos": (len(names),), "joint_vel": (len(names),),
                      "body_pos_w": (len(bodies), 3), "body_quat_w": (len(bodies), 4),
                      "body_lin_vel_w": (len(bodies), 3), "body_ang_vel_w": (len(bodies), 3)}.items():
        if a[key].shape != (n, *tail) or not np.isfinite(a[key]).all():
            raise ValueError("Invalid motion shape/nonfinite values: " + key)
    settings = load_json(Path(root) / "configs/reference/stable_runtime_settings.json")
    fps = a["fps"].reshape(-1)
    if fps.size != 1 or not np.isfinite(fps).all() or not np.isclose(fps[0] * settings["policy_dt"], 1):
        raise ValueError("Prepared reference FPS must equal the inherited policy rate")
    speed = a["nominal_reference_speed_mps"].reshape(-1)
    if speed.size != 1 or not np.isfinite(speed).all() or speed[0] <= 0:
        raise ValueError("Actual selected forward-template nominal speed required")
    support = a["support_frames"]
    if (support.ndim != 1 or len(np.unique(support)) < 2 or not np.isfinite(support).all()
            or np.any(support != np.floor(support)) or np.any(support < 0) or np.any(support >= n)):
        raise ValueError("Invalid support-frame candidates")
    # These are software validation tolerances, not measured hardware tolerances.
    if not np.allclose(np.linalg.norm(a["body_quat_w"], axis=-1), 1, rtol=1e-4, atol=1e-4):
        raise ValueError("Reference quaternions must be unit wxyz")
    lower = np.array([c["joint_properties"][name]["limit"]["lower"] for name in names])
    upper = np.array([c["joint_properties"][name]["limit"]["upper"] for name in names])
    if np.any(a["joint_pos"] < lower - 1e-6) or np.any(a["joint_pos"] > upper + 1e-6):
        raise ValueError("Reference exceeds F1 URDF joint limits; repair separately")
    return a


def prepared_motion(root=ROOT):
    root = Path(root)
    c = load_contract(root)
    manifest = load_json(root / "data/motion_manifest.json")
    if manifest.get("status") != "ready" or not manifest.get("prepared_path"):
        raise ValueError("F1 motion pending repair; prepared NPZ is deliberately absent")
    path = resolve(manifest["prepared_path"], root)
    if manifest.get("urdf_sha256") != c["urdf_sha256"] or sha256(path) != manifest.get("prepared_sha256"):
        raise ValueError("Prepared motion manifest identity mismatch")
    arrays = validate_motion(path, root)
    audit_path = manifest.get("native_audit_path")
    if not audit_path or sha256(resolve(audit_path, root)) != manifest.get("native_audit_sha256"):
        raise ValueError("Actual F1 native FK/period/support audit is required")
    audit = load_json(resolve(audit_path, root))
    if (audit.get("format") != "f1_beyondmimic_native_fk_v1" or audit.get("passed") is not True
            or audit.get("urdf_sha256") != c["urdf_sha256"]
            or audit.get("reference_sha256") != manifest["prepared_sha256"]
            or audit.get("joint_names") != arrays["joint_names"].tolist()
            or audit.get("body_names") != arrays["body_names"].tolist()):
        raise ValueError("Native motion audit is not bound to this exact F1 reference")
    verify_sources(audit.get("sources"), root)
    return path


def reorder_motion_arrays(arrays, joint_names, body_names):
    """Bind reference tensors to actual native names, never dimension equality."""
    import numpy as np
    requested_joints, requested_bodies = list(joint_names), list(body_names)
    stored_joints = arrays['joint_names'].tolist()
    stored_bodies = arrays['body_names'].tolist()
    for stored, requested, label in ((stored_joints, requested_joints, 'joint'),
                                     (stored_bodies, requested_bodies, 'body')):
        if len(stored) != len(set(stored)) or len(requested) != len(set(requested)) or set(stored) != set(requested):
            raise ValueError('Reference/native '+label+' name coverage differs')
    result = dict(arrays)
    joint_order = [stored_joints.index(name) for name in requested_joints]
    body_order = [stored_bodies.index(name) for name in requested_bodies]
    for key in ('joint_pos', 'joint_vel'):
        result[key] = arrays[key][:, joint_order].copy()
    for key in ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        result[key] = arrays[key][:, body_order].copy()
    result['joint_names'] = np.asarray(requested_joints)
    result['body_names'] = np.asarray(requested_bodies)
    return result


def binding(root=ROOT):
    root = Path(root)
    paths = ["configs/robot_contract.json", "configs/parameter_identification.json", "configs/deployment_contract.json",
             "configs/reference/stable_runtime_settings.json", "data/motion_manifest.json",
             "../.oma/impl/source_manifest.json"]
    paths += [p for p in ("configs/control_profile.json", "configs/owner_training_profile.json") if (root / p).is_file()]
    paths += [str(p.relative_to(root)).replace("\\", "/") for p in sorted((root / "source").rglob("*.py"))]
    paths += [str(p.relative_to(root)).replace("\\", "/") for p in sorted((root / "scripts").glob("*.py"))]
    return {p: sha256(root / p) for p in paths}


def verify_assets(root=ROOT):
    root = Path(root)
    manifest = load_json(root.parent / ".oma/impl/source_manifest.json")
    for index, asset in enumerate(manifest["assets"]):
        path = resolve(asset["path"], root)
        try:
            size = path.stat().st_size
        except OSError as error:
            _stage(f"a2-{index}-stat-failed {asset['path']} {type(error).__name__}")
            raise
        _stage(f"a2-{index}/{len(manifest['assets'])} {path.name} {size}")
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != asset["sha256"]:
            raise ValueError("Packaged F1 asset identity changed: " + asset["path"])
    _stage("a2-assets-verified")


def _stage(name):
    # Stage prints go to stdout because the execution host may capture only stdout.
    try:
        from _stage import memory_snapshot
        counters = memory_snapshot()
    except ImportError:
        counters = {}
    print("[F1-BM] check "+name+" "+json.dumps(counters), flush=True)


def assert_can_run(scope="training", root=ROOT):
    if scope not in {"training", "evaluation"}:
        raise ValueError("Execution scope must be training or evaluation")
    root = Path(root)
    _stage("a0-start")
    launch = load_json(root / "configs/launch.json")
    config = load_json(root.parent / ".oma/config.json")
    loop = load_json(root.parent / ".oma/loop.json")
    permission = "training" if scope == "training" else "evaluation"
    paused = "loop" if permission == "training" else "evaluation"
    if (launch.get(permission + "_enabled") is not True
            or not launch.get(permission + "_authorization_ref")
            or config.get("authorization", {}).get(permission) is not True
            or paused in loop.get("pausedScopes", [])):
        raise ValueError(f"{permission} is disabled: current authorization is local scaffolding only")
    _stage("a1-authorization")
    verify_assets(root)
    _stage("a2-assets")
    effective_control(root)
    _stage("a3-control")
    prepared_motion(root)
    _stage("a4-motion")
    evidence = launch.get("preflight_evidence")
    if not evidence or sha256(resolve(evidence["path"], root)) != evidence.get("sha256"):
        raise ValueError("Bind actual per-experiment OMA preflight evidence before execution")
    report = load_json(resolve(evidence["path"], root))
    _stage("a5-permit")
    if report.get("canTrain") is not True or report.get("exp_id") != config.get("activeExpId") or not report.get("exp_id"):
        raise ValueError("OMA preflight did not pass")
    if scope == 'training' and launch.get('admission_mode') == 'owner_direct_formal':
        # The owner explicitly requests one formal job without a separate probe.
        # This records the overridden workflow requirement, never fake prior
        # runtime/deployment success. Actual input/control audits run in the job.
        policy = load_json(root/'configs/training_admission_policy.json')
        verify_sources(policy.get('sources'), root)
        _stage("a6-policy")
        if (report.get('format') != 'f1_beyondmimic_owner_formal_preflight_v1'
                or policy.get('mode') != 'owner_direct_formal'
                or policy.get('prior_runtime_closure_claimed') is not False
                or config['authorization'].get('real_robot') is not False
                or report.get('prior_runtime_closure_verified') is not False
                or report.get('registered_limits') != {key: launch[key] for key in ('num_envs', 'max_iterations', 'seed')}
                or report.get('spend_cap_milli_credit') != 50000):
            raise ValueError('Direct-formal scope, authorization or registered limits mismatch')
        _stage("a7-admitted")
        return launch
    deployment = load_json(root / "configs/deployment_contract.json")
    if deployment.get("status") != "validated":
        raise ValueError("F1 actual deployment contract is pending; formal execution is not admitted")
    verify_sources(deployment.get("sources"), root)
    names = load_contract(root)["model_joint_names"]
    order = deployment.get("hardware_sdk_order")
    if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(names):
        raise ValueError("Bind the actual F1 hardware joint order")
    for key in ("observation_signal_mapping", "control_timing", "action_processing", "state_machine"):
        if not isinstance(deployment.get(key), dict) or not deployment[key]:
            raise ValueError("F1 deployment mapping is missing: " + key)
    inherited = load_json(root / "configs/reference/stable_runtime_settings.json")
    if deployment["control_timing"].get("policy_dt") != inherited["policy_dt"]:
        raise ValueError("Actual F1 policy timing differs; adapt training explicitly")
    if deployment["action_processing"].get("format") != "default_plus_scaled_raw_then_urdf_hard_clip_v1":
        raise ValueError("F1 actual target protections need an explicit native action adapter")
    return launch


def readiness(root=ROOT):
    checks = {}
    for name, check in [("asset", load_contract), ("control", effective_control),
                        ("motion", prepared_motion)]:
        try:
            check(root)
            checks[name] = {"status": "passed"}
        except (ValueError, OSError, KeyError, TypeError) as error:
            checks[name] = {"status": "pending", "reason": str(error)}
    identification = load_json(Path(root) / "configs/parameter_identification.json")
    checks["identification"] = {"status": identification["status"],
        "reason": "Measured identification is separate from the owner-authorized simulation baseline"}
    try:
        assert_can_run("training", root)
        checks["training"] = {"status": "admitted"}
    except (ValueError, OSError, KeyError, TypeError) as error:
        checks["training"] = {"status": "disabled", "reason": str(error)}
    return checks
