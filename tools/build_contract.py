"""Rebuild a static contract from the packaged F1 URDF and frozen G1 source.

All constants below are source values or geometric identities, not hardware estimates.
"""
import ast
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from model_geometry import forward_kinematics, origin, collision_vertices

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build():
    urdf = ROOT / "resources/f1/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"
    robot = ET.parse(urdf).getroot()
    joints = [j for j in robot.findall("joint") if j.get("type") != "fixed"]
    names = [j.get("name") for j in joints]
    snapshot = ROOT / "configs/reference/g1_original_unitree.py"
    tree = ast.parse(snapshot.read_text(encoding="utf-8"))
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "UNITREE_G1_29DOF_CFG" for t in n.targets))
    kw = {n.arg: n.value for n in call.keywords}
    init = ast.literal_eval(next(n.value for n in kw["init_state"].keywords if n.arg == "joint_pos"))
    actuators = [{k.arg: ast.literal_eval(k.value) for k in a.keywords} for a in kw["actuators"].values]

    def source_value(value, name):
        if not isinstance(value, dict):
            return value
        matches = [v for key, v in value.items() if re.fullmatch(key, name)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous source value for {name}: {matches}")
        return matches[0]

    # Preserve G1 knee bend; solve the F1 hip/ankle to center the ankle below the hip.
    angles = dict.fromkeys(names, 0.0)
    knee = source_value(init, "left_knee_joint")
    for side in ("left", "right"):
        angles[f"{side}_knee_pitch_joint"] = knee
    transforms = forward_kinematics(robot, angles)
    for side in ("left", "right"):
        hip_origin = origin(robot.find(f"joint[@name='{side}_hip_pitch_joint']/origin"))[:3, 3]
        delta = transforms[f"{side}_ankle_pitch"][:3, 3] - hip_origin
        hip = math.atan2(delta[0], -delta[2])
        angles[f"{side}_hip_pitch_joint"] = hip
        angles[f"{side}_ankle_pitch_joint"] = -knee - hip
        angles[f"{side}_shoulder_pitch_joint"] = source_value(init, f"{side}_shoulder_pitch_joint")
        # F1 positive shoulder roll moves inward on both sides (URDF FK); negative moves outward.
        angles[f"{side}_shoulder_roll_joint"] = -abs(source_value(init, f"{side}_shoulder_roll_joint"))
        angles[f"{side}_elbow_pitch_joint"] = source_value(init, f"{side}_elbow_joint")
        # Wrist and additional elbow yaw use the actual URDF zero, not G1's absent-DOF mapping.
    transforms = forward_kinematics(robot, angles)
    floors, sole, feet = {}, {}, {}
    for side in ("left", "right"):
        link = f"{side}_ankle_roll"
        points = collision_vertices(urdf, robot.find(f"link[@name='{link}']"))
        sole[side] = -float(points[:, 2].min())
        t = transforms[link]
        floors[side] = float((points @ t[:3, :3].T + t[:3, 3])[:, 2].min())
        feet[side] = t[:3, 3].tolist()
    base_height = -min(floors.values())
    properties, profile = {}, {}
    for joint in joints:
        name = joint.get("name")
        limit = {k: float(v) for k, v in joint.find("limit").attrib.items()}
        dynamics = {k: float(v) for k, v in joint.find("dynamics").attrib.items()}
        if not limit["lower"] <= angles[name] <= limit["upper"]:
            raise ValueError(f"Initial pose outside {name} limits")
        source_name = name.replace("lumbar_", "waist_").replace("knee_pitch_", "knee_")
        if "elbow_" in source_name:
            source_name = re.sub(r"elbow_(pitch|yaw)_", "elbow_", source_name)
        matches = [a for a in actuators if any(re.fullmatch(expr, source_name) for expr in a["joint_names_expr"])]
        if len(matches) != 1:
            raise ValueError(f"No unique engineering seed for {name}")
        a = matches[0]
        properties[name] = {"parent": joint.find("parent").get("link"),
                            "child": joint.find("child").get("link"),
                            "axis": [float(v) for v in joint.find("axis").get("xyz").split()],
                            "limit": limit, "dynamics": dynamics}
        profile[name] = {"stiffness": source_value(a["stiffness"], source_name),
                         "pd_damping": source_value(a["damping"], source_name),
                         "armature": source_value(a["armature"], source_name),
                         "source_joint": source_name,
                         "source": "configs/reference/g1_original_unitree.py:UNITREE_G1_29DOF_CFG.actuators"}
    # Read source MDP fields through AST; no duplicated default numerical guesses.
    env_tree = ast.parse((ROOT / "configs/reference/g1_original_velocity_env_cfg.py").read_text(encoding="utf-8"))
    env_class = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotEnvCfg")
    post = next(n for n in env_class.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    settings = {ast.unparse(n.targets[0]): ast.literal_eval(n.value) for n in post.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("self.sim.dt", "self.decimation")}
    actions = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionsCfg")
    action_scale = ast.literal_eval(next(k.value for n in actions.body if isinstance(n, ast.Assign)
                                        for k in n.value.keywords if k.arg == "scale"))
    observations = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    policy = next(n for n in observations.body if isinstance(n, ast.ClassDef) and n.name == "PolicyCfg")
    history = next(ast.literal_eval(n.value) for n in ast.walk(policy) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "self.history_length")
    terms = []
    for n in policy.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            args = {k.arg: k.value for k in n.value.keywords}
            term_name = n.targets[0].id
            dim = len(names) if term_name in ("joint_pos_rel", "joint_vel_rel", "last_action") else 3
            terms.append({"name": term_name, "dim": dim, "scale": ast.literal_eval(args["scale"]) if "scale" in args else 1.0})
    # Match physical clearance above the sole, not an unrelated robot's ankle-frame height.
    g1 = ET.parse(ROOT / "configs/reference/g1_reference.urdf").getroot()
    g1_sole = {}
    for side in ("left", "right"):
        spheres = g1.findall(f"link[@name='{side}_ankle_roll_link']/collision")
        g1_sole[side] = -min(float(s.find("origin").get("xyz").split()[2])
                            - float(s.find("geometry/sphere").get("radius")) for s in spheres)
    rew = next(n for n in env_tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
    clearance_call = next(n.value for n in rew.body if isinstance(n, ast.Assign) and n.targets[0].id == "feet_clearance")
    clearance_params = ast.literal_eval(ast.Dict(
        keys=[k for k, v in zip(next(k.value for k in clearance_call.keywords if k.arg == "params").keys,
                                next(k.value for k in clearance_call.keywords if k.arg == "params").values) if isinstance(v, ast.Constant)],
        values=[v for v in next(k.value for k in clearance_call.keywords if k.arg == "params").values if isinstance(v, ast.Constant)]))
    f1_clearance = clearance_params["target_height"] + np.mean(list(sole.values())) - np.mean(list(g1_sole.values()))
    contract = {"format": "f1_fullbody_velocity_v1", "robot": robot.get("name"),
                "urdf": str(urdf.relative_to(ROOT)).replace("\\", "/"),
                "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                "model_joint_names": names, "default_joint_pos": [angles[n] for n in names],
                "joint_properties": properties, "base_position": [0.0, 0.0, base_height],
                "base_height_target": base_height,
                "sole_offsets": sole, "feet_zero_floor": floors, "feet_positions": feet,
                "foot_clearance_target": float(f1_clearance), "g1_sole_offsets": g1_sole,
                "sim_dt": settings["self.sim.dt"], "decimation": settings["self.decimation"],
                "step_dt": settings["self.sim.dt"] * settings["self.decimation"],
                "action_scale": action_scale, "history_length": history,
                "history_layout": "term_major_oldest_first", "observation_terms": terms,
                "observation_dim": sum(t["dim"] for t in terms) * history,
                "action_dim": len(names), "hardware_sdk_order": None,
                "provenance": {"pose": "G1 knee/shoulder/elbow magnitudes; F1 FK: ankle x=hip x, sum sagittal pitch=0; F1 mirrored negative shoulder roll; extra elbow yaw/wrists=URDF zero",
                               "height": "minus minimum transformed scaled sole collision STL z",
                               "control": "G1 engineering seed, NOT F1 identification; F1 URDF limits and passive dynamics",
                               "timing_observations": "configs/reference/g1_original_velocity_env_cfg.py:ActionsCfg/ObservationsCfg/RobotEnvCfg",
                               "scope": "static initialization, not dynamic performance or hardware validation"}}
    owner_path = ROOT / "configs/owner_training_profile.json"
    if owner_path.is_file():
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if set(owner["joints"]) != set(names):
            raise ValueError("Owner PD profile does not cover the F1 model exactly")
        for name in names:
            for key, expected in owner["urdf_dynamics"].items():
                if properties[name]["dynamics"].get(key) != expected:
                    raise ValueError(f"Owner URDF dynamics not applied: {name}.{key}")
            profile[name] = {**owner["joints"][name], "armature": owner["additional_armature"],
                             "source": "configs/owner_training_profile.json:joints." + name,
                             "source_status": owner["source_status"]}
        contract["provenance"]["control"] = "owner 2026-11-13 specified PD; URDF friction=0/damping=0.05; no added identification, armature or delivery filter"
        contract["owner_profile_revision"] = owner["revision"]
    write_json(ROOT / "configs/robot_contract.json", contract)
    write_json(ROOT / "configs/control_profile.json", {
        "status": "owner_specified_baseline" if owner_path.is_file() else "unverified_engineering_seed", "joints": profile})
    identification = ROOT / "configs/parameter_identification.json"
    if not identification.exists():
        write_json(identification, {"status": "pending", "urdf_sha256": contract["urdf_sha256"], "sources": [],
                                   "joints": {n: {"stiffness": None, "pd_damping": None, "armature": None} for n in names},
                                   "passive_dynamics": "preserve_urdf; no explicit friction added",
                                   "delivery_filter": None})
    elif owner_path.is_file():
        # Preserve old evidence in the experiment's before snapshot; this new baseline
        # explicitly has no identification injection, even if another source was filled.
        previous = json.loads(identification.read_text(encoding="utf-8"))
        if previous["status"] != "pending" or previous["sources"]:
            raise ValueError("Archive/review existing identified values before selecting an un-identified baseline")
        previous["urdf_sha256"] = contract["urdf_sha256"]
        write_json(identification, previous)
    return contract


if __name__ == "__main__":
    c = build()
    print(json.dumps({"action_dim": c["action_dim"], "observation_dim": c["observation_dim"],
                      "base_height_from_STL": c["base_height_target"]}))
