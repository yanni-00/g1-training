"""Portable asset/control binding, independent of Isaac Lab."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_contract(root=ROOT):
    root = Path(root)
    c = load_json(root / "configs/robot_contract.json")
    if sha256(root / c["urdf"]) != c["urdf_sha256"]:
        raise ValueError("F1 URDF has changed; rebuild/review the contract")
    return c


def effective_control(root=ROOT):
    root = Path(root)
    c = load_contract(root)
    profile = load_json(root / "configs/control_profile.json")
    identification = load_json(root / "configs/parameter_identification.json")
    if identification["urdf_sha256"] != c["urdf_sha256"]:
        raise ValueError("Identification is bound to a different URDF")
    if identification["status"] not in ("pending", "partial", "validated"):
        raise ValueError("Identification status must be pending, partial or validated")
    if identification["status"] != "pending":
        if not identification["sources"]:
            raise ValueError("Injected identification requires actual sources")
        for source in identification["sources"]:
            path = Path(source["path"])
            if not path.is_absolute():
                path = root / path
            if sha256(path) != source["sha256"]:
                raise ValueError("Identification source hash mismatch")
    if set(profile["joints"]) != set(c["model_joint_names"]) or set(identification["joints"]) != set(c["model_joint_names"]):
        raise ValueError("Control/identification joint coverage mismatch")
    result = {}
    for name in c["model_joint_names"]:
        values = {k: profile["joints"][name][k] for k in ("stiffness", "pd_damping", "armature")}
        if identification["status"] != "pending":
            if identification["status"] == "validated" and any(identification["joints"][name].get(k) is None for k in values):
                raise ValueError(f"Incomplete identification: {name}")
            values.update({k: identification["joints"][name][k] for k in values
                           if identification["joints"][name].get(k) is not None})
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError(f"Invalid control values for {name}")
        values["passive_damping"] = c["joint_properties"][name]["dynamics"].get("damping", 0.0)
        values["total_damping"] = values["pd_damping"] + values["passive_damping"]
        values["effort_limit"] = c["joint_properties"][name]["limit"]["effort"]
        values["velocity_limit"] = c["joint_properties"][name]["limit"]["velocity"]
        result[name] = values
    return result


def binding(root=ROOT):
    root = Path(root)
    files = ["configs/robot_contract.json", "configs/control_profile.json", "configs/parameter_identification.json",
             "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/robots/f1/velocity_env_cfg.py",
             "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py",
             "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/mdp/rewards.py"]
    if (root / "configs/owner_training_profile.json").is_file():
        files.append("configs/owner_training_profile.json")
    return {p: sha256(root / p) for p in files}
