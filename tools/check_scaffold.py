"""Static and CPU F1 scaffold verification; never starts a simulator or training."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))

from f1_rl_lab.contract import binding, effective_control, load_contract, sha256
from f1_rl_lab.checkpoint import verify_checkpoint, write_sidecar
from f1_rl_lab.policy import make_actor
from model_geometry import forward_kinematics, collision_vertices


class ConfigNode:
    """AST-only configuration evaluator. This does not emulate Isaac physics/APIs."""
    def __init__(self, *args, **kwargs):
        self._args = args
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        node = ConfigNode()
        setattr(self, name, node)
        return node

    def replace(self, **kwargs):
        node = copy.deepcopy(self)
        node.__dict__.update(kwargs)
        return node


for nested in ("OffsetCfg", "Ranges", "JointDriveCfg", "PDGainsCfg"):
    setattr(ConfigNode, nested, ConfigNode)


class ConfigNamespace:
    def __getattr__(self, name):
        if name.endswith("Cfg"):
            return ConfigNode
        def function(*args, **kwargs):
            raise RuntimeError("AST-only evaluator cannot execute simulation functions")
        function.__name__ = name
        return function


def configclass(cls):
    def init(self, **kwargs):
        for base in reversed(cls.mro()):
            for name, value in base.__dict__.items():
                if not name.startswith("_") and not callable(value):
                    setattr(self, name, copy.deepcopy(value))
        self.__dict__.update(kwargs)
        if hasattr(cls, "__post_init__"):
            self.__post_init__()
    cls.__init__ = init
    return cls


def read_env_ast(path):
    """Read actual config expressions while removing all runtime imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    import math
    ns = {name: ConfigNode for name in (
        "ArticulationCfg", "AssetBaseCfg", "ManagerBasedRLEnvCfg", "CurrTerm", "EventTerm", "ObsGroup",
        "ObsTerm", "RewTerm", "SceneEntityCfg", "DoneTerm", "InteractiveSceneCfg", "ContactSensorCfg",
        "RayCasterCfg", "TerrainImporterCfg", "Unoise")}
    ns.update(math=math, sim_utils=ConfigNamespace(), terrain_gen=ConfigNamespace(), patterns=ConfigNamespace(),
              mdp=ConfigNamespace(), ROBOT_CFG=ConfigNode(), configclass=configclass,
              ISAAC_NUCLEUS_DIR="source-only", ISAACLAB_NUCLEUS_DIR="source-only", load_contract=load_contract)
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def reward_signature(cls):
    return {name: {"func": node.func.__name__, "weight": node.weight}
            for name, node in vars(cls).items() if isinstance(node, ConfigNode)}


def validate():
    c = load_contract()
    control = effective_control()
    manifest = json.loads((ROOT / "configs/source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["assets"]:
        if sha256(ROOT / item["destination"]) != item.get("training_sha256", item["sha256"]):
            raise AssertionError(f"Changed F1 asset: {item['destination']}")
    urdf = ROOT / c["urdf"]
    robot = ET.parse(urdf).getroot()
    links = [n.get("name") for n in robot.findall("link")]
    joints = [n.get("name") for n in robot.findall("joint") if n.get("type") != "fixed"]
    assert joints == c["model_joint_names"] and len(set(joints)) == len(joints)
    assert len(set(links)) == len(links)
    for mesh in robot.findall(".//mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        assert path.is_file() and path.is_relative_to((ROOT / "resources/f1").resolve())
        assert not path.read_bytes()[:128].startswith(b"version https://git-lfs")
    angles = dict(zip(joints, c["default_joint_pos"]))
    fk = forward_kinematics(robot, angles)
    for name, angle in angles.items():
        limit = c["joint_properties"][name]["limit"]
        assert limit["lower"] <= angle <= limit["upper"]
        assert control[name]["effort_limit"] == limit["effort"]
        assert control[name]["velocity_limit"] == limit["velocity"]
    floors = []
    for side in ("left", "right"):
        name = f"{side}_ankle_roll"
        points = collision_vertices(urdf, robot.find(f"link[@name='{name}']"))
        t = fk[name]
        floors.append(float((points @ t[:3, :3].T + t[:3, 3])[:, 2].min() + c["base_position"][2]))
        np.testing.assert_allclose(t[:3, :3], np.eye(3), atol=1e-6)  # URDF decimal rotation precision
    np.testing.assert_allclose(floors, [0.0, 0.0], atol=1e-6)
    for link in ("shoulder_pitch_link", "elbow_pitch_link", "wrist_roll_link"):
        left, right = fk[f"left_{link}"][:3, 3], fk[f"right_{link}"][:3, 3]
        np.testing.assert_allclose(left, right * [1, -1, 1], atol=1e-6)
    original = read_env_ast(ROOT / "configs/reference/g1_original_velocity_env_cfg.py")
    adapted = read_env_ast(ROOT / "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/robots/f1/velocity_env_cfg.py")
    assert reward_signature(original["RewardsCfg"]) == reward_signature(adapted["RewardsCfg"])
    assert not any("arm_swing" in name for name in vars(adapted["RewardsCfg"]))
    assert adapted["ActionsCfg"].JointPositionAction.joint_names == joints
    assert adapted["ActionsCfg"].JointPositionAction.preserve_order is True
    assert adapted["RewardsCfg"].base_height.params["target_height"] == c["base_height_target"]
    assert adapted["RewardsCfg"].feet_clearance.params["target_height"] == c["foot_clearance_target"]
    selectors = []
    def visit(obj, seen):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for v in obj.values(): visit(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj: visit(v, seen)
        elif isinstance(obj, ConfigNode) or isinstance(obj, type):
            for key, value in vars(obj).items():
                if key in ("joint_names", "body_names"):
                    candidates = joints if key == "joint_names" else links
                    patterns = [value] if isinstance(value, str) else value
                    for pattern in patterns:
                        assert any(re.fullmatch(pattern, n) for n in candidates), (key, pattern)
                        selectors.append({key: pattern})
                elif not key.startswith("_"): visit(value, seen)
    for cls in ("RobotSceneCfg", "EventCfg", "ActionsCfg", "ObservationsCfg", "RewardsCfg"):
        visit(adapted[cls], set())
    # CPU/ONNX uses temporary synthetic weights; never recorded as a trained model.
    import torch
    from export_policy import export
    reference = json.loads((ROOT / "configs/reference/g1_11k_manifest.json").read_text(encoding="utf-8"))
    torch.manual_seed(reference["checkpoint_iter"])  # repeatable synthetic fixture only
    with tempfile.TemporaryDirectory(prefix="f1_scaffold_cpu_") as temporary:
        checkpoint = Path(temporary) / "synthetic.pt"
        actor = make_actor(c["observation_dim"], c["action_dim"])
        torch.save({"model_state_dict": {"actor." + k: v for k, v in actor.state_dict().items()}, "iter": 0}, checkpoint)
        runtime = {"binding": binding(), "urdf_sha256": c["urdf_sha256"], "policy_joint_names": joints,
                   "observation_dim": c["observation_dim"], "action_dim": c["action_dim"],
                   "empirical_normalization": False}
        write_sidecar(checkpoint, runtime)
        verify_checkpoint(checkpoint)
        parity = export(checkpoint, Path(temporary) / "export")["export_check"]
    collision_links = [link.get("name") for link in robot.findall("link") if link.find("collision") is not None]
    return {"scope": "static configuration and synthetic CPU inference only; no training or simulator execution",
            "passed": True, "assets_verified": len(manifest["assets"]), "links": len(links),
            "movable_joints": len(joints), "actor_dim": c["observation_dim"],
            "reward_terms_unchanged": len(reward_signature(adapted["RewardsCfg"])),
            "selectors_resolved": len(selectors), "sole_floor_after_translation": floors,
            "base_height_STL": c["base_height_target"], "collision_links": collision_links,
            "control_status": json.loads((ROOT / "configs/control_profile.json").read_text(encoding="utf-8"))["status"], "synthetic_onnx_parity": parity,
            "dynamic_isaac_smoke": "not run; Isaac Lab unavailable", "training": "not run",
            "sim2sim": "not run; no trained F1 weights or MuJoCo runtime",
            "behavior_acceptance": "pending actual F1 videos/data"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = validate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
