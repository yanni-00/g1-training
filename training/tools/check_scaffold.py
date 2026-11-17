"""Read-only scaffold checks plus isolated CPU fixtures; never starts simulation."""
import argparse
import ast
import importlib.util
import io
import json
import fnmatch
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT.parent
sys.path.insert(0, str(ROOT / "source/f1_beyondmimic"))
from f1_beyondmimic.contract import load_contract, load_json, sha256, readiness, verify_assets, prepared_motion
from model_geometry import forward_kinematics, collision_vertices


def owned_files(pattern):
    for folder, directories, files in os.walk(ROOT):
        directories[:] = [name for name in directories if name not in {'.runtime', '__pycache__', '.git'}]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                yield Path(folder)/name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--static-only", action="store_true", help="Skip CPU tests when they were already run separately")
    args = parser.parse_args()
    checks = {}
    for path in owned_files("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    checks["python_syntax"] = True
    c = load_contract()
    urdf = ROOT / c["urdf"]
    robot = ET.parse(urdf).getroot()
    verify_assets()
    manifest = load_json(LOOP / ".oma/impl/source_manifest.json")
    project = LOOP.parents[1]
    original_asset_root = project / manifest["asset_source"]
    for entry in manifest["assets"]:
        packaged = ROOT / entry["path"]
        original = original_asset_root / packaged.relative_to(ROOT / "resources/f1")
        if sha256(original) != entry.get("source_sha256", entry["sha256"]):
            raise ValueError("Packaged asset no longer matches the owner-designated SSOT: " + str(original))
    checks["asset_original_bytes_and_packaged_hashes"] = True
    for mesh in robot.findall(".//mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        if not path.is_relative_to((ROOT / "resources/f1").resolve()) or not path.is_file():
            raise ValueError("Missing/nonportable F1 mesh: " + str(path))
        # Windows existence alone cannot prove Linux case-sensitive portability.
        parent = path.parent
        if path.name not in {p.name for p in parent.iterdir()}:
            raise ValueError("Mesh filename case mismatch: " + str(path))
    checks["mesh_references"] = True
    names = [j.get("name") for j in robot.findall("joint") if j.get("type") != "fixed"]
    if names != c["model_joint_names"]:
        raise ValueError("Robot contract joint identity mismatch")
    if not set(c["tracked_body_names"]) <= {l.get("name") for l in robot.findall("link")}:
        raise ValueError("Tracked F1 body absent")
    angles = dict(zip(names, c["default_joint_pos"]))
    for name, angle in angles.items():
        limits = c["joint_properties"][name]["limit"]
        if not limits["lower"] <= angle <= limits["upper"]:
            raise ValueError("Initial pose beyond limits: " + name)
    transforms = forward_kinematics(robot, angles)
    floors = {}
    for name in c["feet"]:
        points = collision_vertices(urdf, robot.find(f"link[@name='{name}']"))
        tf = transforms[name]
        floors[name] = float((points @ tf[:3, :3].T + tf[:3, 3])[:, 2].min() + c["base_position"][2])
        np.testing.assert_allclose(tf[:3, :3], np.eye(3), atol=1e-6)
    if not np.isclose(min(floors.values()), 0, atol=1e-8):
        raise ValueError("Static initial sole height differs from collision geometry")
    checks["joint_selectors_limits_and_static_flat_soles"] = True
    motion = load_json(ROOT / "data/motion_manifest.json")
    for entry in motion["source_files"]:
        if sha256(ROOT / motion["raw_source"] / entry["name"]) != entry["sha256"]:
            raise ValueError("Raw owner motion changed during scaffold: " + entry["name"])
    checks["raw_motion_unchanged"] = True
    status = readiness()
    if motion['status'] == 'ready':
        prepared_motion()
    elif motion["status"] not in {"pending_repair", "regenerated_candidate_pending_acceptance"} or motion["prepared_path"] is not None:
        raise ValueError("Motion must be an explicit candidate or an audited prepared input")
    identification = load_json(ROOT / "configs/parameter_identification.json")
    if (identification["status"] != "pending" or identification["sources"]
            or any(value is not None for values in identification["joints"].values() for value in values.values())):
        raise ValueError("No identification values were authorized for this scaffold")
    if status['identification']['status'] != 'pending':
        raise ValueError('This owner baseline does not certify measured identification')
    if (ROOT/'configs/control_profile.json').is_file() and status['control']['status'] != 'passed':
        raise ValueError('Owner-authorized control baseline did not validate')
    # Pre-training baseline starts disabled; the owner-admitted permit legitimately flips it.
    if status["training"]["status"] not in ("disabled", "admitted"):
        raise ValueError("Unexpected training permission")
    checks["deferred_inputs_and_no_training"] = True
    no_models = not any(p for pattern in ('*.pt', '*.onnx') for p in owned_files(pattern))
    if not no_models:
        raise ValueError("Unexpected model artifact in untrained scaffold")
    checks["no_policy_weights_or_fake_motion"] = True
    allowed_motion = set()
    for candidate in motion.get("repair_candidates", []):
        path = (ROOT / candidate["path"]).resolve()
        if sha256(path) != candidate["sha256"]:
            raise ValueError("Repair candidate identity mismatch")
        with np.load(path, allow_pickle=False) as z:
            if (bool(z["training_ready"].item()) or not bool(z["kinematic_checks_passed"].item())
                    or str(z["target_urdf_sha256"].item()) not in {c["urdf_sha256"], c.get('geometry_ssot_sha256')}
                    or str(z["source_npz_sha256"].item()) not in {f["sha256"] for f in motion["source_files"]}):
                raise ValueError("Candidate provenance/readiness flags are invalid")
        allowed_motion.add(path)
    inventory = motion.get('regenerated_artifact_inventory')
    if inventory:
        inventory_path = (ROOT/inventory['path']).resolve()
        if sha256(inventory_path) != inventory['sha256']:
            raise ValueError('Regenerated data inventory identity mismatch')
        for entry in load_json(inventory_path)['npz_files']:
            path = (ROOT/entry['path']).resolve()
            if not path.is_relative_to((ROOT/'data/regenerated').resolve()) or sha256(path) != entry['sha256']:
                raise ValueError('Unbound regenerated artifact: '+str(path))
            allowed_motion.add(path)
    if motion.get('prepared_path'):
        allowed_motion.add(prepared_motion().resolve())
    if {p.resolve() for p in owned_files("*.npz")} != allowed_motion:
        raise ValueError("Unregistered NPZ found; repaired candidates must stay explicit and separate from prepared input")
    checks["candidate_provenance_and_no_training_admission"] = True
    stream = io.StringIO()
    result = None
    if not args.static_only:
        suite = unittest.TestLoader().discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"scope": "Local static/CPU software checks only", "checks": checks,
              "cpu_tests_run": result.testsRun if result else 0, "cpu_tests_passed": result.wasSuccessful() if result else None,
              "cpu_tests_skipped": args.static_only,
              "test_output": stream.getvalue(), "all_passed": all(checks.values()) and (result is None or result.wasSuccessful()),
              "readiness": status, "asset_count": len(manifest["assets"]),
              "model_joint_count": len(names), "static_sole_floor_z_m": floors,
              "not_executed": ["Isaac Lab asset load", "native training/smoke", "F1 dynamic replay",
                               "MuJoCo dynamic tracking/sim2sim (offline FK audited separately)", "cloud tasks", "real robot", "remote publish"],
              "behavior_acceptance": False, "training_admission": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_passed", "checks", "cpu_tests_run", "cpu_tests_passed", "readiness")},
                     ensure_ascii=False, indent=2))
    if result is not None and not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
