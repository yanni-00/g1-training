"""Build a traceable MuJoCo retarget model and transfer learned slots to SSOT.

Only generated output directories are written. The supplied UMR and SSOT assets
are read as provenance, and every native body frame is checked against URDF FK.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from model_geometry import forward_kinematics

TRAINING = Path(__file__).resolve().parents[1]
PROJECT = TRAINING.parents[2]
UMR = PROJECT / "f1_UMR"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def vector(text):
    return np.array([float(x) for x in text.split()])


def string(values):
    return " ".join(format(float(x), ".17g") for x in values)


def pose_attrs(origin):
    if origin is None:
        return {}
    rpy = vector(origin.get("rpy", "0 0 0"))
    xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    return {"pos": origin.get("xyz", "0 0 0"), "quat": string(xyzw[[3, 0, 1, 2]])}


def build_model(urdf, out):
    robot = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = robot.findall("joint")
    child_names = {joint.find("child").get("link") for joint in joints}
    root_name, = set(links)-child_names
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    xml = ET.Element("mujoco", model="f1_ssot_kinematics")
    ET.SubElement(xml, "compiler", angle="radian", autolimits="true", balanceinertia="true", fusestatic="false")
    ET.SubElement(xml, "option", gravity="0 0 -9.81")
    asset = ET.SubElement(xml, "asset")
    world = ET.SubElement(xml, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1", group="2", rgba="0.8 0.8 0.8 1")

    mesh_evidence = []
    mesh_dir = out.parent/"meshes"
    mesh_dir.mkdir()

    def visit(name, parent, incoming=None):
        body = ET.SubElement(parent, "body", name=name, **pose_attrs(incoming.find("origin") if incoming is not None else None))
        if incoming is None:
            ET.SubElement(body, "freejoint", name="floating_base")
        elif incoming.get("type") != "fixed":
            if incoming.get("type") != "revolute":
                raise ValueError("Unsupported SSOT joint: "+incoming.get("name"))
            limit = incoming.find("limit")
            attrs = {"name": incoming.get("name"), "type": "hinge", "axis": incoming.find("axis").get("xyz"),
                     "range": limit.get("lower")+" "+limit.get("upper")}
            dynamics = incoming.find("dynamics")
            if dynamics is not None:
                attrs.update(damping=dynamics.get("damping", "0"), frictionloss=dynamics.get("friction", "0"))
            ET.SubElement(body, "joint", **attrs)
        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            full = np.array([[float(inertia.get("i"+min(a, b)+max(a, b))) for b in "xyz"] for a in "xyz"])
            rpy = vector(inertial.find("origin").get("rpy", "0 0 0")) if inertial.find("origin") is not None else np.zeros(3)
            r = Rotation.from_euler("xyz", rpy).as_matrix()
            full = r @ full @ r.T
            pos = inertial.find("origin").get("xyz", "0 0 0") if inertial.find("origin") is not None else "0 0 0"
            ET.SubElement(body, "inertial", pos=pos, mass=inertial.find("mass").get("value"),
                          fullinertia=string(full[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
        for kind in ("visual", "collision"):
            for i, geom in enumerate(link.findall(kind)):
                mesh = geom.find("geometry/mesh")
                if mesh is None:
                    raise ValueError("Expected actual SSOT mesh geometry")
                path = (urdf.parent/mesh.get("filename")).resolve()
                mesh_name = name+"_"+kind+"_"+str(i)
                # SSOT contains ASCII STL. OBJ keeps all source triangles and
                # avoids the MuJoCo binary-STL face-count/format restrictions.
                surface = trimesh.load_mesh(path, process=False)
                exported = mesh_dir/(mesh_name+".obj")
                exported.write_text(surface.export(file_type="obj", include_normals=False,
                                                   include_color=False, digits=17), encoding="ascii")
                mesh_evidence.append({"source": str(path), "source_sha256": sha(path),
                                      "derived": str(exported), "derived_sha256": sha(exported),
                                      "faces": len(surface.faces), "vertices": len(surface.vertices),
                                      "operation": "format-only OBJ export; no triangle simplification"})
                print("[SSOT mesh]", mesh_name, "faces", len(surface.faces), flush=True)
                ET.SubElement(asset, "mesh", name=mesh_name, file="meshes/"+exported.name, scale=mesh.get("scale", "1 1 1"))
                attrs = dict(name=mesh_name, type="mesh", mesh=mesh_name, **pose_attrs(geom.find("origin")))
                attrs.update(contype="0" if kind == "visual" else "1", conaffinity="0" if kind == "visual" else "1",
                             group="1" if kind == "visual" else "0", density="0")
                if kind == "visual":
                    attrs["rgba"] = "0.5 0.65 0.8 1"
                else:
                    attrs["rgba"] = "0.5 0.65 0.8 0"
                ET.SubElement(body, "geom", **attrs)
        for joint in children.get(name, []):
            visit(joint.find("child").get("link"), body, joint)
    visit(root_name, world)
    ET.indent(xml)
    ET.ElementTree(xml).write(out, encoding="utf-8", xml_declaration=True)
    write_json(out.parent/"mesh_provenance.json", mesh_evidence)
    return robot


def native_model(path):
    """Native Windows MuJoCo accepts relative ASCII paths from a Unicode cwd."""
    path = Path(path)
    old_cwd = Path.cwd()
    try:
        os.chdir(path.parent)
        return mujoco.MjModel.from_xml_path(path.name)
    finally:
        os.chdir(old_cwd)


def set_angles(model, data, angles):
    mujoco.mj_resetData(model, data)
    for name, angle in angles.items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory")
    out.mkdir(parents=True)
    contract = json.loads((TRAINING/"configs/robot_contract.json").read_text(encoding="utf-8"))
    urdf = TRAINING/contract["urdf"]
    if sha(urdf) != contract["urdf_sha256"]:
        raise ValueError("Packaged SSOT identity changed")
    model_path = out/"f1_ssot.xml"
    robot = build_model(urdf, model_path)
    target = native_model(model_path)
    data = mujoco.MjData(target)
    sys.path[:0] = [str(UMR/"scripts"), str(UMR)]
    import retarget_smpl_to_humanoid_surface_vector as retarget
    from humanoid_retarget_config import load_config, resolve_path
    from mujoco_point_cloud_center import point_cloud_center_frame
    config = load_config(UMR/"configs/weichai_f1_29_base.json")
    tpose = config["robot"]["tpose_qpos"]
    checks = []
    poses = [{name: 0. for name in tpose}, tpose]
    source_paths = sorted((UMR/"umr_data/weichai_f1_29_accad_walking").glob("*.npz"))
    for path in source_paths:
        with np.load(path, allow_pickle=True) as raw:
            names = raw["robot_joint_names"].tolist()
            q = raw["qpos"]
            for frame in (0, len(q)//2, len(q)-1):
                poses.append(dict(zip(names, q[frame, 7:].astype(float))))
    for angles in poses:
        expected = forward_kinematics(robot, angles)
        set_angles(target, data, angles)
        pos_error, rot_error = 0., 0.
        for name, transform in expected.items():
            i = target.body(name).id
            pos_error = max(pos_error, float(np.abs(data.xpos[i]-transform[:3, 3]).max()))
            rot_error = max(rot_error, float(np.abs(data.xmat[i].reshape(3, 3)-transform[:3, :3]).max()))
        checks.append({"max_position_component_error_m": pos_error, "max_rotation_matrix_component_error": rot_error})
    if any(c["max_position_component_error_m"] > 1e-8 or c["max_rotation_matrix_component_error"] > 1e-8 for c in checks):
        raise ValueError("Native SSOT model differs from independent URDF FK")
    old_path = resolve_path(config["robot"]["xml"], config)
    old = native_model(old_path)
    old_ref = mujoco.MjData(old)
    set_angles(old, old_ref, tpose)
    set_angles(target, data, tpose)
    slots_path = resolve_path(config["correspondence"]["slots"], config)
    with np.load(slots_path, allow_pickle=False) as z:
        names = z["names"].astype(str).tolist()
        modes = z["center_modes"].astype(str)
        slots = z["reconstructed_slots"].copy()
        target_points = z["target_points"].copy()
    robot_index = names.index(config["robot"]["name"])
    matrix = np.asarray(config["robot"].get("frame_transform", {}).get("smpl_to_robot_root_matrix", retarget.SMPL_TO_ROBOT_ROOT_MATRIX))
    new_center, new_rotation, _ = point_cloud_center_frame(target, data, config["robot"]["point_cloud_center"])
    mapping_counts = {}
    for label, points in (("reconstructed_slots", slots), ("target_points", target_points)):
        binding = retarget.bind_robot_slots(old, SimpleNamespace(config_data=config), points[robot_index], source_model_type="smplx")
        transferred = np.empty_like(points[robot_index])
        for geom_id in np.unique(binding["geom_ids"]):
            mask = binding["geom_ids"] == geom_id
            old_id = old.geom_bodyid[geom_id]
            old_name = old.body(old_id).name
            new_name = old_name
            if mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_BODY, new_name) < 0 and new_name.endswith("_link"):
                new_name = new_name[:-5]
            new_id = target.body(new_name).id
            world = binding["local_pos"][mask] @ old_ref.geom_xmat[geom_id].reshape(3, 3).T + old_ref.geom_xpos[geom_id]
            body_local = (world-old_ref.xpos[old_id]) @ old_ref.xmat[old_id].reshape(3, 3)
            new_world = body_local @ data.xmat[new_id].reshape(3, 3).T + data.xpos[new_id]
            root_points = (new_world-new_center) @ new_rotation
            transferred[mask] = root_points @ matrix
            if label == "reconstructed_slots":
                mapping_counts[new_name] = mapping_counts.get(new_name, 0)+int(mask.sum())
        points[robot_index] = transferred
    adapted_slots = out/"ssot_correspondence_slots.npz"
    np.savez_compressed(adapted_slots, names=np.asarray(names), center_modes=modes,
                        reconstructed_slots=slots, target_points=target_points,
                        source_slots_sha256=np.array(sha(slots_path)), urdf_sha256=np.array(sha(urdf)),
                        adaptation=np.array("old surface binding -> body local -> SSOT T-pose surface positions"))
    config = {k: v for k, v in config.items() if not k.startswith("_config")}
    config["smplx_model_dir"] = str((UMR/"smpl").resolve())
    config["robot"]["xml"] = str(model_path)
    config["robot"]["xml_policy"] = {"add_freejoint_root": False}
    config["correspondence"]["slots"] = str(adapted_slots)
    config["correspondence"]["shared_smplx_template"]["surface_binding"] = str(out/"human_surface_binding.npz")
    write_json(out/"umr_ssot.json", config)
    native_names = [target.joint(i).name for i in range(target.njnt) if target.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    report = {"urdf_path": str(urdf), "urdf_sha256": sha(urdf), "native_model": str(model_path), "native_model_sha256": sha(model_path),
              "source_mjcf": str(old_path), "source_mjcf_sha256": sha(old_path), "source_slots": str(slots_path), "source_slots_sha256": sha(slots_path),
              "adapted_slots": str(adapted_slots), "adapted_slots_sha256": sha(adapted_slots),
              "joint_names": native_names, "body_names": [target.body(i).name for i in range(1, target.nbody)],
              "fk_checks": checks, "fk_numerical_component_tolerance": 1e-8, "fk_checks_passed": True,
              "slot_body_counts": mapping_counts, "source_code_sha256": sha(Path(__file__)), "mujoco_version": mujoco.__version__}
    write_json(out/"model_audit.json", report)
    print(json.dumps({"fk_passed": True, "nq": target.nq, "bodies": target.nbody, "output": str(out)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
