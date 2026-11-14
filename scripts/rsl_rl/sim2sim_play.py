"""Exact F1 saved actor in MuJoCo; adapted from the project's G1 sim2sim route.

Source URDF geometry/inertia, named joint order, owner PD, and recorded training
timing are preserved. MuJoCo engine/contact/drive differences remain sim2sim
limitations. No G1 joint gains, armature, target clipping, or initial height leak in.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(rpy):
    import numpy as np
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def build_model(contract, controls, output, gravity):
    import mujoco
    import numpy as np
    urdf = ROOT / contract["urdf"]
    assert sha(urdf) == contract["urdf_sha256"]
    source = ET.parse(urdf).getroot()
    fixed = mujoco.MjModel.from_xml_path(str(urdf))
    intermediate = output / "urdf_import.xml"
    mujoco.mj_saveLastXML(str(intermediate), fixed)
    tree = ET.parse(intermediate)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(urdf.parent))
    compiler.set("balanceinertia", "true")  # Existing source URDF compiler option.
    compiler.set("fusestatic", "false")
    root.set("model", "F1_29dof_ownerpd_sim2sim")
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError("URDF import must preserve the named base frame")
    ET.SubElement(base, "freejoint", name="floating_base")
    # Restore source inertia precision after MuJoCo's text serializer. The same
    # operation is used by the SSOT conversion builder; automatic balancing is audited.
    for link in source.findall("link"):
        inertial = link.find("inertial")
        body = root.find(f".//body[@name='{link.get('name')}']")
        if inertial is None or body is None:
            continue
        target = body.find("inertial")
        if target is None:
            target = ET.SubElement(body, "inertial")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        tensor = np.array([[float(inertia.get(k)) for k in row]
                           for row in (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))])
        r = rotation([float(v) for v in origin.get("rpy", "0 0 0").split()])
        tensor = r @ tensor @ r.T
        target.attrib.clear()
        target.set("pos", origin.get("xyz", "0 0 0"))
        target.set("mass", inertial.find("mass").get("value"))
        target.set("fullinertia", " ".join(format(tensor[i, j], ".17g")
                   for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(contract["sim_dt"]))
    option.set("gravity", " ".join(str(v) for v in gravity))
    option.set("integrator", "implicitfast")  # Explicit MuJoCo execution choice, not identification.
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.1",
                  rgba="0.85 0.87 0.9 1", friction="1 0 0")
    # Sliding friction=1 from the F1 environment's physics_material; no added
    # rolling/torsional identification. Both plane and collision geoms are explicit.
    for geom in base.iter("geom"):
        if geom.get("group") == "1":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        else:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("friction", "1 0 0")
    actuators = ET.SubElement(root, "actuator")
    for name in contract["model_joint_names"]:
        control = controls[name]
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing named joint {name}")
        joint.set("damping", str(control["passive_damping"]))
        joint.set("frictionloss", str(contract["joint_properties"][name]["dynamics"]["friction"]))
        joint.set("armature", str(control["armature"]))
        effort = control["effort_limit"]
        ET.SubElement(actuators, "general", name=name, joint=name, gaintype="fixed", biastype="affine",
                      gainprm=str(control["stiffness"]),
                      biasprm=f"0 {-control['stiffness']} {-control['pd_damping']}",
                      forcelimited="true", forcerange=f"{-effort} {effort}", ctrllimited="false")
    ET.indent(tree)
    xml = output / "f1_scene.xml"
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(xml)), xml


def rpy(q):
    import numpy as np
    w, x, y, z = q
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def main():
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import torch
    from f1_rl_lab.assets_archive import ensure_assets
    from f1_rl_lab.checkpoint import verify_checkpoint
    from f1_rl_lab.contract import load_contract, effective_control
    from f1_rl_lab.interface import ActorHistory, joint_targets
    from f1_rl_lab.policy import load_actor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_assets()
    assert sha(args.checkpoint) == args.sha256
    identity = verify_checkpoint(args.checkpoint)
    runtime = identity["runtime"]
    contract = load_contract()
    controls = effective_control()
    protocol = json.loads((ROOT / "configs/eval_schedule.json").read_text())
    commands = [s["command"] for s in protocol["segments"]
                for _ in range(round(s["duration_s"] / contract["step_dt"]))]
    assert len(commands) == args.max_steps
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint, runtime)
    model, xml = build_model(contract, controls, args.output, runtime["gravity"])
    data = mujoco.MjData(model)
    names = contract["model_joint_names"]
    jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
    aids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
    assert min(jids) >= 0 and min(aids) >= 0 and model.nu == len(names)
    qadr, vadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    feet = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
             contract["joint_properties"][s+"_ankle_roll_joint"]["child"]) for s in ("left", "right")}
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data.qpos[:3] = contract["base_position"]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr] = contract["default_joint_pos"]
    mujoco.mj_forward(model, data)
    audit = {"checkpoint_sha256": sha(args.checkpoint), "urdf_sha256": contract["urdf_sha256"],
             "mujoco_version": mujoco.__version__, "torch_version": str(torch.__version__),
             "policy_joint_names": names, "model_nq": model.nq, "model_nv": model.nv,
             "model_nu": model.nu, "sim_dt": model.opt.timestep, "control_dt": contract["step_dt"],
             "kp": model.actuator_gainprm[aids, 0].tolist(),
             "pd_kd": (-model.actuator_biasprm[aids, 2]).tolist(),
             "passive_damping": model.dof_damping[vadr].tolist(),
             "armature": model.dof_armature[vadr].tolist(),
             "frictionloss": model.dof_frictionloss[vadr].tolist(),
             "body_mass": {model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)},
             "body_inertia": {model.body(i).name: model.body_inertia[i].tolist() for i in range(1, model.nbody)},
             "initial_feet_positions": {s: data.xpos[i].tolist() for s,i in feet.items()},
             "gravity": model.opt.gravity.tolist(), "integrator": "implicitfast",
             "protocol": protocol,
             "limitations": ["MuJoCo contact/solver and implicit servo differ from PhysX",
                 "No extra joint velocity clamp: source limits are recorded, not imitated by an unverified limiter",
                 "Source URDF balanceinertia=true retained; compiled body inertias recorded",
                 "Clean flat-plane sim2sim, not domain-randomized training or real robot validation"],
             "visual_settings": {"resolution": [1280,720], "camera_distance": 3.2,
                 "camera_azimuth": 130, "camera_elevation": -12,
                 "source": "existing G1 sim2sim_play.py rendering; engineering display settings"}}
    assert np.array_equal(audit["kp"], runtime["kp_policy"])
    assert np.array_equal(audit["pd_kd"], runtime["pd_damping_policy"])
    assert np.array_equal(audit["passive_damping"], runtime["passive_damping_policy"])
    assert np.array_equal(audit["armature"], runtime["armature_policy"])
    (args.output/"runtime_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.prepare_only:
        print("F1_MUJOCO_PREPARE_OK", flush=True)
        return
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 130, -12
    history = ActorHistory(contract)
    previous = np.zeros(len(names), dtype=np.float32)
    rows = []
    first_fall = None
    video = args.output / "playback.mp4"
    writer = imageio.get_writer(str(video), fps=round(1/contract["step_dt"]),
                               codec="libx264", quality=8, macro_block_size=None)
    try:
        for step, command in enumerate(commands):
            spatial = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, spatial, 1)
            orientation = data.xmat[base_id].reshape(3,3)
            observation = history.push({"base_ang_vel": spatial[:3],
                "projected_gravity": orientation.T @ np.array([0.,0.,-1.]),
                "velocity_commands": command,
                "joint_pos_rel": data.qpos[qadr]-contract["default_joint_pos"],
                "joint_vel_rel": data.qvel[vadr], "last_action": previous})
            with torch.inference_mode():
                action = actor(torch.from_numpy(observation[None])).numpy()[0]
            target = joint_targets(action, contract)
            data.ctrl[aids] = target
            for _ in range(contract["decimation"]):
                mujoco.mj_step(model, data)
            # Refresh cached poses for the just-integrated state before metrics/render.
            mujoco.mj_forward(model, data)
            roll, pitch, yaw = rpy(data.qpos[3:7])
            vx,vy = data.qvel[:2]
            fall = bool(data.qpos[2] < protocol["termination"]["minimum_height"] or
                -data.xmat[base_id].reshape(3,3)[2,2] > -np.cos(protocol["termination"]["limit_angle"]))
            if fall and first_fall is None:
                first_fall = float(data.time)
            row = {"t": float(data.time), "cmd_vx": command[0], "cmd_vy": command[1], "cmd_wz": command[2],
                "base_z": float(data.qpos[2]), "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
                "roll_deg": float(np.degrees(roll)), "pitch_deg": float(np.degrees(pitch)),
                "vx_yaw": float(np.cos(yaw)*vx+np.sin(yaw)*vy),
                "vy_yaw": float(-np.sin(yaw)*vx+np.cos(yaw)*vy), "wz": float(data.qvel[5]),
                "fall": int(fall)}
            fz = dict(left=0., right=0.)
            contacts = dict(left=0, right=0)
            for i in range(data.ncon):
                contact = data.contact[i]
                if floor not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == floor else contact.geom1
                for side, bid in feet.items():
                    if model.geom_bodyid[other] == bid:
                        contacts[side] = 1
                        force = np.empty(6)
                        mujoco.mj_contactForce(model, data, i, force)
                        world_force = contact.frame.reshape(3,3).T @ force[:3]
                        fz[side] += float(world_force[2] * (1 if contact.geom1 == floor else -1))
            for side,bid in feet.items():
                row[side+"_fz"] = fz[side]
                row[side+"_contact"] = contacts[side]
                row[side+"_foot_z"] = float(data.xpos[bid,2])
            for i,name in enumerate(names):
                row[name] = float(data.qpos[qadr[i]])
                row[name+"_velocity"] = float(data.qvel[vadr[i]])
                row[name+"_desired"] = float(target[i])
                row[name+"_torque"] = float(data.actuator_force[aids[i]])
                row[name+"_err_deg"] = float(np.degrees(target[i]-data.qpos[qadr[i]]))
            rows.append(row)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())
            previous = action.copy()
            if (step+1) % 250 == 0 or fall:
                print(json.dumps({"step":step+1,"t":row["t"],"base_z":row["base_z"],
                                  "fall":fall,"vx_yaw":row["vx_yaw"]}), flush=True)
            if fall:
                break  # Preserve first failure; do not reset into a manufactured success.
    finally:
        writer.close()
        renderer.close()
        with (args.output/"trajectory.csv").open("w",newline="") as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
    summary = {"requested_steps": len(commands), "completed_steps": len(rows),
               "fall_time_s": first_fall,"checkpoint_sha256":args.sha256,
               "video_sha256":sha(video),"trajectory_sha256":sha(args.output/"trajectory.csv"),
               "runtime_audit_sha256":sha(args.output/"runtime_audit.json")}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("F1_MUJOCO_PLAY_COMPLETE "+json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
