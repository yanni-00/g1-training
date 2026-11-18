"""Exact saved F1 BeyondMimic actor in server MuJoCo; no PPO or automatic resets."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import csv
import hashlib
# The existing server gmr interpreter is3.10. Supply only the stdlib hashing
# compatibility operation; leave every trained source/config byte unchanged.
if not hasattr(hashlib, "file_digest"):
    def _file_digest(stream, algorithm):
        result=hashlib.new(algorithm)
        for block in iter(lambda:stream.read(1<<20), b""): result.update(block)
        return result
    hashlib.file_digest=_file_digest
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[1] / "training"
sys.path[:0] = [str(ROOT / "source/f1_beyondmimic"), str(ROOT / "scripts")]
from f1_beyondmimic.assets_archive import ensure_assets
from f1_beyondmimic.contract import load_contract, load_json, effective_control, verify_assets, sha256
from f1_beyondmimic.utils.checkpoint import load_checkpoint, position_targets
from f1_beyondmimic.utils.runtime_policy import RuntimeActor
from f1_beyondmimic.utils.actor_observations import proprioceptive_observation
from f1_beyondmimic.utils.velocity_gait_reference import VelocityGaitReference
from mujoco_model import build_model


def matrix(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def velocity(model, data, body_id, local=False):
    value = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, value, int(local))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-scene", type=Path)
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.sha256: raise ValueError("Wrong source checkpoint")
    ensure_assets(ROOT); verify_assets()
    saved = load_checkpoint(args.checkpoint)  # exact trained source/config/URDF binding
    runtime = saved["infos"]["f1_runtime"]
    cfg, controls = load_contract(), effective_control()
    protocol = load_json(args.protocol)
    if protocol["format"] != "f1_fixed_weight_mujoco_replay_v1": raise ValueError("Wrong protocol")
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"]); np.random.seed(protocol["seed"])
    dt = runtime["policy_dt"]; physics_dt = protocol["physics_dt"]
    substeps = protocol["physics_substeps"]
    if not math.isclose(dt, physics_dt * substeps): raise ValueError("Timing mismatch")
    frozen = load_json(ROOT / "configs/reference/stable_runtime_settings.json")
    refinement=protocol.get("physics_refinement",1)
    if refinement not in (1,2,4): raise ValueError("Unsupported registered numerical diagnostic")
    if physics_dt != frozen["physics_dt"] / refinement or substeps != frozen["decimation"] * refinement:
        raise ValueError("This replay must preserve the registered original physics setting")
    schedule = protocol["schedule"]
    counts = [round(s["duration_s"] / dt) for s in schedule]
    if sum(counts) != args.max_steps or any(abs(n*dt-s["duration_s"])>1e-7 for s,n in zip(schedule,counts)):
        raise ValueError("Actual protocol does not consume the registered step limit")
    for s in schedule:
        if len(s["command"]) != 3: raise ValueError("Invalid command")
        if any(not lo <= v <= hi for v,(lo,hi) in zip(s["command"], runtime["command_limits"])):
            raise ValueError("Command outside this checkpoint limits")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    if args.prepared_scene:
        scene = args.prepared_scene.resolve()
        prior = load_json(scene.parent / "runtime_audit.json")
        if sha256(scene) != prior["scene_sha256"] or prior["urdf_sha256"] != cfg["urdf_sha256"]:
            raise ValueError("Prepared scene identity mismatch")
        model = mujoco.MjModel.from_xml_path(str(scene))
    else:
        model, scene = build_model(cfg, controls, output, protocol["gravity"], physics_dt)
    model.opt.timestep=physics_dt
    if protocol.get("integrator") != "discrete": raise ValueError("Frozen discrete integrator required")
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_DISCRETE
    data = mujoco.MjData(model)
    names, body_names = runtime["joint_names"], runtime["body_names"]
    joints = np.array([model.joint(n).id for n in names]); actuators = np.array([model.actuator(n).id for n in names])
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    if len(set(qids)) != 29 or model.nu != 29: raise ValueError("Non-bijective F1 control mapping")
    base_id = model.body("base_link").id
    anchor_id = model.body(runtime["anchor_body_name"]).id
    anchor_ref_id = body_names.index(runtime["anchor_body_name"])
    root_ref_id = body_names.index("base_link")
    foot_ids = [model.body(n).id for n in cfg["feet"]]
    end_ids = [model.body(n).id for n in cfg["end_effectors"]]
    end_ref_ids = [body_names.index(n) for n in cfg["end_effectors"]]
    control = {k:v.cpu().numpy() for k,v in runtime["control"].items()}
    for key, actual in {"kp": model.actuator_gainprm[actuators,0],
            "kd_total": -model.actuator_biasprm[actuators,2]+model.dof_damping[vids],
            "armature": model.dof_armature[vids], "joint_lower": model.jnt_range[joints,0],
            "joint_upper": model.jnt_range[joints,1]}.items():
        np.testing.assert_allclose(actual, control[key], rtol=2e-6, atol=2e-6)
    wrist_slots=np.array([i for i,n in enumerate(names) if "wrist" in n])
    wrist_act=actuators[wrist_slots]
    wrist_effort=np.array([controls[names[i]]["effort_limit"] for i in wrist_slots])
    model.actuator_forcelimited[wrist_act]=False
    endstep_peak=np.zeros(len(wrist_slots))
    substep_trace=[]
    reference = VelocityGaitReference(runtime["motion"], 1, runtime["command_limits"],
                                     acceleration=runtime["phase_acceleration"])
    reference.reset(torch.tensor([0]), torch.tensor([protocol["initial_support_index"]]), torch.zeros(1,2), torch.zeros(1))
    if protocol.get("initial_mode") == "moving":
        reference.request(torch.tensor([schedule[0]["command"]]),torch.tensor([True]))
        reference.initialize_rate(torch.tensor([0]))
    data.qpos[:3] = reference.body_pos_w[0,root_ref_id].numpy()
    data.qpos[3:7] = reference.body_quat_w[0,root_ref_id].numpy()
    data.qpos[qids] = reference.joint_pos[0].numpy(); data.qvel[:] = 0
    if protocol.get("initial_mode") == "moving":
        data.qvel[:3]=reference.body_lin_vel_w[0,root_ref_id].numpy()
        data.qvel[3:6]=matrix(data.qpos[3:7]).T@reference.body_ang_vel_w[0,root_ref_id].numpy()
        data.qvel[vids]=reference.joint_vel[0].numpy()
    mujoco.mj_forward(model, data)
    # FK catches frame/order/model conversion errors before policy inference.
    actual_positions = np.array([data.xpos[model.body(n).id] for n in body_names])
    actual_rotations = np.array([data.xmat[model.body(n).id].reshape(3,3) for n in body_names])
    reference_positions = reference.body_pos_w[0].numpy()
    reference_rotations = np.array([matrix(q) for q in reference.body_quat_w[0].numpy()])
    fk_pos_error = float(np.max(np.abs(actual_positions-reference_positions)))
    fk_rotation_error = float(np.max(np.abs(actual_rotations-reference_rotations)))
    # Existing package export parity uses2e-4; this is a conversion check, not a behavior gate.
    np.testing.assert_allclose(actual_positions, reference_positions, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual_rotations, reference_rotations, rtol=2e-4, atol=2e-4)
    audit = {"format":"f1_mujoco_replay_audit_v1", "iteration": int(saved["iter"]),
        "checkpoint_sha256": sha256(args.checkpoint), "binding_matches": True,
        "urdf_sha256": cfg["urdf_sha256"], "scene_sha256": sha256(scene), "scene": str(scene),
        "protocol_sha256": sha256(args.protocol), "entry_sha256": sha256(Path(__file__)),
        "mujoco_version": mujoco.__version__, "torch_version":str(torch.__version__),
        "integrator":"discrete", "wrist_force_check":"No instantaneous clipping of wrist drive before implicit solve; actual poststep PD effort checked eachphysicsstep; abort beyond original effort. Experimental translation, not native equivalence.", "velocity_limit_sim_mapping":"not_implemented; diagnostic limitation", "policy_dt":dt,"physics_dt":float(model.opt.timestep),"physics_substeps":substeps,
        "planned_steps":args.max_steps,"physics_refinement":refinement,"initial_mode":protocol.get("initial_mode","standing"),"joint_names":names,"body_names":body_names,
        "kp": model.actuator_gainprm[actuators,0].tolist(),"pd_kd":(-model.actuator_biasprm[actuators,2]).tolist(),
        "passive_damping":model.dof_damping[vids].tolist(),"armature":model.dof_armature[vids].tolist(),
        "effort_limits": model.actuator_forcerange[actuators].tolist(),
        "initial_fk_position_max_abs_error_m":fk_pos_error,"initial_fk_rotation_max_abs_error":fk_rotation_error,
        "body_mass":{model.body(i).name:float(model.body_mass[i]) for i in range(1,model.nbody)},
        "body_inertia":{model.body(i).name:model.body_inertia[i].tolist() for i in range(1,model.nbody)},
        "angular_velocity_frame":"mjOBJ_XBODY local regular base frame; no inertial-frame gyro leakage",
        "target_processing":"saved default plus saved action scale times raw action, URDF hard clipping, implicit affine servo with original effort bounds",
        "optimizer_updates":0,"hashing_compatibility":"sha256 streaming fallback on Python3.10; trained sources untouched","simulation_only":True,"limitations":protocol["scope"]}
    (output/"runtime_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if args.prepare_only:
        print(json.dumps({"prepare":"passed", "fk_position_error":fk_pos_error,"fk_rotation_error":fk_rotation_error}),flush=True)
        return
    actor = RuntimeActor(saved).eval()
    import imageio.v2 as imageio
    width,height=protocol["render"]["resolution"]
    model.vis.global_.offwidth=width; model.vis.global_.offheight=height
    renderer=mujoco.Renderer(model,height=height,width=width)
    camera=mujoco.MjvCamera();camera.distance=3.2;camera.azimuth=130;camera.elevation=-12
    writer=imageio.get_writer(str(output/"replay.mp4"),fps=1/(dt*protocol["render"]["stride"]),codec="libx264")
    rows=[]; last_action=np.zeros(29,dtype=np.float32);terminated=False;reason=[]
    threshold=protocol["termination_thresholds"]
    force_threshold=runtime["native_contact_audit"]["force_threshold"]
    try:
        with torch.inference_mode():
            for stage,count in zip(schedule,counts):
                reference.request(torch.tensor([stage["command"]]),torch.tensor([stage["walk_enabled"]]))
                for _ in range(count):
                    if len(rows)>=args.max_steps: raise ValueError("Replay exceeded bound")
                    mujoco.mj_forward(model,data)
                    root_r=data.xmat[base_id].reshape(3,3);anchor_r=data.xmat[anchor_id].reshape(3,3)
                    ref_q=reference.joint_pos[0].numpy();ref_dq=reference.joint_vel[0].numpy()
                    obs=proprioceptive_observation(ref_q,ref_dq,anchor_r.T@matrix(reference.body_quat_w[0,anchor_ref_id].numpy()),
                        velocity(model,data,base_id,True)[:3],data.qpos[qids],data.qvel[vids],last_action,control["default_joint_pos"])
                    raw=actor(torch.from_numpy(obs[None]))[0]
                    target=position_targets(raw,runtime).numpy()
                    if not np.isfinite(target).all(): raise ValueError("Nonfinite policy action")
                    body_velocity=root_r.T@velocity(model,data,base_id)[3:]
                    row={"step":len(rows),"time_s":len(rows)*dt,"stage":stage["name"],"walk_enabled":stage["walk_enabled"],
                        "command_vx":stage["command"][0],"command_vy":stage["command"][1],"command_yaw_rate":stage["command"][2],
                        "actual_vx":float(body_velocity[0]),"actual_vy":float(body_velocity[1]),
                        "actual_yaw_rate":float(velocity(model,data,base_id)[2]),
                        "yaw":math.atan2(root_r[1,0],root_r[0,0]),"root_x":float(data.xpos[base_id,0]),
                        "root_y":float(data.xpos[base_id,1]),
                        **{"root_q"+axis:float(v) for axis,v in zip("wxyz",data.qpos[3:7])},"root_z":float(data.xpos[base_id,2]),
                        "root_tilt_rad":math.acos(float(np.clip(root_r[2,2],-1,1))),
                        "phase":float(reference.clock.phase[0]),"reference_rate":float(reference.clock.rate[0]),
                        "reference_cycles":int(reference.clock.cycles[0])}
                    for rid,bid,name in zip(end_ref_ids,end_ids,cfg["end_effectors"]):
                        row[name+":height_error_m"]=float(reference.body_pos_w[0,rid,2])-float(data.xpos[bid,2])
                    for prefix,values in (("q",data.qpos[qids]),("dq",data.qvel[vids]),("raw_action",raw.numpy()),("target",target),("q_ref",ref_q),("observation",obs)):
                        labels=names if len(values)==29 else [str(i) for i in range(len(values))]
                        row.update({prefix+":"+name:float(v) for name,v in zip(labels,values)})
                    forces=np.zeros((2,3)); f=np.zeros(6)
                    floor=model.geom("floor").id
                    for i in range(data.ncon):
                        contact=data.contact[i]
                        for side,foot in enumerate(foot_ids):
                            b1,b2=model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
                            if (b1==foot and contact.geom2==floor) or (b2==foot and contact.geom1==floor):
                                mujoco.mj_contactForce(model,data,i,f)
                                forces[side]+=contact.frame.reshape(3,3).T@f[:3]*(1 if b2==foot else -1)
                    for side,foot,name in zip(range(2),foot_ids,cfg["feet"]):
                        row[name+":force_N"]=float(np.linalg.norm(forces[side]));row[name+":contact"]=bool(np.linalg.norm(forces[side])>force_threshold)
                        row[name+":body_planar_speed_proxy"]=float(np.linalg.norm(velocity(model,data,foot)[3:5]))
                        row[name+":z"]=float(data.xpos[foot,2])
                        for axis,value in zip("xy",data.xpos[foot,:2]):row[name+":"+axis]=float(value)
                    if len(rows)%protocol["render"]["stride"]==0:
                        camera.lookat[:]=data.xpos[base_id];renderer.update_scene(data,camera);writer.append_data(renderer.render())
                    data.ctrl[actuators]=target
                    for substep in range(substeps):
                        time_before=float(data.time)
                        mujoco.mj_step(model,data)
                        mujoco.mj_forward(model,data)
                        wrist_force=data.actuator_force[wrist_act].copy()
                        endstep_peak=np.maximum(endstep_peak,np.abs(wrist_force))
                        substep_trace.append({"policy_step":len(rows),"substep":substep,"time_s":float(data.time),
                            **{"effort:"+names[i]:float(f) for i,f in zip(wrist_slots,wrist_force)},
                            **{"velocity:"+names[i]:float(v) for i,v in zip(wrist_slots,data.qvel[vids[wrist_slots]])}})
                        if abs(float(data.time)-time_before-physics_dt)>1e-7:reason.append("physics_time_reset")
                        if not np.isfinite(wrist_force).all():reason.append("nonfinite_endstep_effort")
                        if np.any(np.abs(wrist_force)>wrist_effort+1e-7):reason.append("wrist_endstep_effort_exceeded")
                        if reason:break
                    mujoco.mj_forward(model,data)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():reason.append("nonfinite_physics")
                    anchor_ref_r=matrix(reference.body_quat_w[0,anchor_ref_id].numpy())
                    anchor_actual_r=data.xmat[anchor_id].reshape(3,3)
                    if abs(float(reference.body_pos_w[0,anchor_ref_id,2])-data.xpos[anchor_id,2])>threshold["anchor_z_m"]:reason.append("anchor_z")
                    if abs(anchor_ref_r[2,2]-anchor_actual_r[2,2])>threshold["anchor_gravity_z"]:reason.append("anchor_ori")
                    if max(abs(float(reference.body_pos_w[0,rid,2])-data.xpos[bid,2]) for rid,bid in zip(end_ref_ids,end_ids))>threshold["end_effector_z_m"]:reason.append("end_effector_z")
                    for rid,bid,name in zip(end_ref_ids,end_ids,cfg["end_effectors"]):
                        row[name+":post_step_height_error_m"]=float(reference.body_pos_w[0,rid,2])-float(data.xpos[bid,2])
                    terminated=bool(reason);row["step_done"]=terminated;row["termination_reason"]="|".join(reason)
                    rows.append(row);last_action=raw.numpy().copy()
                    if terminated:break
                    reference.step(dt)
                print(json.dumps({"stage":stage["name"],"steps":len(rows),"terminated":terminated}),flush=True)
                if terminated:break
    finally:
        writer.close();renderer.close()
        if substep_trace:
            with (output/"wrist_substeps.csv").open("w",newline="",encoding="utf-8") as stream:
                w=csv.DictWriter(stream,fieldnames=list(substep_trace[0]));w.writeheader();w.writerows(substep_trace)
        if rows:
            with (output/"states.csv").open("w",newline="",encoding="utf-8") as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"iteration":int(saved["iter"]),"checkpoint_sha256":args.sha256,"completed_steps":len(rows),
        "elapsed_s":len(rows)*dt,"completed_protocol":len(rows)==args.max_steps and not terminated,
        "terminated_early":terminated,"termination_reason":reason,"optimizer_updates":0,
        "wrist_endstep_peak_effort":dict(zip([names[i] for i in wrist_slots],endstep_peak.tolist())),"wrist_effort_limits":dict(zip([names[i] for i in wrist_slots],wrist_effort.tolist())),"behavior_acceptance":False,"scope":"MuJoCo sim2sim only; original PhysX behavior and hardware remain unverified",
        "stages":{}}
    for name in dict.fromkeys(r["stage"] for r in rows):
        group=[r for r in rows if r["stage"]==name]
        summary["stages"][name]={"samples":len(group),"duration_s":len(group)*dt,
            "mean_vx":float(np.mean([r["actual_vx"] for r in group])),
            **{axis+"_rmse":float(np.sqrt(np.mean([(r["actual_"+axis]-r["command_"+axis])**2 for r in group]))) for axis in ("vx","vy","yaw_rate")}}
    (output/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":main()
