"""Replay a real F1 actor in an explicitly supplied, reviewed MuJoCo model.

The MJCF must be produced from the current F1 URDF using the SSOT conversion
workflow. This tool does not claim that a file's existence proves conversion parity.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--schedule", type=Path, default=ROOT / "configs/eval_schedule.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from f1_rl_lab.launch import require_launch
    from f1_rl_lab.checkpoint import verify_checkpoint
    from f1_rl_lab.contract import load_contract, sha256
    from f1_rl_lab.interface import ActorHistory, joint_targets
    from f1_rl_lab.policy import load_actor
    require_launch("evaluation", args.checkpoint)
    try:
        import mujoco as mj
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required in the evaluation environment; scaffold checks do not run sim2sim") from exc
    import numpy as np
    import torch
    c = load_contract()
    identity = verify_checkpoint(args.checkpoint)
    runtime = identity["runtime"]
    actor = load_actor(args.checkpoint, runtime)
    model = mj.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mj.MjData(model)
    joints = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in c["model_joint_names"]]
    if min(joints) < 0:
        raise ValueError("MJCF is missing an F1 joint")
    if model.nv != c["action_dim"] + 6:
        raise ValueError("Expected exactly the F1 movable joints and one free base")
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limits = np.array([[c["joint_properties"][n]["limit"][k] for k in ("lower", "upper")] for n in c["model_joint_names"]])
    if not np.allclose(model.jnt_range[joints], limits):
        raise ValueError("MJCF joint limits differ from the F1 URDF")
    root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    if root < 0 or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError("MJCF needs a free base_link")
    if not np.isclose(model.opt.timestep, c["sim_dt"]):
        raise ValueError("Use the actual training physics timestep in the reviewed MJCF")
    if not np.allclose(model.opt.gravity, runtime["gravity"]):
        raise ValueError("MJCF gravity differs from the actual training environment")
    # Explicit torque PD plus physical passive damping. No implicit MJCF drive in parallel.
    model.dof_damping[vids] = runtime["passive_damping_policy"]
    model.dof_armature[vids] = runtime["armature_policy"]
    model.jnt_stiffness[joints] = 0.0
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    data.qpos[:3] = c["base_position"]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qids] = c["default_joint_pos"]
    mj.mj_forward(model, data)
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    history = ActorHistory(c)
    previous = np.zeros(c["action_dim"], dtype=np.float32)
    kp = np.asarray(runtime["kp_policy"])
    kd = np.asarray(runtime["pd_damping_policy"])
    effort = np.asarray(runtime["effort_limit_policy"])
    fields = ["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx_body", "vy_body", "wz", "base_z", "fall"] + c["model_joint_names"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for segment in schedule["segments"]:
            command = np.asarray(segment["command"], dtype=np.float32)
            if command.shape != (3,) or not np.isfinite(command).all() or segment["duration_s"] <= 0:
                raise ValueError("Invalid command schedule")
            for _ in range(round(segment["duration_s"] / c["step_dt"])):
                spatial = np.zeros(6)
                mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, root, spatial, 1)
                rotation = data.xmat[root].reshape(3, 3)
                gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
                observation = history.push({"base_ang_vel": spatial[:3], "projected_gravity": gravity,
                    "velocity_commands": command, "joint_pos_rel": data.qpos[qids] - np.asarray(c["default_joint_pos"]),
                    "joint_vel_rel": data.qvel[vids], "last_action": previous})
                with torch.inference_mode():
                    actions = actor(torch.from_numpy(observation[None, :])).numpy()[0]
                target = joint_targets(actions, c)
                for _ in range(c["decimation"]):
                    data.qfrc_applied[:] = 0.0
                    data.ctrl[:] = 0.0
                    data.qfrc_applied[vids] = np.clip(kp * (target - data.qpos[qids]) - kd * data.qvel[vids], -effort, effort)
                    mj.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Non-finite sim2sim state")
                mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, root, spatial, 1)
                gravity = data.xmat[root].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
                # The original source termination thresholds are frozen in the schedule file.
                fall = bool(data.qpos[2] < schedule["termination"]["minimum_height"]
                            or np.arccos(np.clip(-gravity[2], -1, 1)) > schedule["termination"]["limit_angle"])
                row = dict(zip(fields[:9], [data.time, *command, spatial[3], spatial[4], spatial[2], data.qpos[2], int(fall)]))
                row.update(zip(c["model_joint_names"], data.qpos[qids]))
                writer.writerow(row)
                previous = actions
                if fall:
                    raise RuntimeError(f"F1 fell during replay at t={data.time}; log retained at {args.output}")
    report = {"checkpoint_sha256": identity["checkpoint_sha256"], "xml_sha256": sha256(args.xml),
              "schedule_sha256": sha256(args.schedule), "log_sha256": sha256(args.output),
              "runtime": runtime, "scope": "sim2sim rollout; human quality acceptance still required"}
    Path(str(args.output) + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
