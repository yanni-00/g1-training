"""Native continuous F1 command replay. No auto-reset may count as recovery."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from _bootstrap import ROOT
from f1_beyondmimic.contract import assert_can_run, load_json, load_contract, readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    if "--check" in sys.argv:
        parser.parse_args()
        print(json.dumps({"inputs": readiness(), "protocol": load_json(ROOT / "configs/evaluation.json")},
                         ensure_ascii=False, indent=2))
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        return
    assert_can_run("evaluation")
    protocol = load_json(ROOT / "configs/evaluation.json")
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Freeze actual F1 evaluation stages/durations/commands before replay")
    for stage in schedule:
        command = stage.get("command")
        if (not isinstance(stage.get("name"), str) or not isinstance(stage.get("duration_s"), (int, float))
                or not math.isfinite(stage["duration_s"]) or stage["duration_s"] <= 0
                or not isinstance(command, list) or len(command) != 3
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in command)):
            raise ValueError("Invalid evaluation stage")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None or args.output is None:
        parser.error("--checkpoint and --output are required for replay")
    from f1_beyondmimic.utils.checkpoint import load_checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    os.environ["F1_BM_EXECUTION_SCOPE"] = "evaluation"
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import f1_beyondmimic.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab.utils.math import quat_apply, quat_inv
        from f1_beyondmimic.utils.runtime_policy import RuntimeActor
        from f1_beyondmimic.utils.actor_observations import audit_native_observations
        task = "Tracking-VelocityReference-F1-Proprio-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.episode_length_s = sum(s["duration_s"] for s in schedule) + cfg.sim.dt * cfg.decimation
        cfg.observations.policy.enable_corruption = False
        cfg.commands.motion.evaluation_start_support = 0
        cfg.commands.motion.initial_go_probability = 0.0
        for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            setattr(cfg.events, name, None)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = gym.wrappers.RecordVideo(env, video_folder=str(output / "video"),
                                          step_trigger=lambda step: step == 0,
                                          video_length=math.ceil(cfg.episode_length_s / env.unwrapped.step_dt))
        obs, _ = env.reset()
        base = env.unwrapped
        audit_native_observations(env)
        command = base.command_manager.get_term("motion")
        robot = command.robot
        # Source G1 crosscheck also initializes derived STOP targets without an
        # unrecorded physics step. Prove the reference/physical state stays fixed.
        command.request([0.0, 0.0, 0.0], walk_enabled=True)
        initial = [v.clone() for v in (command.clock.phase, command.clock.rate,
                                       robot.data.root_state_w, robot.data.joint_pos)]
        command._update_command()
        for before, after in zip(initial, (command.clock.phase, command.clock.rate,
                                           robot.data.root_state_w, robot.data.joint_pos)):
            if not torch.equal(before, after):
                raise RuntimeError("Derived STOP initialization changed reference or physical state")
        obs = base.observation_manager.compute(update_history=False)
        runtime = checkpoint["infos"]["f1_runtime"]
        if list(robot.joint_names) != runtime["joint_names"] or list(robot.body_names) != runtime["body_names"]:
            raise ValueError("Evaluation native joint/body order differs from the checkpoint")
        if not math.isclose(base.step_dt, runtime["policy_dt"]):
            raise ValueError("Evaluation timing differs from the checkpoint")
        action_term = base.action_manager.get_term("joint_pos")
        native_scale = torch.as_tensor(action_term._scale).flatten().cpu()
        torch.testing.assert_close(native_scale, runtime["control"]["action_scale"].flatten())
        torch.testing.assert_close(robot.data.default_joint_pos[0].cpu(), runtime["control"]["default_joint_pos"])
        actor = RuntimeActor(checkpoint).to(base.device).eval()
        sensor = base.scene["contact_forces"]
        c = load_contract()
        feet = c["feet"]
        body_ids = [robot.body_names.index(name) for name in feet]
        sensor_ids = [sensor.body_names.index(name) for name in feet]
        rows, elapsed, terminated_early = [], 0.0, False
        previous_yaw, actual_yaw = None, 0.0
        with torch.inference_mode():
            for stage in schedule:
                steps = math.ceil(stage["duration_s"] / base.step_dt)
                for _ in range(steps):
                    command.request(stage["command"], walk_enabled=stage.get("walk_enabled", True))
                    raw = actor(obs["policy"])
                    velocity = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)[0]
                    w, x, y, z = robot.data.root_quat_w[0].tolist()
                    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
                    actual_yaw = yaw if previous_yaw is None else actual_yaw + math.atan2(math.sin(yaw-previous_yaw), math.cos(yaw-previous_yaw))
                    previous_yaw = yaw
                    row = {"time_s": elapsed, "stage": stage["name"], "walk_enabled": stage.get("walk_enabled", True),
                           "command_vx": stage["command"][0], "command_vy": stage["command"][1],
                           "command_yaw_rate": stage["command"][2], "actual_vx": float(velocity[0]),
                           "actual_vy": float(velocity[1]), "actual_yaw_rate": float(robot.data.root_ang_vel_w[0, 2]),
                           "target_yaw": float(command.reference.root.yaw[0]), "actual_yaw": actual_yaw,
                           "phase": float(command.clock.phase[0]), "cycle": int(command.clock.cycles[0]),
                           "reference_rate": float(command.clock.rate[0]), "reference_source": runtime["reference_sha256"]}
                    for axis, value in zip(("vx", "vy", "yaw_rate"), command.reference.effective_command[0].tolist()):
                        row["reference_effective_" + axis] = value
                    row.update({"root_qw": w, "root_qx": x, "root_qy": y, "root_qz": z})
                    for group, values in [("q", robot.data.joint_pos[0]), ("dq", robot.data.joint_vel[0]),
                                          ("reference_q", command.joint_pos[0]), ("raw_action", raw[0])]:
                        row.update({group + ":" + name: float(value) for name, value in zip(robot.joint_names, values)})
                    for name, body_id, sensor_id in zip(feet, body_ids, sensor_ids):
                        force = sensor.data.net_forces_w[0, sensor_id]
                        row[name + ":contact_force_N"] = float(force.norm())
                        row[name + ":contact"] = bool(force.norm() > sensor.cfg.force_threshold)
                        row[name + ":body_planar_speed_proxy"] = float(robot.data.body_lin_vel_w[0, body_id, :2].norm())
                        row[name + ":body_height"] = float(robot.data.body_pos_w[0, body_id, 2])
                    obs, _, terminated, truncated, extras = env.step(raw)
                    row["step_done"] = bool(terminated[0] or truncated[0])
                    row["termination_reason"] = json.dumps({key: float(value) for key, value in
                        extras.get("log", {}).items() if key.startswith("Episode_Termination/")}) if row["step_done"] else ""
                    # Values above are pre-step state; flags describe that same step.
                    rows.append(row)
                    elapsed += base.step_dt
                    if row["step_done"]:
                        terminated_early = True
                        break
                if terminated_early:
                    break
        with (output / "states.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"checkpoint": str(args.checkpoint.resolve()), "elapsed_s": elapsed,
                   "terminated_early": terminated_early, "behavior_acceptance": False,
                   "quality_scope": "foot body-speed is a proxy; contact-surface slip and video need separate review",
                   "protocol": protocol, "stages": {}}
        for name in dict.fromkeys(row["stage"] for row in rows):
            subset = [row for row in rows if row["stage"] == name]
            summary["stages"][name] = {axis + "_rmse": float(np.sqrt(np.mean([
                (row["actual_"+axis] - row["command_"+axis]) ** 2 for row in subset])))
                for axis in ("vx", "vy", "yaw_rate")}
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        if env is not None:
            env.close()
        app.app.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
