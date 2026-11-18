"""Native continuous F1 command replay. No auto-reset may count as recovery."""
import argparse
import csv
import json
import math
import os
import subprocess
import hashlib
import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1] / "training"
sys.path[:0] = [str(ROOT / "source/f1_beyondmimic"), str(ROOT / "scripts")]


def check_inputs(args):
    from f1_beyondmimic.contract import load_json, sha256, verify_assets, effective_control, prepared_motion
    from f1_beyondmimic.utils.checkpoint import load_checkpoint
    request, protocol = load_json(args.request), load_json(args.protocol)
    config, state = load_json(ROOT.parent / ".oma/config.json"), load_json(ROOT.parent / ".oma/loop.json")
    if (request.get("scope") != "fixed_weight_simulation_diagnostic"
            or request.get("authorization", {}).get("simulation_replay") is not True
            or request["authorization"].get("optimizer_updates") is not False
            or request["authorization"].get("real_robot") is not False
            or config.get("authorization", {}).get("evaluation") is not True
            or set(state.get("pausedScopes", [])) & {"loop", "evaluation"}):
        raise ValueError("Fixed-weight simulation replay is not authorized")
    if sha256(args.protocol) != request["protocol_sha256"]:
        raise ValueError("Frozen replay protocol changed")
    if sha256(Path(__file__)) != request["entry_sha256"]:
        raise ValueError("Replay entry changed after review")
    identity = next((c for c in request["checkpoints"] if c["sha256"] == sha256(args.checkpoint)), None)
    if identity is None or args.checkpoint.stat().st_size != identity["bytes"]:
        raise ValueError("Checkpoint is not a frozen source-task artifact")
    verify_assets(); effective_control(); prepared_motion()
    saved = load_checkpoint(args.checkpoint)  # exact original source/data/control binding
    if int(saved["iter"]) != identity["iteration"]:
        raise ValueError("Checkpoint counter mismatch")
    runtime = saved["infos"]["f1_runtime"]
    for stage in protocol["schedule"]:
        for value, (lower, upper) in zip(stage["command"], runtime["command_limits"]):
            if not lower <= value <= upper:
                raise ValueError("Command outside this checkpoint runtime limits")
    expected_steps = sum(math.ceil(s["duration_s"] / runtime["policy_dt"]) for s in protocol["schedule"])
    if expected_steps != request["max_policy_steps"]:
        raise ValueError("Protocol exceeds or differs from the registered policy step limit")
    print(json.dumps({"check": "passed", "iteration": saved["iter"], "binding": "exact",
                      "source_task": request["source_task"], "optimizer_updates": 0}), flush=True)
    return request, protocol



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    if "--check" in sys.argv:
        args = parser.parse_args()
        check_inputs(args)
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        return
    preliminary, _ = parser.parse_known_args()
    if preliminary.checkpoint is None or preliminary.output is None:
        parser.error("--checkpoint and --output are required")
    from f1_beyondmimic.assets_archive import ensure_assets
    ensure_assets(ROOT)
    checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check",
        "--checkpoint", str(preliminary.checkpoint), "--request", str(preliminary.request),
        "--protocol", str(preliminary.protocol)], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    checked.check_returncode()
    protocol = json.loads(preliminary.protocol.read_text(encoding="utf-8"))
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
        from f1_beyondmimic.contract import load_contract, load_json, sha256
        from f1_beyondmimic.utils.checkpoint import load_checkpoint, audit_native_contact
        checkpoint = load_checkpoint(args.checkpoint)
        request = load_json(args.request)
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab.utils.math import quat_apply, quat_inv
        from f1_beyondmimic.utils.runtime_policy import RuntimeActor
        from f1_beyondmimic.utils.actor_observations import audit_native_observations
        task = "Tracking-VelocityReference-F1-Proprio-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        cfg.scene.num_envs = 1
        cfg.seed = protocol["seed"]
        cfg.viewer.resolution = tuple(protocol["render"]["resolution"])
        cfg.sim.device = args.device
        cfg.episode_length_s = sum(s["duration_s"] for s in schedule) + cfg.sim.dt * cfg.decimation
        cfg.observations.policy.enable_corruption = False
        cfg.commands.motion.evaluation_start_support = protocol["initial_support_index"]
        cfg.commands.motion.initial_go_probability = 0.0
        for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            setattr(cfg.events, name, None)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        # F1Environment adds a training/deployment admission guard only. This
        # adapter uses the same native base and unchanged F1 cfg after the
        # explicit fixed-weight simulation authorization and binding checks.
        env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
        video = None
        if args.video:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(output / "replay.mp4"),
                fps=1.0 / (env.step_dt * protocol["render"]["stride"]), codec="libx264")
        obs, _ = env.reset()
        base = env.unwrapped
        observation_audit = audit_native_observations(env)
        contact_audit = audit_native_contact(env)
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
        for key, live in {
            "kp": robot.data.default_joint_stiffness[0],
            "kd_total": robot.data.default_joint_damping[0],
            "armature": robot.data.default_joint_armature[0],
            "joint_lower": robot.data.joint_pos_limits[0, :, 0],
            "joint_upper": robot.data.joint_pos_limits[0, :, 1],
        }.items():
            torch.testing.assert_close(live.cpu(), runtime["control"][key])
        torch.testing.assert_close(torch.tensor(command.command_limits), torch.tensor(runtime["command_limits"]))
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
                    if len(rows) >= request["max_policy_steps"]:
                        raise RuntimeError("Registered replay step limit reached")
                    command.request(stage["command"], walk_enabled=stage.get("walk_enabled", True))
                    raw = actor(obs["policy"])
                    if not torch.isfinite(raw).all():
                        raise RuntimeError("Nonfinite inference action")
                    if video is not None and len(rows) % protocol["render"]["stride"] == 0:
                        target = robot.data.root_pos_w[0].detach().cpu().numpy()
                        eye = target + np.asarray(protocol["render"]["eye_offset"])
                        base.sim.set_camera_view(eye=eye, target=target)
                        video.append_data(env.render())
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
                    row.update({"root_height_m": float(robot.data.root_pos_w[0, 2]), "root_tilt_rad": math.acos(max(-1.0, min(1.0, 1 - 2*(x*x+y*y)))), "root_qw": w, "root_qx": x, "root_qy": y, "root_qz": z})
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
        if video is not None:
            video.close()
            video = None
        with (output / "states.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"checkpoint": str(args.checkpoint.resolve()), "iteration": int(checkpoint["iter"]),
                   "checkpoint_sha256": sha256(args.checkpoint), "source_task": request["source_task"],
                   "binding_matches": True, "optimizer_updates": 0,
                   "native_observation_audit": observation_audit, "native_contact_audit": contact_audit, "elapsed_s": elapsed,
                   "terminated_early": terminated_early, "behavior_acceptance": False,
                   "quality_scope": "foot body-speed is a proxy; contact-surface slip and video need separate review",
                   "protocol": protocol, "stages": {}}
        for name in dict.fromkeys(row["stage"] for row in rows):
            subset = [row for row in rows if row["stage"] == name]
            summary["stages"][name] = {axis + "_rmse": float(np.sqrt(np.mean([
                (row["actual_"+axis] - row["command_"+axis]) ** 2 for row in subset])))
                for axis in ("vx", "vy", "yaw_rate")}
        for name, metrics in summary["stages"].items():
            subset = [r for r in rows if r["stage"] == name]
            metrics.update(samples=len(subset), duration_s=len(subset)*base.step_dt,
                           mean_vx=float(np.mean([r["actual_vx"] for r in subset])),
                           min_root_height_m=min(r["root_height_m"] for r in subset),
                           max_root_tilt_rad=max(r["root_tilt_rad"] for r in subset))
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.bundle:
            args.bundle.parent.mkdir(parents=True, exist_ok=True)
            files = {}
            for path in output.rglob("*"):
                if path.is_file():
                    content = path.read_bytes()
                    files[path.relative_to(output).as_posix()] = {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "content": torch.frombuffer(bytearray(content), dtype=torch.uint8).clone()}
            torch.save({"format": "f1_beyondmimic_native_diagnostic_v1", "summary": summary,
                        "files": files}, args.bundle)
            print("[F1 replay] diagnostic bundle saved; zero optimizer updates", flush=True)
            time.sleep(120)  # Existing G1 SDK upload grace setting.
    finally:
        if locals().get("video") is not None:
            video.close()
        if env is not None:
            env.close()
        app.app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        time.sleep(15)
        raise
