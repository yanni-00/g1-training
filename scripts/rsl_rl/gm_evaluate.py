"""Bounded deterministic playback of an exact F1 checkpoint; never trains PPO."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))


def locate_checkpoint(request):
    # GM mounts the single selected model here; reject any other model identity.
    roots = [ROOT.parent / "checkpoints", Path("/workspace/checkpoints"),
             Path("/workspace/isaaclab/checkpoints"), ROOT / "checkpoints"]
    matches = []
    for directory in roots:
        if directory.is_dir():
            for path in directory.rglob("*.pt"):
                if hashlib.sha256(path.read_bytes()).hexdigest() == request["checkpoint_sha256"]:
                    matches.append(path.resolve())
    matches = sorted(set(matches))
    if not matches:
        raise RuntimeError("Exact registered F1 checkpoint not found in checkpoint mount")
    return matches[0]


def main():
    os.chdir(ROOT)
    from f1_rl_lab.assets_archive import ensure_assets
    ensure_assets()
    launch = json.loads((ROOT / "configs/launch.json").read_text())
    request = json.loads((ROOT / "configs/eval_request.json").read_text())
    if launch["authorization"].get("evaluation") is not True or launch["oma_exp_ref"] != request["exp_id"]:
        raise RuntimeError("Evaluation must match the authorized registered experiment")
    checkpoint = locate_checkpoint(request)
    from f1_rl_lab.checkpoint import verify_checkpoint
    identity = verify_checkpoint(checkpoint)
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["iter"] == request["checkpoint_iteration"]
    del saved
    import argparse
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--enable_cameras"])
    app = AppLauncher(args).app
    try:
        evaluate(request, checkpoint, identity)
    finally:
        app.close()


def evaluate(request, checkpoint, identity):
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    import f1_rl_lab.tasks  # noqa: F401
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
    from f1_rl_lab.contract import load_contract
    from f1_rl_lab.interface import ActorHistory, joint_targets
    from f1_rl_lab.policy import load_actor
    from f1_rl_lab.utils.parser_cfg import parse_env_cfg
    from f1_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

    contract = load_contract()
    schedule = json.loads((ROOT / "configs/eval_schedule.json").read_text())
    commands = []
    for segment in schedule["segments"]:
        commands.extend([segment["command"]] * round(segment["duration_s"] / contract["step_dt"]))
    assert len(commands) == request["max_steps"]
    cfg = parse_env_cfg(request["task"], num_envs=1)
    cfg.seed = request["seed"]
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    horizon = (len(commands) + 1) * contract["step_dt"]
    cfg.episode_length_s = horizon
    cfg.commands.base_velocity.resampling_time_range = (horizon, horizon)
    cfg.viewer.resolution = tuple(request["video_resolution"])
    output = ROOT / "logs/evaluation" / request["exp_id"]
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make(request["task"], cfg=cfg, render_mode="rgb_array")
    raw = env.unwrapped
    runtime = export_deploy_cfg(raw, output)
    assert runtime["binding"] == identity["runtime"]["binding"]
    actor = load_actor(checkpoint, runtime).to(raw.device)
    robot = raw.scene["robot"]
    sensor = raw.scene["contact_forces"]
    indices = runtime["model_to_engine"]
    feet = {s: robot.body_names.index(contract["joint_properties"][s + "_ankle_roll_joint"]["child"])
            for s in ("left", "right")}
    force_ids = {s: sensor.body_names.index(robot.body_names[i]) for s, i in feet.items()}
    history = ActorHistory(contract)
    previous = np.zeros(contract["action_dim"], dtype=np.float32)
    rows = []
    step = -1
    command = commands[0]
    actions_np = previous.copy()
    terminal_snapshot = None

    def vector(x):
        return x.detach().cpu().numpy()

    def snapshot(fall=False):
        data = robot.data
        roll, pitch, yaw = [float(x[0]) for x in euler_xyz_from_quat(data.root_quat_w)]
        # Isaac euler helper uses [0,2pi); write signed angles for G1 comparison.
        roll, pitch, yaw = [(a + np.pi) % (2*np.pi) - np.pi for a in (roll, pitch, yaw)]
        velocity = vector(quat_apply_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w))[0]
        row = dict(t=(step + 1)*raw.step_dt, cmd_vx=command[0], cmd_vy=command[1], cmd_wz=command[2],
                   base_z=float(data.root_pos_w[0, 2]), roll=roll, pitch=pitch, yaw=yaw,
                   roll_deg=float(np.degrees(roll)), pitch_deg=float(np.degrees(pitch)),
                   vx_yaw=float(velocity[0]), vy_yaw=float(velocity[1]), wz=float(data.root_ang_vel_w[0, 2]),
                   fall=int(fall))
        for side, idx in feet.items():
            fz = float(sensor.data.net_forces_w[0, force_ids[side], 2])
            row[side + "_fz"] = fz
            row[side + "_foot_z"] = float(data.body_pos_w[0, idx, 2])
        targets = joint_targets(actions_np, contract)
        for i, name in enumerate(contract["model_joint_names"]):
            j = indices[i]
            row[name] = float(data.joint_pos[0, j])
            row[name + "_velocity"] = float(data.joint_vel[0, j])
            row[name + "_desired"] = float(targets[i])
            row[name + "_torque"] = float(data.applied_torque[0, j])
        return row

    original_reset = raw._reset_idx

    def capture_before_reset(env_ids):
        nonlocal terminal_snapshot
        if step >= 0 and len(env_ids):
            terminal_snapshot = snapshot(fall=bool(raw.reset_terminated[0]))
        return original_reset(env_ids)

    raw._reset_idx = capture_before_reset
    env = gym.wrappers.RecordVideo(env, video_folder=str(output / "videos"),
        step_trigger=lambda s: s == 0, video_length=len(commands), disable_logger=True)
    try:
        with torch.inference_mode():
            env.reset()
            for step, command in enumerate(commands):
                term = raw.command_manager.get_term("base_velocity")
                term.vel_command_b[:] = torch.tensor([command], device=raw.device)
                term.is_standing_env[:] = False
                data = robot.data
                terms = {"base_ang_vel": vector(data.root_ang_vel_b)[0],
                         "projected_gravity": vector(data.projected_gravity_b)[0],
                         "velocity_commands": command,
                         "joint_pos_rel": vector(data.joint_pos-data.default_joint_pos)[0, indices],
                         "joint_vel_rel": vector(data.joint_vel-data.default_joint_vel)[0, indices],
                         "last_action": previous}
                observation = history.push(terms)
                actions = actor(torch.tensor(observation[None], device=raw.device))
                assert torch.isfinite(actions).all(), "Non-finite deterministic action"
                actions_np = vector(actions)[0]
                position = vector(data.root_pos_w)[0]
                raw.sim.set_camera_view(eye=position+np.array(request["camera_offset"]), target=position)
                _, _, terminated, truncated, _ = env.step(actions)
                rows.append(terminal_snapshot if terminal_snapshot is not None else snapshot())
                previous = actions_np.copy()
                if bool(terminated[0]) or bool(truncated[0]):
                    break
                if (step+1) % 100 == 0:
                    print(f"F1_EVAL_STEP {step+1}/{len(commands)}", flush=True)
    finally:
        env.close()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    (output / "trajectory.csv").write_bytes(csv_bytes)
    videos = sorted((output / "videos").glob("*.mp4"))
    if not videos:
        raise RuntimeError("Playback produced no video")
    metadata = {"format": "f1_evaluation_bundle_v1", "exp_id": request["exp_id"],
        "source_checkpoint_sha256": request["checkpoint_sha256"],
        "source_checkpoint_iteration": request["checkpoint_iteration"],
        "requested_steps": len(commands), "completed_steps": len(rows),
        "fall": bool(rows[-1]["fall"]), "runtime": runtime, "protocol": schedule,
        "evaluation_overrides": request["evaluation_overrides"],
        "delivery_note": "Evaluation data container only; model_0.pt is NOT new trained weights"}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # GM transports PT artifacts. Preserve exact CSV/video bytes in a plainly tagged packet.
    torch.save({**metadata, "files": {"trajectory.csv": csv_bytes,
                "playback.mp4": videos[0].read_bytes()}}, output / "model_0.pt")
    print("F1_EVALUATION_DELIVERED " + json.dumps({k: metadata[k] for k in
        ("exp_id", "completed_steps", "requested_steps", "fall")}), flush=True)


if __name__ == "__main__":
    main()
