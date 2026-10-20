#!/usr/bin/env python
"""Sim2Sim play: IsaacLab-trained G1-29dof velocity policy rolled out in MuJoCo.

Server: tare-codex (sandbox ~/F1_train/yanni/play_20k/), env: ~/Anaconda/envs/gmr
Policy: model_19998.pt from TASK_20261019_093 (unitree_rl_lab, IsaacLab 2.3.2)

Joint order + obs layout verified against the OFFICIAL deploy reference
(unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml
+ deploy/include/isaaclab/manager/{observation_manager.h, manager_term_cfg.h}):
  - joint_ids_map: [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
    (SDK joint k -> PhysX articulation index; policy uses ARTICULATION order)
  - obs is TERM-MAJOR: [ang_vel*0.2 x5, grav x5, cmd x5, qpos_rel x5, qvel_rel*0.05 x5, last_action x5]
    each term's 5 frames oldest->newest
  - history init: current obs repeated 5x (deploy reset()), NOT zeros
PD gains / effort limits / default joint pos from training commit 59166d6
(assets/robots/unitree.py UNITREE_G1_29DOF_CFG, velocity_env_cfg.py action scale 0.25).
Actor: MLP 480->512->256->128->29, ELU, empirical_normalization=False (raw obs).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import mujoco

# ---------------- SDK joint order (unitree.py joint_sdk_names) ----------------
SDK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# official deploy.yaml: SDK joint k -> articulation index
JOINT_IDS_MAP = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
                 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
# official deploy.yaml joint_ids_map + deploy C++ usage
# (unitree_articulation.h: joint_pos[i] = motor_state[joint_ids_map[i]]):
#   joint_ids_map[i] = SDK index of articulation joint i  (NOT the inverse)
# This decodes to PhysX BFS-by-kinematic-depth order, children in file order
# (L before R). Cross-verified: official deploy.yaml default_joint_pos in this
# order reproduces the training cfg init values exactly on all 29 joints.
POLICY_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
_sdk_pos = [SDK_JOINTS.index(n) for n in POLICY_JOINTS]
assert _sdk_pos == JOINT_IDS_MAP, f"map decode mismatch: {_sdk_pos}"

# per-joint (default_pos, kp, kd, eff) from training cfg UNITREE_G1_29DOF_CFG
JOINT_PARAMS = {
    "left_hip_pitch_joint": (-0.1, 100, 2.0, 88), "right_hip_pitch_joint": (-0.1, 100, 2.0, 88),
    "left_hip_roll_joint": (0.0, 100, 2.0, 139), "right_hip_roll_joint": (0.0, 100, 2.0, 139),
    "left_hip_yaw_joint": (0.0, 100, 2.0, 88), "right_hip_yaw_joint": (0.0, 100, 2.0, 88),
    "left_knee_joint": (0.3, 150, 4.0, 139), "right_knee_joint": (0.3, 150, 4.0, 139),
    "left_ankle_pitch_joint": (-0.2, 40, 2.0, 25), "right_ankle_pitch_joint": (-0.2, 40, 2.0, 25),
    "left_ankle_roll_joint": (0.0, 40, 2.0, 25), "right_ankle_roll_joint": (0.0, 40, 2.0, 25),
    "waist_yaw_joint": (0.0, 200, 5.0, 88),
    "waist_roll_joint": (0.0, 40, 5.0, 25), "waist_pitch_joint": (0.0, 40, 5.0, 25),
    "left_shoulder_pitch_joint": (0.3, 40, 1.0, 25), "right_shoulder_pitch_joint": (0.3, 40, 1.0, 25),
    "left_shoulder_roll_joint": (0.25, 40, 1.0, 25), "right_shoulder_roll_joint": (-0.25, 40, 1.0, 25),
    "left_shoulder_yaw_joint": (0.0, 40, 1.0, 25), "right_shoulder_yaw_joint": (0.0, 40, 1.0, 25),
    "left_elbow_joint": (0.97, 40, 1.0, 25), "right_elbow_joint": (0.97, 40, 1.0, 25),
    "left_wrist_roll_joint": (0.15, 40, 1.0, 25), "right_wrist_roll_joint": (-0.15, 40, 1.0, 25),
    "left_wrist_pitch_joint": (0.0, 40, 1.0, 5), "right_wrist_pitch_joint": (0.0, 40, 1.0, 5),
    "left_wrist_yaw_joint": (0.0, 40, 1.0, 5), "right_wrist_yaw_joint": (0.0, 40, 1.0, 5),
}
DEFAULT_POS = np.array([JOINT_PARAMS[n][0] for n in POLICY_JOINTS])
KP = np.array([JOINT_PARAMS[n][1] for n in POLICY_JOINTS], dtype=np.float64)
KD = np.array([JOINT_PARAMS[n][2] for n in POLICY_JOINTS], dtype=np.float64)
EFF = np.array([JOINT_PARAMS[n][3] for n in POLICY_JOINTS], dtype=np.float64)

ACTION_SCALE = 0.25
DECIMATION = 4
PHYS_DT = 0.005
POLICY_DT = PHYS_DT * DECIMATION   # 0.02 s -> 50 Hz
HIST = 5
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
INIT_POS_Z = 0.8
ARMATURE = 0.01

# play command segments (play cfg uses limit_ranges: vx [-0.5,1.0], vy [-0.3,0.3], wz [-0.2,0.2])
CMD_SEGMENTS = [
    (4.0, (0.0, 0.0, 0.0), "stand"),
    (8.0, (0.4, 0.0, 0.0), "fwd 0.4"),
    (8.0, (0.6, 0.0, 0.0), "fwd 0.6 (curriculum max)"),
    (6.0, (0.8, 0.0, 0.0), "fwd 0.8 (beyond curriculum)"),
    (6.0, (0.0, 0.3, 0.0), "lateral 0.3"),
    (6.0, (0.4, 0.0, 0.2), "curve vx0.4 wz0.2"),
    (4.0, (0.0, 0.0, 0.0), "stand-end"),
]


def quat_to_rot(q):  # q = (w, x, y, z)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def build_actor(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(480, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 29),
    )
    actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    actor.eval()
    return actor


class TermHistory:
    """Per-term circular buffer, oldest->newest, deploy-style reset(repeat)."""

    def __init__(self, dim, hist=HIST):
        self.hist = hist
        self.buf = np.zeros((hist, dim))
        self.filled = False

    def reset(self, frame):
        self.buf[:] = frame
        self.filled = True

    def push(self, frame):
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = frame

    def flat(self):
        return self.buf.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="g1_description/g1_29dof_rev_1_0.xml")
    ap.add_argument("--ckpt", default="model_19998.pt")
    ap.add_argument("--out", default="play_g1_20k")
    args = ap.parse_args()

    actor = build_actor(args.ckpt)

    model = mujoco.MjModel.from_xml_path(args.xml)
    if abs(model.opt.timestep - PHYS_DT) > 1e-9:
        model.opt.timestep = PHYS_DT
    for j in range(6, 6 + 29):
        model.dof_armature[j] = ARMATURE
    # clean sensors (play = clean obs)
    if model.nsensor:
        model.sensor_noise[:] = 0.0
        model.sensor_cutoff[:] = 0.0
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)

    # policy-ordered index maps into MuJoCo
    qadr = np.zeros(29, dtype=int)
    vadr = np.zeros(29, dtype=int)
    cadr = np.zeros(29, dtype=int)
    jlo = np.zeros(29)
    jhi = np.zeros(29)
    for i, name in enumerate(POLICY_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr[i] = model.jnt_qposadr[jid]
        vadr[i] = model.jnt_dofadr[jid]
        jlo[i], jhi[i] = model.jnt_range[jid]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator for {name} not found"
        cadr[i] = aid
    # pelvis gyro sensor (by name)
    gyro_adr = -1
    for s in range(model.nsensor):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)
        if nm and "pelvis-angular-velocity" in nm:
            gyro_adr = model.sensor_adr[s]
            break
    assert gyro_adr >= 0, "gyro sensor not found"

    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.8]
    cam.distance = 3.2
    cam.azimuth = 130.0
    cam.elevation = -12.0

    # initial state
    data.qpos[:] = 0.0
    data.qpos[2] = INIT_POS_Z
    data.qpos[3] = 1.0
    data.qpos[qadr] = DEFAULT_POS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # term histories (term-major layout), deploy-style: reset = repeat current obs
    h_ang = TermHistory(3)
    h_grav = TermHistory(3)
    h_cmd = TermHistory(3)
    h_qpos = TermHistory(29)
    h_qvel = TermHistory(29)
    h_la = TermHistory(29)

    def obs_terms(cmd):
        quat = data.qpos[3:7].copy()
        R = quat_to_rot(quat)
        ang = data.sensordata[gyro_adr:gyro_adr + 3].copy()
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        q = data.qpos[qadr].copy()
        qd = data.qvel[vadr].copy()
        return (ANG_VEL_SCALE * ang, grav, cmd.copy(), q - DEFAULT_POS,
                JOINT_VEL_SCALE * qd, None)

    def push_all(cmd, last_action):
        ang, grav, c, dq, dqd, _ = obs_terms(cmd)
        h_ang.push(ang); h_grav.push(grav); h_cmd.push(c)
        h_qpos.push(dq); h_qvel.push(dqd); h_la.push(last_action)

    # init histories with the standing obs (last_action = zeros)
    ang, grav, c, dq, dqd, _ = obs_terms(np.zeros(3))
    h_ang.reset(ang); h_grav.reset(grav); h_cmd.reset(np.zeros(3))
    h_qpos.reset(dq); h_qvel.reset(dqd); h_la.reset(np.zeros(29))

    import imageio.v2 as imageio
    frames = []
    rows = []
    seg_stats = []

    total_t = sum(s[0] for s in CMD_SEGMENTS)
    n_steps = int(round(total_t / POLICY_DT))
    print(f"[play] total sim time {total_t:.0f}s, {n_steps} policy steps @ {1/POLICY_DT:.0f}Hz")

    t = 0.0
    seg_i = 0
    seg_t0 = 0.0
    seg_end = CMD_SEGMENTS[0][0]
    cmd = np.array(CMD_SEGMENTS[0][1])
    seg_vx, seg_vy, seg_wz, seg_n = [], [], [], 0
    fallen = False
    fall_t = None
    t0_wall = time.time()

    for step in range(n_steps):
        if t >= seg_end - 1e-9:
            if seg_n:
                seg_stats.append((CMD_SEGMENTS[seg_i][2], cmd.copy(),
                                  np.mean(seg_vx), np.mean(seg_vy), np.mean(seg_wz), seg_n, fallen))
            seg_i += 1
            if seg_i >= len(CMD_SEGMENTS):
                break
            seg_t0 = seg_end
            seg_end += CMD_SEGMENTS[seg_i][0]
            cmd = np.array(CMD_SEGMENTS[seg_i][1])
            seg_vx, seg_vy, seg_wz, seg_n = [], [], [], 0

        # ---- obs: term-major concat ----
        obs = np.concatenate([h_ang.flat(), h_grav.flat(), h_cmd.flat(),
                              h_qpos.flat(), h_qvel.flat(), h_la.flat()])
        with torch.no_grad():
            action = actor(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)).numpy()[0]

        # ---- action -> joint targets (policy order == array order here) ----
        q_des = np.clip(DEFAULT_POS + ACTION_SCALE * action, jlo, jhi)

        # ---- physics: decimation substeps, explicit PD with effort clip ----
        for _ in range(DECIMATION):
            tau = np.clip(KP * (q_des - data.qpos[qadr]) - KD * data.qvel[vadr], -EFF, EFF)
            data.ctrl[cadr] = tau
            mujoco.mj_step(model, data)
        t += POLICY_DT

        # ---- push new obs into history ----
        push_all(cmd, action)

        # ---- metrics ----
        roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
        v_world = data.qvel[0:3].copy()
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_yaw = cy * v_world[0] + sy * v_world[1]
        vy_yaw = -sy * v_world[0] + cy * v_world[1]
        wz = data.sensordata[gyro_adr:gyro_adr + 3][2]
        base_z = data.qpos[2]
        grav_now = h_grav.buf[-1]
        tilt_cos = grav_now[2]
        is_fall = (base_z < 0.2) or (tilt_cos > -np.cos(0.8))
        if is_fall and not fallen:
            fallen = True
            fall_t = t
            print(f"[play] FALL detected at t={t:.2f}s (z={base_z:.3f}, tilt_cos={tilt_cos:.3f})")
        rows.append([f"{t:.3f}", cmd[0], cmd[1], cmd[2],
                     f"{vx_yaw:.4f}", f"{vy_yaw:.4f}", f"{wz:.4f}",
                     f"{base_z:.4f}", f"{roll:.4f}", f"{pitch:.4f}", f"{yaw:.4f}",
                     f"{tilt_cos:.4f}", int(is_fall)])

        if t - seg_t0 > 0.5:
            seg_vx.append(vx_yaw)
            seg_vy.append(vy_yaw)
            seg_wz.append(wz)
            seg_n += 1

        # ---- render (follow camera) ----
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = 0.8
        renderer.update_scene(data, cam)
        pix = renderer.render()
        frames.append(pix.copy())

        if step % 250 == 0:
            print(f"[play] t={t:5.1f}s  z={base_z:.3f}  vx={vx_yaw:+.2f} (cmd {cmd[0]:+.1f})  "
                  f"|a|max={np.abs(action).max():.2f}  wall={time.time()-t0_wall:.0f}s")

    if seg_n:
        seg_stats.append((CMD_SEGMENTS[seg_i][2], cmd.copy(),
                          np.mean(seg_vx), np.mean(seg_vy), np.mean(seg_wz), seg_n, fallen))

    # ---- outputs ----
    fps = int(round(1 / POLICY_DT))
    mp4_path = f"{args.out}.mp4"
    imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=None)
    print(f"[play] video: {mp4_path} ({len(frames)} frames, {fps}fps, "
          f"{os.path.getsize(mp4_path)/1e6:.1f}MB)")

    csv_path = f"{args.out}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx_yaw", "vy_yaw", "wz",
                    "base_z", "roll", "pitch", "yaw", "tilt_cos_up", "fall"])
        w.writerows(rows)
    print(f"[play] metrics: {csv_path} ({len(rows)} rows)")

    print("\n===== segment summary (cmd -> actual, yaw-frame) =====")
    for name, c, vx, vy, wz, n, fl in seg_stats:
        print(f"{name:32s} cmd=({c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f})  "
              f"act=({vx:+.2f},{vy:+.2f},{wz:+.2f})  n={n}{'  FALLEN' if fl else ''}")
    if fall_t is not None:
        print(f"FALL at t={fall_t:.2f}s")
    else:
        print("NO FALL for entire rollout")
    print(f"wall time: {time.time()-t0_wall:.1f}s")


if __name__ == "__main__":
    main()
