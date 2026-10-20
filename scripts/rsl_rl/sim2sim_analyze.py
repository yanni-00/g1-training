#!/usr/bin/env python
"""Deep-dive analysis of the play rollout: arm asymmetry + stride asymmetry.
Replicates sim2sim_play.py's exact rollout (same commands, same init) while
logging per-step joint positions and foot world positions."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
import mujoco
from sim2sim_play import (POLICY_JOINTS, DEFAULT_POS, KP, KD, EFF, ACTION_SCALE,
                          DECIMATION, PHYS_DT, HIST, ANG_VEL_SCALE, JOINT_VEL_SCALE,
                          INIT_POS_Z, ARMATURE, CMD_SEGMENTS, quat_to_rot, quat_to_rpy,
                          TermHistory, build_actor)

actor = build_actor("model_19998.pt")
model = mujoco.MjModel.from_xml_path("g1_description/g1_29dof_rev_1_0.xml")
model.opt.timestep = PHYS_DT
for j in range(6, 6 + 29):
    model.dof_armature[j] = ARMATURE
model.sensor_noise[:] = 0.0
model.sensor_cutoff[:] = 0.0
data = mujoco.MjData(model)

qadr = np.zeros(29, dtype=int)
vadr = np.zeros(29, dtype=int)
cadr = np.zeros(29, dtype=int)
jlo, jhi = np.zeros(29), np.zeros(29)
for i, name in enumerate(POLICY_JOINTS):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr[i] = model.jnt_qposadr[jid]
    vadr[i] = model.jnt_dofadr[jid]
    jlo[i], jhi[i] = model.jnt_range[jid]
    cadr[i] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
gyro_adr = -1
for s in range(model.nsensor):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)
    if nm and "pelvis-angular-velocity" in nm:
        gyro_adr = model.sensor_adr[s]
foot_bid = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f + "_ankle_roll_link")
            for f in ("left", "right")}

data.qpos[:] = 0.0
data.qpos[2] = INIT_POS_Z
data.qpos[3] = 1.0
data.qpos[qadr] = DEFAULT_POS
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

h_ang, h_grav, h_cmd = TermHistory(3), TermHistory(3), TermHistory(3)
h_qpos, h_qvel, h_la = TermHistory(29), TermHistory(29), TermHistory(29)

def obs_terms(cmd):
    quat = data.qpos[3:7].copy()
    R = quat_to_rot(quat)
    ang = data.sensordata[gyro_adr:gyro_adr + 3].copy()
    grav = R.T @ np.array([0.0, 0.0, -1.0])
    q = data.qpos[qadr].copy()
    qd = data.qvel[vadr].copy()
    return (ANG_VEL_SCALE * ang, grav, cmd.copy(), q - DEFAULT_POS,
            JOINT_VEL_SCALE * qd, None)

ang, grav, c, dq, dqd, _ = obs_terms(np.zeros(3))
h_ang.reset(ang); h_grav.reset(grav); h_cmd.reset(np.zeros(3))
h_qpos.reset(dq); h_qvel.reset(dqd); h_la.reset(np.zeros(29))

# per-step logs
log_q = []        # joint pos (policy order)
log_foot = []     # (left xyz, right xyz)
log_base = []     # base x,y,yaw
log_t = []

total_t = sum(s[0] for s in CMD_SEGMENTS)
n_steps = int(round(total_t / (PHYS_DT * DECIMATION)))
t = 0.0
seg_i, seg_end = 0, CMD_SEGMENTS[0][0]
cmd = np.array(CMD_SEGMENTS[0][1])
last_action = np.zeros(29)

for step in range(n_steps):
    if t >= seg_end - 1e-9:
        seg_i += 1
        if seg_i >= len(CMD_SEGMENTS):
            break
        seg_end += CMD_SEGMENTS[seg_i][0]
        cmd = np.array(CMD_SEGMENTS[seg_i][1])
    obs = np.concatenate([h_ang.flat(), h_grav.flat(), h_cmd.flat(),
                          h_qpos.flat(), h_qvel.flat(), h_la.flat()])
    with torch.no_grad():
        action = actor(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)).numpy()[0]
    q_des = np.clip(DEFAULT_POS + ACTION_SCALE * action, jlo, jhi)
    for _ in range(DECIMATION):
        tau = np.clip(KP * (q_des - data.qpos[qadr]) - KD * data.qvel[vadr], -EFF, EFF)
        data.ctrl[cadr] = tau
        mujoco.mj_step(model, data)
    t += DECIMATION * PHYS_DT
    # push obs
    ang, grav, c_, dq, dqd, _ = obs_terms(cmd)
    h_ang.push(ang); h_grav.push(grav); h_cmd.push(c_)
    h_qpos.push(dq); h_qvel.push(dqd); h_la.push(action)
    # logs
    log_q.append(data.qpos[qadr].copy())
    log_foot.append((data.xpos[foot_bid["left"]].copy(),
                     data.xpos[foot_bid["right"]].copy()))
    _, _, yaw = quat_to_rpy(data.qpos[3:7])
    log_base.append((data.qpos[0], data.qpos[1], yaw))
    log_t.append(t)

log_q = np.array(log_q)              # (N, 29) policy order
log_t = np.array(log_t)
log_base = np.array(log_base)
name_idx = {n: i for i, n in enumerate(POLICY_JOINTS)}

print("=========== 1. ARM JOINT POSITIONS (mean +/- std over fwd walking t=4-20s) ===========")
m_fwd = (log_t >= 4.0) & (log_t < 20.0)
m_stand = log_t < 4.0
ARM = [("shoulder_pitch", 0.3), ("shoulder_roll", 0.25), ("shoulder_yaw", 0.0),
       ("elbow", 0.97), ("wrist_roll", 0.15)]
print(f"{'joint':28s} {'default':>8s} {'LEFT mean+/-std (walk)':>26s} {'RIGHT mean+/-std (walk)':>26s}")
for jn, dflt in ARM:
    lname = f"left_{jn}_joint"
    rname = f"right_{jn}_joint"
    # right default signs mirrored for roll/wrist_roll
    rd = dflt if "roll" not in jn else -dflt
    lv = log_q[m_fwd, name_idx[lname]]
    rv = log_q[m_fwd, name_idx[rname]]
    print(f"{jn:28s} L{dflt:+.2f}/R{rd:+.2f}  {lv.mean():+7.3f} +/-{lv.std():5.3f}          {rv.mean():+7.3f} +/-{rv.std():5.3f}")
print("\n-- standing phase t<4s --")
for jn, dflt in ARM:
    rd = dflt if "roll" not in jn else -dflt
    lv = log_q[m_stand, name_idx[f"left_{jn}_joint"]]
    rv = log_q[m_stand, name_idx[f"right_{jn}_joint"]]
    print(f"{jn:28s} L{dflt:+.2f}/R{rd:+.2f}  {lv.mean():+7.3f} +/-{lv.std():5.3f}          {rv.mean():+7.3f} +/-{rv.std():5.3f}")

print("\n=========== 2. LEG JOINTS asymmetry (mean |q-default|, fwd walk) ===========")
LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
for jn in LEG:
    lv = np.abs(log_q[m_fwd, name_idx[f"left_{jn}_joint"]] - DEFAULT_POS[name_idx[f"left_{jn}_joint"]]).mean()
    rv = np.abs(log_q[m_fwd, name_idx[f"right_{jn}_joint"]] - DEFAULT_POS[name_idx[f"right_{jn}_joint"]]).mean()
    print(f"{jn:14s} |dev| L={lv:.3f}  R={rv:.3f}  ratio L/R={lv/rv:.2f}")

print("\n=========== 3. FOOT STRIDE ANALYSIS (touchdown-to-touchdown distance) ===========")
# touchdown detection: foot z falls below 0.055
for side in ("left", "right"):
    fi = 0 if side == "left" else 1
    fz = np.array([f[fi][2] for f in log_foot])
    fx = np.array([f[fi][0] for f in log_foot])
    fy = np.array([f[fi][1] for f in log_foot])
    # swing when z > 0.06; touchdown = transition swing->stance
    swing = fz > 0.062
    td_events = []  # (t, x, y)
    for k in range(1, len(fz)):
        if swing[k - 1] and not swing[k]:
            td_events.append((log_t[k], fx[k], fy[k]))
    print(f"{side:6s}: {len(td_events)} touchdowns; stride lengths (m):")
    sl = []
    for a, b in zip(td_events[:-1], td_events[1:]):
        dxy = np.hypot(b[1] - a[1], b[2] - a[2])
        sl.append(dxy)
    if sl:
        sl = np.array(sl)
        fwd_sl = sl[(np.array([e[0] for e in td_events[1:]]) >= 4) &
                    (np.array([e[0] for e in td_events[1:]]) < 26)]
        print(f"        all: mean={sl.mean():.3f} std={sl.std():.3f} min={sl.min():.3f} max={sl.max():.3f}")
        if len(fwd_sl):
            print(f"        fwd-walk only (4-26s): mean={fwd_sl.mean():.3f} n={len(fwd_sl)}")
    # also mean z over fwd walk (foot height proxy) and swing durations
    print(f"        mean foot z during walk: {fz[m_fwd].mean():.3f}")

print("\n=========== 4. PATH / HEADING ===========")
for a, b, n in [(4, 12, "fwd0.4"), (12, 20, "fwd0.6"), (20, 26, "fwd0.8")]:
    m = (log_t >= a) & (log_t < b)
    dyaw = np.degrees(log_base[m, 2][-1] - log_base[m, 2][0])
    dx, dy = log_base[m, 0][-1] - log_base[m, 0][0], log_base[m, 1][-1] - log_base[m, 1][0]
    print(f"{n:8s}: heading drift {dyaw:+6.1f}deg, path displacement ({dx:+.2f},{dy:+.2f})m, |path|={np.hypot(dx,dy):.2f}m")
