#!/usr/bin/env python
"""感知对应度分析：用"人眼能看到什么"的度量重测所有模型。

背景：p2p（峰峰值）被瞬态尖峰污染——一次大幅值抽动就能把 p2p 拉满，但人眼
判断"摆臂"看的是**持续、周期、平滑**的振荡。本脚本测：

摆臂可见性（每段稳态窗，跳过段首 1.5s 瞬态）：
  - arm_fft_amp: 摆臂主频处的正弦等效幅值（周期分量振幅，人眼"摆动幅度"）
  - arm_p10p90: 10-90 分位距（抗尖峰的稳健摆幅）
  - arm_spike: 段首 1.5s 瞬态窗内最大偏移（"抽动"指标）
  - arm_periodicity: 主频±0.15Hz 能量 / 0.3-3Hz 总能量（1.0=纯正弦摆，0=乱抖）
  - lr_corr: 左右臂反相相关（负=自然交叉摆）

步态稳定性（稳态窗）：
  - pitch_wobble: pitch 在 1.5-3.5Hz（步频）带内的 RMS（每步躯干顿挫）
  - roll_sway: roll std（横向摇晃）
  - z_bounce: 基座高度 10-90 分位距（上下颠簸，cm）
  - stride_cv: 步长变异系数（步子忽大忽小）
  - stride_t_cv: 步周期变异系数（节奏乱）
  - heading_drift: 每段航向漂移（歪着走）
  - foot_impact: 触地前 100ms 足部最大垂直速度（跺脚力度，m/s）
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
import mujoco
from sim2sim_play import (POLICY_JOINTS, DEFAULT_POS, KP, KD, EFF, ACTION_SCALE,
                          DECIMATION, PHYS_DT, ANG_VEL_SCALE, JOINT_VEL_SCALE,
                          INIT_POS_Z, ARMATURE, CMD_SEGMENTS, quat_to_rot, quat_to_rpy,
                          TermHistory, build_actor)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "model_14997.pt"
DT = DECIMATION * PHYS_DT  # 0.02
SEGS = [("fwd0.4", 4, 12), ("fwd0.6", 12, 20), ("fwd0.8", 20, 26)]


def fft_amp(x, dt, f_lo, f_hi):
    """返回 (主频, 该频正弦幅值, 周期性占比)。x 先去均值。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    if N < 16:
        return 0.0, 0.0, 0.0
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    amp = 2 * np.abs(X) / N
    total_e = np.sum(amp[band] ** 2)
    if total_e <= 0:
        return 0.0, 0.0, 0.0
    k = np.argmax(amp[band])
    fs = freqs[band][k]
    a = amp[band][k]
    near = (freqs >= fs - 0.15) & (freqs <= fs + 0.15) & band
    periodicity = np.sum(amp[near] ** 2) / total_e
    return fs, a, periodicity


def band_rms(x, dt, f_lo, f_hi):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    X[(freqs < f_lo) | (freqs > f_hi)] = 0
    y = np.fft.irfft(X, n=N)
    return float(np.sqrt(np.mean(y ** 2)))


actor = build_actor(CKPT)
model = mujoco.MjModel.from_xml_path("g1_description/g1_29dof_rev_1_0.xml")
model.opt.timestep = PHYS_DT
for j in range(6, 6 + 29):
    model.dof_armature[j] = ARMATURE
model.sensor_noise[:] = 0.0
model.sensor_cutoff[:] = 0.0
data = mujoco.MjData(model)

qadr = np.zeros(29, int); vadr = np.zeros(29, int); cadr = np.zeros(29, int)
jlo = np.zeros(29); jhi = np.zeros(29)
for i, name in enumerate(POLICY_JOINTS):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr[i] = model.jnt_qposadr[jid]; vadr[i] = model.jnt_dofadr[jid]
    jlo[i], jhi[i] = model.jnt_range[jid]
    cadr[i] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
gyro_adr = -1
for s in range(model.nsensor):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)
    if nm and "pelvis-angular-velocity" in nm:
        gyro_adr = model.sensor_adr[s]
LSP = POLICY_JOINTS.index("left_shoulder_pitch_joint")
RSP = POLICY_JOINTS.index("right_shoulder_pitch_joint")
foot_bid = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f + "_ankle_roll_link")
            for f in ("left", "right")}

data.qpos[:] = 0.0; data.qpos[2] = INIT_POS_Z; data.qpos[3] = 1.0
data.qpos[qadr] = DEFAULT_POS; data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

h_ang, h_grav, h_cmd = TermHistory(3), TermHistory(3), TermHistory(3)
h_qpos, h_qvel, h_la = TermHistory(29), TermHistory(29), TermHistory(29)
def obs_terms(cmd):
    quat = data.qpos[3:7].copy(); R = quat_to_rot(quat)
    ang = data.sensordata[gyro_adr:gyro_adr + 3].copy()
    grav = R.T @ np.array([0., 0., -1.])
    q = data.qpos[qadr].copy(); qd = data.qvel[vadr].copy()
    return (ANG_VEL_SCALE * ang, grav, cmd.copy(), q - DEFAULT_POS,
            JOINT_VEL_SCALE * qd, None)
ang, grav, c, dq, dqd, _ = obs_terms(np.zeros(3))
h_ang.reset(ang); h_grav.reset(grav); h_cmd.reset(np.zeros(3))
h_qpos.reset(dq); h_qvel.reset(dqd); h_la.reset(np.zeros(29))

total_t = sum(s[0] for s in CMD_SEGMENTS)
n_steps = int(round(total_t / DT))
t = 0.0; seg_i, seg_end = 0, CMD_SEGMENTS[0][0]
cmd = np.array(CMD_SEGMENTS[0][1])
log = {k: [] for k in ("t", "lsp", "rsp", "pitch", "roll", "z", "lfz", "rfz", "yaw")}
for step in range(n_steps):
    if t >= seg_end - 1e-9:
        seg_i += 1
        if seg_i >= len(CMD_SEGMENTS): break
        seg_end += CMD_SEGMENTS[seg_i][0]; cmd = np.array(CMD_SEGMENTS[seg_i][1])
    obs = np.concatenate([h_ang.flat(), h_grav.flat(), h_cmd.flat(),
                          h_qpos.flat(), h_qvel.flat(), h_la.flat()])
    with torch.no_grad():
        action = actor(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)).numpy()[0]
    q_des = np.clip(DEFAULT_POS + ACTION_SCALE * action, jlo, jhi)
    for _ in range(DECIMATION):
        tau = np.clip(KP * (q_des - data.qpos[qadr]) - KD * data.qvel[vadr], -EFF, EFF)
        data.ctrl[cadr] = tau
        mujoco.mj_step(model, data)
    t += DT
    ang, grav, c_, dq, dqd, _ = obs_terms(cmd)
    h_ang.push(ang); h_grav.push(grav); h_cmd.push(c_)
    h_qpos.push(dq); h_qvel.push(dqd); h_la.push(action)
    roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
    log["t"].append(t)
    log["lsp"].append(np.degrees(data.qpos[qadr][LSP]))
    log["rsp"].append(np.degrees(data.qpos[qadr][RSP]))
    log["pitch"].append(np.degrees(pitch)); log["roll"].append(np.degrees(roll))
    log["yaw"].append(yaw)
    log["z"].append(data.qpos[2])
    log["lfz"].append(data.xpos[foot_bid["left"]][2])
    log["rfz"].append(data.xpos[foot_bid["right"]][2])
    # foot world pos for stride length
    log.setdefault("lfx", []).append(data.xpos[foot_bid["left"]][0])
    log.setdefault("lfx2", []).append(data.xpos[foot_bid["left"]][1])
    log.setdefault("rfx", []).append(data.xpos[foot_bid["right"]][0])
    log.setdefault("rfx2", []).append(data.xpos[foot_bid["right"]][1])

for k in log:
    log[k] = np.array(log[k])
t = log["t"]

print(f"CKPT={CKPT}")
print(f"{'seg':8s} | {'armFFT Hz':>8s} {'amp°':>6s} {'p10p90°':>8s} {'spike°':>7s} {'peri':>5s} {'LRc':>6s} | "
      f"{'pitchW°':>8s} {'roll°':>6s} {'zB cm':>6s} {'strCV':>6s} {'tCV':>5s} {'hd°':>6s} {'imp m/s':>8s}")
for name, a, b in SEGS:
    steady = (t >= a + 1.5) & (t < b)
    trans = (t >= a) & (t < a + 1.5)
    m = steady; mt = trans
    # arm
    fl, al, pl = fft_amp(log["lsp"][m], DT, 0.3, 2.5)
    fr, ar_, pr = fft_amp(log["rsp"][m], DT, 0.3, 2.5)
    p10p90 = np.percentile(log["lsp"][m], 90) - np.percentile(log["lsp"][m], 10)
    spike = np.max(np.abs(log["lsp"][mt] - log["lsp"][m].mean())) if mt.any() else 0
    lr = np.corrcoef(log["lsp"][m], log["rsp"][m])[0, 1] if m.sum() > 10 else 0
    # gait
    pw = band_rms(log["pitch"][m], DT, 1.5, 3.5)
    rs = log["roll"][m].std()
    zb = (np.percentile(log["z"][m], 95) - np.percentile(log["z"][m], 5)) * 100
    # strides (touchdowns in window)
    fz = log["lfz"]; tds = []
    for k in range(1, len(fz)):
        if (t[k] >= a + 1.5) and (t[k] < b) and fz[k - 1] > 0.062 and fz[k] <= 0.062:
            tds.append(k)
    sl = [np.hypot(log["lfx"][j] - log["lfx"][i], log["lfx2"][j] - log["lfx2"][i])
          for i, j in zip(tds[:-1], tds[1:])] if len(tds) > 1 else [0]
    st = [(t[j] - t[i]) for i, j in zip(tds[:-1], tds[1:])] if len(tds) > 1 else [0]
    scv = np.std(sl) / np.mean(sl) if np.mean(sl) > 0 else 9
    tcv = np.std(st) / np.mean(st) if np.mean(st) > 0 else 9
    hd = abs(np.degrees(log["yaw"][m][-1] - log["yaw"][m][0]))
    # foot impact: max |vz| in 0.1s before each touchdown
    imps = []
    for k in tds:
        lo = max(0, k - 5)
        vz = np.diff(log["lfz"][lo:k + 1]) / DT
        imps.append(np.max(np.abs(vz)) if len(vz) else 0)
    imp = np.mean(imps) if imps else 0
    print(f"{name:8s} | {fl:8.2f} {al:6.1f} {p10p90:8.2f} {spike:7.1f} {pl:5.2f} {lr:+6.2f} | "
          f"{pw:8.2f} {rs:6.2f} {zb:6.1f} {scv:6.2f} {tcv:5.2f} {hd:6.1f} {imp:8.2f}")
