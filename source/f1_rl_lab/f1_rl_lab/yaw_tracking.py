"""Continuous command-relative yaw-rate error, matching the existing body-z frame."""
import torch


def yaw_rate_error_l2(env, command_name="base_velocity", asset_name="robot"):
    command = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene[asset_name].data.root_ang_vel_b[:, 2]
    return torch.square(command - actual)


def track_ang_vel_z_window_exp(env, command_name="base_velocity", asset_name="robot",
                               std=0.5, window_s=1.0, return_debug=False):
    """Yaw tracking scored on the recent body-wz average while turning forward.

    The window is used only while walking forward with a nonzero wz command; straight
    (wz=0) and stopped states always score the instantaneous body wz, so natural drift
    during straight walking is never credited by the window average.
    """
    command = env.command_manager.get_command(command_name)
    wz = env.scene[asset_name].data.root_ang_vel_b[:, 2]
    cmd = command[:, :3]
    num_envs = cmd.shape[0]
    device = cmd.device
    dt = getattr(env, "step_dt", None)
    if dt is None:
        cfg = getattr(env, "cfg", None)
        dt = float(cfg.sim.dt) * int(cfg.decimation) if cfg is not None else 0.02
    n = max(1, int(round(float(window_s) / float(dt))))
    state = getattr(env, "_f1_yaw_window_state", None)
    if state is None or state["n"] != n or state["num_envs"] != num_envs:
        state = {"n": n, "num_envs": num_envs,
                 "buffer": torch.zeros((num_envs, n), device=device, dtype=wz.dtype),
                 "count": torch.zeros(num_envs, device=device, dtype=torch.long),
                 "prev_cmd": cmd.detach().clone()}
        env._f1_yaw_window_state = state
    active = (cmd[:, 0] > 0) & (cmd[:, 2] != 0)
    clear = (cmd != state["prev_cmd"]).any(dim=1) | (cmd[:, 0] <= 0) | ~active
    episode = getattr(env, "episode_length_buf", None)
    if episode is not None:
        clear = clear | (episode <= 0)
    for attr in ("reset_buf", "reset_terminated", "reset_time_outs"):
        flag = getattr(env, attr, None)
        if flag is not None:
            clear = clear | flag.bool()
    buffer = state["buffer"]; count = state["count"]
    if clear.any():
        buffer[clear] = 0.0; count[clear] = 0
    buffer = torch.cat((buffer[:, 1:], wz.reshape(-1, 1)), dim=1)
    count = torch.clamp(count + 1, max=n)
    if clear.any():
        count[clear] = 1; buffer[clear] = 0.0; buffer[clear, -1] = wz[clear]
    state["buffer"], state["count"], state["prev_cmd"] = buffer, count, cmd.detach().clone()
    use_window = (count >= n) & (~clear)
    average = buffer.sum(dim=1) / count.clamp(min=1).to(wz.dtype)
    actual = torch.where(use_window, average, wz)
    reward = torch.exp(-torch.square(cmd[:, 2] - actual) / float(std) ** 2)
    if return_debug:
        return reward, {"used_window": use_window, "count": count, "average": average, "actual": actual, "clear": clear}
    return reward

