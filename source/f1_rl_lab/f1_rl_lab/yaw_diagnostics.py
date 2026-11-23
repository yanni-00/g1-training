"""Log-only per-group signed-turn diagnostics.

Records effective samples, actual body yaw rate, tracking error, the unweighted
yaw-tracking reward and the dead-zone fraction for the reserved signed-turn
groups and for the native command groups. Nothing here changes commands,
observations, rewards or the PPO update; it only aggregates state each step and
prints one JSON line per reporting window.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

BUCKETS = (
    "positive_turn",
    "negative_turn",
    "straight_prefix",
    "zero_command",
    "native_nonzero",
)


def _blank():
    return {
        name: {"samples": 0, "cumulative": 0, "sum_cmd": 0.0, "sum_wz": 0.0,
               "sum_e": 0.0, "sum_abs_e": 0.0, "sum_e2": 0.0, "sum_reward": 0.0, "dead": 0}
        for name in BUCKETS
    }


@contextmanager
def install_yaw_diagnostics(runner, log_dir, *, std, every=10, dead_reward=0.05, report_prefix="[F1_YAW_DIAG]"):
    """Aggregate yaw-tracking statistics per command group without touching training."""
    vec_env = runner.env
    env = vec_env.unwrapped
    term = env.command_manager.get_term("base_velocity")
    exposure = getattr(term.cfg, "forward_exposure", None)
    if not exposure or "signed_turn" not in exposure:
        yield None
        return

    modulus = int(exposure["group_modulus"])
    turn_time_left = getattr(term, "_turn_time_left", None)
    device = term.device
    groups = torch.arange(term.num_envs, device=device).remainder(modulus)
    left_group = groups == 0
    right_group = groups == 1
    dead_abs_error = float(std) * math.sqrt(-math.log(dead_reward))

    acc = _blank()
    state = {"updates": 0, "iteration": -1}
    native_step = vec_env.step
    native_update = runner.alg.update
    stream_path = Path(log_dir) / "params" / "yaw_diagnostics.jsonl"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.write_text("", encoding="utf-8")

    def record(bucket, mask, command_z, yaw_rate_z):
        if mask is None or not bool(mask.any()):
            return
        cmd = command_z[mask]
        wz = yaw_rate_z[mask]
        error = cmd - wz
        reward = torch.exp(-(error / float(std)) ** 2)
        item = acc[bucket]
        n = int(mask.sum().item())
        item["samples"] += n
        item["cumulative"] += n
        item["sum_cmd"] += float(cmd.sum().item())
        item["sum_wz"] += float(wz.sum().item())
        item["sum_e"] += float(error.sum().item())
        item["sum_abs_e"] += float(error.abs().sum().item())
        item["sum_e2"] += float((error * error).sum().item())
        item["sum_reward"] += float(reward.sum().item())
        item["dead"] += int((error.abs() > dead_abs_error).sum().item())

    def observe():
        command = term.command
        yaw_rate = env.scene["robot"].data.root_ang_vel_b
        command_z = command[:, 2]
        yaw_rate_z = yaw_rate[:, 2]
        moving = ~term.is_standing_env
        turning = (term.time_left <= turn_time_left) if turn_time_left is not None else torch.zeros_like(moving)
        record("positive_turn", left_group & moving & turning & (command_z > 1e-6), command_z, yaw_rate_z)
        record("negative_turn", right_group & moving & turning & (command_z < -1e-6), command_z, yaw_rate_z)
        record("straight_prefix", (left_group | right_group) & moving & ~turning, command_z, yaw_rate_z)
        native = (~left_group) & (~right_group) & moving
        record("zero_command", native & (command_z.abs() <= 1e-6), command_z, yaw_rate_z)
        record("native_nonzero", native & (command_z.abs() > 1e-6), command_z, yaw_rate_z)

    def guarded_step(actions):
        try:
            observe()
        except Exception:  # diagnostics must never break training
            pass
        return native_step(actions)

    def summarise(item):
        n = item["samples"]
        if not n:
            return {"samples": 0, "cumulative": item["cumulative"]}
        mean_abs = item["sum_abs_e"] / n
        return {
            "samples": n,
            "cumulative": item["cumulative"],
            "mean_command_rad_s": item["sum_cmd"] / n,
            "mean_actual_rad_s": item["sum_wz"] / n,
            "mean_error_rad_s": item["sum_e"] / n,
            "mean_abs_error_rad_s": mean_abs,
            "rms_error_rad_s": math.sqrt(max(item["sum_e2"], 0.0) / n),
            "mean_unweighted_reward": item["sum_reward"] / n,
            "dead_fraction": item["dead"] / n,
        }

    def emit(final=False):
        line = {
            "iteration": state["iteration"],
            "window_updates": state["updates"],
            "std": float(std),
            "dead_abs_error_rad_s": dead_abs_error,
            "final": final,
        }
        line.update({name: summarise(item) for name, item in acc.items()})
        payload = json.dumps(line, allow_nan=False)
        print(f"{report_prefix} {payload}", flush=True)
        with stream_path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")

    def reset_window():
        for name in BUCKETS:
            keep = acc[name]["cumulative"]
            acc[name] = _blank()[name]
            acc[name]["cumulative"] = keep

    def guarded_update(*args, **kwargs):
        result = native_update(*args, **kwargs)
        try:
            state["updates"] += 1
            state["iteration"] = int(getattr(runner, "current_learning_iteration", state["iteration"]))
            if state["updates"] % int(every) == 0:
                emit()
                reset_window()
        except Exception:
            pass
        return result

    vec_env.step = guarded_step
    runner.alg.update = guarded_update
    audit = {
        "method": "log_only_signed_turn_diagnostics_v1",
        "std": float(std),
        "dead_reward_threshold": float(dead_reward),
        "dead_abs_error_rad_s": dead_abs_error,
        "report_every_updates": int(every),
        "stream": str(stream_path),
        "buckets": list(BUCKETS),
    }
    try:
        yield audit
    finally:
        try:
            vec_env.step = native_step
            runner.alg.update = native_update
            emit(final=True)
        except Exception:
            pass
