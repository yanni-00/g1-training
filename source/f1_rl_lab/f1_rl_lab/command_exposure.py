"""Add forward-walking command samples to the existing Isaac velocity term."""

import math

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


SOURCE_PATH = "source/f1_rl_lab/f1_rl_lab/command_exposure.py"


class ForwardExposureVelocityCommand(UniformVelocityCommand):
    """Retain standing and general commands alongside two forward groups."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        exposure = self.cfg.forward_exposure
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        moving = ids[~self.is_standing_env[ids]]
        groups = moving.remainder(exposure["group_modulus"])
        straight = moving[groups == 0]
        steering = moving[groups == 1]
        # Steering keeps the parent sampler's lateral and angular commands.
        self.vel_command_b[straight, 0] = exposure["vx"]
        self.vel_command_b[straight, 1:] = 0
        self.vel_command_b[steering, 0] = exposure["vx"]

    def audit_forward_commands(self, *, after_step=True):
        """Verify actual commands after reset or a native feedback step."""
        groups = torch.arange(self.num_envs, device=self.device).remainder(
            self.cfg.forward_exposure["group_modulus"])
        moving = ~self.is_standing_env
        straight = moving & (groups == 0)
        steering = moving & (groups == 1)
        command = self.command
        vx = self.cfg.forward_exposure["vx"]
        if not (torch.all(command[straight | steering, 0] == vx)
                and torch.all(command[straight, 1:] == 0)):
            raise ValueError("Native commands do not match the registered forward exposure")
        # Isaac applies the standing mask in its command update, after reset.
        if after_step and not torch.all(command[~moving] == 0):
            raise ValueError("A native standing command is nonzero after command update")
        steering_nonzero = steering & (command[:, 1:] != 0).any(dim=1)
        counts = {
            "environments": self.num_envs,
            "standing": int((~moving).sum().item()),
            "pure_forward": int(straight.sum().item()),
            "forward_with_steering": int(steering_nonzero.sum().item()),
        }
        if not counts["pure_forward"] or not counts["forward_with_steering"]:
            raise ValueError("Native command snapshot lacks a targeted forward group")
        return counts


class SignedTurnVelocityCommand(UniformVelocityCommand):
    """Keep forward motion while introducing equally sampled signed yaw steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        exposure = cfg.forward_exposure
        groups = torch.arange(self.num_envs, device=self.device).remainder(exposure["group_modulus"])
        self._left = groups == 0
        self._right = groups == 1
        turn_s = exposure["signed_turn"].get("turn_s")
        if turn_s is not None:
            if not 0 < float(turn_s) < float(cfg.resampling_time_range[0]) - float(exposure["signed_turn"]["straight_s"]):
                raise ValueError("Registered turn_s must leave a positive straight lead-in inside the cycle")
            self._turn_time_left = float(turn_s)
        else:
            self._turn_time_left = cfg.resampling_time_range[0] - exposure["signed_turn"]["straight_s"]
        self._ordinary = ~(self._left | self._right)
        self._ordinary_resample_s = exposure.get("ordinary_resample_s")
        self._prev_cmd = self.command.clone()
        self._walk_to_stop = 0
        self._turn_to_stop = 0
        self._stop_transitions_since_reset = 0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        targeted = ids[(self._left[ids] | self._right[ids]) & ~self.is_standing_env[ids]]
        self.vel_command_b[targeted, 0] = self.cfg.forward_exposure["vx"]
        self.vel_command_b[targeted, 1:] = 0
        # Single native countdown: CommandTerm._resample already wrote cfg.resampling_time_range
        # into time_left before this call, so override per group here (ordinary 10 s, turn 20 s).
        ordinary_ids = ids[self._ordinary[ids]]
        turn_ids = ids[self._left[ids] | self._right[ids]]
        if self._ordinary_resample_s and ordinary_ids.numel():
            self.time_left[ordinary_ids] = float(self._ordinary_resample_s)
        if turn_ids.numel():
            self.time_left[turn_ids] = float(self.cfg.resampling_time_range[1])

    def _turn_masks(self):
        turning = (self.time_left <= self._turn_time_left) & ~self.is_standing_env
        return turning & self._left, turning & self._right

    def _update_command(self):
        previous = self._prev_cmd.clone()
        super()._update_command()
        left, right = self._turn_masks()
        yaw_rate = self.cfg.forward_exposure["signed_turn"]["yaw_rate"]
        self.vel_command_b[left, 2] = yaw_rate
        self.vel_command_b[right, 2] = -yaw_rate
        current = self.command
        valid = torch.ones_like(previous[:, 0], dtype=torch.bool)
        episode = getattr(getattr(self, "_env", None), "episode_length_buf", None)
        if episode is not None:
            valid = episode > 1
        stopped = (previous.abs().sum(dim=1) > 1e-6) & (current.abs().sum(dim=1) <= 1e-6) & valid
        self._walk_to_stop += int(stopped.sum().item())
        self._turn_to_stop += int((stopped & (self._left | self._right)).sum().item())
        self._stop_transitions_since_reset = self._walk_to_stop
        self._prev_cmd = current.clone()

    def audit_forward_commands(self, *, after_step=True):
        moving = ~self.is_standing_env
        targeted = moving & (self._left | self._right)
        left, right = self._turn_masks()
        straight = targeted & ~(left | right)
        exposure = self.cfg.forward_exposure
        command = self.command
        yaw_rate = exposure["signed_turn"]["yaw_rate"]
        if not (torch.all(command[targeted, 0] == exposure["vx"])
                and torch.all(command[targeted, 1] == 0)
                and torch.all(command[straight, 2] == 0)
                and torch.all(command[left, 2] == yaw_rate)
                and torch.all(command[right, 2] == -yaw_rate)):
            raise ValueError("Native commands do not match the signed turn schedule")
        if after_step and not torch.all(command[~moving] == 0):
            raise ValueError("A native standing command is nonzero after command update")
        counts = {
            "environments": self.num_envs,
            "standing": int((~moving).sum().item()),
            "pure_forward": int(straight.sum().item()),
            "left_group_moving": int((moving & self._left).sum().item()),
            "right_group_moving": int((moving & self._right).sum().item()),
            "positive_yaw_active": int(left.sum().item()),
            "negative_yaw_active": int(right.sum().item()),
            "turn_window_s": float(self._turn_time_left),
            "ordinary_resample_s": self._ordinary_resample_s,
            "walk_to_stop_unreset": int(self._walk_to_stop),
            "turn_to_stop_unreset": int(self._turn_to_stop),
        }
        if not after_step and (not counts["left_group_moving"] or not counts["right_group_moving"]):
            raise ValueError("Native command snapshot lacks a signed turn group")
        return counts


class WalkingRouteVelocityCommand(ForwardExposureVelocityCommand):
    """Practice forward walking, path recovery and a gradual stop."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        groups = torch.arange(self.num_envs, device=self.device).remainder(
            cfg.forward_exposure["group_modulus"])
        self._forward = groups < 2
        self._route = groups == 1
        self._route_origin = torch.zeros_like(self.vel_command_b[:, :2])
        self._route_heading = torch.zeros_like(self.vel_command_b[:, 0])

    def _target_speed(self):
        return min(self.cfg.forward_exposure["vx"], self.cfg.ranges.lin_vel_x[1])

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        selected = ids[self._forward[ids] & ~self.is_standing_env[ids]]
        self._route_origin[selected] = self.robot.data.root_pos_w[selected, :2]
        self._route_heading[selected] = self.robot.data.heading_w[selected]
        self.vel_command_b[selected, 0] = self._target_speed()
        self.vel_command_b[selected, 1:] = 0

    def _update_command(self):
        super()._update_command()
        active = self._forward & ~self.is_standing_env
        steering = self._route & active
        cfg = self.cfg.forward_exposure["route_stop"]
        envelope = ((self.time_left - cfg["hold_s"]) / cfg["ramp_s"]).clamp(0, 1)
        forward = self._target_speed()
        offset = self.robot.data.heading_w - self._route_heading
        error = torch.atan2(offset.sin(), offset.cos())
        xy = self.robot.data.root_pos_w[:, :2] - self._route_origin
        cross_track = -self._route_heading.sin() * xy[:, 0] + self._route_heading.cos() * xy[:, 1]
        lateral = -error.sin() * forward - error.cos() * cfg["cross_track_gain"] * cross_track
        yaw = -cfg["heading_gain"] * error
        self.vel_command_b[active, 0] = forward * envelope[active]
        self.vel_command_b[steering, 1] = lateral[steering].clamp(*self.cfg.ranges.lin_vel_y) * envelope[steering]
        self.vel_command_b[steering, 2] = yaw[steering].clamp(*self.cfg.ranges.ang_vel_z) * envelope[steering]

    def audit_forward_commands(self, *, after_step=True):
        active = self._forward & ~self.is_standing_env
        straight = active & ~self._route
        steering = active & self._route
        if not (torch.all(self.command[active, 0] >= 0) and torch.all(self.command[straight, 1:] == 0)):
            raise ValueError("Forward-route samples contain reverse or nonstraight commands")
        if after_step and not torch.all(self.command[self.is_standing_env] == 0):
            raise ValueError("Standing commands are nonzero after command update")
        return {
            "environments": self.num_envs,
            "standing": int(self.is_standing_env.sum().item()),
            "pure_forward": int(straight.sum().item()),
            "forward_route": int(steering.sum().item()),
            "targeted_stopped": int((active & (self.command[:, 0] == 0)).sum().item()),
        }


def install_forward_exposure(cfg, exposure):
    required = {"vx", "group_modulus"}
    allowed = required | {"route_stop", "signed_turn", "ordinary_resample_s"}
    if not required <= set(exposure) or set(exposure) - allowed:
        raise ValueError("Forward exposure requires its nominal speed and group modulus")
    vx, modulus = exposure["vx"], exposure["group_modulus"]
    routes = exposure.get("route_stop")
    lo, hi = cfg.limit_ranges.lin_vel_x if routes is not None else cfg.ranges.lin_vel_x
    if not (type(modulus) is int and modulus >= 3 and math.isfinite(vx) and vx > 0 and lo <= vx <= hi):
        raise ValueError("Invalid forward exposure or speed outside the registered range")
    if cfg.class_type is not UniformVelocityCommand or cfg.heading_command:
        raise ValueError("Forward exposure requires the original body-frame velocity command")
    if routes is not None:
        if (set(routes) != {"heading_gain", "cross_track_gain", "ramp_s", "hold_s"}
                or not all(math.isfinite(v) and v > 0 for v in routes.values())
                or routes["ramp_s"] + routes["hold_s"] >= cfg.resampling_time_range[0]
                or cfg.ranges.lin_vel_x[1] <= 0):
            raise ValueError("Invalid route correction or stopping schedule")
        cfg.class_type = WalkingRouteVelocityCommand
    elif "signed_turn" in exposure:
        turn = exposure["signed_turn"]
        period_lo, period_hi = cfg.resampling_time_range
        turn_fields = {"straight_s", "yaw_rate"} | ({"turn_s"} if "turn_s" in turn else set())
        ordinary = exposure.get("ordinary_resample_s")
        if (set(turn) != turn_fields
                or period_lo != period_hi
                or not 0 < turn["straight_s"] < period_lo
                or not 0 < turn["yaw_rate"] < math.inf
                or ("turn_s" in turn and not 0 < turn["turn_s"] < period_lo - turn["straight_s"])
                or (ordinary is not None and not 0 < float(ordinary) <= period_lo)):
            raise ValueError("Invalid signed turn timing or yaw rate")
        yaw_lo, yaw_hi = cfg.ranges.ang_vel_z
        if not (yaw_lo <= -turn["yaw_rate"] and turn["yaw_rate"] <= yaw_hi):
            raise ValueError("Signed turn yaw exceeds the restored command range")
        cfg.class_type = SignedTurnVelocityCommand
    else:
        cfg.class_type = ForwardExposureVelocityCommand
    cfg.forward_exposure = dict(exposure)
