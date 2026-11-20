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
    if set(exposure) - {"vx", "group_modulus", "route_stop"} or not {"vx", "group_modulus"} <= exposure.keys():
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
    else:
        cfg.class_type = ForwardExposureVelocityCommand
    cfg.forward_exposure = dict(exposure)
