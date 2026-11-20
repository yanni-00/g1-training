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


def install_forward_exposure(cfg, exposure):
    if set(exposure) != {"vx", "group_modulus"}:
        raise ValueError("Forward exposure requires its nominal speed and group modulus")
    vx, modulus = exposure["vx"], exposure["group_modulus"]
    lo, hi = cfg.ranges.lin_vel_x
    if not (type(modulus) is int and modulus >= 3 and math.isfinite(vx) and vx > 0 and lo <= vx <= hi):
        raise ValueError("Invalid forward exposure or speed outside the registered range")
    if cfg.class_type is not UniformVelocityCommand or cfg.heading_command:
        raise ValueError("Forward exposure requires the original body-frame velocity command")
    cfg.forward_exposure = dict(exposure)
    cfg.class_type = ForwardExposureVelocityCommand
