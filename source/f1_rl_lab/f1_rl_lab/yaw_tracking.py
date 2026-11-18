"""Continuous command-relative yaw-rate error, matching the existing body-z frame."""
import torch


def yaw_rate_error_l2(env, command_name="base_velocity", asset_name="robot"):
    command = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene[asset_name].data.root_ang_vel_b[:, 2]
    return torch.square(command - actual)
