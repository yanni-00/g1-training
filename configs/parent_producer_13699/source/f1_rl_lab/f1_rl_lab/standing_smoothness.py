"""Training-only ankle action-rate cost for exactly zero velocity commands."""
import torch


def zero_command_ankle_action_rate_l2(env, ankle_action_ids, command_name="base_velocity"):
    command = env.command_manager.get_command(command_name)
    standing = torch.all(command[:, :3] == 0, dim=1)
    delta = (env.action_manager.action - env.action_manager.prev_action)[:, ankle_action_ids]
    return standing.to(delta.dtype) * torch.sum(delta.square(), dim=1)
