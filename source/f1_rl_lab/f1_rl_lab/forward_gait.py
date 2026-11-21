"""Forward gait credit requires the commanded stance foot to bear support."""

import torch

SOURCE_PATH = "source/f1_rl_lab/f1_rl_lab/forward_gait.py"
METHOD = "single_support_with_body_forward_progress"


def forward_gait_credit(expected_contact, actual_contact, command_x, body_vx, activation_speed):
    """Score matching single support and scale it by forward progress."""
    single_support = expected_contact.sum(dim=-1) == 1
    matched = (expected_contact == actual_contact).all(dim=-1)
    progress = (body_vx / command_x.clamp_min(activation_speed)).clamp(0, 1)
    return (single_support & matched).to(progress.dtype) * (1 + progress)


def feet_gait_with_forward_progress(
    env, period, offset, sensor_cfg, activation_speed, threshold=0.5, command_name=None,
):
    from .tasks.locomotion.mdp.rewards import feet_gait

    original = feet_gait(env, period, offset, sensor_cfg, threshold, command_name)
    command_x = env.command_manager.get_command(command_name)[:, 0]
    sensor = env.scene.sensors[sensor_cfg.name]
    actual_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    gait_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = (gait_phase + command_x.new_tensor(offset)) % 1.0
    expected_contact = leg_phase < threshold
    body_vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    forward_credit = forward_gait_credit(expected_contact, actual_contact, command_x, body_vx, activation_speed)
    return torch.where(command_x >= activation_speed, forward_credit, original)
