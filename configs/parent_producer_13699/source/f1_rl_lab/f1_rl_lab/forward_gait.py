"""Forward gait credit requires the commanded stance foot to bear support."""

import torch

SOURCE_PATH = "source/f1_rl_lab/f1_rl_lab/forward_gait.py"
METHOD = "single_support_with_body_forward_progress"
ALTERNATING_METHOD = "alternating_support_with_body_forward_progress"
SPEED_MATCH_METHOD = "alternating_support_with_body_speed_match"
ALTERNATING_ADAPTER = "source/f1_rl_lab/f1_rl_lab/alternating_support_reward.py"


def forward_gait_credit(expected_contact, actual_contact, command_x, body_vx, activation_speed):
    """Score matching single support and scale it by forward progress."""
    single_support = expected_contact.sum(dim=-1) == 1
    matched = (expected_contact == actual_contact).all(dim=-1)
    return _credit(single_support & matched, command_x, body_vx, activation_speed)


def _credit(eligible, command_x, body_vx, activation_speed, *, match_speed=False):
    ratio = body_vx / command_x.clamp_min(activation_speed)
    progress = (torch.minimum(ratio, 2 - ratio) if match_speed else ratio).clamp(0, 1)
    return eligible.to(progress.dtype) * (1 + progress)


class AlternatingSupport:
    """Track actual double-to-single support transfers independently per environment."""

    def __init__(self, num_envs, device):
        self.last_side = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.credited_side = torch.full_like(self.last_side, -1)
        self.was_double = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_side[ids] = -1
        self.credited_side[ids] = -1
        self.was_double[ids] = False

    def reward(self, contact, air_time, command_x, body_vx, activation_speed, swing_limit_s,
               *, match_speed=False):
        active = command_x >= activation_speed
        left, right = contact.unbind(dim=-1)
        side = right.long()
        single = active & (left ^ right)
        new_support = single & self.was_double & (side != self.last_side)
        ongoing = single & (side == self.credited_side)
        credited = new_support | ongoing
        self.last_side = torch.where(active, torch.where(new_support, side, self.last_side), -1)
        self.credited_side = torch.where(credited, side, -1)
        self.was_double = left & right
        swing_time = torch.where(right, air_time[:, 0], air_time[:, 1])
        return _credit(credited & (swing_time <= swing_limit_s), command_x, body_vx, activation_speed,
                       match_speed=match_speed)


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
