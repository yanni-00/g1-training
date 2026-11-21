"""Isaac Lab reward term for alternating weight-bearing support."""

import torch
from isaaclab.managers import ManagerTermBase

from .forward_gait import AlternatingSupport
from .tasks.locomotion.mdp.rewards import feet_gait


class AlternatingSupportGait(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.support = AlternatingSupport(env.num_envs, env.device)

    def reset(self, env_ids=None):
        self.support.reset(env_ids)

    def __call__(self, env, period, offset, sensor_cfg, activation_speed,
                 threshold=0.5, command_name=None):
        command_x = env.command_manager.get_command(command_name)[:, 0]
        sensor = env.scene.sensors[sensor_cfg.name]
        contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
        air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
        body_vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
        forward = self.support.reward(contact, air_time, command_x, body_vx,
                                      activation_speed, period * (1 - threshold))
        original = feet_gait(env, period, offset, sensor_cfg, threshold, command_name)
        return torch.where(command_x >= activation_speed, forward, original)
