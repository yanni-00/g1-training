"""Velocity-command root reference, shared by native training and offline play.

This module plans reference coordinates only. It never writes robot state or
claims that a commanded trajectory is actual locomotion. Joint gait templates,
support-aware braking and physical tracking are separate required components.
"""
import math

import torch


def joystick_axes(ly, lx, rx):
    """F1 raw joystick mapping is deferred until the actual deployment audit."""
    raise NotImplementedError('Use physical body vx/vy/yaw_rate; F1 joystick axes are not yet bound')


class VelocityReferenceRoot:
    def __init__(self, count, limits, device='cpu'):
        if count <= 0:
            raise ValueError('Reference count must be positive')
        self.device = torch.device(device)
        self.limits = torch.as_tensor(limits, dtype=torch.float32, device=self.device)
        if self.limits.shape != (3, 2) or not torch.isfinite(self.limits).all():
            raise ValueError('Explicit finite vx/vy/yaw_rate limits required')
        if (self.limits[:, 0] > self.limits[:, 1]).any():
            raise ValueError('Invalid command limits')
        self.position = torch.zeros(count, 2, device=self.device)
        self.yaw = torch.zeros(count, device=self.device)
        self.raw_command = torch.zeros(count, 3, device=self.device)
        self.command = torch.zeros_like(self.raw_command)
        self.saturated = torch.zeros(count, 3, dtype=torch.bool, device=self.device)
        self.walk_enabled = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=self.device)

    def reset(self, ids, position, yaw):
        """Call only with a genuine policy/episode initialization, never on GO/STOP."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        position = torch.as_tensor(position, dtype=torch.float32, device=self.device)
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device)
        if position.shape != (ids.numel(), 2) or yaw.shape != (ids.numel(),):
            raise ValueError('Reference reset shape mismatch')
        if not torch.isfinite(position).all() or not torch.isfinite(yaw).all():
            raise ValueError('Non-finite reference reset')
        self.position[ids] = position
        self.yaw[ids] = yaw
        self.raw_command[ids] = 0
        self.command[ids] = 0
        self.saturated[ids] = False
        self.walk_enabled[ids] = False
        self.initialized[ids] = True

    def request(self, command, walk_enabled):
        """Receive physical body vx/vy/yaw_rate, not a raw-motion playback flag."""
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        enabled = torch.as_tensor(walk_enabled, dtype=torch.bool, device=self.device)
        if command.shape != self.command.shape or enabled.shape != self.walk_enabled.shape:
            raise ValueError('Command packet shape mismatch')
        if not torch.isfinite(command).all():
            raise ValueError('Non-finite velocity command')
        fixed_axis = self.limits[:, 0] == self.limits[:, 1]
        if (enabled[:, None] & fixed_axis & (command != self.limits[:, 0])).any():
            raise ValueError('Requested axis is outside the explicitly supported command set')
        bounded = torch.minimum(torch.maximum(command, self.limits[:, 0]), self.limits[:, 1])
        self.raw_command.copy_(command)
        self.walk_enabled.copy_(enabled)
        self.saturated.copy_(bounded != command)
        self.command.copy_(torch.where(enabled[:, None], bounded, torch.zeros_like(bounded)))

    def step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Finite positive reference dt required')
        if not self.initialized.all():
            raise RuntimeError('Reference must be initialized before advancing')
        angle = self.command[:, 2] * dt
        mid_yaw = self.yaw + angle * 0.5
        # Exact constant-body-twist SE(2) integration, including yaw_rate==0.
        scale = dt * torch.sinc(angle / (2 * math.pi))
        cosine, sine = torch.cos(mid_yaw), torch.sin(mid_yaw)
        vx, vy = self.command[:, 0], self.command[:, 1]
        self.position[:, 0] += scale * (cosine * vx - sine * vy)
        self.position[:, 1] += scale * (sine * vx + cosine * vy)
        self.yaw += angle
        return self.position, self.yaw

    @property
    def velocity_world(self):
        cosine, sine = torch.cos(self.yaw), torch.sin(self.yaw)
        vx, vy = self.command[:, 0], self.command[:, 1]
        return torch.stack((cosine * vx - sine * vy, sine * vx + cosine * vy), dim=-1)
