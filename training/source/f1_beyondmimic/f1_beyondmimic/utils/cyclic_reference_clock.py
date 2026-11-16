"""Support-aware cyclic gait phase; reference wrap never resets robot state."""
import math

import torch


class CyclicSupportPhaseClock:
    def __init__(self, count, fps, cycle_frames, support_frames, device='cpu', acceleration=2.0):
        if (count <= 0 or not math.isfinite(fps) or fps <= 0 or
                not math.isfinite(cycle_frames) or cycle_frames < 2 or int(cycle_frames) != cycle_frames or
                not math.isfinite(acceleration) or acceleration <= 0):
            raise ValueError('Invalid cyclic clock configuration')
        self.fps, self.period, self.acceleration = float(fps), int(cycle_frames), float(acceleration)
        self.support = torch.as_tensor(support_frames, dtype=torch.float32, device=device).unique(sorted=True)
        if self.support.numel() < 2 or not torch.isfinite(self.support).all():
            raise ValueError('At least two finite support candidates required')
        if (self.support < 0).any() or (self.support >= self.period).any():
            raise ValueError('Support phases must lie inside the unique gait cycle')
        self.phase = torch.zeros(count, device=device)
        self.cycles = torch.zeros(count, dtype=torch.long, device=device)
        self.rate = torch.zeros(count, device=device)
        self.was_go = torch.zeros(count, dtype=torch.bool, device=device)
        self.remaining = torch.zeros(count, device=device)
        self.target_phase = torch.zeros(count, device=device)
        self.target_cycle = torch.zeros_like(self.cycles)

    def reset(self, ids, support_indices):
        """Only alongside a genuine policy/episode initialization."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.phase.device)
        indices = torch.as_tensor(support_indices, dtype=torch.long, device=self.phase.device)
        if indices.shape != ids.shape or (indices < 0).any() or (indices >= len(self.support)).any():
            raise ValueError('Invalid cyclic support reset')
        self.phase[ids] = self.support[indices]
        self.cycles[ids] = 0
        self.rate[ids] = 0
        self.was_go[ids] = False
        self.remaining[ids] = 0
        self.target_phase[ids] = self.phase[ids]
        self.target_cycle[ids] = 0

    @property
    def unwrapped_phase(self):
        return self.cycles.to(torch.float64)*self.period+self.phase.to(torch.float64)

    def step(self, target_rate, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Finite positive dt required')
        target = torch.as_tensor(target_rate, dtype=torch.float32, device=self.phase.device)
        if target.shape != self.rate.shape or not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError('Invalid desired gait rate')
        go = target > 0
        falling = self.was_go & ~go
        earliest = self.phase+self.rate.square()*self.fps/(2*self.acceleration)+self.rate*self.fps*dt
        offsets = torch.floor(earliest/self.period).long()
        local = earliest-offsets*self.period
        indices = torch.searchsorted(self.support, local)
        wrapped = indices == len(self.support)
        offsets += wrapped.long()
        indices = torch.where(wrapped, torch.zeros_like(indices), indices)
        stop_phase = self.support[indices]
        distance = offsets*self.period+stop_phase-self.phase
        self.remaining[falling] = distance[falling]
        self.target_phase[falling] = stop_phase[falling]
        self.target_cycle[falling] = self.cycles[falling]+offsets[falling]
        advancing_rate = self.rate+(target-self.rate).clamp(-self.acceleration*dt, self.acceleration*dt)
        cap = torch.sqrt(2*self.acceleration*self.remaining.clamp_min(0)/self.fps)
        braking_rate = torch.minimum(self.rate, cap)
        rate = torch.where(go, advancing_rate, braking_rate)
        advance = (self.rate+rate)*0.5*dt*self.fps
        advance = torch.where(go, advance, torch.minimum(advance, self.remaining))
        local_next = self.phase+advance
        wraps = torch.floor(local_next/self.period).long()
        self.phase.copy_(local_next-wraps*self.period)
        self.cycles += wraps
        self.remaining = torch.where(go, self.remaining, (self.remaining-advance).clamp_min(0))
        stopped = ~go & (self.remaining <= 1e-5)
        self.phase[stopped] = self.target_phase[stopped]
        self.cycles[stopped] = self.target_cycle[stopped]
        rate[stopped] = 0
        self.rate = rate
        self.was_go.copy_(go)
        return self.phase, self.rate
