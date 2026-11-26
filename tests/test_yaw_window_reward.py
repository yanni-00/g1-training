"""Cross-step yaw reward: symmetry, zero-net rejection, reset and command-change isolation."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
from f1_rl_lab.yaw_tracking import track_ang_vel_z_window_exp as window_reward


class Env:
    def __init__(self, command=(0.5, 0.0, 0.1), wz=0.0, step_dt=0.02):
        self.step_dt = step_dt
        self.command = torch.tensor([command], dtype=torch.float32)
        self.velocity = torch.tensor([[0.0, 0.0, wz]], dtype=torch.float32)
        self.episode_length_buf = torch.ones(1, dtype=torch.long)
        self.scene = {"robot": NS(data=NS(root_ang_vel_b=self.velocity))}
        self.command_manager = NS(get_command=lambda name: self.command)

    def step(self, command, wz):
        self.command = torch.tensor([command], dtype=torch.float32)
        self.velocity = torch.tensor([[0.0, 0.0, wz]], dtype=torch.float32)
        self.scene["robot"].data.root_ang_vel_b = self.velocity
        return window_reward(self, std=0.3, window_s=1.0, return_debug=True)


class WindowRewardTests(unittest.TestCase):
    def test_positive_and_negative_commands_score_symmetrically(self):
        env = Env()
        for _ in range(55):
            reward_p, _ = env.step((0.5, 0.0, 0.1), 0.1)
        for _ in range(55):
            reward_n, _ = env.step((0.5, 0.0, -0.1), -0.1)
        self.assertAlmostEqual(float(reward_p), 1.0, places=6)
        self.assertAlmostEqual(float(reward_n), 1.0, places=6)

    def test_zero_net_swing_cannot_get_full_turn_reward(self):
        env = Env()
        values = []
        for i in range(100):
            reward, _ = env.step((0.5, 0.0, 0.1), 0.5 if i % 2 == 0 else -0.5)
            values.append(float(reward))
        self.assertLess(values[-1], 0.95)
        self.assertLess(values[-1], 0.9)
        matched = Env()
        for _ in range(100):
            reward_match, _ = matched.step((0.5, 0.0, 0.1), 0.1)
        self.assertGreater(float(reward_match), values[-1])

    def test_window_does_not_cross_command_change(self):
        env = Env()
        for _ in range(60):
            env.step((0.5, 0.0, 0.1), 0.1)
        reward, dbg = env.step((0.5, 0.0, -0.1), 0.1)
        self.assertFalse(bool(dbg["used_window"][0]))
        self.assertEqual(int(dbg["count"][0]), 1)
        self.assertLess(float(reward), 0.9)  # instantaneous error 0.2, not the cleared history
        for _ in range(60):
            reward, dbg = env.step((0.5, 0.0, -0.1), -0.1)
        self.assertTrue(bool(dbg["used_window"][0]))
        self.assertAlmostEqual(float(reward), 1.0, places=6)

    def test_straight_wz_zero_never_uses_window(self):
        env = Env(command=(0.5, 0.0, 0.0))
        for _ in range(60):
            reward, dbg = env.step((0.5, 0.0, 0.0), 0.135)
        self.assertFalse(bool(dbg["used_window"][0]))
        self.assertLess(float(reward), 0.9)  # instantaneous |0.135| vs cmd 0 keeps reward low
        env2 = Env(command=(0.5, 0.0, 0.0))
        for _ in range(60):
            reward0, _ = env2.step((0.5, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(float(reward0), 1.0, places=6)

    def test_window_activates_again_when_wz_becomes_nonzero(self):
        env = Env(command=(0.5, 0.0, 0.0))
        for _ in range(60):
            env.step((0.5, 0.0, 0.0), 0.135)
        _, dbg = env.step((0.5, 0.0, -0.1), -0.1)
        self.assertFalse(bool(dbg["used_window"][0]))  # command change cleared
        self.assertEqual(int(dbg["count"][0]), 1)
        for _ in range(60):
            reward, dbg = env.step((0.5, 0.0, -0.1), -0.1)
        self.assertTrue(bool(dbg["used_window"][0]))
        self.assertAlmostEqual(float(reward), 1.0, places=6)

    def test_reset_and_stop_clear_the_window(self):
        env = Env()
        for _ in range(60):
            env.step((0.5, 0.0, 0.1), 0.1)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)
        _, dbg = env.step((0.5, 0.0, 0.1), 0.1)
        self.assertFalse(bool(dbg["used_window"][0]))
        self.assertEqual(int(dbg["count"][0]), 1)
        env.episode_length_buf = torch.ones(1, dtype=torch.long)
        for _ in range(60):
            env.step((0.5, 0.0, 0.1), 0.1)
        _, dbg = env.step((0.0, 0.0, 0.0), 0.0)
        self.assertFalse(bool(dbg["used_window"][0]))
        self.assertEqual(int(dbg["count"][0]), 1)


if __name__ == "__main__":
    unittest.main()