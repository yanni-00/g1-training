"""Cross-step yaw reward: symmetry, zero-net rejection, reset and command-change isolation."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


_stub("isaaclab")
_stub("isaaclab.utils")
_stub("isaaclab.utils.math", quat_apply_inverse=lambda *a, **k: None)
_stub("isaaclab.assets", Articulation=object, RigidObject=object)
_stub("isaaclab.managers", SceneEntityCfg=type("SceneEntityCfg", (), {"__init__": lambda self, name: setattr(self, "name", name)}))
_stub("isaaclab.sensors", ContactSensor=object)
_stub("isaaclab.envs", ManagerBasedRLEnv=object)

spec = importlib.util.spec_from_file_location(
    "f1_rewards_window_test",
    ROOT / "source/f1_rl_lab/f1_rl_lab/tasks/locomotion/mdp/rewards.py")
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


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
        return rewards.track_ang_vel_z_window_exp(
            self, std=0.3, window_s=1.0, return_debug=True)


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