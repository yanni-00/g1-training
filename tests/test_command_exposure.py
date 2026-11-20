import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class UniformCommandFixture:
    """Exercise the subclass with Isaac's resample-then-mask ordering."""

    device = "cpu"
    num_envs = 10

    def __init__(self, cfg):
        self.cfg = cfg
        self.vel_command_b = torch.zeros(self.num_envs, 3)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool)

    @property
    def command(self):
        return self.vel_command_b

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = torch.tensor([0.2, -0.15, 0.07])
        self.is_standing_env[env_ids] = env_ids == 5


class ForwardExposureTests(unittest.TestCase):
    def test_targeted_sampling_keeps_stand_and_other_commands(self):
        isaac_module = ModuleType("isaaclab.envs.mdp.commands.velocity_command")
        isaac_module.UniformVelocityCommand = UniformCommandFixture
        root = Path(__file__).resolve().parents[1]
        path = root / "source/f1_rl_lab/f1_rl_lab/command_exposure.py"
        spec = importlib.util.spec_from_file_location("command_exposure_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {isaac_module.__name__: isaac_module}):
            spec.loader.exec_module(module)

        cfg = SimpleNamespace(class_type=UniformCommandFixture, heading_command=False,
                              ranges=SimpleNamespace(lin_vel_x=(-0.5, 1.0)))
        module.install_forward_exposure(cfg, {"vx": 0.5, "group_modulus": 5})
        term = cfg.class_type(cfg)
        term._resample_command(torch.arange(term.num_envs))
        torch.testing.assert_close(term.command[0], torch.tensor([0.5, 0., 0.]))
        torch.testing.assert_close(term.command[1], torch.tensor([0.5, -0.15, 0.07]))
        torch.testing.assert_close(term.command[2], torch.tensor([0.2, -0.15, 0.07]))
        self.assertEqual(term.audit_forward_commands(after_step=False),
                         {"environments": 10, "standing": 1, "pure_forward": 1,
                          "forward_with_steering": 2})
        term.command[term.is_standing_env] = 0
        self.assertEqual(term.audit_forward_commands()["standing"], 1)
        snapshot = term.command.clone()
        term._resample_command(torch.tensor([1, 8]))
        torch.testing.assert_close(term.command, snapshot)


if __name__ == "__main__":
    unittest.main()
