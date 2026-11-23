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

    def __init__(self, cfg, env=None):
        self.cfg = cfg
        self.vel_command_b = torch.zeros(self.num_envs, 3)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool)
        self.time_left = torch.full((self.num_envs,), 10.)
        self.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(self.num_envs, 3),
                                                          heading_w=torch.zeros(self.num_envs)))

    @property
    def command(self):
        return self.vel_command_b

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = torch.tensor([0.2, -0.15, 0.07])
        self.is_standing_env[env_ids] = env_ids == 5

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0


class ForwardExposureTests(unittest.TestCase):
    @staticmethod
    def module():
        isaac_module = ModuleType("isaaclab.envs.mdp.commands.velocity_command")
        isaac_module.UniformVelocityCommand = UniformCommandFixture
        root = Path(__file__).resolve().parents[1]
        path = root / "source/f1_rl_lab/f1_rl_lab/command_exposure.py"
        spec = importlib.util.spec_from_file_location("command_exposure_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {isaac_module.__name__: isaac_module}):
            spec.loader.exec_module(module)
        return module

    def test_targeted_sampling_keeps_stand_and_other_commands(self):
        module = self.module()

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

    def test_route_stays_forward_and_practises_stopping(self):
        module = self.module()
        cfg = SimpleNamespace(class_type=UniformCommandFixture, heading_command=False,
                              resampling_time_range=(10., 10.),
                              ranges=SimpleNamespace(lin_vel_x=(-.1, .1), lin_vel_y=(-.1, .1), ang_vel_z=(-.1, .1)),
                              limit_ranges=SimpleNamespace(lin_vel_x=(-.5, 1.)))
        module.install_forward_exposure(cfg, {"vx": .5, "group_modulus": 5,
            "route_stop": {"heading_gain": 1., "cross_track_gain": .5, "ramp_s": .8, "hold_s": 2.}})
        term = cfg.class_type(cfg, None)
        term._resample_command(torch.arange(term.num_envs))
        term.robot.data.heading_w[1] = 2.
        term.robot.data.root_pos_w[1, 1] = .4
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.1, 0., 0.]))
        self.assertAlmostEqual(float(term.command[1, 0]), .1)
        self.assertAlmostEqual(float(term.command[1, 2]), -.1)
        torch.testing.assert_close(term.command[2], torch.tensor([.2, -.15, .07]))
        torch.testing.assert_close(term.command[5], torch.zeros(3))
        term.time_left[1] = 2.4
        term._update_command()
        self.assertAlmostEqual(float(term.command[1, 0]), .05, places=6)
        term.time_left[1] = 2.
        term._update_command()
        torch.testing.assert_close(term.command[1], torch.zeros(3))
        self.assertEqual(term.audit_forward_commands()["targeted_stopped"], 1)

    def test_signed_yaw_starts_after_walking_and_resets_to_straight(self):
        module = self.module()
        cfg = SimpleNamespace(class_type=UniformCommandFixture, heading_command=False,
                              resampling_time_range=(10., 10.),
                              ranges=SimpleNamespace(lin_vel_x=(-.5, 1.), ang_vel_z=(-.1, .1)))
        exposure = {"vx": .5, "group_modulus": 5, "signed_turn": {"straight_s": 5., "yaw_rate": .1}}
        module.install_forward_exposure(cfg, exposure)
        term = cfg.class_type(cfg, None)
        term._resample_command(torch.arange(term.num_envs))
        torch.testing.assert_close(term.command[:2], torch.tensor([[.5, 0., 0.], [.5, 0., 0.]]))
        self.assertEqual(term.audit_forward_commands(after_step=False)["pure_forward"], 3)

        term.time_left[0] = 5.02
        term.time_left[1] = 5.
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.5, 0., 0.]))
        torch.testing.assert_close(term.command[1], torch.tensor([.5, 0., -.1]))
        term.time_left[0] = 4.98
        term.time_left[5] = 4.98
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.5, 0., .1]))
        torch.testing.assert_close(term.command[2], torch.tensor([.2, -.15, .07]))
        torch.testing.assert_close(term.command[5], torch.zeros(3))
        counts = term.audit_forward_commands()
        self.assertEqual((counts["positive_yaw_active"], counts["negative_yaw_active"]), (1, 1))

        term.time_left[0] = 10.
        term._resample_command(torch.tensor([0]))
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.5, 0., 0.]))
        self.assertEqual(term.audit_forward_commands()["positive_yaw_active"], 0)

        cfg.class_type = UniformCommandFixture
        cfg.ranges.ang_vel_z = (-.05, .1)
        with self.assertRaisesRegex(ValueError, "restored command range"):
            module.install_forward_exposure(cfg, exposure)


if __name__ == "__main__":
    unittest.main()
