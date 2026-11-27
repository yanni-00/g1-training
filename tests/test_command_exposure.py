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


class NativeCycleFixture(UniformCommandFixture):
    """Emulate Isaac CommandTerm timing exactly.

    compute(dt) decrements time_left; an env whose time_left <= 0 is resampled, and
    _resample writes a fresh uniform(resampling_time_range) into time_left BEFORE calling
    _resample_command (so the subclass can override the period per group). reset() resamples.
    """

    def __init__(self, cfg, env=None):
        super().__init__(cfg, env)
        self.time_left = torch.zeros(self.num_envs)

    def _resample(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long)
        lo, hi = self.cfg.resampling_time_range
        self.time_left[env_ids] = torch.empty(env_ids.numel()).uniform_(lo, hi)
        self._resample_command(env_ids)

    def reset(self, env_ids):
        self._resample(env_ids)

    def compute(self, dt):
        self.time_left -= dt
        ids = (self.time_left <= 0.0).nonzero().flatten()
        if ids.numel():
            self._resample(ids)
        self._update_command()


def load_module(base):
    isaac_module = ModuleType("isaaclab.envs.mdp.commands.velocity_command")
    isaac_module.UniformVelocityCommand = base
    root = Path(__file__).resolve().parents[1]
    path = root / "source/f1_rl_lab/f1_rl_lab/command_exposure.py"
    spec = importlib.util.spec_from_file_location("command_exposure_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {isaac_module.__name__: isaac_module}):
        spec.loader.exec_module(module)
    return module


class ForwardExposureTests(unittest.TestCase):
    @staticmethod
    def module():
        return load_module(UniformCommandFixture)

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


class SignedTurnWindowTests(ForwardExposureTests):
    """turn_s override: continuous turning longer than the 5 s default inside the cycle."""

    def test_turn_s_extends_window_and_keeps_straight_lead(self):
        module = self.module()
        cfg = SimpleNamespace(class_type=UniformCommandFixture, heading_command=False,
                              resampling_time_range=(20., 20.),
                              ranges=SimpleNamespace(lin_vel_x=(-.5, 1.), ang_vel_z=(-.1, .1)))
        exposure = {"vx": .5, "group_modulus": 5,
                    "signed_turn": {"straight_s": 5., "yaw_rate": .1, "turn_s": 13.}}
        module.install_forward_exposure(cfg, exposure)
        term = cfg.class_type(cfg, None)
        term._resample_command(torch.arange(term.num_envs))
        # cycle 20 s: straight for 7 s (time_left > 13), turning for 13 s
        term.time_left[0] = 13.02
        term.time_left[1] = 13.
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.5, 0., 0.]))
        torch.testing.assert_close(term.command[1], torch.tensor([.5, 0., -.1]))
        term.time_left[0] = 12.98
        term._update_command()
        torch.testing.assert_close(term.command[0], torch.tensor([.5, 0., .1]))
        counts = term.audit_forward_commands()
        self.assertAlmostEqual(counts["turn_window_s"], 13.)
        # invalid: turn_s leaving no straight lead-in must be rejected at install time
        with self.assertRaises(ValueError):
            module.install_forward_exposure(cfg, {**exposure,
                "signed_turn": {"straight_s": 5., "yaw_rate": .1, "turn_s": 16.}})


if __name__ == "__main__":
    unittest.main()


class OrdinaryResampleTests(ForwardExposureTests):
    def test_single_native_timer_ordinary_10s_turn_20s_no_double(self):
        module = load_module(NativeCycleFixture)
        cfg = SimpleNamespace(class_type=NativeCycleFixture, heading_command=False,
                              resampling_time_range=(20., 20.),
                              ranges=SimpleNamespace(lin_vel_x=(-.5, 1.), ang_vel_z=(-.1, .1)))
        exposure = {"vx": .5, "group_modulus": 5, "ordinary_resample_s": 10.,
                    "signed_turn": {"straight_s": 5., "yaw_rate": .1, "turn_s": 13.}}
        module.install_forward_exposure(cfg, exposure)
        term = cfg.class_type(cfg, None)
        dt = 0.02
        t = 0.0
        term.reset(torch.arange(term.num_envs))  # native reset draws a period; subclass overrides per group
        ordinary_env, left_env, right_env = 2, 0, 1  # groups 2 / 0 / 1 modulo 5
        self.assertTrue(bool(term._ordinary[ordinary_env]) and bool(term._left[left_env])
                        and bool(term._right[right_env]))
        self.assertAlmostEqual(float(term.time_left[ordinary_env]), 10.0, places=6)
        self.assertAlmostEqual(float(term.time_left[left_env]), 20.0, places=6)

        events = []  # (t, env_id) at each actual _resample_command call (catches direct-call regressions)
        orig = term._resample_command

        def tracked(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
            events.extend((round(t, 3), int(e)) for e in ids.tolist())
            return orig(ids)

        term._resample_command = tracked
        turn_active_t = []
        for step in range(1, 1001):            # 20 s at 0.02 s
            t = round(step * dt, 3)
            term.compute(dt)
            if float(term.command[left_env, 2]) > 0 and not turn_active_t:
                turn_active_t.append(t)

        def times_for(env):
            return [et for (et, e) in events if e == env]

        def near(ts, target):
            return [x for x in ts if abs(x - target) <= 0.04]

        ord_times = times_for(ordinary_env)
        left_times = times_for(left_env)
        # ordinary group: exactly one sample at 10 s and exactly one at 20 s (no double sample)
        self.assertEqual(len(near(ord_times, 10.0)), 1, ord_times)
        self.assertEqual(len(near(ord_times, 20.0)), 1, ord_times)
        # turn group: single sample at the 20 s cycle boundary, none at 10 s
        self.assertEqual(len(near(left_times, 20.0)), 1, left_times)
        self.assertEqual(len(near(left_times, 10.0)), 0, left_times)
        # turn starts after the 7 s straight lead-in inside the 20 s turn cycle
        self.assertTrue(turn_active_t and abs(turn_active_t[0] - 7.0) <= 0.05, turn_active_t)

    def test_reset_reanchors_ordinary_cycle_without_off_phase_sample(self):
        module = load_module(NativeCycleFixture)
        cfg = SimpleNamespace(class_type=NativeCycleFixture, heading_command=False,
                              resampling_time_range=(20., 20.),
                              ranges=SimpleNamespace(lin_vel_x=(-.5, 1.), ang_vel_z=(-.1, .1)))
        exposure = {"vx": .5, "group_modulus": 5, "ordinary_resample_s": 10.,
                    "signed_turn": {"straight_s": 5., "yaw_rate": .1, "turn_s": 13.}}
        module.install_forward_exposure(cfg, exposure)
        term = cfg.class_type(cfg, None)
        dt = 0.02
        term.reset(torch.arange(term.num_envs))
        env = 2  # ordinary group
        t = 0.0
        events = []
        orig = term._resample_command

        def tracked(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
            events.extend((round(t, 3), int(e)) for e in ids.tolist())
            return orig(ids)

        term._resample_command = tracked
        # advance 7 s without any expiry, then reset the ordinary env at t=7 s
        for _ in range(350):
            t = round(t + dt, 3)
            term.compute(dt)
        self.assertTrue(all(abs(et - 7.0) > 0.5 for (et, e) in events if e == env), events)
        term.reset(torch.tensor([env]))
        # next ordinary expiry is 10 s after the reset (t=17 s), not the old off-phase t=10 s
        for _ in range(600):
            t = round(t + dt, 3)
            term.compute(dt)
        env_times = [et for (et, e) in events if e == env]
        self.assertEqual(len([x for x in env_times if abs(x - 10.0) <= 0.04]), 0, env_times)
        self.assertEqual(len([x for x in env_times if abs(x - 17.0) <= 0.04]), 1, env_times)

    def test_unreset_walk_to_stop_transitions_are_counted(self):
        module = self.module()
        cfg = SimpleNamespace(class_type=UniformCommandFixture, heading_command=False,
                              resampling_time_range=(20., 20.),
                              ranges=SimpleNamespace(lin_vel_x=(-.5, 1.), ang_vel_z=(-.1, .1)))
        exposure = {"vx": .5, "group_modulus": 5, "ordinary_resample_s": 10.,
                    "signed_turn": {"straight_s": 5., "yaw_rate": .1, "turn_s": 13.}}
        module.install_forward_exposure(cfg, exposure)
        term = cfg.class_type(cfg, None)
        term._env = SimpleNamespace(step_dt=0.02, episode_length_buf=torch.full((10,), 10, dtype=torch.long))
        term.vel_command_b[:] = 0.0
        term._prev_cmd = torch.full((10, 3), 0.1)
        term.is_standing_env[:] = False
        term.time_left[:] = 14.0  # outside the turn window so turn groups stay stopped
        term._update_command()
        self.assertEqual(term._walk_to_stop, 10)
        self.assertEqual(term._turn_to_stop, 4)


if __name__ == "__main__":
    unittest.main()
