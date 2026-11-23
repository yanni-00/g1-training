"""Check the log-only yaw diagnostics against a synthetic native vector environment."""

import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))

from f1_rl_lab.yaw_diagnostics import install_yaw_diagnostics


class NativeFixture:
    def __init__(self):
        n = 12
        self.command = torch.zeros(n, 3)
        self.command[:, 0] = 0.5
        # idx0 group0 pre-turn straight; idx10 group0 turning +0.1; idx1 group1 turning -0.1;
        # idx11 group1 standing; idx2 group2 zero command; idx4 group4 native nonzero.
        self.command[10, 2] = 0.1
        self.command[1, 2] = -0.1
        self.command[11, 2] = 0.1
        self.command[4, 2] = 0.03
        self.is_standing_env = torch.zeros(n, dtype=torch.bool)
        self.is_standing_env[11] = True
        self.time_left = torch.full((n,), 8.0)
        self.time_left[1] = 3.0
        self.time_left[10] = 3.0
        self.time_left[11] = 3.0
        self.cfg = SimpleNamespace(forward_exposure={"vx": 0.5, "group_modulus": 10,
                                                      "signed_turn": {"straight_s": 5.0, "yaw_rate": 0.1}})
        self.device = torch.device("cpu")
        self.num_envs = n
        self._turn_time_left = 5.0
        self.steps = 0


def build():
    term = NativeFixture()
    yaw_rate = torch.zeros(12, 3)
    yaw_rate[1, 2] = 0.25
    yaw_rate[2, 2] = 0.05
    yaw_rate[4, 2] = 0.2
    robot = SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=yaw_rate))
    unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term),
                                scene={"robot": robot})
    vec = SimpleNamespace(step=lambda actions: "stepped", unwrapped=unwrapped)
    alg = SimpleNamespace(update=lambda *a, **k: "updated")
    runner = SimpleNamespace(env=vec, alg=alg, current_learning_iteration=40)
    return runner, term, vec


class YawDiagnosticsTests(unittest.TestCase):
    def test_bucket_aggregation(self):
        runner, term, vec = build()
        native_step = vec.step
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as out:
            with install_yaw_diagnostics(runner, tmp, std=0.1, every=1) as audit:
                self.assertEqual(audit["method"], "log_only_signed_turn_diagnostics_v1")
                self.assertAlmostEqual(audit["dead_abs_error_rad_s"], 0.1 * math.sqrt(-math.log(0.05)))
                vec.step(None)  # one native step records the fixture state
                runner.alg.update()  # one update emits the window
            lines = [json.loads(l.split("[F1_YAW_DIAG]", 1)[1]) for l in out.getvalue().splitlines()
                     if "[F1_YAW_DIAG]" in l]
        self.assertEqual(len(lines), 2)  # window + final
        window = lines[0]
        self.assertEqual(window["iteration"], 40)
        dead = 0.1 * math.sqrt(-math.log(0.05))

        pos = window["positive_turn"]
        self.assertEqual(pos["samples"], 1)  # standing env 3 excluded
        self.assertAlmostEqual(pos["mean_error_rad_s"], 0.1)
        self.assertAlmostEqual(pos["mean_unweighted_reward"], math.exp(-1.0))
        self.assertEqual(pos["dead_fraction"], 0.0 if 0.1 < dead else 1.0)

        neg = window["negative_turn"]
        self.assertEqual(neg["samples"], 1)
        self.assertAlmostEqual(neg["mean_error_rad_s"], -0.35)
        self.assertAlmostEqual(neg["dead_fraction"], 1.0)

        straight = window["straight_prefix"]
        self.assertEqual(straight["samples"], 1)  # env 5 pre-turn
        self.assertAlmostEqual(straight["mean_command_rad_s"], 0.0)

        zero = window["zero_command"]
        self.assertEqual(zero["samples"], 7)  # groups 2,3,5,6,7,8,9 move with zero yaw
        self.assertAlmostEqual(zero["mean_actual_rad_s"], 0.05 / 7)

        native = window["native_nonzero"]
        self.assertEqual(native["samples"], 1)
        self.assertAlmostEqual(native["mean_abs_error_rad_s"], 0.17)
        self.assertAlmostEqual(native["dead_fraction"], 0.0)  # 0.17 < 0.1731

        self.assertEqual(lines[1]["final"], True)
        self.assertEqual(vec.step, native_step)  # wrappers restored

    def test_uniform_bucket_is_untouched(self):
        runner, term, vec = build()
        term.cfg.forward_exposure = None
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as out:
            with install_yaw_diagnostics(runner, tmp, std=0.1) as audit:
                self.assertIsNone(audit)
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()