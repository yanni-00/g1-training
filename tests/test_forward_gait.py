import importlib.util
from pathlib import Path
import unittest

import torch


path = Path(__file__).resolve().parents[1] / "source/f1_rl_lab/f1_rl_lab/forward_gait.py"
spec = importlib.util.spec_from_file_location("forward_gait", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ForwardGaitTests(unittest.TestCase):
    def test_speed_matching_preserves_credit_at_and_below_command_and_reduces_overspeed(self):
        command = torch.full((6,), 0.5)
        speeds = torch.tensor([-0.2, 0.0, 0.25, 0.5, 0.75, 1.0])
        eligible = torch.tensor([True, True, True, True, True, True])
        original = module._credit(eligible, command, speeds, 0.1)
        matched = module._credit(eligible, command, speeds, 0.1, match_speed=True)
        torch.testing.assert_close(matched[:4], original[:4])
        torch.testing.assert_close(matched[4:], torch.tensor([1.5, 1.0]))
        self.assertEqual(module._credit(~eligible, command, speeds, 0.1, match_speed=True).sum(), 0)

    def test_credit_requires_the_correct_single_support_and_favors_forward_motion(self):
        expected = torch.tensor([[True, False]] * 6 + [[True, True], [False, True]])
        actual = torch.tensor([
            [True, True], [False, False], [False, True], [True, False],
            [True, False], [True, False], [True, True], [False, True],
        ])
        command = torch.tensor([0.4] * 7 + [0.1])
        velocity = torch.tensor([0.4, 0.4, 0.4, 0, -0.2, 0.2, 0.4, 0.1])
        credit = module.forward_gait_credit(expected, actual, command, velocity, 0.1)
        torch.testing.assert_close(credit, torch.tensor([0, 0, 0, 1, 1, 1.5, 0, 2.0]))

    def test_actual_transfers_cannot_repeat_one_support_or_reward_flight(self):
        state = module.AlternatingSupport(1, "cpu")

        def sample(left, right, *, command=0.4, vx=0.2, swing=0.1):
            return state.reward(
                torch.tensor([[left, right]], dtype=torch.bool),
                torch.tensor([[swing, swing]]),
                torch.tensor([command]), torch.tensor([vx]), 0.1, 0.36,
            ).item()

        self.assertEqual(sample(True, True, command=0), 0)
        self.assertEqual(sample(True, False), 1.5)
        self.assertEqual(sample(True, False, swing=0.37), 0)
        self.assertEqual(sample(True, True), 0)
        self.assertEqual(sample(True, False), 0)
        self.assertEqual(sample(False, False), 0)
        self.assertEqual(sample(False, True), 0)
        self.assertEqual(sample(True, True), 0)
        self.assertEqual(sample(False, True, command=0.1, vx=0.1), 2)
        state.reset()
        self.assertEqual(sample(True, True), 0)
        self.assertEqual(sample(False, True, command=0.1, vx=0), 1)

    def test_reset_clears_only_the_selected_environment(self):
        state = module.AlternatingSupport(2, "cpu")
        command, vx = torch.full((2,), 0.4), torch.zeros(2)
        air = torch.zeros(2, 2)
        double = torch.ones(2, 2, dtype=torch.bool)
        left = torch.tensor([[True, False], [True, False]])
        self.assertEqual(state.reward(double, air, command, vx, 0.1, 0.36).tolist(), [0, 0])
        self.assertEqual(state.reward(left, air, command, vx, 0.1, 0.36).tolist(), [1, 1])
        state.reset([0])
        self.assertEqual(state.reward(double, air, command, vx, 0.1, 0.36).tolist(), [0, 0])
        self.assertEqual(state.reward(left, air, command, vx, 0.1, 0.36).tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
