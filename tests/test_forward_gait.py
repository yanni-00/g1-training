import importlib.util
from pathlib import Path
import unittest

import torch


path = Path(__file__).resolve().parents[1] / "source/f1_rl_lab/f1_rl_lab/forward_gait.py"
spec = importlib.util.spec_from_file_location("forward_gait", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ForwardGaitTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
