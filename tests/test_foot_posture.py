import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("foot_posture", ROOT/"source/f1_rl_lab/f1_rl_lab/foot_posture.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FootPostureTests(unittest.TestCase):
    def test_roll_contact_gating_and_mirror(self):
        # Test fixtures, not measurements: roll +/-30deg has sin²=1/4.
        angle = torch.tensor(torch.pi/12, dtype=torch.float64)
        q = torch.tensor([[[angle.cos(), angle.sin(), 0., 0.],
                           [angle.cos(), -angle.sin(), 0., 0.]]], dtype=torch.float64)
        for contacts in ([[True, True]], [[True, False]], [[False, True]]):
            actual = module.contact_lateral_tilt_l2(q, torch.tensor(contacts))
            self.assertTrue(torch.allclose(actual, torch.tensor([.25], dtype=torch.float64)))
        self.assertEqual(module.contact_lateral_tilt_l2(q, torch.zeros(1, 2, dtype=torch.bool)).item(), 0.)
        self.assertTrue(torch.allclose(module.contact_lateral_tilt_l2(q, torch.ones(1, 2, dtype=torch.bool)),
                                       module.contact_lateral_tilt_l2(-q, torch.ones(1, 2, dtype=torch.bool))))

    def test_pure_pitch_and_yaw_do_not_cost_lateral_tilt(self):
        angle = torch.tensor(.3, dtype=torch.float64)
        q = torch.tensor([[[angle.cos(), 0., angle.sin(), 0.],
                           [angle.cos(), 0., 0., angle.sin()]]], dtype=torch.float64)
        self.assertEqual(module.contact_lateral_tilt_l2(q, torch.ones(1, 2, dtype=torch.bool)).item(), 0.)

    def test_gradient_and_bound(self):
        torch.manual_seed(42)
        raw = torch.randn(128, 2, 4, dtype=torch.float64, requires_grad=True)
        q = raw/raw.norm(dim=-1, keepdim=True)
        result = module.contact_lateral_tilt_l2(q, torch.ones(128, 2, dtype=torch.bool))
        self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))
        result.mean().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool((raw.grad.abs().sum(dim=(1, 2)) > 0).all()))


if __name__ == "__main__":
    unittest.main()
