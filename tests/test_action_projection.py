"""Check the runaway-feedback failure boundary, not native simulator readiness."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
from f1_rl_lab.action_projection import install_projection, project_numpy, projection_contract
from f1_rl_lab.action_limits import action_source_binding, position_clip_contract, verify_runtime_action_contract
from f1_rl_lab.contract import load_contract
from f1_rl_lab.interface import joint_targets


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.spec = projection_contract(self.contract)

    def test_extreme_policy_samples_do_not_reenter_history_or_quadratic_cost(self):
        class Manager:
            action = torch.zeros(2, 29)
            prev_action = torch.zeros(2, 29)
            def process_action(self, action):
                self.prev_action = self.action.clone()
                self.action = action.clone()
        env = SimpleNamespace(device="cpu", action_manager=Manager())
        install_projection(env, self.spec, self.contract)
        for sign in (1, -1, 1):
            policy_sample = torch.full((2, 29), sign * 1e20)
            retained = policy_sample.clone()
            env.action_manager.process_action(policy_sample)
            torch.testing.assert_close(policy_sample, retained, rtol=0, atol=0)
            applied = env.action_manager.action
            np.testing.assert_allclose(applied.numpy(), project_numpy(policy_sample.numpy(), self.spec, self.contract))
            rate_cost = (applied - env.action_manager.prev_action).square().sum(-1)
            self.assertTrue(bool(torch.isfinite(rate_cost).all()))
            bound = torch.tensor(np.subtract(self.spec["upper"], self.spec["lower"])).square().sum()
            self.assertTrue(bool((rate_cost <= bound + 1e-3).all()))
        self.assertEqual(env.f1_action_projection_audit["calls"], 3)
        with self.assertRaises(FloatingPointError):
            env.action_manager.process_action(torch.full((2, 29), float("nan")))

    def test_projection_preserves_existing_saturated_physical_target(self):
        for raw in (np.zeros(29), np.full(29, -1e20), np.full(29, 1e20), np.linspace(-8, 8, 29)):
            applied = project_numpy(raw, self.spec, self.contract)
            np.testing.assert_allclose(joint_targets(applied, self.contract, position_clip_contract(self.contract)),
                                       joint_targets(raw, self.contract, position_clip_contract(self.contract)),
                                       rtol=1e-6, atol=1e-6)

    def test_projection_semantics_and_source_identity_cannot_be_silently_changed(self):
        runtime = {"format": "f1_fullbody_runtime_v3", "action_clip": position_clip_contract(self.contract, projected=True),
                   "action_input_projection": self.spec,
                   "action_interface_sources": action_source_binding(ROOT, projection=True)}
        verify_runtime_action_contract(runtime, self.contract, ROOT)
        wrong = copy.deepcopy(runtime)
        wrong["action_input_projection"]["lower"][0] -= .1
        with self.assertRaises(ValueError): verify_runtime_action_contract(wrong, self.contract, ROOT)
        wrong = copy.deepcopy(runtime)
        wrong["format"] = "f1_fullbody_runtime_v2"
        with self.assertRaises(ValueError): verify_runtime_action_contract(wrong, self.contract, ROOT)


if __name__ == "__main__": unittest.main()
