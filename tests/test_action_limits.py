"""Failure-mode checks for the new action contract; not simulator readiness."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
from f1_rl_lab.action_limits import action_source_binding, position_clip_contract, verify_runtime_action_contract
from f1_rl_lab.contract import load_contract
from f1_rl_lab.continuation import apply_delta
from f1_rl_lab.interface import ActorHistory, joint_targets


class ActionLimitTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.clip = position_clip_contract(self.contract)

    def test_both_limit_boundaries_and_interior_match_torch_processed_positions(self):
        for raw in (np.zeros(29, dtype=np.float32), np.full(29, -100, dtype=np.float32),
                    np.full(29, 100, dtype=np.float32), np.linspace(-4, 4, 29, dtype=np.float32)):
            original = raw.copy()
            unbounded = np.array(self.contract["default_joint_pos"]) + raw * self.contract["action_scale"]
            np.testing.assert_array_equal(joint_targets(raw, self.contract), unbounded)
            expected = torch.clamp(torch.tensor(raw) * self.contract["action_scale"]
                                   + torch.tensor(self.contract["default_joint_pos"]),
                                   min=torch.tensor(self.clip["lower"]), max=torch.tensor(self.clip["upper"]))
            np.testing.assert_allclose(joint_targets(raw, self.contract, self.clip), expected.numpy(), rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(raw, original)

    def test_raw_last_action_history_is_preserved_when_targets_saturate(self):
        raw = np.full(29, 100, dtype=np.float32)
        clipped = joint_targets(raw, self.contract, self.clip)
        self.assertTrue(np.all(clipped <= self.clip["upper"]))
        terms = {term["name"]: np.zeros(term["dim"], dtype=np.float32)
                 for term in self.contract["observation_terms"]}
        terms["last_action"] = raw
        observation = ActorHistory(self.contract).push(terms)
        np.testing.assert_array_equal(observation[-5 * 29:].reshape(5, 29), np.tile(raw, (5, 1)))

    def test_changed_joint_order_or_limits_are_rejected(self):
        for key in ("joint_names", "lower", "upper"):
            wrong = copy.deepcopy(self.clip)
            wrong[key] = wrong[key][::-1]
            if wrong == self.clip:
                wrong[key][0] += 0.1
            with self.assertRaises(ValueError):
                joint_targets(np.zeros(29), self.contract, wrong)

    def test_v2_execution_source_binding_rejects_stale_code_and_keeps_legacy_rule(self):
        verify_runtime_action_contract({"action_clip": None}, self.contract, ROOT)
        runtime = {"format": "f1_fullbody_runtime_v2", "action_clip": self.clip,
                   "action_interface_sources": action_source_binding(ROOT)}
        verify_runtime_action_contract(runtime, self.contract, ROOT)
        runtime["action_interface_sources"][next(iter(runtime["action_interface_sources"]))] = "stale"
        with self.assertRaisesRegex(ValueError, "source binding"):
            verify_runtime_action_contract(runtime, self.contract, ROOT)

    def test_continuation_changes_only_native_target_clip_not_parent_rewards(self):
        parent = json.loads((ROOT / "configs/training_delta.json").read_text(encoding="utf-8"))["parent_reward_weights"]
        rewards = {name: NS(weight=weight) for name, weight in parent.items()}
        rewards["gait"].weight = .5
        rewards["track_ang_vel_z"].params = {"std": .5}
        env = NS(rewards=NS(**rewards), actions=NS(JointPositionAction=NS(clip=None)),
                 commands=NS(base_velocity=NS(ranges=NS())))
        delta = {"reward_override": {"gait": {"from": .5, "to": 1.}},
                 "reward_param_override": {"track_ang_vel_z": {"std": {"from": .5, "to": .1}}},
                 "parent_reward_weights": parent, "resume_command_ranges": {"lin_vel_x": [-.5, 1.]},
                 "action_postprocessing_override": {"parent_action_clip": None, "to": self.clip}}
        result = apply_delta(env, delta)
        self.assertEqual(result["reward_weights_after"], parent)
        self.assertEqual(env.actions.JointPositionAction.clip,
                         {name: (lo, hi) for name, lo, hi in zip(self.clip["joint_names"], self.clip["lower"], self.clip["upper"])})
        with self.assertRaises(ValueError):
            apply_delta(env, delta)


if __name__ == "__main__":
    unittest.main()
