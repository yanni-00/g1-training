"""Command-relative cost/registration/identity checks; not simulator readiness."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
from f1_rl_lab import continuation
from f1_rl_lab.action_limits import action_source_binding, verify_runtime_action_contract
from f1_rl_lab.contract import load_contract, sha256
from f1_rl_lab.yaw_tracking import yaw_rate_error_l2


class YawTrackingTests(unittest.TestCase):
    def test_matching_turn_command_is_free_and_body_frame_is_explicit(self):
        angular = torch.tensor([[9., -4., .2], [9., -4., -.2], [0., 0., .3]])
        commands = torch.tensor([[0., 0., .2], [0., 0., -.2], [.5, 0., 0.]])
        env = NS(scene={"robot": NS(data=NS(root_ang_vel_b=angular, root_ang_vel_w=torch.ones_like(angular)*99))},
                 command_manager=NS(get_command=lambda name: commands))
        torch.testing.assert_close(yaw_rate_error_l2(env), torch.tensor([0., 0., .09]))
        angular[:, 2] *= -1; commands[:, 2] *= -1
        torch.testing.assert_close(yaw_rate_error_l2(env), torch.tensor([0., 0., .09]))

    def test_cost_keeps_increasing_outside_narrow_exponential_band(self):
        angular = torch.tensor([[0., 0., .1], [0., 0., .5], [0., 0., 1.]], requires_grad=True)
        env = NS(scene={"robot": NS(data=NS(root_ang_vel_b=angular))},
                 command_manager=NS(get_command=lambda name: torch.zeros(3, 3)))
        cost = yaw_rate_error_l2(env)
        self.assertTrue(bool((cost[1:] > cost[:-1]).all()))
        cost.sum().backward()
        torch.testing.assert_close(angular.grad[:, 2], 2 * angular.detach()[:, 2])

    def test_addition_preserves_parent_inventory_and_does_not_allow_positive_cost(self):
        env = NS(rewards=NS(gait=NS(weight=.5), track_ang_vel_z=NS(weight=.5, params={"std": .5})),
                 commands=NS(base_velocity=NS(ranges=NS())))
        delta = {"reward_override": {"gait": {"from": .5, "to": 1.}},
                 "reward_param_override": {"track_ang_vel_z": {"std": {"from": .5, "to": .1}}},
                 "parent_reward_weights": {"gait": 1., "track_ang_vel_z": .5},
                 "reward_addition": {"yaw_rate_error_l2": {"from": 0., "to": -.5}},
                 "resume_command_ranges": {}}
        with patch.object(continuation, "yaw_tracking_term", side_effect=lambda w: NS(weight=w)):
            result = continuation.apply_delta(copy.deepcopy(env), delta)
        self.assertEqual(result["reward_weights_after"], {**delta["parent_reward_weights"], "yaw_rate_error_l2": -.5})
        for target in [0., .5, float("nan")]:
            wrong = copy.deepcopy(delta); wrong["reward_addition"]["yaw_rate_error_l2"]["to"] = target
            with self.assertRaises(ValueError): continuation.apply_delta(copy.deepcopy(env), wrong)

    def test_v4_requires_exact_raw_action_and_custom_reward_sources(self):
        runtime = {"format": "f1_fullbody_runtime_v4", "action_clip": None,
                   "action_input_projection": None, "action_interface_sources": action_source_binding(ROOT, projection=True),
                   "training_objective": {"custom_reward_sources": {
                       "source/f1_rl_lab/f1_rl_lab/yaw_tracking.py": sha256(ROOT / "source/f1_rl_lab/f1_rl_lab/yaw_tracking.py")}}}
        verify_runtime_action_contract(runtime, load_contract(), ROOT)
        wrong = copy.deepcopy(runtime)
        wrong["training_objective"]["custom_reward_sources"]["source/f1_rl_lab/f1_rl_lab/yaw_tracking.py"] = "stale"
        with self.assertRaises(ValueError): verify_runtime_action_contract(wrong, load_contract(), ROOT)

    def test_nonfinite_native_loss_records_specific_return_field(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.std = torch.nn.Parameter(torch.ones(2))
        runner = NS(alg=NS(policy=Policy(), update=lambda: {"value_function": float("nan"), "surrogate": .1}),
                    current_learning_iteration=11999)
        with tempfile.TemporaryDirectory() as directory:
            continuation.install_finite_guard(runner, directory)
            with self.assertRaisesRegex(FloatingPointError, "value_function"): runner.alg.update()
            evidence = json.loads((Path(directory) / "params/numerical_failure.json").read_text())
            self.assertEqual(set(evidence["invalid_losses"]), {"value_function"})


if __name__ == "__main__": unittest.main()
