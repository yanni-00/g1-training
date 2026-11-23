"""Check failure transitions against the native vector environment interface."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))

from f1_rl_lab.contract import load_contract
from f1_rl_lab.rollout_guard import full_travel_raw_action_sq
from f1_rl_lab.rollout_recovery import TERM_NAME, raw_action_failure, recover_rollout_actions


class NativeRolloutFixture:
    clip_actions = None

    def __init__(self):
        contract = load_contract()
        joint_count = len(contract["model_joint_names"])
        self.timeout = torch.ones(3, dtype=torch.bool)
        self.termination_manager = SimpleNamespace(active_terms=[TERM_NAME], terminated=None)
        self.action_manager = SimpleNamespace(action=torch.zeros(3, joint_count))
        self.episode_length_buf = torch.ones(3, dtype=torch.long)
        self.command_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(
            cfg=SimpleNamespace(forward_exposure={"group_modulus": 10}), command=torch.zeros(3, 3)
        ))
        self.unwrapped = self
        self.forwarded_actions = []

        self.action = self.action_manager.action.clone()
        self.action[0, 0] = full_travel_raw_action_sq(contract) ** 0.5 + 1
        self.action[1:, 0] = 0.4
        self.action_manager.action.copy_(self.action)  # No jump, only sustained magnitude.
        transition = SimpleNamespace(
            actions=self.action, action_mean=self.action.clone(), action_sigma=torch.ones_like(self.action),
            values=torch.zeros(3, 1), actions_log_prob=torch.zeros(3), observations={"policy": torch.zeros(3, 4)},
        )
        self.runner = SimpleNamespace(
            env=self, alg=SimpleNamespace(transition=transition, update=lambda: {"loss": 0.5}),
            num_steps_per_env=2, current_learning_iteration=12700,
        )

    def step(self, actions):
        self.forwarded_actions.append(actions)
        self.action_manager.action.copy_(actions)
        terminated = self.termination_manager.terminated = raw_action_failure(self)
        dones = terminated | self.timeout
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        self.action_manager.action[dones] = 0
        return {}, torch.ones(3), dones.long(), {"time_outs": self.timeout}


class RolloutRecoveryTests(unittest.TestCase):
    def test_finite_excursion_fails_only_its_row_without_bootstrapping(self):
        env = NativeRolloutFixture()
        env.timeout[1] = False
        original_action = env.action.clone()
        native_step = env.step
        native_update = env.runner.alg.update
        with TemporaryDirectory() as path, recover_rollout_actions(env.runner, path) as audit:
            self.assertEqual(audit["max_recoveries_per_update"], 2)
            _, _, dones, extras = env.step(env.action)
            self.assertEqual(dones.tolist(), [1, 0, 1])
            self.assertEqual(extras["time_outs"].tolist(), [False, False, True])
            self.assertEqual(env.episode_length_buf.tolist(), [0, 2, 0])
            self.assertEqual(env.termination_manager.terminated.tolist(), [True, False, False])
            self.assertFalse(raw_action_failure(env).any())
            torch.testing.assert_close(env.forwarded_actions[-1][0], torch.zeros_like(env.action[0]))
            torch.testing.assert_close(env.forwarded_actions[-1][1:], original_action[1:])
            torch.testing.assert_close(env.runner.alg.transition.actions, original_action)
            torch.testing.assert_close(env.runner.alg.transition.actions_log_prob, torch.zeros(3))
            event = json.loads((Path(path) / "params/rollout_recovery.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["action_delta_sq"], 0.0)
            self.assertGreater(event["action_sq"], event["full_travel_raw_action_sq"])

            regular = torch.ones_like(env.action) * 0.1
            env.step(regular)
            self.assertIs(env.forwarded_actions[-1], regular)
        self.assertEqual(env.step, native_step)
        self.assertEqual(env.runner.alg.update, native_update)

    def test_recovery_budget_resets_on_update_and_fails_closed_on_burst(self):
        env = NativeRolloutFixture()
        with TemporaryDirectory() as path, recover_rollout_actions(env.runner, path):
            env.step(env.action)
            env.step(env.action)
            self.assertEqual(env.runner.alg.update(), {"loss": 0.5})
            env.step(env.action)
            env.step(env.action)
            with self.assertRaisesRegex(FloatingPointError, "recovery_budget_exceeded"):
                env.step(env.action)
            self.assertEqual(len(env.forwarded_actions), 4)
            report = json.loads((Path(path) / "params/rollout_outlier.json").read_text(encoding="utf-8"))
            self.assertEqual(report["reason"], "recovery_budget_exceeded")
            self.assertEqual(report["recoveries_this_update_before_step"], 2)

    def test_nonfinite_action_or_ppo_sample_aborts_before_physics(self):
        for field in ("action", "value", "log_prob"):
            with self.subTest(field=field), TemporaryDirectory() as path:
                env = NativeRolloutFixture()
                if field == "action":
                    env.action[1, 0] = float("nan")
                elif field == "value":
                    env.runner.alg.transition.values[1, 0] = float("inf")
                else:
                    env.runner.alg.transition.actions_log_prob[1] = float("-inf")
                with recover_rollout_actions(env.runner, path):
                    with self.assertRaisesRegex(FloatingPointError, "nonfinite_action_or_ppo_sample"):
                        env.step(env.action)
                self.assertEqual(env.forwarded_actions, [])

    def test_missing_native_termination_fails_closed(self):
        env = NativeRolloutFixture()
        env.termination_manager.active_terms.clear()
        with TemporaryDirectory() as path, self.assertRaisesRegex(ValueError, "native failure term"):
            with recover_rollout_actions(env.runner, path):
                self.fail("Recovery enabled without native resets")

    def test_simulator_reset_is_required_after_a_fallback(self):
        env = NativeRolloutFixture()
        env.step = lambda _: ({}, torch.ones(3), torch.zeros(3), {})
        with TemporaryDirectory() as path, recover_rollout_actions(env.runner, path):
            with self.assertRaisesRegex(FloatingPointError, "invalid_native_reset_transition"):
                env.step(env.action)
            self.assertFalse(raw_action_failure(env).any())


if __name__ == "__main__":
    unittest.main()
