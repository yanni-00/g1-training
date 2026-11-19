"""Protect moving commands and strict parent identity while changing standing cost."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_rl_lab"))
from f1_rl_lab.standing_smoothness import zero_command_ankle_action_rate_l2
from f1_rl_lab import continuation


class StandingSmoothnessTests(unittest.TestCase):
    def test_sampling_only_preserves_yaw_and_all_rewards(self):
        env=NS(rewards=NS(gait=NS(weight=.5),track_ang_vel_z=NS(weight=.5,params={'std':.5})),
               commands=NS(base_velocity=NS(ranges=NS(),rel_standing_envs=.02)))
        delta={'reward_override':{'gait':{'from':.5,'to':1.}},
               'reward_param_override':{'track_ang_vel_z':{'std':{'from':.5,'to':.1}}},
               'parent_reward_weights':{'gait':1.,'track_ang_vel_z':.5,'yaw_rate_error_l2':-.5},
               'retained_custom_rewards':{'yaw_rate_error_l2':-.5},'resume_command_ranges':{},
               'reward_addition':{},'command_sampling_override':{'rel_standing_envs':{'from':.02,'to':.2}}}
        with patch.object(continuation,'yaw_tracking_term',side_effect=lambda w:NS(weight=w)):
            working=copy.deepcopy(env);result=continuation.apply_delta(working,delta)
            self.assertEqual(result['reward_weights_after'],delta['parent_reward_weights'])
            self.assertEqual(working.commands.base_velocity.rel_standing_envs,.2)
            self.assertFalse(hasattr(working.rewards,'zero_command_ankle_action_rate_l2'))
            for value in [0.,1.,float('nan')]:
                bad=copy.deepcopy(delta);bad['command_sampling_override']['rel_standing_envs']['to']=value
                with self.assertRaises(ValueError):continuation.apply_delta(copy.deepcopy(env),bad)
            bad=copy.deepcopy(delta);bad['command_sampling_override']['rel_standing_envs']['from']=.1
            with self.assertRaises(ValueError):continuation.apply_delta(copy.deepcopy(env),bad)

    def test_pure_turn_translation_and_tiny_commands_are_not_penalized(self):
        commands = torch.tensor([[0.,0.,0.], [.5,0.,0.], [0.,.3,0.], [0.,0.,.2], [0.,0.,-1e-8]])
        action = torch.arange(30., dtype=torch.float32).reshape(5,6)
        old = torch.zeros_like(action)
        env = NS(command_manager=NS(get_command=lambda _: commands),
                 action_manager=NS(action=action.clone(), prev_action=old.clone()))
        result = zero_command_ankle_action_rate_l2(env, [1,3,4,5])
        torch.testing.assert_close(result, torch.tensor([51.,0.,0.,0.,0.]))
        torch.testing.assert_close(env.action_manager.action, action)
        torch.testing.assert_close(env.action_manager.prev_action, old)
        # Only ankle changes count; fixed ankle commands have no new cost.
        env.action_manager.prev_action[:, [1,3,4,5]] = action[:, [1,3,4,5]]
        self.assertTrue(bool((zero_command_ankle_action_rate_l2(env,[1,3,4,5]) == 0).all()))

    def test_retained_yaw_and_existing_rewards_cannot_silently_change(self):
        env = NS(rewards=NS(gait=NS(weight=.5), track_ang_vel_z=NS(weight=.5, params={"std":.5})),
                 commands=NS(base_velocity=NS(ranges=NS())))
        delta = {"reward_override":{"gait":{"from":.5,"to":1.}},
                 "reward_param_override":{"track_ang_vel_z":{"std":{"from":.5,"to":.1}}},
                 "retained_custom_rewards":{"yaw_rate_error_l2":-.5},
                 "parent_reward_weights":{"gait":1.,"track_ang_vel_z":.5,"yaw_rate_error_l2":-.5},
                 "reward_addition":{"zero_command_ankle_action_rate_l2":{"from":0.,"to":-.45}},
                 "resume_command_ranges":{}}
        with patch.object(continuation,"yaw_tracking_term",side_effect=lambda w:NS(weight=w)), \
             patch.object(continuation,"standing_smoothness_term",side_effect=lambda w:NS(weight=w)):
            result = continuation.apply_delta(copy.deepcopy(env),delta)
            self.assertEqual(result['reward_weights_after'],{**delta['parent_reward_weights'], 'zero_command_ankle_action_rate_l2':-.45})
            for key in ['parent_reward_weights','retained_custom_rewards']:
                bad=copy.deepcopy(delta);bad[key]['yaw_rate_error_l2']=-.6
                with self.assertRaises(ValueError):continuation.apply_delta(copy.deepcopy(env),bad)
            for value in [0.,.45,float('nan')]:
                bad=copy.deepcopy(delta);bad['reward_addition']['zero_command_ankle_action_rate_l2']['to']=value
                with self.assertRaises(ValueError):continuation.apply_delta(copy.deepcopy(env),bad)


if __name__ == '__main__': unittest.main()
