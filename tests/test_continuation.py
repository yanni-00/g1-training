import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("f1_continuation",ROOT/"source/f1_rl_lab/f1_rl_lab/continuation.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContinuationTests(unittest.TestCase):
    def test_contact_tilt_addition_preserves_actual_parent_objective(self):
        import copy
        env=NS(rewards=NS(gait=NS(weight=.5),track_ang_vel_z=NS(weight=.5,params={"std":.5})),
               commands=NS(base_velocity=NS(ranges=NS())))
        delta={"reward_override":{"gait":{"from":.5,"to":1.}},
               "parent_reward_weights":{"gait":1.,"track_ang_vel_z":.5},
               "reward_param_override":{"track_ang_vel_z":{"std":{"from":.5,"to":.1}}},
               "reward_addition":{"contact_lateral_tilt":{"from":0.,"to":-5.}},
               "resume_command_ranges":{"ang_vel_z":[-.1,.1]}}
        original=copy.deepcopy(env)
        from unittest.mock import patch
        with patch.object(module, "contact_tilt_term", return_value=NS(weight=-5.)) as build_term:
            result=module.apply_delta(env,delta)
        build_term.assert_called_once_with(-5., aggregation="equal_contact")
        self.assertEqual(result["reward_weights_after"],{**delta["parent_reward_weights"],"contact_lateral_tilt":-5.})
        self.assertEqual(env.rewards.track_ang_vel_z.params["std"],.1)
        for value in (0.,1.,float("nan")):
            bad=copy.deepcopy(delta);bad["reward_addition"]["contact_lateral_tilt"]["to"]=value
            with self.assertRaises(ValueError):module.apply_delta(copy.deepcopy(original),bad)

    def test_contact_tilt_and_zero_command_ankle_coexist_on_parent_inventory(self):
        import copy
        from unittest.mock import patch
        env=NS(rewards=NS(gait=NS(weight=.5),track_ang_vel_z=NS(weight=.5,params={"std":.3})),
               commands=NS(base_velocity=NS(ranges=NS())))
        parent={"gait":1.,"track_ang_vel_z":.5}
        delta={"reward_override":{"gait":{"from":.5,"to":1.}},
               "parent_reward_weights":parent,
               "reward_addition":{"contact_lateral_tilt":{"from":0.,"to":-5.,"aggregation":"positive_world_fz_moving"},
                                  "zero_command_ankle_action_rate_l2":{"from":0.,"to":-0.10}},
               "resume_command_ranges":{"ang_vel_z":[-.1,.1]}}
        original=copy.deepcopy(env)
        with patch.object(module,"contact_tilt_term",return_value=NS(weight=-5.)) as ct, \
             patch.object(module,"standing_smoothness_term",return_value=NS(weight=-.10)) as ss:
            result=module.apply_delta(env,delta)
        ct.assert_called_once_with(-5., aggregation="positive_world_fz_moving")
        ss.assert_called_once_with(-0.10)
        self.assertEqual(result["reward_weights_after"], {**parent,"contact_lateral_tilt":-5.,"zero_command_ankle_action_rate_l2":-0.10})
        for bad in (0.,float("nan"),1.):
            b=copy.deepcopy(delta); b["reward_addition"]["zero_command_ankle_action_rate_l2"]["to"]=bad
            with self.assertRaises(ValueError): module.apply_delta(copy.deepcopy(original),b)

    def test_yaw_width_preserves_actual_parent_weights_and_other_parameters(self):
        import copy
        env=NS(rewards=NS(gait=NS(weight=.5,params={"period":.8}),
                         track_ang_vel_z=NS(weight=.5,params={"std":.5,"command_name":"base_velocity"}),
                         track_lin_vel_xy=NS(weight=1.,params={"std":.5})),
               commands=NS(base_velocity=NS(ranges=NS())))
        delta={"reward_override":{"gait":{"from":.5,"to":1.}},
               "parent_reward_weights":{"gait":1.,"track_ang_vel_z":.5,"track_lin_vel_xy":1.},
               "reward_param_override":{"track_ang_vel_z":{"std":{"from":.5,"to":.1}}},
               "resume_command_ranges":{"ang_vel_z":[-.1,.1]}}
        original=copy.deepcopy(env)
        result=module.apply_delta(env,delta)
        self.assertEqual(env.rewards.track_ang_vel_z.params,{"std":.1,"command_name":"base_velocity"})
        self.assertEqual(env.rewards.track_lin_vel_xy.params,original.rewards.track_lin_vel_xy.params)
        self.assertEqual(env.rewards.gait.params,original.rewards.gait.params)
        self.assertEqual(result["reward_weights_after"],delta["parent_reward_weights"])
        for invalid in (0.,float("nan"),float("inf")):
            bad=copy.deepcopy(delta);bad["reward_param_override"]["track_ang_vel_z"]["std"]["to"]=invalid
            with self.assertRaises(ValueError):module.apply_delta(copy.deepcopy(original),bad)
        bad=copy.deepcopy(delta);bad["parent_reward_weights"]["gait"]=.5
        with self.assertRaises(ValueError):module.apply_delta(copy.deepcopy(original),bad)

    def test_only_registered_reward_changes_and_commands_restore(self):
        env=NS(rewards=NS(gait=NS(weight=.5),alive=NS(weight=.15)),
               commands=NS(base_velocity=NS(ranges=NS(lin_vel_x=(-.1,.1)))))
        delta={"reward_override":{"gait":{"from":.5,"to":1.}},
               "resume_command_ranges":{"lin_vel_x":[-.5,1.]}}
        result=module.apply_delta(env,delta)
        self.assertEqual(env.rewards.alive.weight,.15)
        self.assertEqual(env.rewards.gait.weight,1.)
        self.assertEqual(env.commands.base_velocity.ranges.lin_vel_x,(-.5,1.))
        self.assertFalse(result["episode_state_restored"])
        delta["reward_override"]["alive"]={"from":.15,"to":0}
        with self.assertRaises(ValueError):module.apply_delta(env,delta)

    def test_loaded_optimizer_rate_restores_adaptive_controller(self):
        state={"state":{0:{"step":200020}}}
        optimizer=NS(param_groups=[{"lr":2.2500000000000008e-5}],state_dict=lambda:state)
        runner=NS(alg=NS(optimizer=optimizer,learning_rate=.001),current_learning_iteration=10000)
        parent={"iteration":10000,"optimizerStep":200020,"optimizer_learning_rate":2.2500000000000008e-5}
        module.restore_adaptive_lr(runner,parent)
        self.assertEqual(runner.alg.learning_rate,optimizer.param_groups[0]["lr"])
        parent["optimizerStep"]+=1
        with self.assertRaises(ValueError):module.restore_adaptive_lr(runner,parent)

    def test_numerical_guard_stops_invalid_noise(self):
        import tempfile
        import torch
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.std=torch.nn.Parameter(torch.ones(2))
        policy=Policy()
        def invalid_update():
            with torch.no_grad():policy.std[0]=-1
            return {"loss":1.}
        runner=NS(alg=NS(policy=policy,update=invalid_update),current_learning_iteration=10000)
        with tempfile.TemporaryDirectory() as directory:
            module.install_finite_guard(runner,directory)
            with self.assertRaises(FloatingPointError):runner.alg.update()
            self.assertTrue((Path(directory)/"params/numerical_failure.json").is_file())


if __name__ == "__main__":unittest.main()


class CommandCycleTests(unittest.TestCase):
    def test_cycle_override_changes_only_the_registered_cycle(self):
        import copy
        env=NS(rewards=NS(gait=NS(weight=1.),track_ang_vel_z=NS(weight=.5,params={"std":.3})),
               commands=NS(base_velocity=NS(ranges=NS(),resampling_time_range=(10.,10.))))
        delta={"reward_override":{"gait":{"from":1.,"to":1.}},
               "parent_reward_weights":{"gait":1.,"track_ang_vel_z":.5},
               "reward_addition":{"contact_lateral_tilt":{"from":0.,"to":-5.}},
               "resume_command_ranges":{},
               "command_cycle_override":{"resampling_time_s":20.,"from":10.}}
        from unittest.mock import patch
        with patch.object(module, "contact_tilt_term", return_value=NS(weight=-5.)):
            result=module.apply_delta(env,delta)
        self.assertEqual(env.commands.base_velocity.resampling_time_range,(20.,20.))
        self.assertEqual(result["command_sampling_changes"]["command_cycle"],{"from":10.,"to":20.})
        bad=copy.deepcopy(delta); bad["command_cycle_override"]["resampling_time_s"]=0.
        with self.assertRaises(ValueError): module.apply_delta(copy.deepcopy(env),bad)
        wrong=copy.deepcopy(delta); wrong["command_cycle_override"]["from"]=12.
        with self.assertRaises(ValueError): module.apply_delta(copy.deepcopy(env),wrong)
