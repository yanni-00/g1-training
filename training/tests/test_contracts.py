"""Synthetic software fixtures only: no trained policy, FK acceptance or locomotion."""
import copy
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/f1_beyondmimic"))
from f1_beyondmimic.contract import (FORMAT, ROOT, assert_can_run, binding, effective_control,
    load_contract, load_json, prepared_motion, resolve, sha256, validate_motion, reorder_motion_arrays)
from f1_beyondmimic.utils.actor_observations import contract, proprioceptive_observation, audit_native_observations
from f1_beyondmimic.utils.runtime_policy import RuntimeActor, verify_runtime_checkpoint
from f1_beyondmimic.utils.velocity_gait_reference import VelocityGaitReference
from f1_beyondmimic.utils.checkpoint import load_checkpoint, position_targets


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def temporary_inputs(directory):
    root = Path(directory)
    c = load_contract()
    dump(root / "configs/robot_contract.json", c)
    destination = root / c["urdf"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / c["urdf"], destination)
    dump(root / "configs/reference/stable_runtime_settings.json", load_json(ROOT / "configs/reference/stable_runtime_settings.json"))
    return root, c


def fixture(c, frames=8):
    # Values below are arbitrary analytic fixtures, never hardware parameters.
    bodies, names = c["link_names"], c["model_joint_names"]
    positions = np.zeros((frames, len(bodies), 3), dtype=np.float32)
    quat = np.zeros((frames, len(bodies), 4), dtype=np.float32)
    quat[..., 0] = 1
    return dict(fps=np.array([1 / load_json(ROOT / "configs/reference/stable_runtime_settings.json")["policy_dt"]]),
                cycle_frames=np.array([frames]), support_frames=np.array([0, frames // 2]),
                nominal_reference_speed_mps=np.array([0.5]), urdf_sha256=np.array(c["urdf_sha256"]),
                reference_space=np.array("cyclic_pelvis_xy_local_v1"), joint_names=np.array(names), body_names=np.array(bodies),
                joint_pos=np.tile(c["default_joint_pos"], (frames, 1)).astype(np.float32),
                joint_vel=np.zeros((frames, len(names)), dtype=np.float32),
                body_pos_w=positions, body_quat_w=quat,
                body_lin_vel_w=np.zeros_like(positions), body_ang_vel_w=np.zeros_like(positions))


class InputContracts(unittest.TestCase):
    def test_windows_audit_paths_resolve_under_linux_semantics(self):
        # PurePosixPath reproduces Linux even when this test runs on Windows.
        root = PurePosixPath('/workspace/training')
        with patch('f1_beyondmimic.contract.Path', PurePosixPath):
            actual = resolve(r'data\regenerated\20261109\final\walking_cycle_50hz.npz', root)
        self.assertEqual(actual, root/'data/regenerated/20261109/final/walking_cycle_50hz.npz')

    def test_native_reference_name_binding_and_coverage(self):
        c = load_contract()
        a = fixture(c)
        for i in range(len(c['link_names'])):
            a['body_pos_w'][:, i, 0] = i
            a['body_lin_vel_w'][:, i, 1] = -i
        joint_order = c['model_joint_names'][::-1]
        body_order = c['link_names'][::-1]
        reordered = reorder_motion_arrays(a, joint_order, body_order)
        np.testing.assert_array_equal(reordered['joint_pos'][:, ::-1], a['joint_pos'])
        np.testing.assert_array_equal(reordered['body_pos_w'][:, ::-1], a['body_pos_w'])
        np.testing.assert_array_equal(reordered['body_lin_vel_w'][:, ::-1], a['body_lin_vel_w'])
        self.assertEqual(reordered['body_names'].tolist(), body_order)
        self.assertEqual(a['body_names'].tolist(), c['link_names'])
        for bad in (body_order[:-1], body_order[:-1]+[body_order[0]]):
            with self.assertRaisesRegex(ValueError, 'coverage'):
                reorder_motion_arrays(a, joint_order, bad)

    def test_stride_local_reference_marker_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root, c = temporary_inputs(directory)
            arrays = fixture(c)
            arrays['reference_space'] = np.array('cyclic_heading_stride_local_v1')
            path = root/'reference.npz'
            np.savez(path, **arrays)
            self.assertEqual(str(validate_motion(path, root)['reference_space']), 'cyclic_heading_stride_local_v1')
            arrays['reference_space'] = np.array('arbitrary_unregistered_frame')
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'reference space'):
                validate_motion(path, root)

    def test_current_scaffold_rejects_execution_before_isaac(self):
        def disabled_json(path):
            value = load_json(path)
            if Path(path).name == 'launch.json':
                value = {**value, 'training_enabled': False, 'evaluation_enabled': False}
            return value
        with patch('f1_beyondmimic.contract.load_json', side_effect=disabled_json):
            for scope in ("training", "evaluation"):
                with self.assertRaisesRegex(ValueError, "disabled"):
                    assert_can_run(scope)
        self.assertNotIn("isaaclab", sys.modules)
        controls = effective_control()
        self.assertEqual(set(controls), set(load_contract()['model_joint_names']))
        self.assertEqual(load_json(ROOT/'configs/parameter_identification.json')['status'], 'pending')
        self.assertTrue(prepared_motion().is_file())

    def test_owner_control_baseline_needs_exact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root, c = temporary_inputs(directory)
            source = ROOT/'configs/owner_training_profile.json'
            shutil.copy2(source, root/'configs/owner_training_profile.json')
            profile = load_json(ROOT/'configs/control_profile.json')
            dump(root/'configs/control_profile.json', profile)
            identification = load_json(ROOT/'configs/parameter_identification.json')
            dump(root/'configs/parameter_identification.json', identification)
            controls = effective_control(root)
            expected = load_json(source)
            for name, values in expected['joints'].items():
                self.assertEqual(controls[name]['stiffness'], values['stiffness'])
                self.assertEqual(controls[name]['pd_damping'], values['pd_damping'])
                self.assertEqual(controls[name]['total_damping'], values['pd_damping']+c['joint_properties'][name]['dynamics']['damping'])
            self.assertEqual(load_json(root/'configs/parameter_identification.json')['status'], 'pending')
            with (root/'configs/owner_training_profile.json').open('a') as stream:
                stream.write('\n ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                effective_control(root)

    def test_identification_requires_bound_complete_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root, c = temporary_inputs(directory)
            report = root / "measurement.txt"
            report.write_text("Synthetic unit test source; not hardware data.", encoding="utf-8")
            data = dict(status="validated", urdf_sha256=c["urdf_sha256"], delivery_filter=None,
                        sources=[dict(path="measurement.txt", sha256=sha256(report), description="synthetic test")],
                        joints={name: dict(stiffness=1.0, pd_damping=0.0, armature=0.0) for name in c["model_joint_names"]})
            path = root / "configs/parameter_identification.json"
            dump(path, data)
            controls = effective_control(root)
            self.assertEqual(set(controls), set(c["model_joint_names"]))
            name = c["model_joint_names"][0]
            self.assertEqual(controls[name]["passive_damping"], c["joint_properties"][name]["dynamics"]["damping"])
            for invalid in (None, -1, float("nan"), True):
                bad = copy.deepcopy(data)
                bad["joints"][name]["stiffness"] = invalid
                dump(path, bad)
                with self.assertRaises(ValueError):
                    effective_control(root)
            dump(path, data)
            report.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                effective_control(root)

    def test_raw_umr_is_not_a_prepared_reference(self):
        manifest = load_json(ROOT / "data/motion_manifest.json")
        raw = ROOT / manifest["raw_source"] / manifest["source_files"][0]["name"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_motion(raw)

    def test_prepared_reference_rejects_wrong_model_limits_and_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root, c = temporary_inputs(directory)
            arrays = fixture(c)
            path = root / "fixture.npz"
            np.savez(path, **arrays)
            result = validate_motion(path, root)
            self.assertEqual(result["joint_names"].tolist(), c["model_joint_names"])
            for field, value in [("urdf_sha256", np.array("wrong")),
                                 ("support_frames", np.array([0.25, 4.0])),
                                 ("joint_pos", arrays["joint_pos"] + 100),
                                 ("body_quat_w", np.zeros_like(arrays["body_quat_w"]))]:
                bad = dict(arrays)
                bad[field] = value
                np.savez(path, **bad)
                with self.assertRaises(ValueError):
                    validate_motion(path, root)


class ReferenceContracts(unittest.TestCase):
    def setUp(self):
        arrays = fixture(load_contract())
        self.engine = VelocityGaitReference(arrays, 1, ((0, .5), (0, 0), (-.2, .2)))
        self.engine.reset([0], [0], [[0, 0]], [0])
        self.dt = 1 / float(arrays["fps"][0])

    def test_continuous_cycles_stop_hold_restart_without_root_reset(self):
        e = self.engine
        previous = e.clock.unwrapped_phase.clone()
        for command, steps in [([.5, 0, 0], 120), ([.25, 0, 0], 80), ([0, 0, 0], 400)]:
            e.request([command], [True])
            for _ in range(steps):
                e.step(self.dt)
                self.assertTrue(torch.all(e.clock.unwrapped_phase >= previous))
                previous = e.clock.unwrapped_phase.clone()
        self.assertGreater(int(e.clock.cycles[0]), 1)
        self.assertEqual(float(e.clock.rate[0]), 0)
        self.assertIn(float(e.clock.phase[0]), e.clock.support.tolist())
        held = e.body_pos_w.clone()
        for _ in range(20):
            e.step(self.dt)
        torch.testing.assert_close(e.body_pos_w, held, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(e.joint_vel), 0)
        self.assertEqual(torch.count_nonzero(e.body_lin_vel_w), 0)
        position = e.root.position.clone()
        e.request([[.5, 0, 0]], [True])
        e.step(self.dt)
        self.assertTrue(torch.all(e.clock.unwrapped_phase > previous))
        self.assertGreater(float(e.root.position[0, 0]), float(position[0, 0]))

    def test_first_version_rejects_backward_lateral_and_in_place_turn(self):
        for command in [[-.2, 0, 0], [.2, .1, 0], [0, 0, .1], [float("nan"), 0, 0]]:
            with self.assertRaises(ValueError):
                self.engine.request([command], [True])

    def test_turn_and_zero_yaw_heading_hold(self):
        e = self.engine
        e.request([[.5, 0, .1]], [True])
        e.initialize_rate([0])
        for _ in range(50):
            e.step(self.dt)
        self.assertGreater(float(e.root.yaw[0]), 0)
        yaw = e.root.yaw.clone()
        e.request([[.5, 0, 0]], [True])
        for _ in range(50):
            e.step(self.dt)
        torch.testing.assert_close(e.root.yaw, yaw, rtol=0, atol=0)


class Normalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("_mean", torch.full((1, size), .1))
        self.register_buffer("_std", torch.full((1, size), .5))
        self.eps = .01

    def forward(self, obs):
        return (obs - self._mean) / (self._std + self.eps)


class ActorContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        count = len(load_contract()["model_joint_names"])
        self.policy = nn.Module()
        self.policy.actor = nn.Sequential(nn.Linear(contract()["dimension"], 8), nn.ELU(), nn.Linear(8, count))
        self.policy.actor_obs_normalizer = Normalizer(contract()["dimension"])
        self.policy.actor_obs_normalization = True
        self.runtime = dict(format=FORMAT, urdf_sha256=load_contract()["urdf_sha256"], binding=binding(),
                            actor_observation_contract=contract(), actor_obs_normalization=True,
                            normalizer_epsilon=self.policy.actor_obs_normalizer.eps)
        self.checkpoint = dict(model_state_dict=self.policy.state_dict(), infos={"f1_runtime": self.runtime})

    def test_normalized_export_parity_and_training_flags(self):
        self.policy.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save(self.checkpoint, path)
            verify_runtime_checkpoint(self.policy, path)
            self.assertTrue(self.policy.training)
            actor = RuntimeActor(load_checkpoint(path)).eval()
            inputs = torch.randn(3, actor.observation_dim)
            torch.testing.assert_close(actor(inputs), self.policy.actor(self.policy.actor_obs_normalizer(inputs)))
            script = torch.jit.trace(actor, torch.zeros(1, actor.observation_dim))
            torch.testing.assert_close(script(inputs), actor(inputs))
            bad = copy.deepcopy(self.checkpoint)
            bad["infos"]["f1_runtime"]["format"] = "beyondmimic_g1_velocity_proprio_runtime_v1"
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, "F1 task"):
                load_checkpoint(path)

    def test_observation_assembly_and_truth_leakage_rejection(self):
        import types
        count = len(load_contract()["model_joint_names"])
        q, dq, ref, dref, prev, default = [np.arange(count, dtype=np.float32) * .01 for _ in range(6)]
        obs = proprioceptive_observation(ref, dref, np.eye(3), np.zeros(3), q, dq, prev, default)
        self.assertEqual(obs.shape, (contract()["dimension"],))
        cfg = types.SimpleNamespace(motion_anchor_pos_b=None, base_lin_vel=None)
        expected = contract()
        body_count = len(load_contract()["tracked_body_names"])
        manager = types.SimpleNamespace(active_terms={"policy": expected["terms"]},
            group_obs_dim={"policy": (expected["dimension"],), "critic": (expected["dimension"] + 6 + 9 * body_count,)})
        env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(observation_manager=manager,
            cfg=types.SimpleNamespace(observations=types.SimpleNamespace(policy=cfg),
                                     commands=types.SimpleNamespace(motion=types.SimpleNamespace(body_names=list(range(body_count)))))))
        audit_native_observations(env)
        cfg.base_lin_vel = object()
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            audit_native_observations(env)

    def test_target_equation_hard_limits(self):
        runtime = {"joint_names": ["a", "b"], "control": {
            "default_joint_pos": torch.tensor([0., .1]), "action_scale": torch.tensor([.5, 1.]),
            "joint_lower": torch.tensor([-.2, -.3]), "joint_upper": torch.tensor([.3, .4])}}
        torch.testing.assert_close(position_targets([[10, -10]], runtime), torch.tensor([[.3, -.3]]))

    def test_export_cli_preserves_source_and_refuses_replacement(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "synthetic.pt"
            torch.save(self.checkpoint, checkpoint)
            source_hash = sha256(checkpoint)
            output = directory / "new-package"
            command = [sys.executable, str(ROOT / "scripts/export.py"), "--checkpoint", str(checkpoint),
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = load_json(output / "manifest.json")
            self.assertEqual(manifest["source_checkpoint_sha256"], source_hash)
            self.assertFalse(manifest["hardware_deployment_approved"])
            self.assertEqual(sha256(checkpoint), source_hash)
            self.assertNotEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
