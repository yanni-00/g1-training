"""Independent FK/Jacobian checks and signal-processing guard tests."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source/f1_beyondmimic"))
from f1_beyondmimic.contract import load_contract, load_json
from model_geometry import forward_kinematics
from motion_kinematics import F1Kinematics, SIDES, orientation_angle
from repair_walking_motion import load_motion, smooth_non_leg_velocity


class MotionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.model = F1Kinematics(ROOT / cls.contract["urdf"])
        cls.settings = load_json(ROOT / "configs/motion_repair.json")

    def test_batched_fk_agrees_with_independent_scalar_urdf_fk(self):
        path = ROOT / "../../../f1-retargeting/F1/UMR/weichai_f1_29_accad_walking/B3_walk1_weichai_f1_29.npz"
        _, q, names, _ = load_motion(path)
        for index in (0, len(q)//2, len(q)-1):
            scalar = forward_kinematics(self.model.robot, dict(zip(names, q[index, 7:])))
            for side in SIDES:
                indices = self.model.leg_indices(side, names)
                p, r = self.model.leg_fk(side, q[index:index+1, 7:][:, indices])
                expected = scalar[side+"_ankle_roll"]
                np.testing.assert_allclose(p[0], expected[:3, 3], atol=1e-12)
                np.testing.assert_allclose(r[0], expected[:3, :3], atol=1e-12)

    def test_analytic_position_jacobian_matches_finite_difference(self):
        q = np.array([[.1, -.1, .15, .6, -.3, .1]])
        epsilon = 1e-7
        for side in SIDES:
            _, _, jacobian, _ = self.model.leg_fk(side, q, jacobian=True)
            for i in range(6):
                plus, minus = q.copy(), q.copy()
                plus[0, i] += epsilon
                minus[0, i] -= epsilon
                numerical = (self.model.leg_fk(side, plus)[0]-self.model.leg_fk(side, minus)[0])/(2*epsilon)
                np.testing.assert_allclose(jacobian[0, :, i], numerical[0], rtol=1e-6, atol=1e-8)

    def test_ik_recovers_known_reachable_poses(self):
        nominal = dict(zip(self.contract["model_joint_names"], self.contract["default_joint_pos"]))
        for side in SIDES:
            target_q = np.array([[nominal[j["name"]] for j in self.model.legs[side]]])
            p, r = self.model.leg_fk(side, target_q)
            initial = target_q + np.array([[.02, -.01, .015, .03, -.02, .01]])
            solved, state = self.model.solve_leg(side, initial, p, r, self.settings["ik"])
            self.assertTrue(state["converged"].all())
            actual_p, actual_r = self.model.leg_fk(side, solved)
            self.assertLessEqual(float(np.linalg.norm(actual_p-p)), self.settings["ik"]["position_tolerance_m"])
            self.assertLessEqual(float(orientation_angle(actual_r, r)[0]), self.settings["ik"]["orientation_tolerance_rad"])

    def test_filter_detects_alternating_jitter_and_preserves_other_joints(self):
        # Synthetic alternating noise: central differences alone miss its interior velocity.
        names = self.contract["model_joint_names"]
        q = np.zeros((120, 7+len(names)))
        q[:, 3] = 1
        joint = names.index("left_wrist_roll_joint")
        q[:, 7+joint] = np.where(np.arange(len(q)) % 2, .1, -.1)
        original = q.copy()
        changes = smooth_non_leg_velocity(self.model, q, names, 120, self.settings["non_leg_velocity_filter"])
        self.assertEqual([v["joint"] for v in changes], ["left_wrist_roll_joint"])
        self.assertLessEqual(float(np.abs(np.diff(q[:, 7+joint].astype(np.float32).astype(float))*120).max()), 2.5)
        unchanged = [i for i in range(q.shape[1]) if i != 7+joint]
        np.testing.assert_array_equal(q[:, unchanged], original[:, unchanged])
        self.assertGreaterEqual(q[:, 7+joint].min(), original[:, 7+joint].min())
        self.assertLessEqual(q[:, 7+joint].max(), original[:, 7+joint].max())


if __name__ == "__main__":
    unittest.main()
