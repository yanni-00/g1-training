"""Analytic software fixtures for periodic interpolation and angular velocity."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from export_ssot_motion import angular_velocity, periodic_pose


class PeriodicReferenceMath(unittest.TestCase):
    def test_periodic_quaternion_derivative_crosses_sign_seam(self):
        n, fps = 40, 20.
        rate = 2*np.pi/(n/fps)
        rotation = Rotation.from_rotvec(np.column_stack([np.zeros(n), np.zeros(n), np.arange(n)*rate/fps]))
        xyzw = rotation.as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        wxyz[7:18] *= -1
        actual = angular_velocity(wxyz[:, None, :], fps, periodic=True)
        expected = np.tile([0., 0., rate], (n, 1))[:, None, :]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_periodic_interpolation_preserves_nonnegative_knee_fixture(self):
        n, fps = 80, 80.
        phase = np.linspace(0., 1., n+1)
        q = np.zeros((n+1, 8))
        q[:, 0] = phase
        q[:, 2] = 1.
        q[:, 3] = 1.
        q[:, 7] = .2*(1-np.cos(2*np.pi*phase))
        output, speed, _ = periodic_pose(q, 0, n, fps, 50.)
        self.assertGreaterEqual(float(output[:, 7].min()), 0.)
        self.assertLessEqual(float(output[:, 7].max()), .4)
        np.testing.assert_allclose(output[:, 0], 0., atol=1e-14)
        self.assertEqual(speed, 1.)


if __name__ == '__main__':
    unittest.main()
