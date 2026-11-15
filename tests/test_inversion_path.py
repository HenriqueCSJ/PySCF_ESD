"""Coordinate and rigid-motion checks without an electronic calculation."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.inversion_path import InversionPath
from pyscf_esd.inversion_space import InversionSpace


@unittest.skipUnless(find_spec("scipy"), "SciPy is required for the inversion arclength")
class InversionPathTests(unittest.TestCase):
    def setUp(self):
        self.ground = np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [-1.1, 1.7, 0.0], [-1.1, -1.7, 0.0]])
        excited = np.array([[-0.06, 0.0, 0.40], [2.3, 0.0, 0.05], [-1.05, 1.76, 0.07], [-1.05, -1.76, 0.07]])
        angle = 0.3
        rotation = np.array([[np.cos(angle), 0.0, -np.sin(angle)], [0.0, 1.0, 0.0],
                             [np.sin(angle), 0.0, np.cos(angle)]])
        self.ground += np.array([-2.0, 1.0, 3.0])
        self.excited = excited @ rotation + np.array([2.0, 1.0, -3.0])
        self.masses = np.array([12.0, 16.0, 1.0, 1.0])
        self.path = InversionPath(self.ground, self.excited, self.masses)

    def test_volume_contains_both_wells_and_the_planar_saddle(self):
        planar = np.array([[0., 0., 0.], [2.4, 0., 0.], [-1.15, 1.82, 0.], [-1.15, -1.82, 0.]])
        space = InversionSpace(self.path, planar)
        np.testing.assert_allclose(space.directions.T @ space.directions, np.eye(3), atol=1e-12)
        for q in (0., 1., -1.):
            np.testing.assert_allclose(space.geometry(q, q*q, 0.), self.path.geometry(q), atol=1e-12)
        reconstructed = space.geometry(*space.saddle_coordinates)
        np.testing.assert_allclose(reconstructed[:, 2], 0, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(reconstructed[:, None] - reconstructed[None, :], axis=-1),
            np.linalg.norm(planar[:, None] - planar[None, :], axis=-1), atol=1e-12,
        )
        np.testing.assert_allclose(space.geometry(-0.6, 0.4, 3.),
                                   space.geometry(0.6, 0.4, 3.) * [1, 1, -1], atol=1e-12)

    def test_alignment_endpoints_and_reflection(self):
        path = self.path
        ground = self.ground - np.average(self.ground, weights=self.masses, axis=0)
        excited = self.excited - np.average(self.excited, weights=self.masses, axis=0)
        np.testing.assert_allclose(path.geometry(0), ground, atol=2e-14)
        np.testing.assert_allclose(path.geometry(1), excited @ path.excited_rotation, atol=2e-14)
        self.assertAlmostEqual(np.linalg.det(path.excited_rotation), 1.0)
        self.assertGreater(path.geometry(1)[0, 2], 0)
        for q in (0.25, 1.0, 2.5):
            geometry = path.geometry(q)
            np.testing.assert_allclose(path.geometry(-q), geometry * [1, 1, -1], atol=1e-14)
            np.testing.assert_allclose(np.average(geometry, weights=self.masses, axis=0), 0, atol=1e-14)
            self.assertEqual(path.arclength(-q), -path.arclength(q))
            self.assertGreater(path.arclength(q), 0)

    def test_tangent_and_metric(self):
        path = self.path
        q = 0.7
        h = 1e-5
        tangent = path.tangent(q)
        geometry = path.geometry(q)
        fd_tangent = (path.geometry(q + h) - path.geometry(q - h)) / (2 * h)
        np.testing.assert_allclose(tangent, fd_tangent, rtol=1e-8, atol=1e-10)
        internal = path.local_tangent(q)
        self.assertAlmostEqual(np.linalg.norm(internal), 1.0)
        weighted_internal = np.sqrt(path.masses_au)[:, None] * internal
        np.testing.assert_allclose(weighted_internal.sum(axis=0), 0, atol=1e-13)
        np.testing.assert_allclose(np.cross(geometry, weighted_internal).sum(axis=0), 0, atol=1e-13)
        self.assertLess(path.metric(q), np.sum(path.masses_au[:, None] * tangent**2))
        self.assertAlmostEqual(path.metric(-q), path.metric(q))
        fd_speed = (path.arclength(q + h) - path.arclength(q - h)) / (2 * h)
        self.assertAlmostEqual(fd_speed, np.sqrt(path.metric(q)), places=7)

    def test_rejects_a_planar_well(self):
        with self.assertRaisesRegex(ValueError, "zero or undefined internal speed"):
            InversionPath(self.ground, self.ground, self.masses)


if __name__ == "__main__":
    unittest.main()
