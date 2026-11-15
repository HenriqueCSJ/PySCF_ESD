"""Synthetic scan data exercise electronic root and phase transport."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd.anharmonic_pyscf import track_anharmonic_scan
from pyscf_esd.state_tracking import StateTrackingError


class _Molecule:
    natm = 2

    def atom_symbol(self, index):
        return ("H", "F")[index]


def _point(ground, index, phase, direction=(1, 0, 0)):
    amplitudes = np.eye(3)
    amplitudes[0] = direction
    if index == 1:
        amplitudes[[0, 1]] = amplitudes[[1, 0]]
    amplitudes[index] *= phase
    dipoles = np.zeros((3, 3))
    dipoles[index] = (phase, 0, 0)
    return SimpleNamespace(
        mol=_Molecule(), nocc=1, mo_coeff=np.eye(4), settings="same recipe",
        ground_energy_hartree=ground, excitation_energies_hartree=np.array([0.1, 0.3, 0.5]),
        amplitudes=amplitudes[:, None, :], transition_dipoles_au=dipoles,
    )


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for cross-geometry overlaps")
class AnharmonicPySCFTests(unittest.TestCase):
    @patch("pyscf.gto.intor_cross", return_value=np.eye(4))
    def test_periodic_phase_and_reordering_on_common_energies(self, overlap):
        points = [_point(-6, 0, 1), _point(-5.99, 1, -1), _point(-5.98, 0, 1)]
        result = track_anharmonic_scan(points, target_index=0, periodic=True)
        np.testing.assert_allclose(result.ground_potential_hartree, [0.0, 0.01, 0.02])
        np.testing.assert_allclose(result.excited_potential_hartree, [0.1, 0.31, 0.12])
        np.testing.assert_array_equal(result.transition_dipole_au, [[1, 0, 0]] * 3)
        np.testing.assert_array_equal(result.root_indices, [0, 1, 0])
        np.testing.assert_array_equal(result.phases, [1, -1, 1])
        self.assertEqual(overlap.call_count, 4)

    @patch("pyscf.gto.intor_cross", return_value=np.eye(4))
    def test_rejects_nontrivial_periodic_sign_holonomy(self, overlap):
        points = []
        for angle in np.arange(8) * np.pi / 8:
            c, s = np.cos(angle), np.sin(angle)
            point = _point(-6, 0, 1, (c, s, 0))
            point.amplitudes[1, 0] = (-s, c, 0)
            points.append(point)
        opened = track_anharmonic_scan(points, target_index=0, periodic=False)
        np.testing.assert_array_equal(opened.phases, np.ones(8))
        with self.assertRaisesRegex(StateTrackingError, "does not close"):
            track_anharmonic_scan(points, target_index=0, periodic=True)


if __name__ == "__main__":
    unittest.main()
