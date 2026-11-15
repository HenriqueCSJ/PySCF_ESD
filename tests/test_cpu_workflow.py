"""State-specific derivative and restart wiring with synthetic electronic data."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd.cpu import Settings
from pyscf_esd.cpu_workflow import run_cpu_vibronic
from pyscf_esd.pyscf_adapter import VerticalGradientParameters
from pyscf_esd.state_tracking import StateTrackingError


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the workflow")
class CpuWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.geometry = np.zeros((2, 3))
        self.geometry[1, 2] = 2
        self.calls = []
        self.gradients = []
        self.model = VerticalGradientParameters(
            np.array([0.01]), np.array([2.0]), 0.1, np.array([1.0, 0, 0]), 0.001,
            np.ones((1, 2, 3)),
        )
        self.settings = Settings("synthetic", "hf", nstates=3)
        patches = (
            patch("pyscf.gto.intor_cross", return_value=np.eye(4)),
            patch("pyscf_esd.cpu_workflow.build_molecule", return_value=object()),
            patch("pyscf_esd.cpu_workflow.solve_point", side_effect=self.solve_point),
            patch("pyscf_esd.cpu_workflow.vertical_gradient", return_value=self.model),
        )
        for mock in patches:
            mock.start()
            self.addCleanup(mock.stop)

    def solve_point(self, symbols, coordinates, settings, *, previous=None):
        self.calls.append(coordinates.copy())
        displacement = (coordinates - self.geometry).ravel()
        displaced = np.any(displacement)
        sign = -1 if displacement.sum() < 0 else 1
        root = int(displaced)
        amplitudes = (np.eye(3) / np.sqrt(2)).reshape(3, 1, 3)
        if displaced:
            amplitudes[[0, 1]] = amplitudes[[1, 0]]
        amplitudes[root] *= sign
        dipoles = np.zeros((3, 3))
        dipoles[root, 0] = sign * (1 + displacement.sum())
        energies = np.array([0.1, 0.2, 0.3])
        if displaced:
            energies[1] += displacement @ displacement

        def gradient(index):
            self.gradients.append(index)
            if index != root:
                raise AssertionError("gradient evaluated at the wrong root")
            return 2 * displacement.reshape(2, 3)

        return SimpleNamespace(
            mol=object(), reference_stability=lambda: (True, True), ground_gradient=lambda: np.zeros((2, 3)),
            ground_hessian=lambda: np.zeros((2, 2, 3, 3)), excited_gradient=gradient,
            mo_coeff=np.eye(4), nocc=1, amplitudes=amplitudes, excitation_energies_hartree=energies,
            transition_dipoles_au=dipoles, ground_energy_hartree=-1.0, attempts=(),
            scf_residual=1e-12, tda_residuals=np.full(3, 1e-12),
            tda_route="davidson",
        )

    def run_workflow(self, folder, **options):
        return run_cpu_vibronic(("H", "H"), self.geometry, self.settings, target_index=0,
                                checkpoint_dir=folder, **options)

    def test_default_uses_center_only_and_returns_fc_sticks(self):
        with TemporaryDirectory() as folder:
            checkpoint_dir = Path(folder) / "stencil"
            result = self.run_workflow(checkpoint_dir)
            self.assertFalse(checkpoint_dir.exists())

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.gradients, [0])
        self.assertEqual(result.displacements, ())
        self.assertIsNone(result.energy_difference_gradient_hartree_per_bohr)
        self.assertIsNone(result.excited_hessian_hartree_per_bohr2)
        self.assertIsNone(result.hessian_asymmetry_hartree_per_bohr2)
        self.assertIsNone(result.dipole_jacobian_au_per_bohr)
        self.assertIsNone(result.normal_dipole_derivatives_au)
        np.testing.assert_array_equal(result.spectrum().dipole_strength_au2,
                                      result.spectrum(include_ht=False).dipole_strength_au2)
        with self.assertRaisesRegex(ValueError, "HT sticks require dipole derivatives"):
            result.spectrum(include_ht=True)

    def test_ht_only_cache_can_be_reused_and_upgraded_to_include_gradients(self):
        with TemporaryDirectory() as folder:
            ht_only = self.run_workflow(folder, include_ht=True)
            self.assertEqual(len(self.calls), 13)
            self.assertEqual(self.gradients, [0])
            self.assertIsNone(ht_only.excited_hessian_hartree_per_bohr2)
            self.assertIsNone(ht_only.hessian_asymmetry_hartree_per_bohr2)
            np.testing.assert_allclose(ht_only.normal_dipole_derivatives_au, [[6, 0, 0]])
            np.testing.assert_array_equal(ht_only.spectrum().dipole_strength_au2,
                                          ht_only.spectrum(include_ht=True).dipole_strength_au2)
            self.assertFalse(np.allclose(ht_only.spectrum().dipole_strength_au2,
                                         ht_only.spectrum(include_ht=False).dipole_strength_au2))

            checkpoint_path = Path(folder) / "axis_000_minus.npz"
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                original_config = checkpoint["config"].item()
                self.assertNotIn("gradient", checkpoint.files)

            self.calls.clear()
            self.gradients.clear()
            resumed = self.run_workflow(folder, include_ht=True)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.gradients, [0])
            self.assertTrue(all(point.cached for point in resumed.displacements))

            self.calls.clear()
            self.gradients.clear()
            hessian_only = self.run_workflow(folder, compute_excited_hessian=True)
            self.assertEqual(len(self.calls), 13)
            self.assertEqual(self.gradients, [0] + [1] * 12)
            self.assertIsNone(hessian_only.dipole_jacobian_au_per_bohr)
            self.assertIsNone(hessian_only.normal_dipole_derivatives_au)
            np.testing.assert_allclose(hessian_only.excited_hessian_hartree_per_bohr2, 2 * np.eye(6))
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                self.assertEqual(checkpoint["config"].item(), original_config)
                self.assertIn("gradient", checkpoint.files)

            self.calls.clear()
            self.gradients.clear()
            combined = self.run_workflow(folder, include_ht=True, compute_excited_hessian=True)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.gradients, [0])
            self.assertTrue(all(point.cached for point in combined.displacements))
            np.testing.assert_array_equal(combined.dipole_jacobian_au_per_bohr,
                                          ht_only.dipole_jacobian_au_per_bohr)
            np.testing.assert_array_equal(combined.excited_hessian_hartree_per_bohr2,
                                          hessian_only.excited_hessian_hartree_per_bohr2)

    def test_root_swaps_phase_and_partial_resume(self):
        with TemporaryDirectory() as folder:
            options = dict(include_ht=True, compute_excited_hessian=True)
            result = self.run_workflow(folder, **options)
            self.assertEqual(self.gradients, [0] + [1] * 12)
            self.assertEqual([p.match.phase for p in result.displacements], [-1, 1] * 6)
            np.testing.assert_allclose(result.dipole_jacobian_au_per_bohr[:, 0], 1)
            np.testing.assert_allclose(result.excited_hessian_hartree_per_bohr2, 2 * np.eye(6))

            (Path(folder) / "axis_000_minus.npz").unlink()
            self.calls.clear()
            self.gradients.clear()
            resumed = self.run_workflow(folder, **options)
            self.assertEqual(len(self.calls), 2)
            self.assertEqual(self.gradients, [0, 1])
            self.assertEqual(sum(p.cached for p in resumed.displacements), 11)
            np.testing.assert_array_equal(resumed.dipole_jacobian_au_per_bohr, result.dipole_jacobian_au_per_bohr)
            with self.assertRaisesRegex(ValueError, "different calculation"):
                self.run_workflow(folder, step_bohr=0.006, **options)

    def test_ambiguous_point_stops_before_gradient(self):
        with patch("pyscf_esd.cpu_workflow._match_point", side_effect=StateTrackingError("ambiguous")):
            with self.assertRaisesRegex(StateTrackingError, "axis 0, displacement -1: ambiguous"):
                self.run_workflow(None, compute_excited_hessian=True)
        self.assertEqual(self.gradients, [0])

    def test_completed_points_can_be_rephased_to_a_fresh_center(self):
        with TemporaryDirectory() as folder:
            options = dict(include_ht=True, compute_excited_hessian=True)
            result = self.run_workflow(folder, **options)
            original_solve = self.solve_point

            def reversed_center(symbols, coordinates, settings, *, previous=None):
                point = original_solve(symbols, coordinates, settings, previous=previous)
                np.testing.assert_array_equal(coordinates, self.geometry)
                point.amplitudes[0] *= -1
                point.transition_dipoles_au[0] *= -1
                return point

            with patch("pyscf_esd.cpu_workflow.solve_point", side_effect=reversed_center):
                resumed = self.run_workflow(folder, **options)
            np.testing.assert_array_equal(resumed.dipole_jacobian_au_per_bohr,
                                          -result.dipole_jacobian_au_per_bohr)


if __name__ == "__main__":
    unittest.main()
