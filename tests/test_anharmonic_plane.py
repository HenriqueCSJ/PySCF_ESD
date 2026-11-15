"""Check the sine Galerkin spectrum against analytically or directly assembled matrices."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.anharmonic_plane import SineGalerkin2D


@unittest.skipUnless(find_spec("scipy"), "SciPy is required for the anharmonic plane solver")
class SineGalerkinTests(unittest.TestCase):
    def setUp(self):
        self.grid = SineGalerkin2D(10.0, (-3.0, 9.0), (5, 4))
        nx, ny = self.grid.basis_shape
        self.x_modes = np.arange(1, nx + 1)
        self.y_modes = np.arange(1, ny + 1)
        self.kinetic = 0.5 * ((np.pi * self.x_modes[:, None] / 20)**2 + (np.pi * self.y_modes[None, :] / 12)**2)

    def test_constant_potential_and_sampled_orthonormality(self):
        grid = self.grid
        result = grid.solve(lambda x, y: -0.2, n_levels=13)
        order = np.argsort(self.kinetic.ravel())
        exact_energies = self.kinetic.ravel()[order] - 0.2
        np.testing.assert_allclose(result.energies_hartree, exact_energies[:13], atol=1e-14)
        expected_parity = np.repeat((-1) ** (self.x_modes + 1), len(self.y_modes))[order]
        np.testing.assert_array_equal(result.parity, expected_parity[:13])
        self.assertAlmostEqual(result.next_energy_hartree, exact_energies[13])
        np.testing.assert_allclose(grid.transition_matrix(result, result, lambda x, y: 1), np.eye(13), atol=1e-13)

    def test_nonseparable_potential_and_odd_transition_operator(self):
        grid = self.grid
        x = grid.x[:, None]
        y = grid.y[None, :]
        potential = 0.0002 * (x**2 - 16)**2 + 0.004 * (y - 1 - x**2 / 20)**2
        result = grid.solve(potential, n_levels=self.kinetic.size)
        basis_x = np.sqrt(2 / 20) * np.sin(np.pi * np.outer((grid.x + 10) / 20, self.x_modes))
        basis_y = np.sqrt(2 / 12) * np.sin(np.pi * np.outer((grid.y + 3) / 12, self.y_modes))
        basis = np.einsum("xi,yj->xyij", basis_x, basis_y).reshape(-1, self.kinetic.size)
        weights = grid.x_weights[:, None] * grid.y_weights[None, :]
        direct = np.diag(self.kinetic.ravel()) + basis.T @ ((potential * weights).ravel()[:, None] * basis)
        np.testing.assert_allclose(result.energies_hartree, np.linalg.eigvalsh(direct), atol=1e-14)
        self.assertIsNone(result.next_energy_hartree)
        np.testing.assert_allclose(result.wavefunctions[::-1], result.wavefunctions * result.parity, atol=1e-13)
        dipole = grid.transition_matrix(result, result, lambda x, y: x * (1 + y / 20))
        same_parity = result.parity[:, None] == result.parity[None, :]
        np.testing.assert_allclose(dipole[same_parity], 0, atol=1e-13)
        self.assertGreater(np.linalg.norm(dipole[~same_parity]), 1)
        exact = np.sum((weights * potential)[:, :, None] * result.wavefunctions**2, axis=(0, 1))
        np.testing.assert_allclose(grid.expectation(result, potential), exact, atol=1e-13)

    def test_rejects_asymmetry_and_unresolved_quadrature(self):
        with self.assertRaisesRegex(ValueError, "potential must be even"):
            self.grid.solve(lambda x, y: x, n_levels=2)
        with self.assertRaisesRegex(ValueError, "quadrature_shape must resolve"):
            SineGalerkin2D(10, (-3, 9), (4, 4), (4, 4))


if __name__ == "__main__":
    unittest.main()
