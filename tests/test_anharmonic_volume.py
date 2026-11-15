"""Check parity-resolved energies and spectral quadrature against dense tensor references."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.anharmonic_volume import SineDVR3D


@unittest.skipUnless(find_spec("scipy"), "SciPy is required for the anharmonic volume solver")
class SineDVRTests(unittest.TestCase):
    def test_constant_potential_eigenvalues_and_reflection(self):
        bounds = ((-9.0, 9.0), (-3.0, 8.0), (-5.0, 8.0))
        for nx in (5, 6):
            grid = SineDVR3D(bounds, (nx, 4, 3), workers=1)
            potential = np.full(grid.shape, -113.91)
            kinetic = [
                0.5 * (np.pi * np.arange(1, n + 1) / (right - left))**2
                for n, (left, right) in zip(grid.shape, bounds)
            ]
            for offset, parity in enumerate(("even", "odd")):
                with self.subTest(nx=nx, parity=parity):
                    levels = grid.solve(potential, parity=parity, n_levels=5)
                    exact = (
                        kinetic[0][offset::2, None, None] + kinetic[1][None, :, None] + kinetic[2][None, None, :]
                    ).ravel()
                    np.testing.assert_allclose(levels.energies, potential.flat[0] + np.sort(exact)[:5], atol=2e-14, rtol=0)
                    np.testing.assert_array_equal(levels.vectors[:, ::-1], (1 - 2 * offset) * levels.vectors)
                    states = levels.vectors.reshape(5, -1)
                    np.testing.assert_allclose(states @ states.T, np.eye(5), atol=2e-14)
                    self.assertLess(levels.residuals.max(), 1e-12)

    def test_dense_nonseparable_hamiltonian_and_gaussian_response(self):
        grid = SineDVR3D(((-30, 30), (-16, 24), (-27, 27)), (5, 4, 3), workers=1)
        x, y, z = np.meshgrid(*grid.axes, indexing="ij", sparse=True)
        potential = 2e-8 * (x**2 - 145)**2 + 1e-5 * (y + x**2 / 150)**2 + 2e-5 * (z - y / 3)**2
        initial = grid.solve(potential, parity="even", n_levels=1).vectors[0]
        source = (x * (1.0 + y / 40) + 0.2) * initial
        odd_source = 0.5 * (source - source[::-1])
        nodes = np.linspace(0.004, 0.027, 400)
        sigma = 50 / 219474.6313632

        basis_factors = []
        kinetic_axes = []
        for axis, (n, (left, right)) in enumerate(zip(grid.shape, grid.bounds)):
            mode = np.arange(1, n + 1)
            if axis == 0:
                mode = mode[1::2]
            basis_factors.append(np.sqrt(2 / (n + 1)) * np.sin(np.pi * np.outer(np.arange(1, n + 1), mode) / (n + 1)))
            kinetic_axes.append(0.5 * (np.pi * mode / (right - left))**2)
        basis = np.kron(np.kron(*basis_factors[:2]), basis_factors[2])
        kx, ky, kz = kinetic_axes
        kinetic = kx[:, None, None] + ky[None, :, None] + kz[None, None, :]
        dense = np.diag(kinetic.ravel()) + basis.T @ (potential.ravel()[:, None] * basis)
        exact_energies, exact_vectors = np.linalg.eigh(dense)
        exact_weights = (exact_vectors.T @ (basis.T @ odd_source.ravel()))**2

        levels = grid.solve(potential, parity="odd", n_levels=3)
        np.testing.assert_allclose(levels.energies, exact_energies[:3], atol=1e-13, rtol=0)
        short, complete, repeated = grid.spectral_measures(potential, source, parity="odd", orders=(8, 24, 32))
        self.assertEqual(short.order, 8)
        self.assertEqual(complete.order, 24)
        self.assertEqual(repeated.order, 24)
        self.assertLess(complete.max_reorthogonalization, 1e-13)
        np.testing.assert_allclose(complete.energies, exact_energies, atol=1e-13, rtol=0)
        np.testing.assert_allclose(complete.weights, exact_weights, atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(complete.weights.sum(), np.linalg.norm(odd_source.ravel())**2, rtol=1e-13)
        for degree in range(3):
            np.testing.assert_allclose(short.weights @ short.energies**degree, exact_weights @ exact_energies**degree, rtol=1e-12)

        def profile(energies, weights):
            return np.exp(-0.5 * ((nodes[:, None] - energies) / sigma)**2) @ weights

        np.testing.assert_allclose(profile(complete.energies, complete.weights), profile(exact_energies, exact_weights), atol=1e-10)

    def test_projection_and_roundoff_breakdown(self):
        grid = SineDVR3D(((-5, 5), (-4, 4), (-7, 7)), (3, 2, 1), workers=1)
        potential = np.zeros(grid.shape)
        self.assertEqual(grid.solve(potential, parity="odd", n_levels=2).energies.size, 2)
        level = grid.solve(potential, parity="even", n_levels=1)
        allowed = grid.spectral_measure(potential, 2 * level.vectors[0], parity="even", order=5)
        self.assertEqual(allowed.order, 1)
        self.assertLess(allowed.terminal_residual, 1e-13)
        np.testing.assert_allclose(allowed.weights, [4.0], atol=1e-14)
        np.testing.assert_allclose(allowed.energies, level.energies, atol=1e-14)
        forbidden = grid.spectral_measure(potential, level.vectors[0], parity="odd", order=5)
        self.assertEqual(forbidden.order, 0)
        self.assertEqual(forbidden.projected_norm_squared, 0)
        self.assertEqual(forbidden.weights.size, 0)
        potential[0] = 0.1
        with self.assertRaisesRegex(ValueError, "potential must be even"):
            grid.solve(potential, parity="odd", n_levels=1)


if __name__ == "__main__":
    unittest.main()
