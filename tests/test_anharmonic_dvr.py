"""Analytic energy and quadrature checks for the one-dimensional DVR."""

import unittest

import numpy as np

from pyscf_esd.anharmonic_dvr import solve_dvr


class DVRTests(unittest.TestCase):
    def test_free_rotor_with_odd_and_even_grids(self):
        offset, inertia = -120.35, 3850.0
        for n_points in (23, 24):
            with self.subTest(n_points=n_points):
                angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
                result = solve_dvr(
                    angles, np.full(n_points, offset), boundary="periodic",
                    n_levels=7, kinetic_coefficient=1 / inertia,
                )
                unsigned_modes = np.sort(np.abs(np.arange(n_points) - n_points // 2))
                expected = offset + unsigned_modes**2 / (2 * inertia)
                np.testing.assert_allclose(result.all_energies_hartree, expected, rtol=0, atol=2e-14)
                np.testing.assert_allclose(result.energies_hartree, expected[:7], rtol=0, atol=2e-14)
                np.testing.assert_allclose(result.domain, [-np.pi, np.pi])
                np.testing.assert_allclose(
                    np.abs(result.wavefunctions[:, 0]), 1 / np.sqrt(2 * np.pi), atol=2e-14,
                )

    def test_periodic_grid_supports_an_explicit_period(self):
        period = np.pi
        grid = np.linspace(0.1, 0.1 + period, 32, endpoint=False)
        inertia = 10.0
        result = solve_dvr(
            grid, np.zeros_like(grid), boundary="periodic", n_levels=5,
            kinetic_coefficient=1 / inertia, period=period,
        )
        np.testing.assert_allclose(
            result.energies_hartree, np.array([0, 1, 1, 4, 4]) * 2 / inertia, atol=1e-14,
        )
        np.testing.assert_allclose(result.domain, [0.1, 0.1 + period], rtol=0, atol=1e-14)

    def test_hindered_rotor_matches_a_fourier_basis_reference(self):
        barrier, inertia, fold = 0.006, 625.0, 3
        # V(q) = barrier/2 * (1 - cos(fold*q)) couples Fourier modes that
        # differ by ±fold. Build the reference without sampling the potential.
        modes = np.arange(-48, 49)
        reference = np.diag(barrier / 2 + modes**2 / (2 * inertia))
        reference[np.abs(np.subtract.outer(modes, modes)) == fold] = -barrier / 4
        n_levels = 8
        expected = np.linalg.eigvalsh(reference)[:n_levels]

        for n_points in (33, 64):
            with self.subTest(n_points=n_points):
                grid = np.linspace(0.23, 0.23 + 2 * np.pi, n_points, endpoint=False)
                potential = barrier / 2 * (1 - np.cos(fold * grid))
                result = solve_dvr(
                    grid, potential, boundary="periodic", n_levels=n_levels,
                    kinetic_coefficient=1 / inertia,
                )
                np.testing.assert_allclose(result.energies_hartree, expected, rtol=0, atol=1e-12)

    def test_dirichlet_box_energies_and_wavefunctions(self):
        left, right = -4.5, 9.5
        n_points, n_levels = 63, 8
        grid = np.linspace(left, right, n_points + 2)[1:-1]
        offset = 0.21
        result = solve_dvr(grid, np.full(n_points, offset), boundary="dirichlet", n_levels=n_levels)
        modes = np.arange(1, n_levels + 1)
        expected_energies = offset + (np.pi * modes / (right - left)) ** 2 / 2
        expected_psi = np.sqrt(2 / (right - left)) * np.sin(
            np.pi * np.outer((grid - left) / (right - left), modes)
        )
        overlaps = result.spacing * expected_psi.T @ result.wavefunctions
        np.testing.assert_allclose(result.energies_hartree, expected_energies, rtol=0, atol=1e-13)
        np.testing.assert_allclose(np.abs(overlaps), np.eye(n_levels), atol=1e-12)
        np.testing.assert_allclose(result.domain, [left, right])
        np.testing.assert_allclose(result.coefficients.T @ result.coefficients, np.eye(n_levels), atol=1e-14)

    def test_morse_low_levels_and_grid_dipole_integral(self):
        n_levels = 4
        grid = np.linspace(-55, 175, 202)[1:-1]
        depth, inverse_length = 0.06, 0.06
        electronic_offset = -84.25
        potential = electronic_offset + depth * (1 - np.exp(-inverse_length * grid))**2
        result = solve_dvr(grid, potential, boundary="dirichlet", n_levels=n_levels)
        oscillator_energy = inverse_length * np.sqrt(2 * depth)
        quanta = np.arange(n_levels) + 0.5
        expected = electronic_offset + oscillator_energy * quanta - inverse_length**2 * quanta**2 / 2
        np.testing.assert_allclose(result.energies_hartree, expected, rtol=0, atol=2e-9)

        # Compare the quadrature with the analytic ground-state <exp(-a*q)>.
        # With y = (2*sqrt(2*depth)/a)*exp(-a*q), the density over y is a
        # gamma density with shape 2*sqrt(2*depth)/a - 1 and unit scale.
        observable = np.exp(-inverse_length * grid)
        ground = result.coefficients[:, 0]
        exact_dipole = 1 - inverse_length / (2 * np.sqrt(2 * depth))
        self.assertAlmostEqual(ground @ (observable * ground), exact_dipole, places=11)

    def test_result_preserves_the_grid_and_retains_only_requested_vectors(self):
        grid = np.arange(5.0)
        result = solve_dvr(grid, np.zeros(5), boundary="dirichlet", n_levels=2, kinetic_coefficient=2)
        grid[0] = -2
        np.testing.assert_array_equal(result.grid, np.arange(5.0))
        self.assertEqual(result.coefficients.shape, (5, 2))
        self.assertEqual(result.all_energies_hartree.shape, (5,))
        self.assertEqual(result.energies_hartree.shape, (2,))
        self.assertEqual(result.kinetic_coefficient, 2)
        for array in (result.grid, result.coefficients, result.energies_hartree, result.all_energies_hartree):
            self.assertFalse(array.flags.writeable)

    def test_malformed_grids_and_potentials(self):
        for grid, potential, message in (
            ([0, 1], [0, 0], "grid must contain"),
            (np.arange(1025), np.zeros(1025), "grid must contain"),
            ([[0, 1, 2]], [0, 0, 0], "grid must be a real vector"),
            ([0j, 1, 2], [0, 0, 0], "grid must be a real vector"),
            ([0, 1, np.inf], [0, 0, 0], "grid must be finite"),
            ([0, 0, 1], [0, 0, 0], "strictly increasing"),
            ([2, 1, 0], [0, 0, 0], "strictly increasing"),
            ([0, 1, 3], [0, 0, 0], "uniformly spaced"),
            ([0, 1, 2], [0, 0], "one energy per grid point"),
            ([0, 1, 2], [0, np.nan, 0], "potential_hartree must be finite"),
            ([0, 1, 2], [0, 1j, 0], "potential_hartree must be a real vector"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                solve_dvr(grid, potential, boundary="dirichlet", n_levels=1)

    def test_malformed_boundaries_and_solver_options(self):
        grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        potential = np.zeros_like(grid)
        with self.assertRaisesRegex(ValueError, "repeated endpoint"):
            solve_dvr(np.linspace(0, 2 * np.pi, 24), potential, boundary="periodic", n_levels=1)
        with self.assertRaisesRegex(ValueError, "period is only defined"):
            solve_dvr(grid, potential, boundary="dirichlet", n_levels=1, period=2 * np.pi)
        with self.assertRaisesRegex(ValueError, "boundary must be"):
            solve_dvr(grid, potential, boundary="free", n_levels=1)
        with self.assertRaisesRegex(ValueError, "period must be"):
            solve_dvr(grid, potential, boundary="periodic", n_levels=1, period=np.nan)
        for n_levels in (False, np.bool_(True), 1.5, 0, 25):
            with self.subTest(n_levels=n_levels), self.assertRaisesRegex(ValueError, "n_levels"):
                solve_dvr(grid, potential, boundary="periodic", n_levels=n_levels)
        for kinetic_coefficient in (False, -1, 0, np.nan, np.inf, 1j, [1]):
            with self.subTest(kinetic_coefficient=kinetic_coefficient):
                with self.assertRaisesRegex(ValueError, "kinetic_coefficient"):
                    solve_dvr(
                        grid, potential, boundary="periodic", n_levels=1,
                        kinetic_coefficient=kinetic_coefficient,
                    )


if __name__ == "__main__":
    unittest.main()
