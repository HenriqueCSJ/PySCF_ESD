"""Synthetic checks of separable DVR sticks and their coverage diagnostics."""

from dataclasses import replace
import unittest

import numpy as np

from pyscf_esd.anharmonic import HarmonicSpectators, anharmonic_spectrum
from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.harmonic import harmonic_spectrum


class AnharmonicSpectrumTests(unittest.TestCase):
    def test_displaced_active_oscillator_matches_harmonic_product(self):
        grid = np.linspace(-140.0, 140.0, 121)
        omega_i, omega_f = 0.007, 0.009
        active_shift, spectator_shift = 10.0, -6.0
        gap = 0.12
        initial = solve_dvr(grid, 0.5 * (omega_i * grid)**2, boundary="dirichlet", n_levels=1)
        final = solve_dvr(grid, gap + 0.5 * (omega_f * (grid + active_shift))**2,
                          boundary="dirichlet", n_levels=4)
        spectators = HarmonicSpectators([0.004], [0.006], 0, 3, shift_au=[spectator_shift])
        dipole = [0.2, -0.7, 0.4]
        result = anharmonic_spectrum(initial, final, transition_dipole_au=dipole,
                                     spectators=spectators)

        reference = harmonic_spectrum(
            [omega_i, 0.004], [omega_f, 0.006],
            gap_00_hartree=gap + (omega_f - omega_i + 0.006 - 0.004) / 2,
            transition_dipole_au=dipole, shift_au=[active_shift, spectator_shift],
            initial_max_quanta=0, final_max_quanta=6,
        )
        ref_sticks = {
            tuple(pair): (energy, strength)
            for pair, energy, strength in zip(reference.final_quanta, reference.photon_energy_hartree,
                                              reference.dipole_strength_au2)
            if np.all(pair <= 3)
        }
        actual = np.column_stack((result.photon_energy_hartree, result.dipole_strength_au2))
        expected = [ref_sticks[(active, bath)] for active, bath in
                    zip(result.final_level, result.final_spectator_quanta[:, 0])]
        self.assertEqual(len(actual), 16)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(np.sum(result.transition_dipole_au**2, axis=1), result.dipole_strength_au2)
        np.testing.assert_array_equal(result.population, 1.0)
        self.assertEqual(result.initial_population_coverage, 1.0)
        self.assertAlmostEqual(result.final_dipole_strength_coverage,
                               result.dipole_strength_au2.sum() / np.dot(dipole, dipole))

    def test_absorption_and_emission_have_reciprocal_line_strengths(self):
        grid = np.linspace(-90.0, 90.0, 81)
        initial = solve_dvr(grid, 3e-5 * grid**2 + 1e-8 * grid**4,
                            boundary="dirichlet", n_levels=3)
        final = solve_dvr(grid, 0.12 + 4e-5 * (grid - 8)**2 + 1.4e-8 * (grid - 8)**4,
                          boundary="dirichlet", n_levels=4)
        dipole = np.column_stack((grid * 0.012, np.cos(0.03 * grid), 0.05 * np.ones_like(grid)))
        kwargs = dict(transition_dipole_au=dipole, temperature_kelvin=2200)
        absorption = anharmonic_spectrum(initial, final, **kwargs)
        emission = anharmonic_spectrum(final, initial, process="emission", **kwargs)
        a = np.lexsort((absorption.final_level, absorption.initial_level))
        e = np.lexsort((emission.initial_level, emission.final_level))
        np.testing.assert_array_equal(absorption.initial_level[a], emission.final_level[e])
        np.testing.assert_array_equal(absorption.final_level[a], emission.initial_level[e])
        np.testing.assert_allclose(absorption.photon_energy_hartree[a], emission.photon_energy_hartree[e])
        np.testing.assert_allclose(absorption.dipole_strength_au2[a], emission.dipole_strength_au2[e])
        np.testing.assert_allclose(absorption.transition_dipole_au[a], emission.transition_dipole_au[e], atol=1e-14)
        self.assertLess(absorption.initial_population_coverage, 1)
        self.assertLess(emission.initial_population_coverage, 1)

    def test_grid_dipole_strength_obeys_parity_and_full_final_closure(self):
        grid = np.linspace(-80.0, 80.0, 81)
        potential = 4e-5 * grid**2 + 5e-9 * grid**4
        initial = solve_dvr(grid, potential, boundary="dirichlet", n_levels=1)
        final = solve_dvr(grid, 0.1 + potential, boundary="dirichlet", n_levels=len(grid))
        dipole = np.column_stack((0.02 * grid, np.zeros_like(grid), 1e-6 * grid**3))
        result = anharmonic_spectrum(initial, final, transition_dipole_au=dipole)
        np.testing.assert_allclose(result.dipole_strength_au2[::2], 0, atol=1e-25)
        self.assertGreater(result.dipole_strength_au2[1], 0)
        self.assertAlmostEqual(result.final_dipole_strength_coverage, 1.0)
        mean_square_dipole = initial.spacing * np.dot(
            initial.wavefunctions[:, 0]**2, np.sum(dipole**2, axis=1)
        )
        self.assertAlmostEqual(result.dipole_strength_au2.sum(), mean_square_dipole)

    def test_rotor_partition_and_manifold_cutoff(self):
        # Nine momenta in each direction plus the unique m=0 ground state.
        grid = np.linspace(0, 2 * np.pi, 19, endpoint=False)
        rotor = dict(boundary="periodic", kinetic_coefficient=0.002)
        temperature = 400
        initial = solve_dvr(grid, np.zeros_like(grid), n_levels=3, **rotor)
        final = solve_dvr(grid, np.full(grid.shape, 0.1), n_levels=5, **rotor)
        result = anharmonic_spectrum(initial, final, transition_dipole_au=[1, 0, 0],
                                     temperature_kelvin=temperature)
        momenta = np.arange(-9, 10)
        partition = np.exp(-0.001 * momenta**2 / (3.166811563e-6 * temperature)).sum()
        expected = np.exp(-np.array([0, 0.001, 0.001]) / (3.166811563e-6 * temperature)) / partition
        levels = np.arange(3)
        np.testing.assert_allclose(result.population, expected[result.initial_level])
        self.assertAlmostEqual(result.initial_population_coverage, expected.sum())
        self.assertAlmostEqual(result.final_dipole_strength_coverage, 1.0)
        np.testing.assert_allclose(np.bincount(result.initial_level, result.dipole_strength_au2),
                                   np.ones_like(levels))

        rotated_coefficients = final.coefficients.copy()
        rotated_coefficients[:, 1:3] = final.coefficients[:, 1:3] @ np.array(
            [[0.6, -0.8], [0.8, 0.6]]
        )
        rotated = anharmonic_spectrum(initial, replace(final, coefficients=rotated_coefficients),
                                      transition_dipole_au=[1, 0, 0], temperature_kelvin=temperature)
        block = (result.initial_level > 0) & np.isin(result.final_level, [1, 2])
        rotated_block = (rotated.initial_level > 0) & np.isin(rotated.final_level, [1, 2])
        self.assertAlmostEqual(result.thermal_strength_au2[block].sum(),
                               rotated.thermal_strength_au2[rotated_block].sum())

        partial = solve_dvr(grid, np.zeros_like(grid), n_levels=2, **rotor)
        options = dict(transition_dipole_au=[1, 0, 0], temperature_kelvin=temperature)
        with self.assertRaisesRegex(ValueError, "initial DVR cutoff splits"):
            anharmonic_spectrum(partial, final, **options)
        with self.assertRaisesRegex(ValueError, "final DVR cutoff splits"):
            anharmonic_spectrum(initial, partial, **options)

    def test_positive_filter_follows_joint_energy_and_coverage(self):
        grid = np.linspace(-50.0, 50.0, 51)
        potential = 5e-5 * grid**2
        initial = solve_dvr(grid, potential, boundary="dirichlet", n_levels=1)
        final = solve_dvr(grid, potential - 0.01, boundary="dirichlet", n_levels=1)
        spectators = HarmonicSpectators([0.010], [0.014], 0, 2, shift_au=[10.0])
        dipole = [1, 0, 0]
        with_bath = anharmonic_spectrum(initial, final, transition_dipole_au=dipole,
                                        spectators=spectators)
        without_bath = anharmonic_spectrum(initial, final, transition_dipole_au=dipole)
        self.assertEqual(without_bath.photon_energy_hartree.size, 0)
        np.testing.assert_array_equal(with_bath.final_spectator_quanta[:, 0], [1, 2])
        np.testing.assert_allclose(with_bath.photon_energy_hartree, [0.006, 0.020])
        self.assertGreater(with_bath.thermal_strength_au2.sum(), 0)
        self.assertGreater(with_bath.final_dipole_strength_coverage, with_bath.thermal_strength_au2.sum())

    def test_joint_partition_and_dipole_closure_have_independent_cutoffs(self):
        grid = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        rotor = dict(boundary="periodic", kinetic_coefficient=0.001)
        initial = solve_dvr(grid, np.zeros_like(grid), n_levels=3, **rotor)
        final = solve_dvr(grid, np.full(grid.shape, 0.1), n_levels=1, **rotor)
        frequency, temperature = 0.002, 450.0
        spectators = HarmonicSpectators([frequency], [frequency], 2, 1)
        result = anharmonic_spectrum(initial, final, transition_dipole_au=[0, 1, 0],
                                     temperature_kelvin=temperature, spectators=spectators)
        beta = 1 / (3.166811563e-6 * temperature)
        rotor_factor = np.exp(-0.0005 * beta)
        spectator_factor = np.exp(-frequency * beta)
        rotor_partition = np.sum(rotor_factor**(np.arange(-4, 5)**2))
        self.assertAlmostEqual(result.initial_population_coverage,
                               (1 + 2 * rotor_factor) / rotor_partition * (1 - spectator_factor**3))
        self.assertAlmostEqual(result.final_dipole_strength_coverage,
                               (1 - spectator_factor**2) / (1 - spectator_factor**3)
                               / (1 + 2 * rotor_factor))
        self.assertAlmostEqual(result.thermal_strength_au2.sum(),
                               (1 - spectator_factor**2) / rotor_partition)

    def test_incompatible_inputs_and_dark_dipole(self):
        grid = np.linspace(-1.0, 1.0, 5)
        initial = solve_dvr(grid, np.zeros_like(grid), boundary="dirichlet", n_levels=1)
        final = solve_dvr(grid, np.ones_like(grid), boundary="dirichlet", n_levels=3)
        other = solve_dvr(grid + 1.0, np.ones_like(grid), boundary="dirichlet", n_levels=3)
        with self.assertRaisesRegex(ValueError, "same grid"):
            anharmonic_spectrum(initial, other, transition_dipole_au=[1, 0, 0])
        with self.assertRaisesRegex(ValueError, "shape"):
            anharmonic_spectrum(initial, final, transition_dipole_au=np.zeros((5, 1)))
        with self.assertRaisesRegex(ValueError, "must be real"):
            anharmonic_spectrum(initial, final, transition_dipole_au=[1j, 0, 0])
        with self.assertRaisesRegex(ValueError, "exceeds max_transitions"):
            anharmonic_spectrum(initial, final, transition_dipole_au=[1, 0, 0], max_transitions=2)
        result = anharmonic_spectrum(initial, final, transition_dipole_au=[0, 0, 0])
        np.testing.assert_array_equal(result.dipole_strength_au2, 0)
        self.assertIsNone(result.final_dipole_strength_coverage)


if __name__ == "__main__":
    unittest.main()
