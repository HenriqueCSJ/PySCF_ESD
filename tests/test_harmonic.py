"""Analytic and quadrature checks for the harmonic reference solver."""

from math import factorial
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss, hermval

from pyscf_esd import harmonic_spectrum


class HarmonicSpectrumTests(unittest.TestCase):
    def test_displaced_equal_curvature_poisson_series(self):
        omega, gap, huang_rhys = 0.004, 0.12, 0.75
        cutoff = 7
        spectrum = harmonic_spectrum(
            [omega], [omega], gap_00_hartree=gap, transition_dipole_au=[1, 0, 0],
            shift_au=[np.sqrt(2 * huang_rhys / omega)], initial_max_quanta=0,
            final_max_quanta=cutoff,
        )
        levels = np.arange(cutoff + 1)
        expected = np.exp(-huang_rhys) * np.array([
            huang_rhys**n / factorial(n) for n in levels
        ])
        np.testing.assert_array_equal(spectrum.final_quanta[:, 0], levels)
        np.testing.assert_allclose(spectrum.photon_energy_hartree, gap + omega * levels)
        np.testing.assert_allclose(spectrum.thermal_strength_au2, expected, rtol=1e-12)
        self.assertEqual(spectrum.initial_population_coverage, 1)
        self.assertAlmostEqual(spectrum.final_dipole_strength_coverage, expected.sum())

    def test_emission_photon_energies_and_positive_energy_filter(self):
        spectrum = harmonic_spectrum(
            [0.02], [0.02], gap_00_hartree=-0.05, transition_dipole_au=[1, 0, 0],
            shift_au=[6.0], initial_max_quanta=0, final_max_quanta=4, process="emission",
        )
        np.testing.assert_array_equal(spectrum.final_quanta[:, 0], [2, 1, 0])
        np.testing.assert_allclose(spectrum.photon_energy_hartree, [0.01, 0.03, 0.05])
        self.assertGreater(spectrum.final_dipole_strength_coverage, spectrum.thermal_strength_au2.sum())

    def test_unequal_curvature_parity_and_ground_overlap(self):
        omega_i, omega_f = 0.006, 0.013
        spectrum = harmonic_spectrum(
            [omega_i], [omega_f], gap_00_hartree=0.1, transition_dipole_au=[0, 1, 0],
            initial_max_quanta=0, final_max_quanta=8,
        )
        expected = 2 * np.sqrt(omega_i * omega_f) / (omega_i + omega_f)
        self.assertAlmostEqual(spectrum.dipole_strength_au2[0], expected)
        np.testing.assert_array_equal(spectrum.dipole_strength_au2[1::2], 0)
        self.assertGreater(spectrum.dipole_strength_au2[2], 0)

    def test_identical_surfaces_keep_absolute_boltzmann_populations(self):
        omega, temperature = 0.001, 300.0
        spectrum = harmonic_spectrum(
            [omega], [omega], gap_00_hartree=0.1, transition_dipole_au=[0, 0, 1],
            initial_max_quanta=2, final_max_quanta=1, temperature_kelvin=temperature,
        )
        expected_ratio = np.exp(-omega / (3.166811563e-6 * temperature))
        self.assertAlmostEqual(spectrum.initial_population_coverage, 1 - expected_ratio**3)
        self.assertAlmostEqual(spectrum.final_dipole_strength_coverage,
                               (1 - expected_ratio**2) / (1 - expected_ratio**3))
        diagonal = (spectrum.initial_quanta == spectrum.final_quanta).all(axis=1)
        np.testing.assert_allclose(spectrum.photon_energy_hartree[diagonal], 0.1)
        np.testing.assert_allclose(spectrum.dipole_strength_au2[~diagonal], 0, atol=1e-28)
        np.testing.assert_allclose(np.sort(spectrum.population[diagonal]),
                                   (1 - expected_ratio) * np.array([expected_ratio, 1]))

    def test_linear_ht_selection_rule_for_a_dark_electronic_transition(self):
        omega, derivative = 0.006, 0.03
        spectrum = harmonic_spectrum(
            [omega], [omega], gap_00_hartree=0.1, transition_dipole_au=[0, 0, 0],
            initial_max_quanta=0, final_max_quanta=3,
            dipole_derivatives_au=[[0, derivative, 0]],
        )
        np.testing.assert_allclose(spectrum.dipole_strength_au2,
                                   [0, derivative**2 / (2 * omega), 0, 0], atol=1e-14)
        self.assertAlmostEqual(spectrum.final_dipole_strength_coverage, 1)

    def test_rotated_two_mode_fcht_matches_independent_quadrature(self):
        omega_i = np.array([0.004, 0.007])
        omega_f = np.array([0.006, 0.009])
        angle = 0.4
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        shift = np.array([5.0, -8.0])
        dipole = np.array([0.4, -0.3, 0.2])
        derivatives = np.array([[0.03, 0.12, 0.0], [-0.02, 0.01, 0.04]])
        options = dict(gap_00_hartree=0.1, transition_dipole_au=dipole,
                       initial_max_quanta=2, final_max_quanta=3,
                       temperature_kelvin=1000, shift_au=shift)
        spectrum = harmonic_spectrum(omega_i, omega_f, rotation=rotation,
                                     dipole_derivatives_au=derivatives, **options)

        # Integrate the explicit normalized coordinate-space wavefunctions.
        # Scale the quadrature grid using the initial frequency only.
        nodes, weights = hermgauss(80)
        grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
        coordinates = grid / np.sqrt(omega_i)
        final_coordinates = coordinates @ rotation.T + shift
        quadrature = (np.outer(weights, weights).ravel() * np.exp((grid**2).sum(axis=1))
                      / np.sqrt(np.prod(omega_i)))
        moment = dipole + coordinates @ derivatives

        def wavefunction(omega, quanta, q):
            scaled = q * np.sqrt(omega)
            psi = (omega / np.pi)**0.25 * np.exp(-scaled**2 / 2)
            for mode, n in enumerate(quanta):
                psi[:, mode] *= hermval(scaled[:, mode], [0] * n + [1]) / np.sqrt(2**n * factorial(n))
            return psi.prod(axis=1)

        np.testing.assert_allclose(
            np.sum(spectrum.transition_dipole_au**2, axis=-1), spectrum.dipole_strength_au2,
        )
        for i, f, stick_dipole in zip(spectrum.initial_quanta, spectrum.final_quanta,
                                      spectrum.transition_dipole_au):
            integrand = (wavefunction(omega_i, i, coordinates)
                         * wavefunction(omega_f, f, final_coordinates))
            amplitude = (quadrature * integrand) @ moment
            np.testing.assert_allclose(stick_dipole, amplitude, atol=1e-12)

        # Reverse and rephase initial normal modes without changing the spectrum.
        basis_change = np.array([[0, -1], [1, 0]])
        changed = harmonic_spectrum(
            omega_i[::-1], omega_f, rotation=rotation @ basis_change.T,
            dipole_derivatives_au=basis_change @ derivatives, **options,
        )
        np.testing.assert_allclose(changed.photon_energy_hartree, spectrum.photon_energy_hartree)
        original_strengths = {
            (tuple(i), tuple(f)): strength
            for i, f, strength in zip(spectrum.initial_quanta, spectrum.final_quanta,
                                      spectrum.dipole_strength_au2)
        }
        for i, f, strength in zip(changed.initial_quanta, changed.final_quanta,
                                  changed.dipole_strength_au2):
            self.assertAlmostEqual(strength, original_strengths[(tuple(i[::-1]), tuple(f))])

    def test_incompatible_inputs_are_rejected(self):
        options = dict(gap_00_hartree=0.1, transition_dipole_au=[0, 0, 1],
                       initial_max_quanta=2, final_max_quanta=3)
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            harmonic_spectrum([0.01], [0.01], rotation=[[0.9]], **options)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            harmonic_spectrum([0.01], [-0.01], **options)
        with self.assertRaisesRegex(ValueError, "max_overlaps"):
            harmonic_spectrum([0.01], [0.01], max_overlaps=3, **options)


if __name__ == "__main__":
    unittest.main()
