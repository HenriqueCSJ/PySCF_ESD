"""Verify all-quanta harmonic FC/HT with assigned lines and oscillator closure."""

import unittest

import numpy as np

from pyscf_esd.anharmonic import HarmonicSpectators
from pyscf_esd.harmonic import _KB_HARTREE_PER_K, harmonic_spectrum
from pyscf_esd.spectral_properties import HARTREE_TO_WAVENUMBER as CM
from pyscf_esd.thermal_separable import _harmonic_characteristic, thermal_harmonic_spectrum


class ThermalHarmonicHTTests(unittest.TestCase):
    def test_characteristic_against_assigned_fc_ht_at_zero_and_finite_temperature(self):
        angle = 0.29
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        omega_i, omega_f = np.array([[690, 830], [720, 915]]) / CM
        shift = [8.2, -5.1]
        dipole = np.array([0.27, -0.14, 0.03])
        derivatives = np.array([[0.011, -0.006, 0.005], [-0.008, -0.013, 0.009]])
        times = np.array([0.0, 0.001, 0.013, 0.471, np.pi])
        for temperature, process in ((0, "emission"), (298, "absorption")):
            with self.subTest(temperature=temperature, process=process):
                sign = 1 if process == "absorption" else -1
                gap = sign * 30000 / CM
                thermal_energy = _KB_HARTREE_PER_K * temperature
                bath = HarmonicSpectators(omega_i, omega_f, 0, 0, rotation, shift)
                model = _harmonic_characteristic(bath, thermal_energy, sign, dipole=dipole, derivatives=derivatives)
                sticks = harmonic_spectrum(
                    omega_i, omega_f, gap_00_hartree=gap, transition_dipole_au=dipole,
                    dipole_derivatives_au=derivatives, initial_max_quanta=8, final_max_quanta=15,
                    process=process, temperature_kelvin=temperature, rotation=rotation, shift_au=shift,
                )
                occupation = 0 if temperature == 0 else 1 / np.expm1(omega_i / thermal_energy)
                closure = dipole @ dipole + np.sum((2 * occupation + 1) * np.sum(derivatives**2, axis=1) / (2 * omega_i))
                strength = sticks.thermal_strength_au2
                np.testing.assert_allclose(strength.sum(), closure, atol=2e-11, rtol=0)
                energies = (sticks.photon_energy_hartree - sign * gap) * CM + model.zero_point
                expected = np.exp(-1j * times[:, None] * energies) @ strength / closure
                np.testing.assert_allclose(model.transform(times), expected, atol=2e-10, rtol=0)
                for parameter in (-1e-4, 1e-4):
                    expected_moment = np.exp(parameter * energies) @ strength / closure
                    np.testing.assert_allclose(np.exp(model.log_moment(parameter)), expected_moment, atol=2e-10, rtol=0)

                width = 85.0
                profile = thermal_harmonic_spectrum(
                    omega_i, omega_f, gap_00_hartree=gap, transition_dipole_au=dipole,
                    dipole_derivatives_au=derivatives, process=process, temperature_kelvin=temperature,
                    rotation=rotation, shift_au=shift, fwhm_cm_inverse=width, step_cm_inverse=1.0,
                )
                self.assertEqual(profile.initial_population_coverage, 1.0)
                self.assertEqual(profile.final_dipole_strength_coverage, 1.0)
                self.assertAlmostEqual(profile.origin_00_cm_inverse, 30000.0)
                self.assertAlmostEqual(profile.signed_dipole_strength_au2, closure)
                exponent = 1 if process == "absorption" else 3
                photon_weights = strength * sticks.photon_energy_hartree**exponent
                np.testing.assert_allclose(profile.positive_photon_weighted_strength, photon_weights.sum(), rtol=2e-9)
                axis = profile.wavenumber_cm_inverse
                reference = np.zeros_like(axis)
                sigma = width / np.sqrt(8 * np.log(2))
                significant = photon_weights > 1e-12 * photon_weights.sum()
                for line, weight in zip(sticks.photon_energy_hartree[significant], photon_weights[significant], strict=True):
                    reference += weight * np.exp(-0.5 * ((axis - CM * line) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
                l1_error = np.sum(np.abs(reference - profile.photon_weighted_density_per_cm_inverse))
                self.assertLess(l1_error / photon_weights.sum(), 2e-9)

    def test_ht_only_equal_surfaces_include_thermal_sidebands_and_positive_cutoff(self):
        omega = 500 / CM
        gap = 230 / CM
        slope = 0.02
        temperature = 298
        occupation = 1 / np.expm1(omega / (_KB_HARTREE_PER_K * temperature))
        profile = thermal_harmonic_spectrum(
            [omega], [omega], gap_00_hartree=gap, transition_dipole_au=[0, 0, 0],
            dipole_derivatives_au=[[slope, 0, 0]], process="absorption", temperature_kelvin=temperature,
            fwhm_cm_inverse=50, step_cm_inverse=0.5,
        )
        raising_strength = slope**2 * (occupation + 1) / (2 * omega)
        lowering_strength = slope**2 * occupation / (2 * omega)
        self.assertAlmostEqual(profile.signed_dipole_strength_au2, raising_strength + lowering_strength)
        self.assertAlmostEqual(profile.positive_photon_weighted_strength, raising_strength * (gap + omega), places=12)
        self.assertAlmostEqual(profile.centroid_cm_inverse, 730.0, places=7)

    def test_zero_slope_matches_condon_and_global_electronic_phase_is_invariant(self):
        options = dict(gap_00_hartree=4300 / CM, shift_au=[5.2], process="absorption", temperature_kelvin=298,
                       fwhm_cm_inverse=50.0, step_cm_inverse=0.6)
        dipole = np.array([0.2, -0.1, 0.05])
        derivative = np.array([[0.013, 0.005, -0.002]])
        omega_i, omega_f = [610 / CM], [700 / CM]
        condon = thermal_harmonic_spectrum(omega_i, omega_f, transition_dipole_au=dipole, **options)
        zero = thermal_harmonic_spectrum(
            omega_i, omega_f, transition_dipole_au=dipole, dipole_derivatives_au=np.zeros((1, 3)), **options,
        )
        np.testing.assert_array_equal(zero.photon_weighted_density_per_cm_inverse, condon.photon_weighted_density_per_cm_inverse)
        plus = thermal_harmonic_spectrum(
            omega_i, omega_f, transition_dipole_au=dipole, dipole_derivatives_au=derivative, **options,
        )
        minus = thermal_harmonic_spectrum(
            omega_i, omega_f, transition_dipole_au=-dipole, dipole_derivatives_au=-derivative, **options,
        )
        np.testing.assert_array_equal(plus.photon_weighted_density_per_cm_inverse, minus.photon_weighted_density_per_cm_inverse)


if __name__ == "__main__":
    unittest.main()
