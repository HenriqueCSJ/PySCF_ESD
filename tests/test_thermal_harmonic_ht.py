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
                    include_dipole_strength_density=True,
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
                reduced_reference = np.zeros_like(axis)
                sigma = width / np.sqrt(8 * np.log(2))
                significant = photon_weights > 1e-12 * photon_weights.sum()
                for line, weight, reduced in zip(
                    sticks.photon_energy_hartree[significant], photon_weights[significant], strength[significant], strict=True,
                ):
                    gaussian = np.exp(-0.5 * ((axis - CM * line) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
                    reference += weight * gaussian
                    reduced_reference += reduced * gaussian
                l1_error = np.sum(np.abs(reference - profile.photon_weighted_density_per_cm_inverse))
                self.assertLess(l1_error / photon_weights.sum(), 2e-9)
                reduced_error = np.sum(np.abs(reduced_reference - profile.dipole_strength_density_per_cm_inverse))
                self.assertLess(reduced_error / strength.sum(), 2e-9)

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

    def test_pure_ht_emission_weights_sticks_before_broadening_at_threshold(self):
        omega = 0.01
        width = 80.0
        sigma = width / np.sqrt(8 * np.log(2))
        for photon_cm in (-1.0, 0.0, 1.0, 500.0):
            gap = -(omega + photon_cm / CM)
            options = dict(
                gap_00_hartree=gap, transition_dipole_au=[0, 0, 0],
                dipole_derivatives_au=[[np.sqrt(2 * omega), 0, 0]], process="emission", temperature_kelvin=0,
            )
            sticks = harmonic_spectrum([omega], [omega], initial_max_quanta=0, final_max_quanta=2, **options)
            expected = float(sticks.thermal_strength_au2 @ sticks.photon_energy_hartree**3)
            for step in (1.0, 0.5):
                with self.subTest(photon_cm=photon_cm, step=step):
                    profile = thermal_harmonic_spectrum(
                        [omega], [omega], fwhm_cm_inverse=width, step_cm_inverse=step,
                        include_dipole_strength_density=True, **options,
                    )
                    self.assertEqual(profile.auxiliary_sigma_cm_inverse, 0.0)
                    self.assertEqual(profile.negative_weight_grid_fraction, 0.0)
                    self.assertAlmostEqual(profile.signed_dipole_strength_au2, 1.0)
                    np.testing.assert_allclose(profile.positive_photon_weighted_strength, expected, atol=0, rtol=1e-12)
                    if photon_cm <= 0:
                        self.assertIsNone(profile.centroid_cm_inverse)
                        self.assertFalse(np.any(profile.photon_weighted_density_per_cm_inverse))
                        self.assertFalse(np.any(profile.dipole_strength_density_per_cm_inverse))
                    else:
                        self.assertAlmostEqual(profile.centroid_cm_inverse, photon_cm, places=10)
                        axis = profile.wavenumber_cm_inverse
                        gaussian = np.exp(-0.5 * ((axis - photon_cm) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
                        np.testing.assert_allclose(profile.dipole_strength_density_per_cm_inverse, gaussian, atol=1e-15)
                        error = np.abs(profile.photon_weighted_density_per_cm_inverse - expected * gaussian).sum() * step
                        self.assertLess(error / expected, 1e-12)

    def test_displaced_threshold_profile_rejects_unresolved_fft_cutoff(self):
        with self.assertRaisesRegex(ValueError, "cannot resolve the positive photon-energy cutoff"):
            thermal_harmonic_spectrum(
                [0.01], [0.01], gap_00_hartree=-1 / CM, transition_dipole_au=[1, 0, 0], shift_au=[5],
                process="emission", temperature_kelvin=0, fwhm_cm_inverse=80, step_cm_inverse=1.0,
            )

    def test_undisplaced_ht_broadens_multiple_off_mesh_lines_at_their_energies(self):
        omega = 600 / CM
        fwhm = 24.0
        line_energy = np.array([2002.0, 2602.0])
        sigma = fwhm / np.sqrt(8 * np.log(2))
        for step in (4, 4.0):
            with self.subTest(step=step):
                profile = thermal_harmonic_spectrum(
                    [omega], [omega], gap_00_hartree=2002 / CM, transition_dipole_au=[1, 0, 0],
                    dipole_derivatives_au=[[np.sqrt(2 * omega), 0, 0]], process="absorption", temperature_kelvin=0,
                    fwhm_cm_inverse=fwhm, step_cm_inverse=step, include_dipole_strength_density=True,
                )
                offset = (profile.wavenumber_cm_inverse[:, None] - line_energy) / sigma
                gaussian = np.exp(-0.5 * offset**2) / (sigma * np.sqrt(2 * np.pi))
                for actual, expected in (
                    (profile.dipole_strength_density_per_cm_inverse, gaussian.sum(axis=1)),
                    (profile.photon_weighted_density_per_cm_inverse, gaussian @ (line_energy / CM)),
                ):
                    self.assertLess(np.max(np.abs(actual - expected)) / expected.max(), 2e-13)

    def test_cold_displaced_emission_respects_upper_photon_support(self):
        for origin in (-1, 500):
            with self.subTest(origin=origin):
                options = dict(
                    gap_00_hartree=-origin / CM, transition_dipole_au=[1, 0, 0], shift_au=[5],
                    process="emission", temperature_kelvin=0,
                )
                profile = thermal_harmonic_spectrum(
                    [0.01], [0.01], fwhm_cm_inverse=80, step_cm_inverse=1.0,
                    include_dipole_strength_density=True, **options,
                )
                self.assertAlmostEqual(profile.signed_dipole_strength_au2, 1.0)
                if origin < 0:
                    self.assertEqual(profile.positive_photon_weighted_strength, 0.0)
                    self.assertIsNone(profile.centroid_cm_inverse)
                    self.assertFalse(np.any(profile.photon_weighted_density_per_cm_inverse))
                    self.assertFalse(np.any(profile.dipole_strength_density_per_cm_inverse))
                else:
                    sticks = harmonic_spectrum([0.01], [0.01], initial_max_quanta=0, final_max_quanta=2, **options)
                    expected = float(sticks.thermal_strength_au2 @ sticks.photon_energy_hartree**3)
                    np.testing.assert_allclose(profile.positive_photon_weighted_strength, expected, atol=0, rtol=1e-9)
                    self.assertAlmostEqual(profile.centroid_cm_inverse, origin, places=7)

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
