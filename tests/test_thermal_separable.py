import unittest
from dataclasses import replace

import numpy as np

from pyscf_esd.anharmonic import HarmonicSpectators, _spectator_factors, anharmonic_spectrum
from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.harmonic import _KB_HARTREE_PER_K
from pyscf_esd.spectral_properties import HARTREE_TO_WAVENUMBER
from pyscf_esd.thermal_separable import _harmonic_characteristic, thermal_separable_spectrum


CM = HARTREE_TO_WAVENUMBER


class ThermalSeparableTests(unittest.TestCase):
    def test_harmonic_characteristic_against_assigned_duschinsky_sticks(self):
        for temperature in (0.0, 298.0):
            with self.subTest(temperature=temperature):
                self._test_harmonic_characteristic(temperature)

    def test_thermal_profile_against_direct_broadening_in_both_directions(self):
        for process in ("absorption", "emission"):
            with self.subTest(process=process):
                self._test_thermal_profile(process)

    def _test_harmonic_characteristic(self, temperature):
        angle = 0.32
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        bath = HarmonicSpectators(
            np.array([700, 885]) / CM, np.array([800, 910]) / CM, 7, 12, rotation, [9.0, -4.0],
        )
        thermal_energy = temperature * _KB_HARTREE_PER_K
        exact = _spectator_factors(bath, thermal_energy, 250_000)
        characteristic = _harmonic_characteristic(bath, thermal_energy, 1)
        energy = exact.energy_difference.ravel() * CM
        weights = (exact.overlap**2 * exact.population[:, None]).ravel()
        times = np.array([0.0, 0.001, 0.002, 0.01, 0.025, 0.4, 2.0])
        expected = np.exp(-1j * times[:, None] * energy) @ weights
        np.testing.assert_allclose(characteristic.transform(times), expected, atol=1e-10, rtol=0)
        parameter = 1e-4
        np.testing.assert_allclose(
            np.exp(characteristic.log_moment(parameter)), np.exp(parameter * energy) @ weights, atol=1e-10, rtol=0,
        )


    def _test_thermal_profile(self, process):
        grid = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ground = solve_dvr(grid, 0.002 * (1 - np.cos(grid)), boundary="periodic",
                           kinetic_coefficient=0.003, n_levels=len(grid))
        excited = solve_dvr(grid, 0.0028 * (1 - np.cos(grid - 0.7)) + 0.022,
                            boundary="periodic", kinetic_coefficient=0.0029, n_levels=len(grid))
        dipole = np.column_stack((0.2 + 0.07 * np.sin(grid), 0.1 * np.cos(grid), np.zeros_like(grid)))
        if process == "absorption":
            initial, final = ground, excited
            bath = HarmonicSpectators([1000 / CM], [1200 / CM], 8, 17, shift_au=[9.0])
            exponent = 1
        else:
            initial, final = excited, ground
            bath = HarmonicSpectators([1200 / CM], [1000 / CM], 8, 17, shift_au=[-9.0])
            exponent = 3
        sticks = anharmonic_spectrum(
            initial, final, transition_dipole_au=dipole, process=process, temperature_kelvin=298.0, spectators=bath,
        )
        spectrum = thermal_separable_spectrum(
            initial, final, transition_dipole_au=dipole, process=process, temperature_kelvin=298.0,
            fwhm_cm_inverse=80.0, spectators=bath, step_cm_inverse=0.5,
        )
        line_energy = sticks.photon_energy_hartree * CM
        line_weight = sticks.thermal_strength_au2 * sticks.photon_energy_hartree**exponent
        sigma = 80 / np.sqrt(8 * np.log(2))
        axis = spectrum.wavenumber_cm_inverse
        reference = np.zeros_like(axis)
        significant = line_weight > line_weight.sum() * 1e-12
        for energy, weight in zip(line_energy[significant], line_weight[significant], strict=True):
            reference += weight * np.exp(-0.5 * ((axis - energy) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        relative_l1 = abs(reference - spectrum.photon_weighted_density_per_cm_inverse).sum() * 0.5 / line_weight.sum()
        self.assertLess(relative_l1, 3e-5)
        np.testing.assert_allclose(spectrum.positive_photon_weighted_strength, line_weight.sum(), rtol=2e-7, atol=0)
        self.assertAlmostEqual(spectrum.centroid_cm_inverse, line_energy @ line_weight / line_weight.sum(), delta=0.001)
        self.assertAlmostEqual(spectrum.initial_population_coverage, 1)
        self.assertAlmostEqual(spectrum.final_dipole_strength_coverage, 1)
        self.assertLess(spectrum.negative_weight_grid_fraction, 1e-8)


    def test_no_spectators_recovers_requested_gaussian_fwhm_at_absolute_origin(self):
        grid = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ground = solve_dvr(grid, np.zeros_like(grid), boundary="periodic", n_levels=len(grid))
        excited = solve_dvr(grid, np.full_like(grid, 1000 / CM), boundary="periodic", n_levels=len(grid))
        spectrum = thermal_separable_spectrum(
            ground, excited, transition_dipole_au=[0.1, 0.0, 0.0], process="absorption",
            temperature_kelvin=298, fwhm_cm_inverse=80.0, step_cm_inverse=0.5,
        )
        axis = spectrum.wavenumber_cm_inverse
        density = spectrum.photon_weighted_density_per_cm_inverse
        self.assertAlmostEqual(spectrum.origin_00_cm_inverse, 1000)
        np.testing.assert_allclose(spectrum.positive_photon_weighted_strength, 0.01 * 1000 / CM, rtol=1e-7, atol=0)
        self.assertAlmostEqual(spectrum.centroid_cm_inverse, 1000)
        maximum = density[axis == 1000].item()
        for half_max in (960, 1040):
            np.testing.assert_allclose(density[axis == half_max].item(), maximum / 2, rtol=1e-7, atol=0)

    def test_equal_spectator_hamiltonians_preserve_zero_photon_cutoff(self):
        grid = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        initial = solve_dvr(grid, np.zeros_like(grid), boundary="periodic", n_levels=len(grid))
        final = solve_dvr(grid, np.full_like(grid, -10 / CM), boundary="periodic", n_levels=len(grid))
        # The constant dipole connects only equal rotor wavefunctions at zero
        # temperature. Restrict both coefficient sets to their ground levels.
        initial = replace(initial, coefficients=initial.coefficients[:, :1])
        final = replace(final, coefficients=final.coefficients[:, :1])
        bath = HarmonicSpectators([1000 / CM], [1000 / CM], 0, 0)
        spectrum = thermal_separable_spectrum(
            initial, final, transition_dipole_au=[0.1, 0.0, 0.0], process="absorption",
            temperature_kelvin=0, fwhm_cm_inverse=80, spectators=bath,
        )
        self.assertAlmostEqual(spectrum.signed_dipole_strength_au2, 0.01)
        self.assertEqual(spectrum.positive_photon_weighted_strength, 0)
        self.assertIsNone(spectrum.centroid_cm_inverse)
        self.assertFalse(np.any(spectrum.photon_weighted_density_per_cm_inverse))
