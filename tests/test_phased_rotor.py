"""Analytic checks of electronic closure and phase-compatible rotor spectra."""

from dataclasses import replace
import unittest

import numpy as np

from pyscf_esd.anharmonic import anharmonic_spectrum
from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.anharmonic_sampling import evaluate_rotor_samples, phase_periodic_scan
from pyscf_esd.state_tracking import StateTrackingError
from pyscf_esd.thermal_separable import thermal_separable_spectrum


def _signed_links(states):
    return np.einsum("ni,ni->n", states, np.roll(states, -1, axis=0))


def _electronic_states(angle):
    return np.column_stack((np.cos(angle / 2), np.sin(angle / 2)))


class PhasedRotorTests(unittest.TestCase):
    def test_rotor_grid_origin_conversion_preserves_seam_sign(self):
        origin, period = -0.41, 4.2
        target = np.array([origin - 1.5 * period, origin - 0.37, origin, origin + 0.29, origin + 1.3 * period])

        def expected(points, closure):
            angle = (points - origin) * 2 * np.pi / period
            if closure < 0:
                return np.column_stack((np.cos(angle / 2), np.sin(3 * angle / 2)))
            return np.column_stack((0.7 + np.cos(2 * angle), np.sin(angle)))

        for count in (9, 10):
            grid = origin + period * np.arange(count) / count
            for closure in (-1, 1):
                with self.subTest(count=count, closure=closure):
                    data = expected(grid, closure)
                    field = evaluate_rotor_samples(grid, data, target, period=period, closure_sign=closure)
                    wrapped = evaluate_rotor_samples(grid, data, target + period, period=period, closure_sign=closure)
                    np.testing.assert_allclose(field, expected(target, closure), atol=1e-14)
                    np.testing.assert_allclose(wrapped, closure * field, atol=1e-14)

    def test_rotor_grid_origin_conversion_uses_real_nyquist_cosine(self):
        grid = 0.31 + 2 * np.pi * np.arange(8) / 8
        targets = grid + 0.11
        field = evaluate_rotor_samples(grid, (-1.) ** np.arange(8), targets)
        np.testing.assert_allclose(field, np.cos(4 * (targets - grid[0])), atol=1e-14)

    def test_free_antiperiodic_rotor_has_half_integer_kinetic_modes(self):
        period, inertia, offset = 3.3, 197.0, -12.0
        for n_points in (23, 24):
            with self.subTest(n_points=n_points):
                grid = np.linspace(-0.7, -0.7 + period, n_points, endpoint=False)
                surface = solve_dvr(
                    grid, np.full(n_points, offset), boundary="antiperiodic",
                    period=period, kinetic_coefficient=1 / inertia, n_levels=6,
                )
                modes = np.arange(n_points) - n_points // 2 + 0.5
                expected = offset + np.sort((2 * np.pi / period * modes)**2) / (2 * inertia)
                np.testing.assert_allclose(surface.all_energies_hartree, expected, rtol=0, atol=1e-14)
                local = (grid - grid[0]) * np.pi / period
                basis = np.column_stack((np.cos(local), np.sin(local))) * np.sqrt(2 / n_points)
                first_pair = surface.coefficients[:, :2]
                np.testing.assert_allclose(first_pair @ first_pair.T, basis @ basis.T, atol=1e-13)

    def test_independent_gauges_and_closed_scan_interpolation(self):
        count, fine_count = 9, 42
        origin = 0.23
        angle = 2 * np.pi * np.arange(count) / count
        periodic = np.ones((count, 1))
        reversing = _electronic_states(angle)
        ground_signs = np.array([1, 1, -1, 1, -1, -1, 1, 1, -1])
        excited_signs = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])

        def potentials(theta):
            return -100 + 0.012 * np.cos(theta), -99.9 + 0.025 * np.sin(2 * theta)

        fine = 2 * np.pi * np.arange(fine_count) / fine_count
        ground_expected, excited_expected = potentials(fine)
        cases = ((1, periodic), (-1, reversing))
        for ground_closure, ground in cases:
            for excited_closure, excited in cases:
                with self.subTest(ground=ground_closure, excited=excited_closure):
                    parity = ground_closure * excited_closure
                    frequency = 0.5 if parity == -1 else 1.0

                    def transition(theta):
                        return np.column_stack((
                            np.cos(frequency * theta), np.sin(3 * frequency * theta), np.zeros_like(theta),
                        ))

                    raw = (ground_signs * excited_signs)[:, None] * transition(angle)
                    result = phase_periodic_scan(
                        origin + angle, *potentials(angle), raw,
                        ground_overlaps=_signed_links(ground_signs[:, None] * ground),
                        excited_overlaps=_signed_links(excited_signs[:, None] * excited), n_points=fine_count,
                    )
                    np.testing.assert_allclose(result.grid, origin + fine, atol=1e-14)
                    np.testing.assert_allclose(result.ground_potential_hartree, ground_expected, rtol=0, atol=5e-14)
                    np.testing.assert_allclose(result.excited_potential_hartree, excited_expected, rtol=0, atol=5e-14)
                    np.testing.assert_allclose(result.transition_dipole.values_au, transition(fine), atol=1e-14)
                    np.testing.assert_array_equal(result.ground_source_phases, ground_signs)
                    np.testing.assert_array_equal(result.excited_source_phases, excited_signs)
                    self.assertEqual(
                        (result.ground_closure_sign, result.excited_closure_sign), (ground_closure, excited_closure),
                    )
                    self.assertEqual(result.transition_dipole.closure_sign, parity)

    def test_unqualified_closing_link_is_rejected(self):
        angle = 2 * np.pi * np.arange(5) / 5
        ground = np.ones(5)
        excited = np.array([1, 1, 1, 1, -0.1])
        dipole = np.zeros((5, 3))
        with self.assertRaisesRegex(StateTrackingError, "excited scan link 4 to 0 has insufficient overlap"):
            phase_periodic_scan(angle, ground, excited, dipole, ground_overlaps=ground, excited_overlaps=excited)
        with self.assertRaisesRegex(ValueError, "including closure"):
            phase_periodic_scan(angle, ground, excited, dipole, ground_overlaps=ground, excited_overlaps=excited[:-1])

    def test_dipole_selection_for_mixed_and_matched_antiperiodic_boundaries(self):
        angle = 2 * np.pi * np.arange(24) / 24
        excited_links = _signed_links(_electronic_states(angle))
        for both in (False, True):
            with self.subTest(both_antiperiodic=both):
                ground_links = excited_links if both else np.ones(len(angle))
                dipole = np.zeros((len(angle), 3))
                dipole[:, 0] = 1 if both else np.cos(angle / 2)
                scan = phase_periodic_scan(
                    angle, np.zeros_like(angle), np.full_like(angle, 0.1), dipole,
                    ground_overlaps=ground_links, excited_overlaps=excited_links,
                )
                ground = solve_dvr(scan.grid, scan.ground_potential_hartree, boundary=scan.ground_boundary,
                                   n_levels=2 if both else 1, kinetic_coefficient=0.01)
                excited = solve_dvr(scan.grid, scan.excited_potential_hartree, boundary=scan.excited_boundary,
                                    n_levels=4, kinetic_coefficient=0.01)
                spectrum = anharmonic_spectrum(ground, excited, transition_dipole_au=scan.transition_dipole)
                expected_strength = 1.0 if both else 0.5
                expected_energy = 0.1 if both else 0.10125
                low_pair = spectrum.final_level < 2
                np.testing.assert_allclose(spectrum.photon_energy_hartree[low_pair], expected_energy, atol=1e-14)
                self.assertAlmostEqual(spectrum.thermal_strength_au2[low_pair].sum(), expected_strength, places=12)
                np.testing.assert_allclose(spectrum.dipole_strength_au2[~low_pair], 0, atol=1e-14)
                self.assertAlmostEqual(spectrum.final_dipole_strength_coverage, 1.0)
                emission = anharmonic_spectrum(
                    excited, ground, transition_dipole_au=scan.transition_dipole, process="emission",
                )
                np.testing.assert_allclose(emission.photon_energy_hartree, expected_energy, atol=1e-14)
                self.assertAlmostEqual(emission.thermal_strength_au2.sum(), 1.0 if both else 0.25, places=12)

                thermal = thermal_separable_spectrum(
                    ground, excited, transition_dipole_au=scan.transition_dipole, process="absorption",
                    fwhm_cm_inverse=60.0, temperature_kelvin=0.0,
                )
                self.assertAlmostEqual(thermal.signed_dipole_strength_au2, expected_strength, places=12)
                self.assertAlmostEqual(thermal.positive_photon_weighted_strength,
                                       expected_strength * expected_energy, places=12)

                with self.assertRaisesRegex(ValueError, "require a phased rotor transition dipole"):
                    anharmonic_spectrum(ground, excited, transition_dipole_au=dipole)
                wrong = replace(scan.transition_dipole, closure_sign=-scan.transition_dipole.closure_sign)
                with self.assertRaisesRegex(ValueError, "closure sign is incompatible"):
                    thermal_separable_spectrum(
                        ground, excited, transition_dipole_au=wrong, process="absorption",
                        fwhm_cm_inverse=60.0, temperature_kelvin=0.0,
                    )
                shifted = replace(scan.transition_dipole, grid=scan.grid + 0.1)
                with self.assertRaisesRegex(ValueError, "common rotor grid"):
                    anharmonic_spectrum(ground, excited, transition_dipole_au=shifted)
