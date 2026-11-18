"""Adaptive spectra from an explicitly selected prepared electronic method."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd import WorkflowIncomplete, run_prepared_ah
from pyscf_esd.harmonic import _KB_HARTREE_PER_K
from pyscf_esd.pyscf_adapter import AdiabaticHessianParameters
from pyscf_esd.spectral_properties import _EINSTEIN_A_FACTOR


def _prepared_oscillator():
    modes = np.array([[[0., 0., -1.], [0., 0., 1.]]])
    return AdiabaticHessianParameters(
        ground_frequencies_hartree=np.array([0.005]),
        excited_frequencies_hartree=np.array([0.005]),
        gap_00_hartree=0.1,
        rotation=np.eye(1),
        shift_au=np.array([np.sqrt(6 / 0.005)]),
        ground_cartesian_mode_derivatives_au=modes,
        excited_cartesian_mode_derivatives_au=modes,
        excited_alignment_rotation=np.eye(3),
        excited_alignment_translation_bohr=np.zeros(3),
        subspace_singular_values=np.ones(1),
        rotation_correction_norm=0.0,
        displacement_norm_au=0.,
        displacement_projection_residual_au=0.,
    )


class PreparedAHTests(unittest.TestCase):
    def test_prepared_method_keeps_the_absolute_origin_and_shared_nuclear_checks(self):
        parameters = _prepared_oscillator()
        with patch("pyscf_esd.workflow._prepare_reference", side_effect=AssertionError("electron solve")):
            result = run_prepared_ah(parameters, [0., 0., 0.2], electronic_method="wavefunction control")
        self.assertEqual(result.model_used, "ah_fc")
        self.assertEqual(result.physical_properties["electronic_method"], "wavefunction control")
        self.assertEqual(result.physical_properties["electronic_validation"], "caller_supplied")
        self.assertEqual(result.diagnostics["predicted_00_lines_hartree"], {"absorption": 0.1, "emission": 0.1})
        self.assertFalse(result.diagnostics["vibronic_terms"]["herzberg_teller"])
        for process in ("absorption", "emission"):
            self.assertGreaterEqual(result.spectra[process].final_dipole_strength_coverage, 0.995)
            self.assertGreater(len(result.diagnostics["basis_attempts"][process]), 1)

    def test_prepared_ht_propagates_partial_coverage_with_method_identity(self):
        with self.assertRaises(WorkflowIncomplete) as error:
            run_prepared_ah(
                _prepared_oscillator(), [0., 0., 0.2], electronic_method="explicit response",
                dipole_derivatives_au=[[0., 0., 0.001]], process="absorption", max_overlaps=3,
            )
        diagnostics = error.exception.diagnostics
        self.assertEqual(diagnostics["electronic_method"], "explicit response")
        self.assertEqual(diagnostics["model_used"], "ah_ht")
        self.assertTrue(diagnostics["vibronic_terms"]["herzberg_teller"])
        self.assertEqual(diagnostics["spectral_status"]["absorption"], "incomplete")
        self.assertIn("absorption", error.exception.partial_spectra)

    def test_prepared_ht_uses_affordable_initial_basis_for_many_modes(self):
        count = 81
        frequencies = np.full(count, .012)
        modes = np.eye(87)[:count].reshape(count, 29, 3)
        parameters = replace(
            _prepared_oscillator(), ground_frequencies_hartree=frequencies, excited_frequencies_hartree=frequencies,
            rotation=np.eye(count), shift_au=np.zeros(count), subspace_singular_values=np.ones(count),
            ground_cartesian_mode_derivatives_au=modes, excited_cartesian_mode_derivatives_au=modes,
        )
        derivatives = np.zeros((count, 3))
        derivatives[0, 0] = .001
        result = run_prepared_ah(
            parameters, [1, 0, 0], electronic_method="equal-curvature HT oscillators",
            dipole_derivatives_au=derivatives, temperature_kelvin=298.,
        )
        for process in ("absorption", "emission"):
            with self.subTest(process=process):
                self.assertEqual(result.diagnostics["spectral_status"][process], "complete")
                attempt, = result.diagnostics["basis_attempts"][process]
                self.assertEqual((attempt["initial_max_quanta"], attempt["final_max_quanta"]), (0, 1))
                self.assertGreaterEqual(attempt["initial_population_coverage"], .995)
                self.assertGreaterEqual(attempt["final_dipole_strength_coverage"], .995)
                for moment in attempt["positive_photon_moments"].values():
                    self.assertGreaterEqual(moment["coverage_lower_bound"], .995)

    def test_prepared_fc_can_start_with_one_overlap(self):
        parameters = replace(_prepared_oscillator(), shift_au=np.zeros(1))
        result = run_prepared_ah(
            parameters, [1, 0, 0], electronic_method="equal-curvature FC oscillator",
            process="absorption", max_overlaps=1,
        )
        attempt, = result.diagnostics["basis_attempts"]["absorption"]
        self.assertEqual((attempt["initial_max_quanta"], attempt["final_max_quanta"]), (0, 0))
        np.testing.assert_allclose(result.spectra["absorption"].dipole_strength_au2, [1.])

    def test_prepared_fc_trades_final_quanta_for_initial_population_within_budget(self):
        count = 8
        frequencies = np.full(count, .006)
        modes = np.eye(15)[:count].reshape(count, 5, 3)
        parameters = replace(
            _prepared_oscillator(), ground_frequencies_hartree=frequencies, excited_frequencies_hartree=frequencies,
            rotation=np.eye(count), shift_au=np.zeros(count), subspace_singular_values=np.ones(count),
            ground_cartesian_mode_derivatives_au=modes, excited_cartesian_mode_derivatives_au=modes,
        )
        result = run_prepared_ah(
            parameters, [1, 0, 0], electronic_method="equal-curvature FC oscillators",
            temperature_kelvin=298., max_overlaps=100,
        )
        for process in ("absorption", "emission"):
            with self.subTest(process=process):
                first, qualified = result.diagnostics["basis_attempts"][process]
                self.assertEqual((first["initial_max_quanta"], first["final_max_quanta"]), (0, 2))
                self.assertLess(first["initial_population_coverage"], .995)
                self.assertEqual((qualified["initial_max_quanta"], qualified["final_max_quanta"]), (1, 1))
                self.assertEqual(result.diagnostics["spectral_status"][process], "complete")

    def test_budget_fallback_checks_both_directions_of_the_harmonic_basis(self):
        omega = .01
        base = replace(_prepared_oscillator(), gap_00_hartree=.005,
                       ground_frequencies_hartree=np.array([omega]), excited_frequencies_hartree=np.array([omega]))
        cases = (
            ("absorption", .2, .02, .95, 4, None, (0, 3)),
            ("emission", .02, .08, .995, 27, [[.04, 0, 0]], (7, 2)),
        )
        for process, displacement, thermal_ratio, coverage, budget, slopes, cutoffs in cases:
            with self.subTest(process=process):
                parameters = replace(base, shift_au=np.array([np.sqrt(2 * displacement / omega)]))
                result = run_prepared_ah(
                    parameters, [1, 0, 0], electronic_method="displaced oscillator", process=process,
                    temperature_kelvin=-omega / (_KB_HARTREE_PER_K * np.log(thermal_ratio)),
                    coverage=coverage, max_overlaps=budget, dipole_derivatives_au=slopes,
                )
                final = result.diagnostics["basis_attempts"][process][-1]
                self.assertEqual((final["initial_max_quanta"], final["final_max_quanta"]), cutoffs)
                self.assertEqual(result.diagnostics["spectral_status"][process], "complete")

    def test_absorption_strength_requires_m1_beyond_dipole_closure(self):
        omega, gap, displacement_factor = .01, 1e-6, .3
        parameters = replace(
            _prepared_oscillator(), gap_00_hartree=gap,
            ground_frequencies_hartree=np.array([omega]), excited_frequencies_hartree=np.array([omega]),
            shift_au=np.array([np.sqrt(2 * displacement_factor / omega)]),
        )
        result = run_prepared_ah(parameters, [0, 0, 1], electronic_method="displaced oscillator", process="absorption")
        attempts = result.diagnostics["basis_attempts"]["absorption"]
        self.assertGreaterEqual(attempts[0]["final_dipole_strength_coverage"], .995)
        self.assertEqual(attempts[0]["positive_photon_moments"]["M1"]["status"], "incomplete")
        moment = result.physical_properties["spectra"]["absorption"]["positive_photon_moments"]["M1"]
        self.assertEqual(moment["status"], "complete")
        true_moment = gap + omega * displacement_factor
        self.assertLessEqual(moment["coverage_lower_bound"], moment["retained"] / true_moment)
        self.assertGreaterEqual(moment["retained"], .995 * true_moment)

    def test_thermal_ht_hot_band_cannot_pass_on_population_and_dipole_closure(self):
        omega, gap, slope, ratio = .01, .01001, .01, .001
        parameters = replace(
            _prepared_oscillator(), gap_00_hartree=gap, shift_au=np.zeros(1),
            ground_frequencies_hartree=np.array([omega]), excited_frequencies_hartree=np.array([omega]),
        )
        options = dict(
            electronic_method="equal-curvature HT oscillator", process="emission",
            dipole_derivatives_au=[[0, 0, slope]], temperature_kelvin=-omega / (_KB_HARTREE_PER_K * np.log(ratio)),
        )
        solve_spectrum = parameters.spectrum
        with patch.object(AdiabaticHessianParameters, "spectrum", autospec=True,
                          side_effect=lambda _, **options: solve_spectrum(**options)) as solve:
            with self.assertRaisesRegex(WorkflowIncomplete, "overlap budget") as error:
                run_prepared_ah(parameters, [0, 0, 0], max_overlaps=6, **options)
        tried = [(call.kwargs["initial_max_quanta"], call.kwargs["final_max_quanta"]) for call in solve.call_args_list]
        self.assertEqual(len(tried), len(set(tried)))
        self.assertNotIn((2, 1), tried)
        attempts = error.exception.diagnostics["basis_attempts"]["emission"]
        first, best, last = attempts[0], attempts[1], attempts[-1]
        self.assertGreaterEqual(first["initial_population_coverage"], .995)
        self.assertGreaterEqual(first["final_dipole_strength_coverage"], .995)
        self.assertEqual(first["positive_photon_moments"]["M3"]["status"], "incomplete")
        self.assertEqual(first["positive_photon_moments"]["M4"]["status"], "incomplete")
        rate_cold = first["positive_photon_moments"]["M3"]["retained"] * _EINSTEIN_A_FACTOR
        self.assertLess(rate_cold, 1.1e-7)
        self.assertLess(last["final_dipole_strength_coverage"], .005)
        partial = error.exception.partial_spectra["emission"]
        self.assertEqual(partial.final_dipole_strength_coverage, best["final_dipole_strength_coverage"])
        self.assertEqual(partial.positive_photon_moments[4].coverage_lower_bound,
                         best["positive_photon_moments"]["M4"]["coverage_lower_bound"])

        result = run_prepared_ah(parameters, [0, 0, 0], **options)
        final = result.diagnostics["basis_attempts"]["emission"][-1]
        self.assertGreater(final["initial_max_quanta"], first["initial_max_quanta"])
        report = result.physical_properties["spectra"]["emission"]
        occupation = ratio / (1 - ratio)
        for order in (3, 4):
            exact = slope**2 / (2 * omega) * (
                (occupation + 1) * (gap - omega)**order + occupation * (gap + omega)**order
            )
            moment = report["positive_photon_moments"][f"M{order}"]
            self.assertEqual(moment["status"], "complete")
            self.assertLessEqual(moment["coverage_lower_bound"], moment["retained"] / exact)
            self.assertGreaterEqual(moment["retained"], .995 * exact)
        self.assertAlmostEqual(report["vacuum_radiative_rate_s_inverse"], .85894, places=5)

    def test_emission_rate_and_radiant_power_have_independent_checks(self):
        omega, ratio = .01, 4e-5
        parameters = replace(
            _prepared_oscillator(), gap_00_hartree=2 * omega, shift_au=np.zeros(1),
            ground_frequencies_hartree=np.array([omega]), excited_frequencies_hartree=np.array([omega]),
        )
        result = run_prepared_ah(
            parameters, [0, 0, 0], electronic_method="equal-curvature HT oscillator", process="emission",
            dipole_derivatives_au=[[0, 0, .01]], temperature_kelvin=-omega / (_KB_HARTREE_PER_K * np.log(ratio)),
        )
        attempts = result.diagnostics["basis_attempts"]["emission"]
        first = attempts[0]["positive_photon_moments"]
        self.assertEqual(first["M3"]["status"], "complete")
        self.assertEqual(first["M4"]["status"], "incomplete")
        self.assertEqual(attempts[-1]["positive_photon_moments"]["M4"]["status"], "complete")
