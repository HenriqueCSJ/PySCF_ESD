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
        with self.assertRaisesRegex(WorkflowIncomplete, "overlap budget") as error:
            run_prepared_ah(parameters, [0, 0, 0], max_overlaps=6, **options)
        first = error.exception.diagnostics["basis_attempts"]["emission"][0]
        self.assertGreaterEqual(first["initial_population_coverage"], .995)
        self.assertGreaterEqual(first["final_dipole_strength_coverage"], .995)
        self.assertEqual(first["positive_photon_moments"]["M3"]["status"], "incomplete")
        self.assertEqual(first["positive_photon_moments"]["M4"]["status"], "incomplete")
        rate_cold = first["positive_photon_moments"]["M3"]["retained"] * _EINSTEIN_A_FACTOR
        self.assertLess(rate_cold, 1.1e-7)

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
