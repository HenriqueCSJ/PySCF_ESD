"""Adaptive spectra from an explicitly selected prepared electronic method."""

import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd import WorkflowIncomplete, run_prepared_ah
from pyscf_esd.pyscf_adapter import AdiabaticHessianParameters


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
