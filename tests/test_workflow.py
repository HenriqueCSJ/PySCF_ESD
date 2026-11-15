import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyscf_esd import WorkflowIncomplete, run_vibronic
from pyscf_esd.cli import main
from pyscf_esd.harmonic import harmonic_spectrum


class _AnharmonicUnavailable(ValueError):
    pass


def spectrum(**options):
    # Identical wells with a displacement yielding a 3-quantum progression.
    return harmonic_spectrum([.005], [.005], gap_00_hartree=.1, shift_au=[np.sqrt(6 / .005)], **options)


class FirstRunTests(unittest.TestCase):
    def setUp(self):
        self.geometry = np.array([[0., 0., 0.], [0., 0., 2.]])
        self.electronic = SimpleNamespace(
            model=SimpleNamespace(spectrum=Mock(side_effect=spectrum), gap_00_hartree=.1),
            ground_coordinates_bohr=self.geometry,
            excited_coordinates_bohr=self.geometry + .1,
            transition_dipole_au=np.array([1., 0., 0.]),
            dipole_derivatives_au=np.zeros((1, 3)),
            diagnostics={"phase_tracking": True},
        )
        self.anharmonic = SimpleNamespace(diagnostics={"status": "harmonic_screened"}, spectrum=Mock())
        self.mock_electronic = SimpleNamespace(
            optimize_ground=Mock(return_value=self.geometry),
            prepare_electronic=Mock(return_value=self.electronic),
        )
        self.mock_anharmonic = SimpleNamespace(
            AnharmonicUnavailableError=_AnharmonicUnavailable,
            prepare_anharmonic=Mock(return_value=self.anharmonic),
        )
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            "pyscf_esd.electronic_workflow": self.mock_electronic,
            "pyscf_esd.automatic_anharmonic": self.mock_anharmonic,
        }).start()

    def test_default_uses_ah_ht_and_screens_then_converges(self):
        result = run_vibronic(("H", "H"), self.geometry, process="absorption")
        self.assertEqual(result.model_used, "ah_ht")
        self.mock_anharmonic.prepare_anharmonic.assert_called_once_with(self.electronic)
        self.anharmonic.spectrum.assert_not_called()
        history = result.diagnostics["basis_attempts"]["absorption"]
        self.assertGreater(history[-1]["final_max_quanta"], history[0]["final_max_quanta"])
        self.assertGreaterEqual(history[-1]["final_dipole_strength_coverage"], .995)
        np.testing.assert_array_equal(
            self.electronic.model.spectrum.call_args.kwargs["dipole_derivatives_au"], np.zeros((1, 3)),
        )

    def test_auto_does_not_substitute_harmonic_when_screening_fails(self):
        self.anharmonic.diagnostics = {"status": "unsupported", "reasons": ["multiple wells"]}
        with self.assertRaisesRegex(WorkflowIncomplete, "multiple wells") as caught:
            run_vibronic(("H", "H"), self.geometry)
        self.assertEqual(caught.exception.diagnostics["anharmonic"]["status"], "unsupported")
        self.electronic.model.spectrum.assert_not_called()

    def test_explicit_ah_can_bypass_anharmonic_screening(self):
        result = run_vibronic(("H", "H"), self.geometry, model="ah", process="absorption")
        self.assertEqual(result.model_used, "ah_ht")
        self.mock_anharmonic.prepare_anharmonic.assert_not_called()

    def test_anharmonic_thermal_tail_is_not_treated_as_qualified(self):
        self.anharmonic.diagnostics = {"status": "corrected"}
        with self.assertRaisesRegex(WorkflowIncomplete, "thermal partition tail"):
            run_vibronic(("H", "H"), self.geometry, temperature_kelvin=700)
        self.anharmonic.spectrum.assert_not_called()

    def test_inadequate_budget_retains_diagnostics(self):
        with self.assertRaisesRegex(WorkflowIncomplete, "overlap budget") as caught:
            run_vibronic(("H", "H"), self.geometry, process="absorption", max_overlaps=3)
        diagnostics = caught.exception.diagnostics
        self.assertEqual(len(diagnostics["spectrum_failures"]["absorption"]["basis_attempts"]), 1)
        self.assertEqual(diagnostics["spectral_status"]["absorption"], "incomplete")
        self.assertIn("absorption", caught.exception.partial_spectra)

    def test_cli_writes_only_diagnostics_for_incomplete_auto(self):
        self.anharmonic.diagnostics = {"status": "unsupported", "reasons": ["multiple wells"]}
        with TemporaryDirectory() as directory:
            xyz = Path(directory) / "raw.xyz"
            xyz.write_text("2\nneutral\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
            fake_pyscf = SimpleNamespace(lib=SimpleNamespace(num_threads=Mock(return_value=2)))
            with patch("sys.stderr"), patch.dict(sys.modules, {"pyscf": fake_pyscf}):
                self.assertEqual(main([str(xyz)]), 2)
            output = xyz.with_suffix(".esd")
            self.assertEqual([path.name for path in output.iterdir()], ["diagnostics.json"])
            report = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["spectra"], {})

    def test_cli_labels_last_valid_spectrum_as_partial(self):
        with TemporaryDirectory() as directory:
            xyz = Path(directory) / "raw.xyz"
            xyz.write_text("2\nneutral\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
            fake_pyscf = SimpleNamespace(lib=SimpleNamespace(num_threads=Mock(return_value=2)))
            with patch("sys.stderr"), patch.dict(sys.modules, {"pyscf": fake_pyscf}):
                self.assertEqual(main([str(xyz), "--process", "absorption", "--max-overlaps", "3"]), 2)
            report = json.loads((xyz.with_suffix(".esd") / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["spectra"]["absorption"]["sticks"], "absorption_partial_sticks.csv")
            self.assertEqual(report["spectra"]["absorption"]["coverage_status"], "incomplete")

    def test_zero_temperature_recurrence_matches_complete_overlap_tensor(self):
        angle = .41
        rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        common = dict(
            gap_00_hartree=.14, shift_au=[2., -3.], transition_dipole_au=[.2, .1, -.3],
            dipole_derivatives_au=[[.02, .04, -.01], [-.02, .01, .02]],
            rotation=rotation, initial_max_quanta=0, final_max_quanta=5,
        )
        args = ([.004, .007], [.006, .008])
        zero = harmonic_spectrum(*args, **common)
        low = harmonic_spectrum(*args, **common, temperature_kelvin=1e-6)
        np.testing.assert_allclose(zero.transition_dipole_au, low.transition_dipole_au, atol=1e-14)
        self.assertAlmostEqual(zero.final_dipole_strength_coverage, low.final_dipole_strength_coverage)
