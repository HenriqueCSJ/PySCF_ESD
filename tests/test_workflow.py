from collections import OrderedDict
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyscf_esd import WorkflowIncomplete, run_vibronic
from pyscf_esd.cli import _write_profile, main
from pyscf_esd.cpu import ElectronicConvergenceError, ReferenceStabilityUnresolvedError, UnstableReferenceError
from pyscf_esd.harmonic import harmonic_spectrum
from pyscf_esd.pyscf_adapter import VerticalGradientParameters
from pyscf_esd.state_tracking import StateTrackingError


class _AnharmonicUnavailable(ValueError):
    pass


def spectrum(**options):
    # Identical wells with a displacement yielding a 3-quantum progression.
    gap = -.1 if options.get("process") == "emission" else .1
    return harmonic_spectrum([.005], [.005], gap_00_hartree=gap, shift_au=[np.sqrt(6 / .005)], **options)


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

    def test_gpu_ah_and_auto_use_the_analytic_electronic_backend(self):
        prepare_gpu = Mock(return_value=self.electronic)
        with patch.dict(sys.modules, {
            "pyscf_esd.gpu_adiabatic": SimpleNamespace(prepare_gpu_electronic=prepare_gpu),
        }):
            ah = run_vibronic(("H", "H"), self.geometry, backend="gpu", model="ah", solvent="water")
            auto = run_vibronic(("H", "H"), self.geometry, backend="gpu")
        self.mock_electronic.prepare_electronic.assert_not_called()
        self.assertEqual(set(ah.spectra), {"absorption", "emission"})
        self.assertEqual(ah.diagnostics["electronic_backend"], "gpu")
        self.assertEqual(ah.diagnostics["solvent"], "water")
        self.assertEqual(prepare_gpu.call_args_list[0].kwargs, {
            "target_index": 0, "direct_dipole": False, "solvent": "water",
        })
        self.assertEqual(prepare_gpu.call_args_list[1].kwargs, {
            "target_index": 0, "direct_dipole": True, "solvent": None,
        })
        self.assertEqual(auto.diagnostics["electronic_backend"], "gpu")
        self.mock_anharmonic.prepare_anharmonic.assert_called_once()

    def test_gpu_analytic_failure_never_restarts_on_cpu(self):
        prepare_gpu = Mock(side_effect=ElectronicConvergenceError("analytic response failed"))
        with patch.dict(sys.modules, {
            "pyscf_esd.gpu_adiabatic": SimpleNamespace(prepare_gpu_electronic=prepare_gpu),
        }), self.assertRaisesRegex(WorkflowIncomplete, "analytic response failed"):
            run_vibronic(("H", "H"), self.geometry, backend="gpu", model="ah")
        self.mock_electronic.prepare_electronic.assert_not_called()

    def test_gpu_vg_uses_the_relaxed_gpu_reference(self):
        model = VerticalGradientParameters(
            np.array([.005]), np.array([np.sqrt(6 / .005)]), .1,
            np.array([1., 0., 0.]), .015, np.zeros((1, 2, 3)),
        )
        vg = SimpleNamespace(model=model, spectrum=lambda **options: spectrum(
            transition_dipole_au=model.transition_dipole_au, **options,
        ))
        prepare_gpu = Mock(return_value=(self.geometry + .1, vg))
        with patch.dict(sys.modules, {
            "pyscf_esd.gpu_adiabatic": SimpleNamespace(prepare_gpu_vg=prepare_gpu),
        }):
            result = run_vibronic(("H", "H"), self.geometry, backend="gpu", model="vg", process="absorption")
        prepare_gpu.assert_called_once()
        np.testing.assert_allclose(result.diagnostics["ground_coordinates_bohr"], self.geometry + .1)
        self.assertEqual(result.model_used, "vg")
        self.mock_electronic.optimize_ground.assert_not_called()

    def test_internal_reference_retry_preserves_functional(self):
        prepare = self.mock_electronic.prepare_electronic
        prepare.side_effect = [
            UnstableReferenceError("internal", "S0 minimum"), self.electronic,
        ]
        result = run_vibronic(("H", "H"), self.geometry, model="ah", process="absorption", fallback_xc=("pbe",))
        self.assertEqual(result.diagnostics["settings_requested"]["xc"], "pbe0")
        self.assertEqual(result.diagnostics["settings"]["xc"], "pbe0")
        self.assertEqual(result.diagnostics["settings"]["stability_checks"], "all")
        self.assertEqual(prepare.call_count, 2)

    def test_external_reference_retry_restarts_entire_electronic_model(self):
        prepare = self.mock_electronic.prepare_electronic
        prepare.side_effect = [
            UnstableReferenceError("restricted_to_unrestricted", "tracked excited minimum"), self.electronic,
        ]
        result = run_vibronic(("H", "H"), self.geometry, model="ah", process="absorption", fallback_xc=("pbe",))
        self.assertEqual([call.args[2].xc for call in prepare.call_args_list], ["pbe0", "pbe"])
        for call in prepare.call_args_list:
            np.testing.assert_equal(call.args[1], self.geometry)
        self.assertEqual(result.diagnostics["settings"]["stability_checks"], "all")
        self.assertEqual(result.diagnostics["settings"]["xc"], "pbe")
        self.assertEqual(result.diagnostics["reference_attempts"][0]["geometry_stage"], "tracked excited minimum")

    def test_external_reference_without_alternate_terminates_with_reason(self):
        prepare = self.mock_electronic.prepare_electronic
        prepare.side_effect = UnstableReferenceError("restricted_to_unrestricted", "Optimized S0 reference")
        with self.assertRaisesRegex(WorkflowIncomplete, "restricted-to-unrestricted") as caught:
            run_vibronic(("H", "H"), self.geometry)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(caught.exception.diagnostics["reference_attempts"][0]["status"], "unstable")

    def test_ambiguous_target_is_reported_without_changing_the_functional(self):
        prepare = self.mock_electronic.prepare_electronic
        prepare.side_effect = StateTrackingError("ambiguous state identity")
        with self.assertRaisesRegex(WorkflowIncomplete, "ambiguous state identity") as caught:
            run_vibronic(("H", "H"), self.geometry, fallback_xc=("pbe",))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(caught.exception.diagnostics["reference_attempts"][0]["failure_kind"], "state_tracking")

    def test_unresolved_stability_does_not_change_the_functional(self):
        self.mock_anharmonic.prepare_anharmonic.side_effect = ReferenceStabilityUnresolvedError(
            "restricted_to_unrestricted", "anharmonic sample", "orbital eigensolver residual exceeded its tolerance"
        )
        with self.assertRaises(WorkflowIncomplete) as caught:
            run_vibronic(("H", "H"), self.geometry, fallback_xc=("pbe",))
        self.assertEqual(self.mock_electronic.prepare_electronic.call_count, 1)
        attempt, = caught.exception.diagnostics["reference_attempts"]
        self.assertEqual(attempt["status"], "stability_unresolved")
        self.assertEqual(attempt["channel"], "restricted_to_unrestricted")
        self.assertEqual(attempt["geometry_stage"], "anharmonic sample")

    def test_instability_during_anharmonic_sampling_restarts_electronic_preparation(self):
        screening = self.mock_anharmonic.prepare_anharmonic
        screening.side_effect = [
            UnstableReferenceError("restricted_to_unrestricted", "anharmonic sample"), self.anharmonic,
        ]
        result = run_vibronic(("H", "H"), self.geometry, process="absorption", fallback_xc=("pbe",))
        calls = self.mock_electronic.prepare_electronic.call_args_list
        self.assertEqual([call.args[2].xc for call in calls], ["pbe0", "pbe"])
        self.assertEqual(result.diagnostics["settings"]["xc"], "pbe")
        self.assertEqual(result.diagnostics["reference_attempts"][0]["geometry_stage"], "anharmonic sample")

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

    def test_default_cli_screens_ah_and_exports_both_directions(self):
        with TemporaryDirectory() as directory:
            xyz = Path(directory) / "raw.xyz"
            xyz.write_text("2\nneutral\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
            fake_pyscf = SimpleNamespace(lib=SimpleNamespace(num_threads=Mock(return_value=2)))
            with patch("sys.stdout"), patch("sys.stderr"), patch.dict(sys.modules, {"pyscf": fake_pyscf}):
                self.assertEqual(main([str(xyz)]), 0)
            output = xyz.with_suffix(".esd")
            report = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model_requested"], "auto")
            self.assertEqual(report["model_used"], "ah_ht")
            self.assertEqual(report["gaussian_sigma_cm^-1"], 50.0)
            self.assertEqual(set(report["spectra"]), {"absorption", "emission"})
            diagnostics = report["diagnostics"]
            self.assertEqual(diagnostics["temperature_kelvin"], 0.0)
            self.assertEqual(diagnostics["reference_target_index"], 0)
            self.assertEqual(diagnostics["settings"]["basis"], "def2-tzvpd")
            self.assertEqual(diagnostics["settings"]["xc"], "pbe0")
            self.assertEqual(diagnostics["settings"]["stability_checks"], "minima")
            self.mock_anharmonic.prepare_anharmonic.assert_called_once_with(self.electronic)
            properties = json.loads((output / "properties.json").read_text(encoding="utf-8"))
            self.assertIn("gaussian_peak_stokes_shift_cm^-1", properties)
            self.assertGreater(properties["spectra"]["absorption"]["population_weighted_oscillator_strength_sum"], 0)
            self.assertGreater(properties["spectra"]["emission"]["vacuum_radiative_rate_s_inverse"], 0)
            with (output / "emission_sticks.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(max(float(row["vacuum_einstein_a_s^-1"]) for row in rows), 0)
            self.assertTrue(all(row["oscillator_strength"] == "" for row in rows))

    def test_explicit_cli_vg_uses_selected_root_and_writes_sticks_only(self):
        vg_model = VerticalGradientParameters(
            np.array([.005]), np.array([np.sqrt(6 / .005)]), .1,
            np.array([1., 0., 0.]), .015, np.zeros((1, 2, 3)),
        )
        vg = SimpleNamespace(model=vg_model, spectrum=lambda **options: spectrum(
            transition_dipole_au=vg_model.transition_dipole_au, **options,
        ))
        with TemporaryDirectory() as directory:
            xyz = Path(directory) / "raw.xyz"
            xyz.write_text("2\nneutral\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
            fake_pyscf = SimpleNamespace(lib=SimpleNamespace(num_threads=Mock(return_value=2)))
            with (patch("sys.stdout"), patch("sys.stderr"), patch.dict(sys.modules, {"pyscf": fake_pyscf}),
                  patch("pyscf_esd.cpu_workflow.run_cpu_vibronic", return_value=vg) as run_vg):
                self.assertEqual(main([
                    str(xyz), "--model", "vg", "--process", "absorption", "--state", "2", "--sigma", "0",
                    "--reference-checks", "all",
                ]), 0)
            report = json.loads((xyz.with_suffix(".esd") / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["model_requested"], "vg")
        self.assertEqual(report["model_used"], "vg")
        self.assertEqual(report["diagnostics"]["vibronic_terms"], {"duschinsky": False, "herzberg_teller": False})
        self.assertEqual(set(report["spectra"]), {"absorption"})
        self.assertNotIn("profile", report["spectra"]["absorption"])
        self.assertEqual(run_vg.call_args.kwargs["target_index"], 1)
        self.assertEqual(run_vg.call_args.args[2].stability_checks, "all")
        self.mock_electronic.prepare_electronic.assert_not_called()
        self.mock_anharmonic.prepare_anharmonic.assert_not_called()

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


class ExcitedRelaxationTrackingTests(unittest.TestCase):
    def test_rejects_an_energy_rank_change_even_with_nearly_perfect_overlap(self):
        from pyscf_esd.electronic_workflow import _ElectronicSession, _TrackedPoint
        from pyscf_esd.state_tracking import StateTrackingError, track_state

        def response(x):
            energies, vectors = np.linalg.eigh([[1 + x, 1e-6], [1e-6, 1 - x]])
            return SimpleNamespace(excitation_energies_hartree=energies, amplitudes=vectors.T[:, None, :])

        previous = _TrackedPoint(np.array([[-0.1, 0, 0]]), response(-0.1), root=0, phase=1)
        trial_coordinates = np.array([[0.1, 0, 0]])
        candidate = response(0.1)
        assignment = track_state(
            previous.point.amplitudes[0], candidate.amplitudes, np.eye(1), np.eye(2),
            candidate_energies_hartree=candidate.excitation_energies_hartree,
            minimum_gap_hartree=1e-4,
        )
        self.assertEqual(assignment.index, 1)
        self.assertGreater(assignment.overlap, 0.99999)

        session = _ElectronicSession.__new__(_ElectronicSession)
        session.tracked_cache = OrderedDict()
        session.anchors = [previous]
        session._calculate = Mock(return_value=candidate)
        session._match = Mock(return_value=assignment)
        with self.assertRaisesRegex(StateTrackingError, "energy rank"):
            session.isolated_point(trial_coordinates, previous)
        self.assertFalse(session.tracked_cache)

        # A geometry already cached by a probe still needs the relaxation check.
        cached = _TrackedPoint(trial_coordinates, candidate, assignment.index, assignment.phase)
        session.tracked_cache[trial_coordinates.tobytes()] = cached
        with self.assertRaisesRegex(StateTrackingError, "energy rank"):
            session.isolated_point(trial_coordinates, previous)


class GaussianProfileTests(unittest.TestCase):
    def test_local_peak_refinement_removes_most_grid_alignment_error(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "profile.csv"
            first = _write_profile(output, np.array([8001., 10003.]), np.array([.05, 1.]), 50.)
            second = _write_profile(output, np.array([10003., 12005.]), np.array([1., .05]), 50.)
            difference = abs(first["local_quadratic_peak_cm^-1"] - second["local_quadratic_peak_cm^-1"])
            self.assertLess(difference, .05)
