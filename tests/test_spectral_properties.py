"""Physical units and population conventions for vibronic spectral summaries."""

import json
from math import pi, sqrt
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd.anharmonic import AnharmonicSpectrum
from pyscf_esd.harmonic import HarmonicSpectrum
from pyscf_esd.spectral_properties import spectral_line_properties, summarize_spectral_properties


class SpectralPropertiesTests(unittest.TestCase):
    def test_single_line_agrees_with_si_electric_dipole_formulas(self):
        # Independent SI expressions for a 2.5 eV line with a 1-Debye dipole.
        electron_charge = 1.602176634e-19
        electron_mass = 9.1093837139e-31
        vacuum_permittivity = 8.8541878188e-12
        speed_of_light = 299792458
        hbar = 6.62607015e-34 / (2 * pi)
        dipole_si = 1e-21 / speed_of_light
        energy_joule = 2.5 * electron_charge
        omega = energy_joule / hbar
        expected_f = 2 * electron_mass * energy_joule * dipole_si**2 / (3 * hbar**2 * electron_charge**2)
        expected_a = omega**3 * dipole_si**2 / (3 * pi * vacuum_permittivity * hbar * speed_of_light**3)
        spectrum = SimpleNamespace(
            photon_energy_hartree=np.array([energy_joule / 4.3597447222060e-18]),
            population=np.array([0.25]),
            dipole_strength_au2=np.array([(dipole_si / 8.4783536198e-30)**2]),
            initial_population_coverage=0.25,
            final_dipole_strength_coverage=1.0,
        )

        absorption = spectral_line_properties(spectrum, "absorption")
        emission = spectral_line_properties(spectrum, "emission")

        self.assertIsNone(absorption.einstein_a_s_inverse)
        self.assertIsNone(emission.oscillator_strength)
        np.testing.assert_allclose(absorption.oscillator_strength, [expected_f], rtol=2e-9)
        np.testing.assert_allclose(absorption.population_weighted_oscillator_strength, [0.25 * expected_f], rtol=2e-9)
        np.testing.assert_allclose(emission.einstein_a_s_inverse, [expected_a], rtol=2e-9)
        np.testing.assert_allclose(emission.population_weighted_einstein_a_s_inverse, [0.25 * expected_a], rtol=2e-9)
        np.testing.assert_allclose(
            emission.population_weighted_radiant_power_w, [0.25 * expected_a * energy_joule], rtol=2e-9, atol=0,
        )
        summary = summarize_spectral_properties({"emission": spectrum})["spectra"]["emission"]
        self.assertAlmostEqual(summary["centroid_energy"]["ev"], 2.5)
        self.assertAlmostEqual(summary["centroid_energy"]["cm_inverse"], energy_joule / (hbar * 2 * pi * speed_of_light * 100))
        self.assertEqual(summary["energy_standard_deviation"]["hartree"], 0)
        self.assertNotIn("initial_assignment", summary["dominant_lines"][0])

    def test_paired_moments_mode_fractions_and_emitter_rates(self):
        absorption = HarmonicSpectrum(
            np.array([0.11, 0.14, 0.18]), np.array([[0, 0], [0, 0], [1, 0]]),
            np.array([[0, 0], [1, 1], [0, 2]]), np.array([0.75, 0.75, 0.2]),
            np.array([0.0, 1.0, 2.0]), 0.95, 0.9,
        )
        emission = HarmonicSpectrum(
            np.array([0.08, 0.11, 0.10]), np.array([[0, 0], [0, 0], [1, 0]]),
            np.array([[1, 1], [0, 0], [1, 0]]), np.array([0.75, 0.75, 0.2]),
            np.array([2.0, 1.0, 0.2]), 0.95, 1.0,
            np.array([[-sqrt(2), 0, 0], [0, 1, 0], [0, 0, sqrt(0.2)]]),
        )
        report = summarize_spectral_properties(
            {"absorption": absorption, "emission": emission},
            spectral_status={"absorption": "incomplete", "emission": "complete"},
            temperature_kelvin=300.0, max_dominant_lines=2, assignment_scheme="harmonic_quanta",
        )
        json.dumps(report, allow_nan=False)
        a = report["spectra"]["absorption"]
        e = report["spectra"]["emission"]
        self.assertEqual(a["coverage_status"], "incomplete")
        self.assertEqual(a["nonzero_weight_line_count"], 2)

        weights_a = np.array([0.0, 0.75 * 0.14, 0.2 * 2 * 0.18])
        centroid_a = float(np.average(absorption.photon_energy_hartree, weights=weights_a))
        spread_a = sqrt(np.average((absorption.photon_energy_hartree - centroid_a)**2, weights=weights_a))
        self.assertAlmostEqual(a["population_weighted_oscillator_strength_sum"], (2 / 3) * weights_a.sum())
        self.assertAlmostEqual(a["population_weighted_dipole_strength_sum_au2"], 0.75 + 0.2 * 2)
        self.assertAlmostEqual(a["centroid_energy"]["hartree"], centroid_a)
        self.assertAlmostEqual(a["energy_standard_deviation"]["hartree"], spread_a)
        self.assertEqual(a["origin_00_line"]["stick_index"], 0)
        self.assertEqual(a["origin_00_line"]["dipole_strength_au2"], 0.0)
        self.assertEqual(a["dominant_lines"][0]["final_assignment"], {"quanta": [1, 1]})
        first_mode, second_mode = a["final_mode_statistics"]["modes"]
        self.assertEqual([first_mode["mode"], second_mode["mode"]], [1, 2])
        self.assertAlmostEqual(first_mode["weight_fraction_with_nonzero_final_quanta"], weights_a[1] / weights_a.sum())
        self.assertAlmostEqual(second_mode["weight_fraction_with_nonzero_final_quanta"], 1.0)
        self.assertAlmostEqual(second_mode["weighted_mean_final_quanta"], (weights_a[1] + 2 * weights_a[2]) / weights_a.sum())

        weights_e = np.array([0.75 * 2 * 0.08**3, 0.75 * 0.11**3, 0.2 * 0.2 * 0.10**3])
        centroid_e = float(np.average(emission.photon_energy_hartree, weights=weights_e))
        self.assertAlmostEqual(e["centroid_energy"]["hartree"], centroid_e)
        self.assertEqual(e["dominant_lines"][0]["stick_index"], 1)
        self.assertEqual(e["dominant_lines"][1]["transition_dipole_au"], [-sqrt(2), 0, 0])
        first_state, second_state = e["emitting_vibrational_states"]
        self.assertEqual(first_state["initial_assignment"], {"quanta": [0, 0]})
        self.assertAlmostEqual(e["represented_initial_population"], 0.95)
        rate = e["vacuum_radiative_rate_s_inverse"]
        reconstructed = sum(state["relative_population"] * state["vacuum_radiative_rate_s_inverse"]
                            for state in (first_state, second_state))
        self.assertAlmostEqual(rate / reconstructed, 1.0)
        self.assertAlmostEqual(rate * e["vacuum_radiative_lifetime_s"], 1.0)
        self.assertAlmostEqual(e["vacuum_radiant_power_w_per_excited_molecule"] / (rate * 4.3597447222060e-18), centroid_e)

        self.assertEqual(report["paired"]["coverage_status"], "incomplete")
        self.assertAlmostEqual(report["paired"]["absorption_minus_emission_centroid_energy"]["hartree"], centroid_a - centroid_e)
        self.assertAlmostEqual(report["paired"]["absorption_minus_emission_strongest_stick_energy"]["hartree"], 0.14 - 0.11)

    def test_dark_empty_and_missing_spectra(self):
        dark = HarmonicSpectrum(
            np.array([0.11]), np.array([[0]]), np.array([[0]]), np.array([1.0]), np.array([0.0]), 1.0, None,
        )
        empty = HarmonicSpectrum(
            np.empty(0), np.empty((0, 1), dtype=int), np.empty((0, 1), dtype=int), np.empty(0), np.empty(0), 1.0, None,
        )
        report = summarize_spectral_properties({"emission": dark, "absorption": empty}, assignment_scheme="harmonic_quanta")
        json.dumps(report, allow_nan=False)
        a = report["spectra"]["absorption"]
        e = report["spectra"]["emission"]
        self.assertEqual(a["line_count"], 0)
        self.assertEqual(a["population_weighted_oscillator_strength_sum"], 0.0)
        self.assertIsNone(a["centroid_energy"])
        self.assertEqual(e["vacuum_radiative_rate_s_inverse"], 0.0)
        self.assertIsNone(e["vacuum_radiative_lifetime_s"])
        self.assertIsNone(e["emitting_vibrational_states"][0]["vacuum_radiative_lifetime_s"])
        self.assertIsNone(e["energy_standard_deviation"])
        self.assertEqual(e["dominant_lines"], [])
        self.assertEqual(e["origin_00_line"]["photon_energy"]["hartree"], 0.11)
        self.assertIsNone(report["paired"]["absorption_minus_emission_strongest_stick_energy"])
        missing = summarize_spectral_properties({"absorption": empty}, spectral_status={"emission": "incomplete"})
        self.assertEqual(missing["spectra"]["emission"], {
            "coverage_status": "incomplete", "line_count": None,
            "positive_photon_moments": {"M3": {"status": "unassessed"}, "M4": {"status": "unassessed"}},
        })
        self.assertIsNone(missing["paired"])

    def test_anharmonic_dvr_labels_and_harmonic_spectators(self):
        spectrum = AnharmonicSpectrum(
            np.array([0.1, 0.13]), np.array([0, 0]), np.array([0, 2]),
            np.array([[0], [0]]), np.array([[0], [3]]), np.array([1.0, 1.0]), np.array([0.1, 0.5]), 1.0, 1.0,
        )
        report = summarize_spectral_properties({"emission": spectrum}, max_dominant_lines=1)
        e = report["spectra"]["emission"]
        dominant = e["dominant_lines"][0]
        self.assertEqual(dominant["initial_assignment"], {"dvr_level": 0, "spectator_quanta": [0]})
        self.assertEqual(dominant["final_assignment"], {"dvr_level": 2, "spectator_quanta": [3]})
        self.assertNotIn("transition_dipole_au", dominant)
        self.assertEqual(e["final_mode_statistics"]["mode_basis"], "harmonic_spectators")
        self.assertEqual(e["origin_00_line"]["stick_index"], 0)

    def test_dvr_wrapper_and_vci_labels_do_not_report_harmonic_mode_statistics(self):
        spectrum = HarmonicSpectrum(
            np.array([0.11, 0.12]), np.array([[0], [0]]), np.array([[0], [1]]), np.array([1.0, 1.0]),
            np.array([0.0, 0.1]), 1.0, None,
        )
        dvr = summarize_spectral_properties({"absorption": spectrum}, assignment_scheme="dvr_levels")["spectra"]["absorption"]
        self.assertEqual(dvr["dominant_lines"][0]["final_assignment"], {"dvr_level": 1})
        self.assertIn("origin_00_line", dvr)
        self.assertNotIn("final_mode_statistics", dvr)
        vci = summarize_spectral_properties(
            {"absorption": spectrum}, assignment_scheme="vci_dominant_harmonic_labels",
        )["spectra"]["absorption"]
        self.assertEqual(vci["dominant_lines"][0]["final_assignment"], {"quanta": [1]})
        self.assertNotIn("final_mode_statistics", vci)
        self.assertNotIn("origin_00_line", vci)


if __name__ == "__main__":
    unittest.main()
