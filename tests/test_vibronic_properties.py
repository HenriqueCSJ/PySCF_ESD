from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd import electronic_relaxation_properties, harmonic_model_properties, harmonic_spectrum
from pyscf_esd.vibronic_properties import condon_ht_line_strengths, condon_ht_properties


class VibronicModelPropertiesTests(unittest.TestCase):
    def test_four_electronic_energies_close_the_vertical_gap_cycle(self):
        report = electronic_relaxation_properties(
            ground_at_ground_hartree=-100,
            ground_at_excited_hartree=-99.99,
            excited_at_ground_hartree=-99.90,
            excited_at_excited_hartree=-99.92,
            transition_dipole_at_ground_au=(0, .3, .4),
        )
        differences = report["energy_differences_hartree"]
        self.assertAlmostEqual(differences["vertical_absorption"], .10)
        self.assertAlmostEqual(differences["vertical_emission"], .07)
        self.assertAlmostEqual(differences["adiabatic_electronic_gap"], .08)
        self.assertAlmostEqual(differences["vertical_absorption_minus_emission"],
                               differences["ground_deformation_at_excited"] + differences["excited_relaxation_from_ground"])
        self.assertEqual(report["vertical_electronic_dipole_strength_au2"], {"absorption": .25})

    def test_rotated_displacements_and_linear_dipole_use_the_same_frame(self):
        model = SimpleNamespace(
            ground_frequencies_hartree=np.array([.005, .01]),
            excited_frequencies_hartree=np.array([.02, .03]),
            gap_00_hartree=.1,
            rotation=np.array([[0, 1], [-1, 0]]),
            shift_au=np.array([2, 3]),
        )
        report = harmonic_model_properties(
            model, dipole_derivatives_au=[[.01, 0, 0], [0, .02, 0]],
            transition_dipole_at_ground_au=[.1, 0, 0],
            direct_transition_dipole_at_excited_au=[.13, -.04, 0],
        )
        ground = report["ground_at_excited"]["modes"]
        excited = report["excited_at_ground"]["modes"]
        self.assertAlmostEqual(ground[0]["huang_rhys_displacement_factor"], .005 * 9 / 2)
        self.assertAlmostEqual(excited[0]["huang_rhys_displacement_factor"], .02 * 4 / 2)
        self.assertAlmostEqual(report["linear_ht_at_excited_minimum"]["vector_error_norm_au"], 0)

    def test_condon_and_ht_reconstruct_the_transition_including_interference(self):
        for temperature in (0, 300):
            options = {
                "gap_00_hartree": .2, "shift_au": [3],
                "transition_dipole_au": [.1, .02, 0],
                "initial_max_quanta": 2, "final_max_quanta": 4, "temperature_kelvin": temperature,
            }
            with_ht = harmonic_spectrum([.01], [.012], dipole_derivatives_au=[[.11, 0, 0]], **options)
            without_ht = harmonic_spectrum([.01], [.012], **options)
            np.testing.assert_allclose(with_ht.condon_transition_dipole_au, without_ht.transition_dipole_au)
            terms = condon_ht_line_strengths(with_ht)
            np.testing.assert_allclose(sum(terms.values()), with_ht.dipole_strength_au2)
            self.assertGreater(np.abs(terms["condon_ht_interference_au2"]).max(), .001)
            report = condon_ht_properties(with_ht, "absorption")
            self.assertAlmostEqual(sum(report["fractions_of_total_signal"].values()), 1)
