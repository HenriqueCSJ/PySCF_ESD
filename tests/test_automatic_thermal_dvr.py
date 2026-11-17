"""Synthetic checks of the automatic DVR thermal trace and its coverage."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.automatic_anharmonic import AnharmonicUnavailableError, prepare_anharmonic
from test_automatic_anharmonic import _surfaces


@unittest.skipUnless(find_spec("scipy"), "the direct one-mode solver interpolates on SciPy")
class AutomaticThermalDVRTests(unittest.TestCase):
    def test_quartic_trace_normalizes_the_retained_sticks_for_either_process(self):
        omega = np.array([0.003])
        temperature = 900.0
        thermal_energy = 3.166811563e-6 * temperature

        def quartic(x):
            return 0.5 * omega[0] * x[0] ** 2 + 0.00007 * x[0] ** 3 + 0.00012 * x[0] ** 4

        def harmonic(x):
            return 0.5 * omega @ x**2

        grid = np.linspace(-13.0, 13.0, 425)[1:-1]
        reference = solve_dvr(
            grid, quartic([grid]), boundary="dirichlet", n_levels=1, kinetic_coefficient=omega[0],
        )
        weights = np.exp(-(reference.all_energies_hartree - reference.energies_hartree[0]) / thermal_energy)
        trace = weights.sum()
        expected = weights / trace
        for process, potentials in (
            ("absorption", (quartic, harmonic)), ("emission", (harmonic, quartic)),
        ):
            with self.subTest(process=process):
                prepared = prepare_anharmonic(_surfaces(omega, omega, *potentials))
                settings = dict(process=process, temperature_kelvin=temperature, initial_max_quanta=1,
                                final_max_quanta=2)
                spectrum = prepared.spectrum(**settings)
                report = prepared.diagnostics["spectrum"][process]
                partition = report["thermal_partition"]
                self.assertAlmostEqual(partition["partition_function"], trace, places=5)
                self.assertIsNone(report["partition_tail"])
                np.testing.assert_allclose(spectrum.population, expected[spectrum.initial_quanta.ravel()],
                                           rtol=0, atol=2e-6)
                self.assertAlmostEqual(spectrum.initial_population_coverage, expected[:2].sum(), places=5)
                inner = partition["comparison_domain_sqrt_electron_mass_bohr"]
                outer = report["domain_sqrt_electron_mass_bohr"]
                self.assertLess(outer[0], inner[0])
                self.assertGreater(outer[1], inner[1])
                self.assertLessEqual(partition["domain_relative_partition_change"], partition["tolerance"])
                evaluations = prepared.diagnostics["energy_evaluations"]
                prepared.spectrum(**settings)
                self.assertEqual(prepared.diagnostics["energy_evaluations"], evaluations)

    def test_hot_shallow_well_fails_thermal_coverage_after_zero_kelvin_succeeds(self):
        omega = np.array([0.005])
        depth = 4.5 * omega[0]

        def morse(x):
            return depth * np.expm1(-x[0] / np.sqrt(9)) ** 2

        def harmonic(x):
            return 0.5 * omega @ x**2

        prepared = prepare_anharmonic(_surfaces(omega, omega, morse, harmonic))
        zero_kelvin = prepared.spectrum(final_max_quanta=2)
        self.assertEqual(zero_kelvin.initial_population_coverage, 1.0)
        with self.assertRaisesRegex(AnharmonicUnavailableError, "thermal partition is not confined"):
            prepared.spectrum(temperature_kelvin=1200, initial_max_quanta=2, final_max_quanta=2)
        attempt = prepared.diagnostics["one_mode_attempts"]["absorption"][-1]
        self.assertEqual(attempt["status"], "incomplete")
        terminal = attempt["domain_attempts"][-1]
        self.assertGreater(terminal["wall_margin_hartree_by_edge_and_surface"][1][0], 0.0)
        partition = terminal["thermal_partition"]
        self.assertLess(partition["thermal_wall_margin_hartree_by_edge"][1], 0.0)
        self.assertGreater(partition["probability_at_or_above_lowest_wall"], partition["tolerance"])
