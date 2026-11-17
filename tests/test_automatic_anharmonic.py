"""Independent checks of the automatic nuclear correction."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.automatic_anharmonic import AnharmonicUnavailableError, prepare_anharmonic
from pyscf_esd.pyscf_adapter import AdiabaticHessianParameters


def _surfaces(omega0, omega1, ground, excited, *, displacement=0.0):
    modes = len(omega0)
    ground_coordinates = np.zeros((2, 3))
    excited_coordinates = ground_coordinates.copy()
    excited_coordinates[0, :modes] = displacement
    cartesian_modes = np.zeros((modes, 2, 3))
    cartesian_modes[:, 0] = np.eye(3)[:modes]
    dipole = np.array([0.3, 0.4, 0.5])
    derivatives = np.tile([0.003, -0.001, 0.0001], (modes, 1))
    model = AdiabaticHessianParameters(
        ground_frequencies_hartree=omega0,
        excited_frequencies_hartree=omega1,
        gap_00_hartree=0.08 + (omega1 - omega0).sum() / 2,
        rotation=np.eye(modes),
        shift_au=-excited_coordinates[0, :modes],
        ground_cartesian_mode_derivatives_au=cartesian_modes,
        excited_cartesian_mode_derivatives_au=cartesian_modes,
        excited_alignment_rotation=np.eye(3),
        excited_alignment_translation_bohr=np.zeros(3),
        subspace_singular_values=np.ones(modes),
        rotation_correction_norm=0.0,
        displacement_norm_au=float(np.linalg.norm(excited_coordinates - ground_coordinates)),
        displacement_projection_residual_au=0.0,
    )

    def evaluate(coordinates):
        q = coordinates[0, :modes]
        dp = excited_coordinates[0, :modes]
        return (
            -1.0 + ground(np.sqrt(omega0) * q),
            -0.92 + excited(np.sqrt(omega1) * (q - dp)),
            dipole + q @ derivatives,
        )

    return SimpleNamespace(
        model=model, evaluate=evaluate,
        ground_coordinates_bohr=ground_coordinates, excited_coordinates_bohr=excited_coordinates,
        transition_dipole_au=dipole, dipole_derivatives_au=derivatives,
        ground_energy_hartree=-1.0, excited_energy_hartree=-0.92,
    )


class AutomaticAnharmonicTests(unittest.TestCase):
    def test_harmonic_rank_tail_counts_all_omitted_states(self):
        from pyscf_esd.automatic_anharmonic import _harmonic_log_rank_tail

        omega = np.array([.004, .002, .003])
        k_thermal = .002
        reference = .001
        index = np.arange(40)
        energies = np.sort((index[:, None, None] * omega[0] + index[None, :, None] * omega[1]
                            + index[None, None, :] * omega[2]).ravel())
        expected = np.exp((reference - energies[9:]) / k_thermal).sum()
        np.testing.assert_allclose(np.exp(_harmonic_log_rank_tail(omega, 9, k_thermal, reference)),
                                   expected, rtol=2e-10)

    @unittest.skipUnless(find_spec("scipy"), "variational selection uses SciPy")
    def test_thermal_vci_retains_soft_mode_energy_ranks_for_each_direction(self):
        from pyscf_esd.harmonic import _KB_HARTREE_PER_K

        w0, w1 = np.array([.0035, .0095]), np.array([.0041, .0102])
        temperature = 298.
        k_thermal = _KB_HARTREE_PER_K * temperature

        def ground(x):
            return .5 * w0 @ x**2 + 8e-6 * x[0]**4

        def excited(x):
            return .5 * w1 @ x**2 + 4e-6 * x[0]**3 + 7e-6 * x[0]**4

        prepared = prepare_anharmonic(_surfaces(w0, w1, ground, excited))
        self.assertEqual(prepared.diagnostics["status"], "corrected")
        for direction, omega, cubic, quartic in (
            ("absorption", w0, 0., 8e-6), ("emission", w1, 4e-6, 7e-6),
        ):
            with self.subTest(direction=direction):
                result = prepared.spectrum(process=direction, temperature_kelvin=temperature,
                                           initial_max_quanta=1, final_max_quanta=2)
                labels, rows = np.unique(result.initial_quanta, axis=0, return_index=True)
                np.testing.assert_array_equal(labels, [[0, 0], [1, 0], [2, 0]])
                x = np.linspace(-8, 8, 161)[1:-1]
                window = .5 + .5 * np.cos(np.pi * np.clip(abs(x) - 3.5, 0, 1))
                potential = .5 * omega[0] * x**2 + window * (cubic * x**3 + quartic * x**4)
                levels = solve_dvr(x, potential, boundary="dirichlet", n_levels=3,
                                   kinetic_coefficient=omega[0]).all_energies_hartree
                weights = np.exp(-(levels - levels[0]) / k_thermal)
                ratio = result.population[rows] / result.population[rows[0]]
                np.testing.assert_allclose(ratio, weights[:3], rtol=.002)
                reference_coverage = weights[:3].sum() / weights.sum() * -np.expm1(-omega[1] / k_thermal)
                self.assertGreater(result.initial_population_coverage, .995)
                self.assertLessEqual(result.initial_population_coverage, reference_coverage + 1e-8)
                partition = prepared.diagnostics["spectrum"][direction]["thermal_partition"]
                self.assertEqual(partition["method"], "windowed_vci_minmax_bound")
                self.assertAlmostEqual(partition["omitted_population_upper_bound"],
                                       1 - result.initial_population_coverage)

    @unittest.skipUnless(find_spec("scipy"), "the direct one-mode solver interpolates on SciPy")
    def test_morse_tail_extends_only_the_unconfined_edge(self):
        omega = np.array([0.004])
        depth = 3 * omega[0]

        def harmonic(x):
            return 0.5 * omega @ x**2

        def morse(x):
            return depth * np.expm1(-x[0] / np.sqrt(6))**2

        prepared = prepare_anharmonic(_surfaces(omega, omega, harmonic, morse))
        predicted = prepared.spectrum(final_max_quanta=4)
        quanta = np.arange(5)
        expected = 0.08 + omega[0] * quanta - omega[0]**2 * (quanta + 0.5)**2 / (4 * depth)
        np.testing.assert_allclose(predicted.photon_energy_hartree, expected, atol=2e-6)
        attempts = prepared.diagnostics["one_mode_attempts"]["absorption"]
        domains = np.array([row["sampled_domain_sqrt_electron_mass_bohr"] for row in attempts[0]["domain_attempts"]])
        self.assertGreater(len(domains), 1)
        np.testing.assert_array_equal(domains[:, 0], domains[0, 0])
        self.assertTrue(np.all(np.diff(domains[:, 1]) > 0))
        prepared.spectrum(process="emission", final_max_quanta=2)
        with self.assertRaisesRegex(AnharmonicUnavailableError, "not confined"):
            prepared.spectrum(final_max_quanta=6)
        terminal = attempts[-1]["domain_attempts"][-1]
        self.assertLess(terminal["wall_margin_hartree_by_edge_and_surface"][1][1], 0)
        partial = prepared.diagnostics["last_converged_sticks"]["absorption"]
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["final_max_quanta"], 4)
        self.assertNotIn("emission", prepared.diagnostics["last_converged_sticks"])
        np.testing.assert_array_equal(partial["photon_energy_hartree"], predicted.photon_energy_hartree)

    @unittest.skipUnless(find_spec("scipy"), "the direct one-mode solver interpolates on SciPy")
    def test_one_mode_direct_quadratic_dipole_and_shared_samples(self):
        omega = np.array([0.007])

        def harmonic(x):
            return 0.5 * omega @ x**2

        surfaces = _surfaces(omega, omega, harmonic, harmonic)
        original = surfaces.evaluate
        quadratic_dipole = np.array([0.0006, 0.0001, -0.00015])

        def direct_dipole(coordinates):
            e0, e1, dipole = original(coordinates)
            return e0, e1, dipole + quadratic_dipole * coordinates[0, 0]**2

        surfaces.evaluate = direct_dipole
        prepared = prepare_anharmonic(surfaces)
        self.assertEqual(prepared.diagnostics["solver"], "adaptive_open_dvr")
        predicted = prepared.spectrum(final_max_quanta=2)
        expected = np.stack([
            surfaces.transition_dipole_au + quadratic_dipole / (2 * omega[0]),
            surfaces.dipole_derivatives_au[0] / np.sqrt(2 * omega[0]),
            np.sqrt(2) * quadratic_dipole / (2 * omega[0]),
        ])
        np.testing.assert_allclose(predicted.photon_energy_hartree, 0.08 + np.arange(3) * omega[0], atol=2e-7)
        np.testing.assert_allclose(predicted.transition_dipole_au, expected, atol=3e-6)
        self.assertAlmostEqual(predicted.final_dipole_strength_coverage, 1.0, places=8)
        evaluations = prepared.diagnostics["energy_evaluations"]
        emitted = prepared.spectrum(process="emission", final_max_quanta=2)
        self.assertEqual(prepared.diagnostics["energy_evaluations"], evaluations)
        np.testing.assert_allclose(emitted.transition_dipole_au, expected[::-1], atol=3e-6)
        thermal = prepared.spectrum(initial_max_quanta=2, final_max_quanta=2, temperature_kelvin=1200.0)
        boltzmann = np.exp(-omega[0] / (1200.0 * 3.166811563e-6))
        occupations = thermal.initial_quanta.ravel()
        np.testing.assert_allclose(thermal.population, (1 - boltzmann) * boltzmann**occupations, atol=1.2e-6)
        self.assertAlmostEqual(thermal.initial_population_coverage, 1 - boltzmann**3, places=6)

    @unittest.skipUnless(find_spec("scipy"), "variational selection uses SciPy")
    def test_cubic_quartic_sticks_match_independent_dvr_and_reverse(self):
        omega0, omega1 = np.array([0.008]), np.array([0.010])

        def ground(x):
            return 0.5 * omega0[0] * x[0]**2 + 0.00007 * x[0]**3 + 0.000022 * x[0]**4

        def excited(x):
            return 0.5 * omega1[0] * x[0]**2 - 0.00005 * x[0]**3 + 0.000016 * x[0]**4

        prepared = prepare_anharmonic(_surfaces(omega0, omega1, ground, excited, displacement=1.5))
        self.assertEqual(prepared.diagnostics["status"], "corrected")
        grid = np.linspace(-8.0, 8.0, 129)[1:-1]
        dvr0 = solve_dvr(grid, ground([grid]), boundary="dirichlet", n_levels=1, kinetic_coefficient=omega0[0])
        dvr1 = solve_dvr(grid, excited([grid]), boundary="dirichlet", n_levels=4, kinetic_coefficient=omega1[0])
        observed = prepared.spectrum(final_max_quanta=3)
        expected = 0.08 + dvr1.energies_hartree - dvr0.energies_hartree[0]
        np.testing.assert_allclose(observed.photon_energy_hartree, expected, rtol=0, atol=8e-7)
        self.assertGreater(abs(observed.photon_energy_hartree[-1] - 0.111), 1e-4)

        options = dict(temperature_kelvin=300.0, initial_max_quanta=2, final_max_quanta=2)
        absorption = prepared.spectrum(**options)
        emission = prepared.spectrum(process="emission", **options)
        for a, (initial, final) in enumerate(zip(absorption.initial_quanta, absorption.final_quanta)):
            is_reverse = np.all(emission.initial_quanta == final, axis=1) & np.all(
                emission.final_quanta == initial, axis=1
            )
            reverse = np.flatnonzero(is_reverse)
            self.assertEqual(reverse.size, 1)
            np.testing.assert_allclose(
                absorption.transition_dipole_au[a], emission.transition_dipole_au[reverse[0]], atol=2e-14,
            )

    def test_mixed_probes_find_pair_cancellation_and_three_mode_fermi_term(self):
        omega = np.array([0.005, 0.006, 0.011])

        def harmonic(x):
            return 0.5 * omega @ x**2

        def coupled(x):
            a, b, c = x
            return harmonic(x) + 0.00012 * a * b * c + 0.000026 * (a**3 * b - a * b**3)

        prepared = prepare_anharmonic(_surfaces(omega, omega, harmonic, coupled))
        self.assertEqual(prepared.diagnostics["status"], "corrected")
        report = prepared.diagnostics["surfaces"]["excited"]
        np.testing.assert_allclose(
            report["pair_probes"][0]["coefficients_hartree"], [0, 0, 0.000026, 0, -0.000026], atol=1e-15,
        )
        np.testing.assert_allclose(
            report["triple_probes"][0]["coefficients_hartree"], [0.00012, 0, 0, 0], atol=1e-15,
        )

    def test_alternate_ground_well_blocks_local_spectrum(self):
        omega = np.array([0.006])
        harmonic = lambda x: 0.5 * omega @ x**2
        electronic = _surfaces(omega, omega, harmonic, harmonic)
        electronic.alternate_ground_coordinates_bohr = (np.ones_like(electronic.ground_coordinates_bohr),)
        prepared = prepare_anharmonic(electronic)
        self.assertEqual(prepared.diagnostics["energy_evaluations"], 0)
        with self.assertRaisesRegex(AnharmonicUnavailableError, "second ground-state minimum"):
            prepared.spectrum()
