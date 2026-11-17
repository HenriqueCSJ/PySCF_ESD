"""Synthetic exact physical-state overlaps for prepared scans."""

import unittest

import numpy as np

from pyscf_esd.anharmonic_sampling import phase_periodic_scan
from pyscf_esd.prepared_state_tracking import track_prepared_states
from pyscf_esd.state_tracking import StateTrackingError


class PreparedStatePathTests(unittest.TestCase):
    def test_open_path_tracks_rectangular_root_pools_and_phases(self):
        basis = np.eye(4)
        states = [basis[:3], basis[[1, 3, 0, 2]], basis[[3, 0, 1]]]
        states[1][[0, 2]] *= -1
        states[2][2] *= -1
        energies = [[0.01, 0.09, 0.50], [0.10, 0.51, 0.02, 0.52], [0.45, 0.03, 0.11]]
        overlaps = [source @ destination.T for source, destination in zip(states, states[1:])]
        result = track_prepared_states(energies, overlaps, initial_indices=(0, 1))

        np.testing.assert_array_equal(result.root_indices, [[0, 1], [2, 0], [1, 2]])
        np.testing.assert_array_equal(result.phases, [[1, 1], [-1, -1], [1, -1]])
        np.testing.assert_array_equal(result.energies_hartree, [[0.01, 0.09], [0.02, 0.10], [0.03, 0.11]])
        np.testing.assert_array_equal(result.signed_link_overlaps, [[-1, -1], [-1, 1]])
        dipoles = np.tile([1.0, -2.0, 3.0], (3, 1))
        np.testing.assert_array_equal(result.align_transition_dipoles(dipoles), dipoles * [[1], [1], [-1]])
        self.assertIsNone(result.closure_signs)
        self.assertFalse(result.root_indices.flags.writeable)
        self.assertFalse(result.phases.flags.writeable)
        self.assertFalse(result.energies_hartree.flags.writeable)
        with self.assertRaisesRegex(ValueError, "one raw Cartesian transition dipole"):
            result.align_transition_dipoles(np.ones((3, 2)))

    def test_periodic_path_retains_both_state_holonomies(self):
        n_points = 8
        angles = np.arange(n_points) * 2 * np.pi / n_points
        e0, e1, e2, e3, e4 = np.eye(5)
        signs = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]] * 2)
        orderings = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]] * 2)
        energies = np.array([0.0, 0.08, 0.25])[orderings]

        for first_turns, second_turns in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            first_sign, second_sign = (-1)**first_turns, (-1)**second_turns
            with self.subTest(closure=(first_sign, second_sign)):
                states = []
                for angle, (first_phase, second_phase), ordering in zip(angles, signs, orderings):
                    first = np.cos(first_turns * angle / 2) * e0 + np.sin(first_turns * angle / 2) * e1
                    second = np.cos(second_turns * angle / 2) * e2 + np.sin(second_turns * angle / 2) * e3
                    states.append(np.stack([first_phase * first, second_phase * second, e4])[ordering])
                overlaps = [source @ destination.T for source, destination in zip(states, states[1:] + states[:1])]
                result = track_prepared_states(energies, overlaps, initial_indices=(0, 1), periodic=True)

                np.testing.assert_array_equal(result.root_indices, np.argsort(orderings, axis=1)[:, :2])
                np.testing.assert_array_equal(result.phases, signs * signs[0])
                self.assertEqual(result.closure_signs, (first_sign, second_sign))
                physical_dipoles = np.cos((first_turns + second_turns) * angles / 2)[:, None] * [1., -2., 3.]
                raw_dipoles = signs.prod(axis=1)[:, None] * physical_dipoles
                anchored_dipoles = signs[0].prod() * physical_dipoles
                rotor = phase_periodic_scan(
                    angles, *result.energies_hartree.T, raw_dipoles,
                    ground_overlaps=result.signed_link_overlaps[:, 0], excited_overlaps=result.signed_link_overlaps[:, 1],
                )
                self.assertEqual(rotor.ground_closure_sign, first_sign)
                self.assertEqual(rotor.excited_closure_sign, second_sign)
                self.assertEqual(rotor.transition_dipole.closure_sign, first_sign * second_sign)
                np.testing.assert_array_equal(rotor.transition_dipole.values_au, anchored_dipoles)
                np.testing.assert_array_equal(result.align_transition_dipoles(raw_dipoles), anchored_dipoles)

    def test_rejects_physical_root_exchange_at_the_seam(self):
        angles = np.linspace(0., np.pi / 2, 5)
        states = [np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) for angle in angles]
        overlaps = [source @ destination.T for source, destination in zip(states, states[1:] + states[:1])]
        with self.assertRaisesRegex(StateTrackingError, "4 to 0: seam does not return to both initial root identities"):
            track_prepared_states([[0.0, 0.1]] * len(states), overlaps, initial_indices=(0, 1), periodic=True)

    def test_qualifies_overlap_margin_collision_reverse_match_and_energy_gap(self):
        energies = [[0.0, 0.2, 0.4]] * 2
        failures = [
            (np.diag([0.5, 1., 1.]), {}, "insufficient overlap"),
            (np.array([[0.70, 0.70, 0.], [-0.70, 0.70, 0.], [0., 0., 1.]]), {}, "ambiguous state identity"),
            (np.array([[0.70, 0., 0.], [0.70, 0., 0.], [0., 0., 0.]]), {}, "same destination root"),
            (np.array([[0.65, 0., 0.], [0., 1., 0.], [0.75, 0., 0.]]), {"minimum_margin": 0.05},
             "reverse assignment does not recover source root 1"),
        ]
        for matrix, options, message in failures:
            with self.subTest(message=message), self.assertRaisesRegex(StateTrackingError, message):
                track_prepared_states(energies, [matrix], initial_indices=(0, 1), **options)

        crowded = [0., 0.2, 1e-5]
        for point_energies in [[crowded, energies[1]], [energies[0], crowded]]:
            with self.assertRaisesRegex(StateTrackingError, "near-degenerate state"):
                track_prepared_states(point_energies, [np.eye(3)], initial_indices=(0, 1))

    def test_requires_real_normalized_overlap_matrices(self):
        energies = [[0., 0.2]] * 2
        invalid_inputs = [
            ([np.eye(2)] * 2, {"periodic": True}, "a closed path requires at least three"),
            ([[[1., 0.], [0., 1.], [0., 0.]]], {}, "dimensions must match"),
            ([np.diag([1.0001, 1.])], {}, "incompatible with normalized orthogonal states"),
            ([np.eye(2) + 0.2 * np.fliplr(np.eye(2))], {}, "incompatible with normalized orthogonal states"),
            ([np.eye(2, dtype=complex)], {}, "real rank-2 array"),
            ([np.diag([np.nan, 1.])], {}, "only finite values"),
            ([np.eye(2)], {"minimum_gap_hartree": 0.}, "finite and positive"),
        ]
        for matrices, options, message in invalid_inputs:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                track_prepared_states(energies, matrices, initial_indices=(0, 1), **options)
        with self.assertRaisesRegex(ValueError, "one signed overlap matrix per link"):
            track_prepared_states(energies + energies[:1], [np.eye(2)] * 2, initial_indices=(0, 1), periodic=True)
        for indices in [(0, 0), (0, 2), (False, 1)]:
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "two distinct valid"):
                track_prepared_states(energies, [np.eye(2)], initial_indices=indices)
