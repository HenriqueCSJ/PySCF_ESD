"""Synthetic controls for electronic state character and phase tracking."""

import unittest

import numpy as np

from pyscf_esd.state_tracking import StateTrackingError, match_state_overlaps, orbital_overlap_blocks, track_state


class StateTrackingTests(unittest.TestCase):
    def test_reordered_root_and_phase_are_recovered(self):
        target = np.array([[1.0, 0.0]])
        distractor = np.array([[0.0, 1.0]])
        result = track_state(
            target, [distractor, -target / np.sqrt(2)], [[1.0]], np.eye(2),
            candidate_energies_hartree=[0.2, 0.3],
        )
        self.assertEqual(result.index, 1)
        self.assertEqual(result.phase, -1)
        self.assertEqual(result.overlap, 1.0)
        self.assertEqual(result.margin, 1.0)
        self.assertAlmostEqual(result.gap_hartree, 0.1)
        np.testing.assert_array_equal(result.overlaps, [0.0, -1.0])

    def test_mo_gauge_and_cross_basis_covariance(self):
        ref_ao = np.array([
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.2, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 0.9],
        ])
        new_ao = np.array([
            [0.0, 0.9, 0.0, 0.0],
            [1.1, 0.0, 0.2, 0.0],
            [0.0, 0.0, 1.4, 0.3],
            [0.0, 0.0, 0.0, 0.8],
        ])
        new_occ_gauge = np.array([[0.6, -0.8], [0.8, 0.6]])
        new_vir_gauge = np.array([[0.0, -1.0], [-1.0, 0.0]])
        new_physical_mo = np.zeros((4, 4))
        new_physical_mo[:2, [0, 2]] = new_occ_gauge
        new_physical_mo[2:, [1, 3]] = new_vir_gauge
        occ, vir = orbital_overlap_blocks(
            np.linalg.inv(ref_ao), np.linalg.solve(new_ao, new_physical_mo),
            ref_ao.T @ new_ao, [2, 2, 0, 0], [2, 0, 2, 0],
        )
        target = np.array([[1.0, 0.2], [0.0, 0.4]])
        distractor = np.array([[-0.2, 1.0], [0.0, 0.0]])
        signed_candidates = [
            new_occ_gauge.T @ distractor @ new_vir_gauge,
            -new_occ_gauge.T @ target @ new_vir_gauge,
        ]
        result = track_state(target, signed_candidates, occ, vir)
        self.assertEqual((result.index, result.phase), (1, -1))
        np.testing.assert_allclose(result.overlaps, [0.0, -1.0], atol=1e-14)

    def test_cross_geometry_loss_is_preserved_in_score(self):
        result = track_state(
            [[1.0, 0.0]], [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[0.8]], np.diag([0.9, 0.95]),
        )
        self.assertAlmostEqual(result.overlap, 0.72)
        self.assertAlmostEqual(result.margin, 0.72)

    def test_rejects_an_ambiguity_even_without_margin_threshold(self):
        with self.assertRaisesRegex(StateTrackingError, "ambiguous"):
            track_state(
                [[1.0, 1.0]], [[[1.0, 0.0]], [[0.0, -1.0]]],
                [[1.0]], np.eye(2), minimum_margin=0.0,
            )

    def test_rejects_a_small_margin(self):
        with self.assertRaisesRegex(StateTrackingError, "ambiguous"):
            track_state(
                [[1.0, 0.98]], [[[1.0, 0.0]], [[0.0, 1.0]]],
                [[1.0]], np.eye(2),
            )

    def test_rejects_low_overlap_and_near_degeneracy(self):
        target = [[1.0, 0.0]]
        candidates = [target, [[0.0, 1.0]]]
        with self.assertRaisesRegex(StateTrackingError, "insufficient overlap"):
            track_state(target, candidates, [[0.3]], np.eye(2))
        with self.assertRaisesRegex(StateTrackingError, "near-degenerate"):
            track_state(
                target, candidates, [[1.0]], np.eye(2),
                candidate_energies_hartree=[0.3, 0.30001], minimum_gap_hartree=1e-4,
            )

    def test_rejects_malformed_orbital_and_amplitude_inputs(self):
        with self.assertRaisesRegex(ValueError, "occupations of 0 or 2"):
            orbital_overlap_blocks(np.eye(3), np.eye(3), np.eye(3), [2, 0, 0], [1, 1, 0])
        with self.assertRaisesRegex(ValueError, "finite nonzero Frobenius norms"):
            track_state([[0.0, 0.0]], [[[1.0, 0.0]], [[0.0, 1.0]]], [[1.0]], np.eye(2))
        with self.assertRaisesRegex(ValueError, "invalid normalized scores"):
            track_state([[1.0, 0.0]], [[[1.0, 0.0]], [[0.0, 1.0]]], [[1.1]], np.eye(2))

    def test_single_candidate_requires_explicit_margin_disable(self):
        args = ([[1.0]], [[[-1.0]]], [[1.0]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "at least two candidates"):
            track_state(*args)
        result = track_state(*args, minimum_margin=0.0)
        self.assertEqual((result.index, result.phase), (0, -1))
        self.assertIsNone(result.margin)
        self.assertIsNone(result.gap_hartree)

    def test_backend_scores_use_the_same_sign_and_gap_policy(self):
        scores = np.array([-0.87, 0.04, 0.31])
        energies = [-8.1, -8.3, -8.09999]
        match = match_state_overlaps(scores, candidate_energies_hartree=energies)
        self.assertEqual((match.index, match.phase), (0, -1))
        self.assertAlmostEqual(match.margin, 0.56)
        self.assertAlmostEqual(match.gap_hartree, 1e-5)
        self.assertFalse(match.overlaps.flags.writeable)
        self.assertTrue(scores.flags.writeable)
        with self.assertRaisesRegex(StateTrackingError, "near-degenerate"):
            match_state_overlaps(scores, candidate_energies_hartree=energies, minimum_gap_hartree=1e-4)
        with self.assertRaisesRegex(ValueError, "valid normalized scores"):
            match_state_overlaps([1.01, 0.04])


if __name__ == "__main__":
    unittest.main()
