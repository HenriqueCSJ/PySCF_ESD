"""Synthetic checks against full occupied spin-orbital determinants."""

from itertools import combinations, product
import unittest

import numpy as np

from pyscf_esd.determinant_overlap import determinant_expansion_overlap


def _direct_overlap(bra, ket, bra_a, bra_b, ket_a, ket_b, alpha, beta):
    def determinant(row, column):
        occupied = np.zeros((len(bra_a[row]) + len(bra_b[row]),) * 2)
        nalpha = len(bra_a[row])
        occupied[:nalpha, :nalpha] = alpha[np.ix_(bra_a[row], ket_a[column])]
        occupied[nalpha:, nalpha:] = beta[np.ix_(bra_b[row], ket_b[column])]
        return np.linalg.det(occupied)

    minors = np.array([
        [determinant(row, column) for column in range(len(ket_a))]
        for row in range(len(bra_a))
    ])
    return bra @ minors @ ket.T


def _occupation_grid(alpha, beta):
    return tuple(np.array(occupations) for occupations in zip(*product(alpha, beta)))


class DeterminantExpansionOverlapTests(unittest.TestCase):
    def test_native_orbital_gauge_occupation_order_and_state_phase(self):
        raw = np.eye(4)[:, [0, 2, 1, 3]]
        raw[:, 0] *= -1
        ket = np.array([[0.6, 0.8], [-0.8, 0.6]])
        # The first ket string cancels the orbital phase with its own odd
        # order. The second ket string has the opposite determinant phase.
        args = ([[0, 1], [1, 3]], [[], []], [[2, 0], [3, 2]], [[], []], raw)
        overlap = determinant_expansion_overlap(np.eye(2), ket, *args)
        np.testing.assert_allclose(overlap, [[0.6, -0.8], [-0.8, -0.6]])
        np.testing.assert_allclose(
            determinant_expansion_overlap([1, 0], ket, *args), overlap[0]
        )
        np.testing.assert_allclose(
            determinant_expansion_overlap(np.eye(2), ket[1], *args), overlap[:, 1]
        )
        self.assertAlmostEqual(determinant_expansion_overlap([1, 0], -ket[0], *args), -0.6)

    def test_factorized_spin_flip_grid_matches_explicit_determinants(self):
        rng = np.random.default_rng(814)
        alpha = np.linalg.qr(rng.normal(size=(9, 9)))[0][:7, :8]
        beta = np.linalg.qr(rng.normal(size=(9, 9)))[0][:8, :7]
        bra_a, bra_b = _occupation_grid(
            combinations(range(5), 4), [[0, 3, p] for p in [1, 2, 5, 6]]
        )
        ket_a, ket_b = _occupation_grid(
            combinations([0, 1, 2, 5, 6], 4), [[1, 4, p] for p in [0, 2, 5]]
        )
        bra_a[0, :2] = bra_a[0, :2][::-1]
        ket_b[3] = ket_b[3, ::-1]
        bra = rng.normal(size=(2, len(bra_a)))
        ket = rng.normal(size=(3, len(ket_a)))
        expected = _direct_overlap(bra, ket, bra_a, bra_b, ket_a, ket_b, alpha, beta)
        actual = determinant_expansion_overlap(
            bra, ket, bra_a, bra_b, ket_a, ket_b, alpha, mo_overlap_beta=beta
        )
        np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=0)
        reverse = determinant_expansion_overlap(
            ket, bra, ket_a, ket_b, bra_a, bra_b, alpha.T, mo_overlap_beta=beta.T
        )
        np.testing.assert_allclose(reverse, expected.T, atol=5e-14, rtol=0)

    def test_singular_hole_block_and_singular_common_core(self):
        alpha = np.eye(4)[:, [0, 1, 3, 2]]
        occupied_alpha, occupied_beta = _occupation_grid(
            combinations(range(3), 2), [[0, p] for p in [1, 2, 3]]
        )
        states = np.eye(len(occupied_alpha))
        for cosine in (0.0, 1e-12):
            with self.subTest(cosine=cosine):
                sine = np.sqrt(1 - cosine**2)
                beta = np.eye(4)
                beta[:2, :2] = [[cosine, -sine], [sine, cosine]]
                expected = _direct_overlap(
                    states, states, occupied_alpha, occupied_beta,
                    occupied_alpha, occupied_beta, alpha, beta,
                )
                self.assertAlmostEqual(expected[0, 0], 1.0)
                actual = determinant_expansion_overlap(
                    states, states, occupied_alpha, occupied_beta,
                    occupied_alpha, occupied_beta, alpha, mo_overlap_beta=beta,
                )
                np.testing.assert_allclose(actual, expected, atol=5e-15, rtol=0)

    def test_sparse_general_occupations_and_repeated_determinants(self):
        rng = np.random.default_rng(196)
        overlap = np.linalg.qr(rng.normal(size=(8, 8)))[0]
        alpha = [[0, 1], [2, 3], [4, 5], [1, 0]]
        beta = [[1, 2], [3, 4], [5, 6], [1, 2]]
        bra, ket = rng.normal(size=(2, 4))
        expected = _direct_overlap(bra, ket, alpha, beta, beta, alpha, overlap, overlap)
        actual = determinant_expansion_overlap(bra, ket, alpha, beta, beta, alpha, overlap)
        self.assertAlmostEqual(actual, expected, places=14)
        # A repeated determinant with its alpha order reversed cancels.
        self.assertEqual(
            determinant_expansion_overlap(
                [1, 1], [1], [[0, 1], [1, 0]], [[2], [2]], [[0, 1]], [[2]], np.eye(3)
            ),
            0,
        )

    def test_malformed_inputs(self):
        args = ([1], [1], [[0, 1]], [[0]], [[0, 1]], [[0]], np.eye(2))
        for position, value, message in (
            (2, [[0, 0]], "repeated orbital"),
            (2, [[0, 2]], "integer orbital indices in range"),
            (2, [[0.0, 1.0]], "integer orbital indices in range"),
            (2, [[False, True]], "integer orbital indices in range"),
            (3, [[0], [1]], "one row per determinant"),
            (4, [[0]], "same electron count"),
            (6, [[np.inf]], "finite"),
        ):
            with self.subTest(position=position, value=value):
                malformed = (*args[:position], value, *args[position + 1 :])
                with self.assertRaisesRegex(ValueError, message):
                    determinant_expansion_overlap(*malformed)


if __name__ == "__main__":
    unittest.main()
