"""Regression tests for the restricted MRSF determinant representation."""

import unittest

import numpy as np

from pyscf_esd.determinant_overlap import determinant_expansion_overlap
from pyscf_esd.mrsf_overlap import expand_restricted_mrsf_singlets


class RestrictedMRSFSingletTests(unittest.TestCase):
    @staticmethod
    def expand(vectors):
        return expand_restricted_mrsf_singlets(vectors, n_core=1, n_orbitals=4)

    def test_diagonal_somo_and_closed_open_phases(self):
        vectors = np.eye(9)[:, (1, 0)]
        expanded = self.expand(vectors)
        terms = {
            (tuple(alpha), tuple(beta)): values
            for alpha, beta, values in zip(expanded.alpha, expanded.beta, expanded.coefficients.T)
        }
        scale = 1 / np.sqrt(2)
        np.testing.assert_allclose(terms[(0, 2), (0, 1)], [scale, 0])
        np.testing.assert_allclose(terms[(0, 1), (0, 2)], [scale, 0])
        np.testing.assert_allclose(terms[(1, 2), (0, 1)], [0, -scale])
        np.testing.assert_allclose(terms[(0, 1), (1, 2)], [0, -scale])
        np.testing.assert_allclose(expanded.overlap(expanded, np.eye(4)), np.eye(2), atol=1e-15)

    def test_mixed_families_are_reciprocal_and_gauge_covariant(self):
        selected = np.eye(9)[:, (0, 1, 2, 3, 4, 6, 7, 8)]
        roots, _ = np.linalg.qr(np.random.default_rng(211).standard_normal((8, 2)))
        vectors = selected @ roots
        source = self.expand(vectors)
        target = self.expand(vectors[:, ::-1])
        metric, _ = np.linalg.qr(np.random.default_rng(234).standard_normal((4, 4)))
        metric *= 0.9
        forward = source.overlap(target, metric)
        reverse = target.overlap(source, metric.T)
        np.testing.assert_allclose(forward, reverse.T, atol=1e-15)
        np.testing.assert_allclose(
            self.expand(vectors * [-1, 1]).overlap(target, metric),
            forward * np.array([[-1], [1]]), atol=1e-15,
        )

    def test_rejects_unused_slot_nonorthogonal_roots_and_expansive_metric(self):
        with self.assertRaisesRegex(ValueError, "unused diagonal SOMO slot"):
            self.expand(np.eye(9)[:, [5]])
        with self.assertRaisesRegex(ValueError, "not orthonormal"):
            self.expand(np.repeat(np.eye(9)[:, [1]], 2, axis=1))
        expansion = self.expand(np.eye(9)[:, [1]])
        with self.assertRaisesRegex(ValueError, "incompatible with orthonormal orbitals"):
            expansion.overlap(expansion, 1.1 * np.eye(4))

    def test_four_blocks_match_explicit_determinants(self):
        rng = np.random.default_rng(8301)

        def expand(ncore, nmo):
            nconfig = (nmo - ncore) * (ncore + 2)
            roots, _ = np.linalg.qr(rng.standard_normal((nconfig - 1, 2)))
            vectors = np.insert(roots, 2 * ncore + 3, 0, axis=0)
            return expand_restricted_mrsf_singlets(vectors, n_core=ncore, n_orbitals=nmo)

        for ncore, nleft, nright in ((0, 2, 4), (1, 5, 4), (3, 7, 8), (7, 12, 12)):
            bra, ket = expand(ncore, nleft), expand(ncore, nright)
            metric = rng.standard_normal((nleft, nright))
            metric *= 0.94 / np.linalg.norm(metric, ord=2)
            singular = metric.copy()
            singular[:, :ncore] = 0
            for cross in (metric, singular):
                with self.subTest(ncore=ncore, shape=cross.shape):
                    expected = determinant_expansion_overlap(
                        bra.coefficients, ket.coefficients, bra.alpha, bra.beta,
                        ket.alpha, ket.beta, cross,
                    )
                    np.testing.assert_allclose(bra.overlap(ket, cross), expected, atol=2e-15, rtol=0)
                    np.testing.assert_allclose(ket.overlap(bra, cross.T).T, expected, atol=2e-15, rtol=0)

    def test_orbital_signs_preserve_the_physical_overlap(self):
        roots = np.eye(9)[:, (0, 1, 8)]
        metric, _ = np.linalg.qr(np.random.default_rng(321).standard_normal((4, 4)))
        phases = np.array([1, -1, 1, -1])
        configurations = (phases[1] * phases[2] * np.outer(phases[1:], phases[:3])).ravel()
        reference = self.expand(roots)
        rephased = self.expand(roots * configurations[:, None])
        np.testing.assert_allclose(
            rephased.overlap(reference, phases[:, None] * metric),
            reference.overlap(reference, metric), atol=2e-15, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
