"""Check CAS overlaps against small explicit full Slater determinant sums."""

from importlib.util import find_spec
from itertools import combinations, product
import unittest

import numpy as np

from pyscf_esd.casscf_tracking import casscf_state_overlap
from pyscf_esd.state_tracking import StateTrackingError


def _occupied_strings(ncore, ncas, count):
    # PySCF orders the active determinant strings by their integer bit masks.
    active = sorted(combinations(range(ncas), count), key=lambda indices: sum(1 << i for i in indices))
    return [tuple(range(ncore)) + tuple(ncore + i for i in indices) for indices in active]


def _full_determinant_overlap(bra, ket, mo_cross, ncore, ncas, nelecas):
    alpha, beta = (_occupied_strings(ncore, ncas, count) for count in nelecas)
    overlap = 0.0
    for i, j, k, l in product(range(len(alpha)), range(len(beta)), range(len(alpha)), range(len(beta))):
        alpha_overlap = np.linalg.det(mo_cross[np.ix_(alpha[i], alpha[k])])
        beta_overlap = np.linalg.det(mo_cross[np.ix_(beta[j], beta[l])])
        overlap += bra[i, j] * ket[k, l] * alpha_overlap * beta_overlap
    return overlap


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for CAS state overlaps")
class CasscfTrackingTests(unittest.TestCase):
    def test_cross_geometry_overlap_with_core_active_mixing(self):
        rng = np.random.default_rng(308)
        reference = np.linalg.qr(rng.normal(size=(6, 6)))[0]
        candidate = np.linalg.qr(rng.normal(size=(7, 7)))[0]
        mo_cross = np.linalg.qr(np.eye(7) + 0.3 * rng.normal(size=(7, 7)))[0][:6]
        ao_cross = reference @ mo_cross @ candidate.T
        bra, ket = rng.normal(size=(2, 3, 3))
        bra /= np.linalg.norm(bra)
        ket /= np.linalg.norm(ket)
        ncore, ncas, nelecas = 2, 3, (2, 1)

        expected = _full_determinant_overlap(bra, ket, mo_cross, ncore, ncas, nelecas)
        block_diagonal = mo_cross.copy()
        block_diagonal[:ncore, ncore:] = 0.0
        block_diagonal[ncore:, :ncore] = 0.0
        without_mixing = _full_determinant_overlap(bra, ket, block_diagonal, ncore, ncas, nelecas)
        self.assertGreater(abs(expected - without_mixing), 1e-3)
        result = casscf_state_overlap(
            bra, ket, reference, candidate, ao_cross, ncore=ncore, ncas=ncas, nelecas=nelecas,
        )
        self.assertAlmostEqual(result, expected, places=13)

    def test_zero_core_preserves_cross_geometry_overlap_and_its_orientation(self):
        bra = np.array([[0.6], [0.8]])
        ket = np.array([[0.8], [-0.6]])
        cross = np.array([[0.76, -0.21], [0.32, 0.69]])
        result = casscf_state_overlap(bra, ket, np.eye(2), np.eye(2), cross, ncore=0, ncas=2, nelecas=(1, 0))
        self.assertAlmostEqual(result, (bra.T @ cross @ ket).item(), places=13)

    def test_orbital_gauge_and_ci_state_phase(self):
        reference = np.eye(4)
        candidate = reference[:, [0, 2, 1, 3]].copy()
        candidate[:, 0] *= -1
        bra = np.zeros((3, 3))
        bra[0, 1] = 1.0
        ket = np.zeros((3, 3))
        # Reversing the first two active orbitals reverses the occupied
        # alpha string, while the single occupied beta orbital keeps its sign.
        ket[0, 0] = -1.0
        for phase in (-1, 1):
            with self.subTest(phase=phase):
                result = casscf_state_overlap(
                    bra, phase * ket, reference, candidate, reference, ncore=1, ncas=3, nelecas=(2, 1),
                )
                self.assertAlmostEqual(result, phase, places=13)

    def test_ill_conditioned_core_can_have_nonzero_total_overlap(self):
        ci = np.array([[1.0], [0.0]]) @ np.array([[1.0, 0.0]])
        for cosine in (0.0, 1e-12):
            with self.subTest(cosine=cosine):
                sine = np.sqrt(1.0 - cosine**2)
                cross = np.eye(4)
                cross[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
                self.assertAlmostEqual(_full_determinant_overlap(ci, ci, cross, 2, 2, (1, 1)), 1.0)
                with self.assertRaisesRegex(StateTrackingError, "common core overlap is singular or ill-conditioned"):
                    casscf_state_overlap(ci, ci, np.eye(4), np.eye(4), cross, ncore=2, ncas=2, nelecas=(1, 1))
