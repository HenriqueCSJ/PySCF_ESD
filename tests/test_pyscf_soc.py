"""Verify native PySCF integral conventions and the restricted TDA adapter."""

import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd import assemble_pyscf_tda_soc, run_pyscf_tda_soc
from pyscf_esd.pyscf_soc import _breit_pauli_somf_ao

try:
    from pyscf import dft, gto
    from pyscf.data.nist import LIGHT_SPEED
    from pyscf.tdscf.rks import TDA
except ImportError:
    gto = None


@unittest.skipIf(gto is None, "requires PySCF")
class PySCFTDASOCTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom="He 0 0 0; He .4 .7 1.2", basis="cc-pvdz", verbose=0)

    def test_direct_somf_matches_explicit_coulomb_and_both_exchange_terms(self):
        mol = self.mol
        generator = np.random.default_rng(207)
        matrix = generator.normal(size=(mol.nao, mol.nao))
        density = matrix @ matrix.T
        g2 = mol.intor("int2e_p1vxp1", comp=3)
        direct = np.einsum("xijkl,lk->xij", g2, density)
        exchange_1 = np.einsum("xijkl,jk->xil", g2, density)
        exchange_2 = np.einsum("xijkl,li->xkj", g2, density)
        nuclear = np.zeros_like(direct)
        for atom, charge in enumerate(mol.atom_charges()):
            with mol.with_rinv_at_nucleus(atom):
                nuclear -= charge * mol.intor("int1e_prinvxp", comp=3)
        expected = 1j * (nuclear + direct - 1.5 * (exchange_1 + exchange_2)) / (2 * LIGHT_SPEED**2)
        actual = _breit_pauli_somf_ao(mol, density)
        self.assertGreater(np.max(np.abs(expected)), 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-17)
        np.testing.assert_allclose(actual, actual.swapaxes(-1, -2).conj(), atol=1e-17)

    def test_native_amplitude_and_interstate_dipole_conventions(self):
        # Populate native objects with a synthetic orthonormal orbital basis
        # and solved-root fixtures. The tests need no SCF or response solve.
        mf = dft.RKS(self.mol, xc="pbe0")
        values, vectors = np.linalg.eigh(mf.get_ovlp())
        mf.mo_coeff = vectors / np.sqrt(values)
        mf.mo_occ = np.zeros(self.mol.nao)
        mf.mo_occ[:2] = 2.
        mf.converged = True
        singlet, triplet = TDA(mf), TDA(mf)
        shape = 2, self.mol.nao - 2
        x1, x2 = np.zeros(shape), np.zeros(shape)
        x1[0, 0] = x2[0, 1] = 1 / np.sqrt(2.)
        for solver, energies in ((singlet, [.30, .40]), (triplet, [.20, .32])):
            solver.xy = [(x1, 0.), (x2, 0.)]
            solver.e = np.array(energies)
            solver.converged = np.ones(2, dtype=bool)
        triplet.singlet = False
        matrices = assemble_pyscf_tda_soc(singlet, triplet, singlet_roots=(1, 0), triplet_roots=(1, 0))
        np.testing.assert_allclose(matrices.spin_free_energies_hartree, [0., .40, .30, .32, .32, .32, .20, .20, .20])
        np.testing.assert_allclose(matrices.dipole_operator_au[0, 1:3], -singlet.transition_dipole()[[1, 0]], atol=1e-14)
        np.testing.assert_array_equal(matrices.dipole_operator_au[:3, 3:], 0.)
        rates = matrices.phosphorescence(0, temperature_kelvin=77)
        np.testing.assert_allclose(rates.photon_energy_hartree, .20, atol=1e-6)
        self.assertIsNotNone(rates.vacuum_radiative_rate_s_inverse)

        singlet.converged[0] = False
        with self.assertRaisesRegex(ValueError, "converge"):
            assemble_pyscf_tda_soc(singlet, triplet)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            run_pyscf_tda_soc(mf, n_triplets=0)
        singlet.converged[:] = True
        with patch.object(mf, "TDA", side_effect=[singlet, triplet]), \
                patch.object(singlet, "kernel"), patch.object(triplet, "kernel"):
            direct = run_pyscf_tda_soc(mf, n_singlets=2, n_triplets=2)
        self.assertEqual(direct.singlet_roots, (0, 1))
        for solver in (singlet, triplet):
            self.assertEqual(solver.conv_tol, 1e-7)
            self.assertLess(np.sqrt(solver.lindep), solver.conv_tol)

    def test_nuclear_soc_sign_and_factor_match_libcint_spinors(self):
        mol = self.mol
        spinor = mol.sph2spinor_coeff()
        scalar = np.einsum("api,pq,aqj->ij", spinor.conj(), mol.intor("int1e_pnucp"), spinor)
        expected = (mol.intor("int1e_spnucsp_spinor") - scalar) / (4 * LIGHT_SPEED**2)
        pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        spatial = _breit_pauli_somf_ao(mol, np.zeros((mol.nao, mol.nao)))
        actual = np.einsum("api,xab,xpq,bqj->ij", spinor.conj(), pauli / 2, spatial, spinor)
        self.assertGreater(np.max(np.abs(expected)), 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-18)
