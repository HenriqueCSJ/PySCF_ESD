"""Analytic SOMF derivative checks at a single geometry."""

import unittest

import numpy as np

from pyscf_esd.pyscf_soc import _breit_pauli_somf_ao
from pyscf_esd.somf_derivatives import (
    breit_pauli_somf_ao_derivatives,
    breit_pauli_somf_mo_derivatives,
)

try:
    from pyscf import gto
    from pyscf.data.nist import LIGHT_SPEED
except ImportError:
    gto = None


def _transpose(x):
    return x.swapaxes(-1, -2)


@unittest.skipIf(gto is None, "requires PySCF")
class SOMFDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom="He 0 0 0; He 0.4 0.7 1.2", basis="cc-pvdz", verbose=0)
        self.rng = np.random.default_rng(6075)

    def test_nuclear_attribution_and_spin_vector_match_native_spinors(self):
        mol = self.mol
        spinor = mol.sph2spinor_coeff()
        pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
        density = np.zeros((mol.nao, mol.nao))
        actual = breit_pauli_somf_ao_derivatives(mol, density)
        actual = np.einsum("api,xab,nuxpq,bqj->nuij", spinor.conj(), pauli / 2, actual, spinor)

        def spin_part(spin_intor, scalar_intor):
            spin = mol.intor(spin_intor, comp=3)
            scalar = mol.intor(scalar_intor, comp=3)
            return spin - np.einsum("api,upq,aqj->uij", spinor.conj(), scalar, spinor)

        full = spin_part("int1e_ipspnucsp_spinor", "int1e_ippnucp")
        expected = np.zeros_like(actual)
        for atom, (_, _, ao_start, ao_end) in enumerate(mol.aoslice_by_atom()):
            ao_mask = np.zeros(mol.nao)
            ao_mask[ao_start:ao_end] = 1.
            # The atom mask is a projector on the bra AO of each spinor.
            bra_atom = np.einsum("api,p,apj->ij", spinor.conj(), ao_mask, spinor)
            # PySCF's real-to-spinor transform is unitary.
            bra = -bra_atom @ full
            with mol.with_rinv_at_nucleus(atom):
                bra -= mol.atom_charge(atom) * spin_part("int1e_ipsprinvsp_spinor", "int1e_ipprinvp")
            expected[atom] = (bra + _transpose(bra.conj())) / (4 * LIGHT_SPEED**2)
        self.assertGreater(np.linalg.norm(expected), 1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-17)

    def test_two_electron_derivative_matches_dense_all_slot_contractions(self):
        mol = self.mol
        matrix = self.rng.normal(size=(mol.nao, mol.nao))
        density = matrix @ matrix.T
        grad_i = mol.intor("int2e_ipspsp1", comp=12).reshape(3, 4, *(mol.nao,) * 4)[:, :3]
        grad_k = mol.intor("int2e_ip1spsp2", comp=12).reshape(3, 4, *(mol.nao,) * 4)[:, :3]
        grad_k = grad_k.transpose(0, 1, 4, 5, 2, 3)
        expected = np.empty((mol.natm, 3, 3, mol.nao, mol.nao), dtype=complex)
        for atom, (_, _, start, end) in enumerate(mol.aoslice_by_atom()):
            i = np.zeros_like(grad_i)
            k = np.zeros_like(grad_k)
            i[:, :, start:end] = -grad_i[:, :, start:end]
            k[:, :, :, :, start:end] = -grad_k[:, :, :, :, start:end]
            dg = i - i.swapaxes(2, 3) + k + k.swapaxes(4, 5)
            direct = np.einsum("uxijkl,lk->uxij", dg, density)
            exchange_1 = np.einsum("uxijkl,jk->uxil", dg, density)
            exchange_2 = np.einsum("uxijkl,li->uxkj", dg, density)
            expected[atom] = 1j * (direct - 1.5 * (exchange_1 + exchange_2)) / (2 * LIGHT_SPEED**2)

        fixed_density = breit_pauli_somf_ao_derivatives(mol, density)
        nuclear = breit_pauli_somf_ao_derivatives(mol, np.zeros_like(density))
        self.assertGreater(np.linalg.norm(expected), 1e-5)
        np.testing.assert_allclose(fixed_density - nuclear, expected, rtol=1e-11, atol=1e-17)
        np.testing.assert_allclose(fixed_density.sum(axis=0), 0., atol=1e-17)
        np.testing.assert_allclose(fixed_density, _transpose(fixed_density.conj()), atol=1e-17)

    def test_orbital_response_and_direct_projection_include_density_response(self):
        mol = self.mol
        coefficients = self.rng.normal(size=(mol.nao, 4))
        occupations = np.array([2, 0, 2, 0])
        derivatives = self.rng.normal(size=(mol.natm, 3, mol.nao, 4))
        occupied = coefficients[:, occupations == 2]
        density = 2 * occupied @ occupied.T
        half_dd = 2 * derivatives[..., occupations == 2] @ occupied.T
        dd = half_dd + _transpose(half_dd)
        eq_ao = _breit_pauli_somf_ao(mol, density)
        frozen = breit_pauli_somf_ao_derivatives(mol, density)
        relaxed = breit_pauli_somf_ao_derivatives(mol, density, dd)
        reference_nuclear = _breit_pauli_somf_ao(mol, np.zeros_like(density))
        expected = np.stack([
            _breit_pauli_somf_ao(mol, direction) - reference_nuclear
            for direction in dd.reshape(-1, mol.nao, mol.nao)
        ]).reshape(relaxed.shape)
        self.assertGreater(np.linalg.norm(expected), 1e-5)
        np.testing.assert_allclose(relaxed - frozen, expected, rtol=1e-11, atol=1e-17)

        actual = breit_pauli_somf_mo_derivatives(mol, coefficients, occupations, derivatives)
        response = _transpose(derivatives)[..., None, :, :] @ eq_ao @ coefficients
        expected_mo = coefficients.T @ relaxed @ coefficients + response + _transpose(response.conj())
        np.testing.assert_allclose(actual, expected_mo, rtol=1e-11, atol=1e-16)

        directions = self.rng.normal(size=(4, mol.natm, 3))
        derivatives_by_direction = np.einsum("dau,auij->dij", directions, derivatives)
        actual_by_direction = breit_pauli_somf_mo_derivatives(
            mol, coefficients, occupations, derivatives_by_direction, directions=directions,
        )
        expected_by_direction = np.einsum("dau,auxij->dxij", directions, expected_mo)
        np.testing.assert_allclose(actual_by_direction, expected_by_direction, rtol=1e-11, atol=1e-16)
