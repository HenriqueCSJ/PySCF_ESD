"""Check analytic CIS curvature and invariance to the orbital gauge."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.cis_hessian import analytic_cis_hessian, analytic_cis_transition_dipole
from pyscf_esd.state_tracking import orbital_overlap_blocks, track_state


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the CIS Hessian")
class CISHessianTests(unittest.TestCase):
    def setUp(self):
        from pyscf import gto, lib

        self.addCleanup(lib.num_threads, lib.num_threads())
        lib.num_threads(1)
        self.mol = gto.M(atom="Li 0 0 0; H 0.7 1.1 3.4", basis="sto-3g", unit="Bohr", verbose=0)

    def electronic(self, mol):
        from pyscf import scf

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        cis = mf.TDA()
        nocc = np.count_nonzero(mf.mo_occ)
        cis.kernel(nstates=1, x0=np.eye(nocc * (mol.nao - nocc)))
        return mf, cis

    def test_matches_native_energy_and_gradient_in_both_gauges(self):
        mf, cis = self.electronic(self.mol)
        amplitude = cis.xy[0][0]
        grad = cis.nuc_grad_method().kernel(state=1)
        canonical = analytic_cis_hessian(mf, amplitude)
        np.testing.assert_allclose(canonical.total_energy_hartree, mf.e_tot + cis.e[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(canonical.gradient, grad, atol=1e-9, rtol=0)
        np.testing.assert_allclose(canonical.hessian.sum(axis=1), 0, atol=1e-10, rtol=0)

        nocc, nvir = amplitude.shape
        rng = np.random.default_rng(37)
        occupied, _ = np.linalg.qr(rng.normal(size=(nocc, nocc)))
        virtual, _ = np.linalg.qr(rng.normal(size=(nvir, nvir)))
        mf.mo_coeff = np.column_stack((mf.mo_coeff[:, :nocc] @ occupied, mf.mo_coeff[:, nocc:] @ virtual))
        rotated = analytic_cis_hessian(mf, occupied.T @ amplitude @ virtual)
        np.testing.assert_allclose(rotated.gradient, canonical.gradient, atol=1e-10, rtol=0)
        np.testing.assert_allclose(rotated.hessian, canonical.hessian, atol=1e-10, rtol=0)

        mf.disp = "d3bj"
        with self.assertRaisesRegex(ValueError, "unmodified RHF Hamiltonian"):
            analytic_cis_hessian(mf, amplitude)

    def test_analytic_dipole_derivative_matches_tracked_native_dipoles(self):
        from pyscf import gto

        water = gto.M(atom="O 0 0 0; H 0.2 1.4 1.1; H 0 -1.5 1.2",
                      basis="sto-3g", unit="Bohr", verbose=0)
        mf, cis = self.electronic(water)
        amplitude = cis.xy[0][0]
        result = analytic_cis_transition_dipole(mf, amplitude)
        np.testing.assert_allclose(result.dipole, cis.transition_dipole()[0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(result.derivative.sum(axis=0), 0, atol=1e-9, rtol=0)
        rotations = np.cross(np.eye(3)[:, None, :], water.atom_coords())
        np.testing.assert_allclose(
            np.einsum("rax,axd->rd", rotations, result.derivative),
            np.cross(np.eye(3), result.dipole), atol=1e-9, rtol=0,
        )
        direction = np.array([[0.2, -0.1, 0.4], [0.3, 0.6, -0.2], [-0.7, 0.1, -0.3]])
        step = 2e-4
        moments = []
        for sign in (-1, 1):
            mol = water.set_geom_(water.atom_coords() + sign * step * direction, unit="Bohr", inplace=False)
            displaced, displaced_cis = self.electronic(mol)
            occ, vir = orbital_overlap_blocks(
                mf.mo_coeff, displaced.mo_coeff, gto.intor_cross("int1e_ovlp", water, mol),
                mf.mo_occ, displaced.mo_occ,
            )
            match = track_state(amplitude, [displaced_cis.xy[0][0]], occ, vir, minimum_margin=0)
            moments.append(match.phase * displaced_cis.transition_dipole()[0])
        expected = (moments[1] - moments[0]) / (2 * step)
        np.testing.assert_allclose(np.einsum("ax,axd->d", direction, result.derivative), expected, atol=2e-7, rtol=0)

        flipped = analytic_cis_transition_dipole(mf, -amplitude)
        np.testing.assert_allclose(flipped.dipole, -result.dipole, atol=1e-12, rtol=0)
        np.testing.assert_allclose(flipped.derivative, -result.derivative, atol=1e-12, rtol=0)
