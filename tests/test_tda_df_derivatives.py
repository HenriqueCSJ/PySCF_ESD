import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required")
class MolecularDFDerivativeTests(unittest.TestCase):
    def setUp(self):
        from pyscf import dft, gto

        self.mol = gto.M(
            atom="O 0 0 0; H .76 .04 .48; H -.60 .16 .61", basis="def2-svp", verbose=0,
        )
        self.mf = dft.RKS(self.mol, xc="pbe0").density_fit(auxbasis="def2-svp-jkfit")
        self.rng = np.random.default_rng(35)
        self.x, self.y = self.rng.normal(size=(2, self.mol.natm, 3)) / 3

    def test_nonsymmetric_batch_jk_matches_native_and_preserves_reciprocity(self):
        from pyscf_esd.tda_df_derivatives import MolecularDFDerivatives

        dm = self.rng.normal(size=(2, self.mol.nao, self.mol.nao))
        fit = MolecularDFDerivatives(self.mf.with_df)
        jet = fit.directional(self.x, self.y)
        actions = jet.jk(dm)
        native = self.mf.with_df.get_jk(dm, hermi=0)
        np.testing.assert_allclose(actions.value, native, atol=2e-12, rtol=0)
        np.testing.assert_allclose(fit.reference_jk(dm), native, atol=2e-12, rtol=0)
        swapped = fit.directional(self.y, self.x).jk(dm)
        np.testing.assert_allclose(actions.first_x, swapped.first_y, atol=2e-12, rtol=0)
        np.testing.assert_allclose(actions.mixed_xy, swapped.mixed_xy, atol=2e-12, rtol=0)

        left_right = jet.bilinear(dm[0], dm[1])
        right_left = jet.bilinear(dm[1], dm[0])
        expected = [np.einsum("ij,kij->k", dm[0], [order[0][1], order[1][1]]) for order in (
            actions.value, actions.first_x, actions.first_y, actions.mixed_xy,
        )]
        np.testing.assert_allclose(
            [left_right.value, left_right.first_x, left_right.first_y, left_right.mixed_xy],
            expected, atol=2e-12, rtol=0,
        )
        np.testing.assert_allclose(
            (left_right.first_x, left_right.mixed_xy), (right_left.first_x, right_left.mixed_xy),
            atol=2e-12, rtol=0,
        )
        exchange = fit.directional(self.x).jk(dm[0], with_j=False)
        self.assertIsNone(exchange.value[0])
        self.assertIsNone(exchange.first_y)
        np.testing.assert_allclose(exchange.first_x[1], actions.first_x[1][0], atol=2e-12, rtol=0)

    def test_fixed_density_contractions_match_native_analytic_df_gradient_and_hessian(self):
        from pyscf.df.grad.rhf import get_jk
        from pyscf.df.hessian.rhf import _partial_hess_ejk

        from pyscf_esd.tda_df_derivatives import MolecularDFDerivatives

        # The native DF routines allow arbitrary fixed coefficients. Zero
        # orbital energies and the supplied density avoid any electronic solve.
        coeff = np.linalg.qr(self.rng.normal(size=(self.mol.nao, self.mol.nao)))[0]
        occ = np.zeros(self.mol.nao)
        occ[:2] = 2
        density = 2 * coeff[:, :2] @ coeff[:, :2].T
        fit = MolecularDFDerivatives(self.mf.with_df)
        bilinear = fit.directional(self.x, self.y).bilinear(density, density)
        grad = self.mf.nuc_grad_method()
        grad.auxbasis_response = True
        potentials = get_jk(grad, dm=density, hermi=1)

        def gradient_along_x(potential):
            ao_term = sum(
                np.einsum("cij,ij,c->", potential[:, p0:p1], density[p0:p1], self.x[atom])
                for atom, (_, _, p0, p1) in enumerate(self.mol.aoslice_by_atom())
            )
            aux_term = np.einsum("ac,ac->", potential.aux[0, 0], self.x)
            return 2 * ao_term + aux_term

        np.testing.assert_allclose(
            np.asarray(bilinear.first_x) / 2, [gradient_along_x(potential) for potential in potentials],
            atol=2e-11, rtol=0,
        )
        hess = self.mf.Hessian()
        hess.auxbasis_response = 2
        _, hj, hk = _partial_hess_ejk(hess, np.zeros_like(occ), coeff, occ)
        np.testing.assert_allclose(
            np.asarray(bilinear.mixed_xy) / (2, 4),
            [np.einsum("abcd,ac,bd->", term, self.x, self.y) for term in (hj, hk)],
            atol=2e-11, rtol=0,
        )
        cartesian = fit.bilinear_hessian(density, density)
        for fast, native, factor in zip(cartesian, (hj, hk), (2, 4)):
            expected = native.transpose(0, 2, 1, 3).reshape(fast.shape)
            np.testing.assert_allclose(fast / factor, expected, atol=1e-10, rtol=0)

    def test_all_cartesian_contractions_cover_nonsymmetric_and_disk_backed_exchange(self):
        from pyscf_esd.tda_df_derivatives import MolecularDFDerivatives

        # A small memory limit exercises multiple auxiliary shell blocks and
        # the disk-backed projected derivative path.
        self.mf.with_df.max_memory = 0.1
        fit = MolecularDFDerivatives(self.mf.with_df)
        size = self.mol.nao
        left = self.rng.normal(size=(size, size)) / size
        right = self.rng.normal(size=(size, 3)) @ self.rng.normal(size=(3, size)) / size
        hessians = fit.bilinear_hessian(left, right)
        for hessian in hessians:
            np.testing.assert_allclose(hessian, hessian.T, atol=1e-14, rtol=0)
            translation = hessian.reshape(3 * self.mol.natm, self.mol.natm, 3).sum(axis=1)
            np.testing.assert_allclose(translation, 0, atol=2e-12, rtol=0)
        expected = fit.directional(self.x, self.y).bilinear(left, right).mixed_xy
        np.testing.assert_allclose(
            [self.x.ravel() @ hessian @ self.y.ravel() for hessian in hessians],
            expected, atol=2e-12, rtol=0,
        )
        exchange_only = fit.bilinear_hessian(left, right, with_j=False)
        self.assertIsNone(exchange_only[0])
        np.testing.assert_allclose(exchange_only[1], hessians[1], atol=2e-12, rtol=0)
        self_exchange = fit.bilinear_hessian(right, right, with_j=False)[1]
        self_mixed = fit.directional(self.x, self.y).bilinear(right, right, with_j=False).mixed_xy[1]
        np.testing.assert_allclose(self.x.ravel() @ self_exchange @ self.y.ravel(), self_mixed,
                                   atol=2e-12, rtol=0)

    def test_single_atom_action_matches_dense_analytic_derivatives(self):
        from pyscf_esd.tda_df_derivatives import MolecularDFDerivatives

        self.mf.with_df.max_memory = 0.1
        fit = MolecularDFDerivatives(self.mf.with_df)
        left, right = self.rng.normal(size=(2, self.mol.nao, 3))
        dm = np.stack((left @ right.T, (left * (1, 1, -1)) @ left.T, np.zeros((self.mol.nao, self.mol.nao))))
        for atom in range(self.mol.natm):
            x = np.zeros_like(self.x)
            x[atom] = self.x[atom]
            jet = fit.directional(x)
            sparse = jet.jk(dm)
            dense = fit.directional(x, self.y).jk(dm)
            np.testing.assert_allclose(sparse.value, dense.value, atol=2e-12, rtol=0)
            np.testing.assert_allclose(sparse.first_x, dense.first_x, atol=2e-12, rtol=0)
            np.testing.assert_allclose(jet.first_jk(dm), dense.first_x, atol=2e-12, rtol=0)
            np.testing.assert_allclose(fit.reference_jk(dm), dense.value, atol=2e-12, rtol=0)
            self.assertIsNone(sparse.first_y)
            self.assertIsNone(sparse.mixed_xy)

    def test_cartesian_ao_and_auxiliary_translation_invariance(self):
        from pyscf import df, gto

        from pyscf_esd.tda_df_derivatives import MolecularDFDerivatives

        mol = gto.M(
            atom="O 0 0 0; H .76 .04 .48; H -.60 .16 .61", basis="def2-svp", cart=True, verbose=0,
        )
        reference = df.DF(mol, auxbasis="def2-svp-jkfit")
        fit = MolecularDFDerivatives(reference)
        shift = np.array([1.0, -0.5, 0.25])
        x_translation = np.broadcast_to(shift, self.x.shape)
        dm = self.rng.normal(size=(mol.nao, mol.nao))
        translation = fit.directional(x_translation, self.y).jk(dm)
        np.testing.assert_array_equal(translation.first_x, 0)
        np.testing.assert_array_equal(translation.mixed_xy, 0)
        native = reference.get_jk(dm, hermi=0)
        np.testing.assert_allclose(translation.value, native, atol=2e-12, rtol=0)
        mixed = fit.directional(self.x, self.y).jk(dm)
        shifted = fit.directional(self.x + shift, self.y - shift).jk(dm)
        np.testing.assert_allclose(mixed.mixed_xy, shifted.mixed_xy, atol=2e-12, rtol=0)
