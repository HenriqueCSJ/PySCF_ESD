"""Same-geometry checks for analytic spin-adapted TDA nuclear response."""

from importlib.util import find_spec
import unittest

import numpy as np


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required")
class TDANuclearResponseTests(unittest.TestCase):
    def _reference(self, *, fitted, level):
        from pyscf import dft, gto

        mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g", verbose=0)
        mf = dft.RKS(mol, xc="b3lyp")
        if fitted:
            mf = mf.density_fit(auxbasis="def2-universal-jkfit")
        mf.small_rho_cutoff = 0
        mf.grids.level = level
        mf.conv_tol = 1e-12
        mf.kernel()
        self.assertTrue(mf.converged)
        return mf

    def _root(self, mf, singlet):
        td = mf.TDA()
        td.singlet = singlet
        td.nstates = 2
        td.conv_tol = 1e-11
        td.kernel()
        self.assertTrue(np.all(td.converged))
        return td

    def test_fitted_gauge_and_operator_bilinears_for_both_spins(self):
        from pyscf import lib

        from pyscf_esd.tda_nuclear_response import analytic_tda_nuclear_response

        with lib.with_omp_threads(1):
            mf = self._reference(fitted=True, level=1)
            evaluator = analytic_tda_nuclear_response(mf)
            nocc = evaluator.shape[0]
            c, s = mf.mo_coeff, mf.get_ovlp()
            c1 = evaluator.mo_coeff_derivatives
            u = c.T @ s @ c1
            self.assertLess(evaluator.orbital_response_residual, 1e-8)
            np.testing.assert_allclose(u[:, :nocc, :nocc], u[:, :nocc, :nocc].swapaxes(-1, -2), atol=1e-12)
            np.testing.assert_allclose(u[:, nocc:, nocc:], u[:, nocc:, nocc:].swapaxes(-1, -2), atol=1e-12)
            np.testing.assert_allclose(c1.reshape(3, 3, *c.shape).sum(axis=0), 0, atol=1e-9)
            for i, direction in enumerate(evaluator.directions):
                s1 = evaluator.core.first_one_electron(direction).overlap
                np.testing.assert_allclose(u[i] + u[i].T + c.T @ s1 @ c, 0, atol=1e-12)

            q = np.random.default_rng(13).uniform(-1, 1, (3, *evaluator.shape))
            left, right = np.array([0, 0, 1, 2, 0]), np.array([0, 1, 2, 0, 2])
            for singlet in (True, False):
                with self.subTest(singlet=singlet):
                    td = self._root(mf, singlet)
                    apply, _ = td.gen_vind()
                    np.testing.assert_allclose(
                        evaluator.operator_action(q, singlet=singlet).reshape(3, -1),
                        apply(q.reshape(3, -1)), atol=1e-12,
                    )
                    a1q = evaluator.operator_derivatives(q, singlet=singlet)
                    np.testing.assert_allclose(a1q.reshape(3, 3, *a1q.shape[1:]).sum(axis=0), 0, atol=1e-9)
                    projected = np.einsum("qia,rpia->rqp", q, a1q)
                    np.testing.assert_allclose(projected, projected.swapaxes(-1, -2), atol=1e-11)
                    bilinear = evaluator.operator_derivative_bilinears(q[left], q[right], singlet=singlet)
                    np.testing.assert_allclose(bilinear, projected[:, left, right], atol=1e-11)
                    root = evaluator.root_response(td, 0)
                    np.testing.assert_allclose(root.amplitude, np.sqrt(2) * td.xy[0][0], atol=1e-12)
                    np.testing.assert_allclose(root.amplitude_derivatives.reshape(3, 3, *q.shape[1:]).sum(axis=0),
                                               0, atol=1e-9)
                    np.testing.assert_allclose(np.einsum("ria,ia->r", root.amplitude_derivatives, root.amplitude),
                                               0, atol=1e-12)
                    self.assertLess(root.max_response_residual, 1e-9)

    def test_conventional_root_energies_with_native_analytic_gradients(self):
        from pyscf import lib

        from pyscf_esd.tda_nuclear_response import analytic_tda_nuclear_response

        with lib.with_omp_threads(1):
            mf = self._reference(fitted=False, level=4)
            # Select the symmetric O-H stretch using both bonds in Bohr.
            direction = np.zeros((mf.mol.natm, 3))
            direction[1:] = mf.mol.atom_coords()[1:] - mf.mol.atom_coords()[0]
            direction /= np.linalg.norm(direction)
            evaluator = analytic_tda_nuclear_response(mf, directions=direction)
            ground = mf.Gradients().kernel()
            for singlet in (True, False):
                with self.subTest(singlet=singlet):
                    td = self._root(mf, singlet)
                    root = evaluator.root_response(td, 0)
                    native = td.Gradients().kernel(state=1) - ground
                    # Conventional PySCF TD gradients omit quadrature motion.
                    # A dense grid makes that contribution small in this check.
                    np.testing.assert_allclose(root.excitation_gradient, [np.vdot(direction, native)], atol=3e-6)
                    bilinear = evaluator.operator_derivative_bilinears(root.amplitude, root.amplitude, singlet=singlet)
                    np.testing.assert_allclose(root.excitation_gradient, bilinear, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
