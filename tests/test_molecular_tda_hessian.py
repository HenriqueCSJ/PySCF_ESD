"""Same-geometry checks of the molecular DF-RKS/TDA Hessian."""

from dataclasses import replace
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class MolecularTDAResponseRefinementTests(unittest.TestCase):
    def test_vacuum_has_zero_fixed_density_solvent_terms(self):
        from pyscf_esd.molecular_tda_hessian import _MolecularTDAOperator

        model = _MolecularTDAOperator.__new__(_MolecularTDAOperator)
        model.mf = SimpleNamespace(mol=SimpleNamespace(natm=2, nao=5))
        gradients, hessian = model.solvent_derivatives()
        self.assertEqual(len(gradients), 3)
        for gradient in gradients:
            self.assertEqual(gradient.shape, (6, 5, 5))
            np.testing.assert_array_equal(gradient, 0)
        self.assertEqual(hessian.shape, (6, 6))
        np.testing.assert_array_equal(hessian, 0)

    def test_refines_only_projected_response_when_relaxation_is_asymmetric(self):
        from pyscf_esd import molecular_tda_hessian
        from pyscf_esd.tda_hessian import assemble_tda_hessian, solve_tda_root_response

        matrix = np.diag([0.3, 0.35, 0.5])
        amplitude = np.array([1.0, 0.0, 0.0])
        actions = np.array([[0.05, 0.0, 0.2], [-0.08, 0.3, 0.0]])
        exact = solve_tda_root_response(matrix, 0.3, amplitude, actions)
        derivatives = exact.amplitude_derivatives.copy()
        derivatives[0, 1] += 2.3e-8
        residual = np.linalg.norm((0.3 * np.eye(3) - matrix) @ derivatives[0] - exact.projected_first_action[0])
        loose = replace(exact, amplitude_derivatives=derivatives, max_response_residual=residual)
        self.assertLess(residual, 1e-10 + 1e-8 * 0.2)
        with self.assertRaisesRegex(ValueError, "asymmetry 1.38e-08"):
            assemble_tda_hessian(np.zeros((2, 2)), np.zeros((2, 2)), loose)

        chunks = (actions[:1], actions[1:])
        events = []
        calls = []

        options = {
            "response_atol": 1e-10, "response_rtol": 1e-8, "root_tolerance": 1e-8,
            "root_gap_tolerance": 1e-4, "root_separation_hartree": 0.05,
        }

        def solve(returned):
            responses = iter(returned)
            calls.clear()
            events.clear()

            def stub_solver(operator, omega, root, first_actions, **solver_options):
                calls.append((tuple(first_actions), solver_options))
                self.assertIs(operator, matrix)
                self.assertEqual(omega, 0.3)
                self.assertIs(root, amplitude)
                return next(responses)

            with patch.object(molecular_tda_hessian, "solve_tda_root_response", side_effect=stub_solver):
                return molecular_tda_hessian._solve_symmetric_tda_response(
                    matrix, 0.3, amplitude, iter(chunks), **options, progress=lambda *args: events.append(args),
                )

        self.assertIs(solve([exact]), exact)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events, [])

        self.assertIs(solve([loose, exact]), exact)
        self.assertEqual(len(calls), 2)
        for batch, _ in calls:
            self.assertEqual(len(batch), len(chunks))
            for reused, original in zip(batch, chunks):
                self.assertIs(reused, original)
        self.assertEqual(calls[0][1], options)
        self.assertEqual(calls[1][1], dict(options, response_atol=1e-10 * 0.1, response_rtol=1e-8 * 0.1))
        self.assertEqual(events, [("tda_root_response_retry", 0, 2), ("tda_root_response_retry", 2, 2)])

        with self.assertRaisesRegex(ValueError, "after one tighter projected solve"):
            solve([loose, loose])
        self.assertEqual(len(calls), 2)


@unittest.skipUnless(find_spec("gpu4pyscf"), "GPU4PySCF is required")
class MolecularTDAHessianTests(unittest.TestCase):
    def test_water_curvature_and_gradient(self):
        import cupy as cp
        from gpu4pyscf import dft
        from pyscf import gto, lib

        from pyscf_esd.molecular_tda_hessian import analytic_molecular_tda_hessian
        from pyscf_esd.tda_xc_curvature import require_semilocal_xc_fourth
        from pyscf_esd.tda_xc_grid import moving_xc_grid_correction
        from pyscf_esd.tda_xc_operator import MovingXCOperator

        try:
            usable_gpu = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            usable_gpu = False
        if not usable_gpu:
            self.skipTest("a CUDA GPU is required")
        try:
            require_semilocal_xc_fourth("PBE0")
        except NotImplementedError as error:
            self.skipTest(str(error))

        mol = gto.M(atom="O 0 0 0; H .75 .02 .51; H -.66 -.13 .58", basis="sto-3g", verbose=0)
        mf = dft.RKS(mol, xc="PBE0").density_fit().PCM()
        mf.grids.level = 1
        mf.small_rho_cutoff = 0
        mf.cphf_grids = mf.grids
        mf.conv_tol, mf.conv_tol_grad, mf.conv_tol_cpscf = 1e-13, 1e-10, 1e-10
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.eps = 2.023
        mf.with_solvent.lebedev_order = 29
        mf.kernel()
        self.assertTrue(mf.converged)
        td = mf.TDA(equilibrium_solvation=True)
        td.conv_tol, td.nstates = 5e-10, 3
        nocc = mol.nelectron // 2
        td.kernel(x0=cp.eye(nocc * (mol.nao - nocc)))
        result = analytic_molecular_tda_hessian(td, 0)
        grad = td.Gradients()
        grad.cphf_conv_tol, grad.cphf_max_cycle = 1e-10, 150
        with lib.temporary_env(mf.with_solvent, equilibrium_solvation=True):
            native = cp.asnumpy(grad.kernel(state=1))
        ground = mf.Gradients()
        ground.grid_response = False
        correction = moving_xc_grid_correction(mf, grad._dmz1doo, grad._dmxpy + grad._dmxpy.T)
        native = native + correction - cp.asnumpy(ground.kernel())
        np.testing.assert_allclose(result.excitation_gradient, native, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.hessian.sum(axis=0), 0, atol=3e-7, rtol=0)
        paired = result.hessian.reshape(3, 3, 3, 3).transpose(1, 0, 3, 2)
        np.testing.assert_allclose(result.hessian, paired, atol=1e-11, rtol=0)
        self.assertLess(result.orbital_adjoint_residual, 1e-8)
        self.assertLess(result.tda_response_residual, 1e-8)

        ground.grid_response = True
        total_gradient = cp.asnumpy(ground.kernel()) + result.excitation_gradient
        # The atom-centered XC and SWIG angular quadratures leave a finite
        # rotation defect even though exact coordinate translations cancel.
        for axis in np.eye(3):
            velocity = np.cross(axis, mol.atom_coords())
            rotation = np.einsum("abxy,by->ax", result.hessian, velocity)
            np.testing.assert_allclose(rotation, np.cross(axis, total_gradient), atol=2e-3, rtol=0)

        # Compare the batched grid contraction with two independent exact
        # directional orders, using both orbital density and grid motion.
        rng = np.random.default_rng(811)
        p = cp.asnumpy(mf.make_rdm1())
        d1 = rng.normal(size=(3 * mol.natm, mol.nao, mol.nao)) * 0.03
        d1 = (d1 + d1.transpose(0, 2, 1)) * 0.5
        r, t = rng.normal(size=(2, mol.nao, mol.nao)) * 0.03
        grid = MovingXCOperator(mf)
        batched = grid.bilinear_hessian(p, d1, r, t)
        x, y = rng.normal(size=(2, mol.natm, 3))
        fixed_r, fixed_t = (r, None, None, None), (t, None, None, None)
        for a, b in ((x, y), (y, x)):
            pa, pb = np.einsum("n,nij->ij", a.ravel(), d1), np.einsum("n,nij->ij", b.ravel(), d1)
            fock, kernel = grid.directional(a, b, (p, pa, pb, None), fixed_t, scalar_probes=(fixed_r, fixed_t))
            self.assertAlmostEqual(a.ravel() @ batched @ b.ravel(), fock[3] + 2 * kernel[3], places=8)
