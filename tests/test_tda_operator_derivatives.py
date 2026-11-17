import numpy as np
import unittest
from importlib.util import find_spec


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required")
class CPCMOperatorDerivativeTests(unittest.TestCase):
    def test_cpcm_action_matches_same_geometry_native_response(self):
        from pyscf import gto
        from pyscf.solvent import pcm
        from pyscf.solvent.hessian.pcm import (
            analytical_grad_vmat, analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver,
        )

        from pyscf_esd.tda_operator_derivatives import (
            cpcm_transition_kernel_action, cpcm_transition_kernel_bilinear,
        )

        mol = gto.M(atom="O 0 0 0; H .76 .04 .48; H -.60 .16 .61", basis="sto-3g", verbose=0)
        solvent = pcm.PCM(mol)
        solvent.method = "C-PCM"
        solvent.eps = 2.023
        solvent.lebedev_order = 7
        solvent.build()
        rng = np.random.default_rng(1284)
        density = rng.normal(size=(mol.nao, mol.nao)) / mol.nao
        density = (density + density.T) / 2
        x, y = rng.normal(size=(2, mol.natm, 3)) / 3
        action = cpcm_transition_kernel_action(solvent, density, x, y)
        bilinear = cpcm_transition_kernel_bilinear(solvent, density, density, x, y)

        np.testing.assert_allclose(action.value, solvent._B_dot_x(density), atol=1e-12, rtol=0)
        # The response is linear in the density at this one geometry.
        grad_v = (analytical_grad_vmat(solvent, density) - analytical_grad_vmat(solvent, -density)) / 2
        np.testing.assert_allclose(action.first_x, np.einsum("axpq,ax->pq", grad_v, x), atol=1e-12, rtol=0)

        def fixed_density_hessian(dm):
            solvent.kernel(dm)
            return (analytical_hess_nuc(solvent, dm) + analytical_hess_qv(solvent, dm)
                    + analytical_hess_solver(solvent, dm))

        # The exact even part of the quadratic PCM energy in density isolates B″.
        response = fixed_density_hessian(density) + fixed_density_hessian(-density)
        response -= 2 * fixed_density_hessian(np.zeros_like(density))
        contracted = np.einsum("abxy,ax,by->", response, x, y)
        np.testing.assert_allclose(bilinear.mixed_xy, contracted, atol=2e-12, rtol=0)
        np.testing.assert_allclose(np.einsum("ij,ji->", density, action.mixed_xy), contracted, atol=2e-12, rtol=0)

    def test_cpcm_fock_potential_includes_moving_nuclei(self):
        from pyscf import gto
        from pyscf.solvent import pcm
        from pyscf.solvent.hessian.pcm import (
            analytical_grad_vmat, analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver,
        )

        from pyscf_esd.tda_operator_derivatives import cpcm_fock_potential_derivatives

        mol = gto.M(atom="O .02 .01 -.03; H .76 .04 .48; H -.60 .16 .61", basis="sto-3g", verbose=0)
        solvent = pcm.PCM(mol)
        solvent.method, solvent.eps, solvent.lebedev_order = "C-PCM", 2.023, 7
        rng = np.random.default_rng(941)
        density, probe = rng.normal(size=(2, mol.nao, mol.nao)) / mol.nao
        density, probe = (density + density.T) / 2, (probe + probe.T) / 2
        x, y = rng.normal(size=(2, mol.natm, 3)) / 3
        derivatives = cpcm_fock_potential_derivatives(solvent, density, x, y)
        np.testing.assert_allclose(derivatives.value, solvent.kernel(density)[1], atol=2e-12, rtol=0)
        native = analytical_grad_vmat(solvent, density)
        for direction, value in ((x, derivatives.first_x), (y, derivatives.first_y)):
            np.testing.assert_allclose(value, np.einsum("acpq,ac->pq", native, direction), atol=2e-12, rtol=0)

        def hessian(dm):
            solvent.kernel(dm)
            return (analytical_hess_nuc(solvent, dm) + analytical_hess_qv(solvent, dm)
                    + analytical_hess_solver(solvent, dm))

        # The solvent energy Hessian is exactly quadratic in the AO density.
        mixed = np.einsum("abxy,ax,by->", (hessian(density + probe) - hessian(density - probe)) / 2, x, y)
        np.testing.assert_allclose(np.einsum("ij,ji->", probe, derivatives.mixed_xy), mixed, atol=2e-12, rtol=0)
        rigid = cpcm_fock_potential_derivatives(solvent, density, np.ones_like(x), y)
        np.testing.assert_allclose(rigid.first_x, 0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(rigid.mixed_xy, 0, atol=1e-14, rtol=0)
