import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pyscf_esd.tda_xc_operator import MovingXCOperator


class MovingXCFirstDerivativeTests(unittest.TestCase):
    def test_batched_ao_projections_match_directional_lda_and_gga(self):
        rng = np.random.default_rng(86)
        natm, nao, ngrid = 3, 7, 13
        ncart = 3 * natm
        ao_bounds = np.array([[0, 2], [2, 5], [5, 7]])
        grids = SimpleNamespace(
            coords=np.arange(ngrid), weights=rng.uniform(.1, .5, ngrid),
            quadrature_weights=np.ones(ngrid), atm_idx=np.array([0, 1, 0, 2, 2, 0, 0, 2, 1, 0, 1, 2, 2]),
        )
        weights1 = rng.normal(size=(natm, 3, ngrid))
        ao = rng.normal(size=(10, nao, ngrid))
        mol = SimpleNamespace(
            natm=natm, nao_nr=lambda: nao,
            aoslice_by_atom=lambda: np.hstack((np.zeros((natm, 2), dtype=int), ao_bounds)),
        )
        ground = rng.normal(size=(nao, nao))
        ground = ground @ ground.T
        probe = rng.normal(size=(nao, nao))
        probe = (probe + probe.T) * .5
        density1 = rng.normal(size=(ncart, nao, nao))

        class HostArrayModule:
            def __getattr__(self, name):
                return np.asarray if name == "asnumpy" else getattr(np, name)

        class NumInt:
            gdftopt = None

            def __init__(self, xctype):
                self._xc_type = lambda xc: xctype

            def eval_xc_eff(self, xc, rho, order, **kwargs):
                rho = np.atleast_2d(rho)
                # Point energy proportional to (v.rho)^3 in this fixture.
                v = np.linspace(.25, .45, len(rho))
                amplitude = v @ rho
                potential = .5 * np.einsum("i,g->ig", v, amplitude**2)
                kernel = np.einsum("i,j,g->ijg", v, v, amplitude)
                third = np.einsum("i,j,k,g->ijkg", v, v, v, np.ones_like(amplitude))
                return (None, potential, kernel, third)[:order + 1]

        def eval_ao(mol, points, *, deriv, **kwargs):
            return ao[:(4 if deriv == 1 else 10), :, points]

        modules = {
            "cupy": HostArrayModule(),
            "gpu4pyscf.dft.numint": SimpleNamespace(eval_ao=eval_ao),
            "gpu4pyscf.hessian.rks": SimpleNamespace(
                get_dweight_dA=lambda mol, grid, rows: weights1[..., slice(*rows)],
                get_d2weight_dAdB=None,
            ),
        }
        with mock.patch.dict(sys.modules, modules):
            for xctype in ("LDA", "GGA"):
                mf = SimpleNamespace(mol=mol, grids=grids, _numint=NumInt(xctype), xc="fixture")
                operator = MovingXCOperator(mf, batch=5)
                for test_density, ground1 in ((None, None), (probe, density1)):
                    with self.subTest(xctype=xctype, probe=test_density is not None):
                        batched = operator.first_derivatives(ground, test_density, ground1)
                        for i, direction in enumerate(np.eye(ncart).reshape(ncart, natm, 3)):
                            ground_jet = ground, (None if ground1 is None else ground1[i])
                            probe_jet = None if test_density is None else (test_density, None)
                            reference = operator.directional(direction, None, ground_jet, probe_jet)
                            expected = reference[int(test_density is not None)][1]
                            np.testing.assert_allclose(batched[i], expected, atol=1e-8, rtol=1e-13)
