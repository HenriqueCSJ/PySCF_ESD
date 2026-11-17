"""Check the actual SA-CASSCF response equation, including analytic retries."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd.casscf_response import SACASSCFResponseError, checked_sacasscf_gradient


def _mock_method(multipliers, native_flags):
    outcomes = iter(zip(multipliers, native_flags))
    configured = []

    class Gradient:
        max_cycle = 45
        level_shift = 1e-8

        def solve_lagrange(self, state):
            configured.append((state, self.conv_atol, self.conv_rtol, self.max_cycle, self.level_shift))
            value, converged = next(outcomes)
            return converged, np.array([value]), np.ones(1), lambda vector: vector, np.ones(1)

        def kernel(self, state):
            self.converged = self.solve_lagrange(state)[0]
            return np.array([[0., 0., 0.1], [0., 0., -0.1]])

    return SimpleNamespace(weights=(0.5, 0.5), mol=SimpleNamespace(natm=2), converged=True,
                           nuc_grad_method=Gradient), configured


class SACASSCFResponseTests(unittest.TestCase):
    def test_retries_a_native_success_that_fails_the_measured_residual(self):
        method, configured = _mock_method([-0.999, -1.], [True, True])
        result = checked_sacasscf_gradient(method, 1)
        self.assertEqual(configured, [(1, 1e-12, 1e-7, 45, 1e-8), (1, 1e-12, 1e-7, 135, 1e-8)])
        self.assertGreater(result.attempts[0].residual_norm, result.attempts[0].residual_limit)
        self.assertEqual(result.attempts[1].residual_norm, 0.)
        self.assertFalse(result.gradient_hartree_per_bohr.flags.writeable)

    def test_exhausted_responses_keep_the_original_residual_limit(self):
        method, configured = _mock_method([-0.999, -0.999, -0.999], [False, False, True])
        with self.assertRaises(SACASSCFResponseError) as error:
            checked_sacasscf_gradient(method, 0, relative_tolerance=1e-9)
        self.assertEqual([item.residual_limit for item in error.exception.attempts], [1e-9] * 3)
        self.assertEqual(configured[-1], (0, 1e-12, 1e-9, 135, 1e-4))

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the native response")
    def test_native_two_singlet_gradient(self):
        from pyscf import gto, mcscf, scf

        molecule = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr", verbose=0)
        reference = scf.RHF(molecule).run(conv_tol=1e-12)
        method = mcscf.CASSCF(reference, 2, 2).fix_spin_(shift=.5, ss=0).state_average_([.5, .5])
        method.kernel()
        checked = checked_sacasscf_gradient(method, 1)
        native = method.nuc_grad_method().kernel(state=1)
        np.testing.assert_allclose(checked.gradient_hartree_per_bohr, native, atol=1e-10)
        self.assertLessEqual(checked.attempts[-1].residual_norm, checked.attempts[-1].residual_limit)
