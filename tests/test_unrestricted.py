"""Prepared UKS endpoint contracts without PySCF runs or CUDA."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyscf_esd.cpu import ElectronicConvergenceError
from pyscf_esd.unrestricted import analytic_uks_triplet_endpoint


class DeviceArray(np.ndarray):
    __cuda_array_interface__ = {}
    device = nullcontext()

    def get(self):
        return np.asarray(self)


def _derivative(value, module):
    derivative = type("Derivative", (), {"__module__": module})()
    derivative.grid_response = None
    derivative.auxbasis_response = None
    derivative.kernel = Mock(return_value=value)
    return derivative


def _reference(backend="pyscf", *, density_fitting=False):
    mean_field = type("UKS", (), {"__module__": f"{backend}.dft.uks"})()
    mean_field.mol = SimpleNamespace(spin=2, nelectron=4, nao_nr=lambda: 4, natm=2, ecp={}, pseudo={})
    mean_field.converged = True
    mean_field.xc = "PBE0"
    mean_field._numint = SimpleNamespace(_xc_type=Mock(return_value="GGA"))
    mean_field.do_nlc = Mock(return_value=False)
    mean_field.do_disp = Mock(return_value=False)
    mean_field.with_df = object() if density_fitting else None
    mean_field.mo_coeff = np.stack((np.eye(4), np.eye(4)))
    mean_field.mo_occ = np.array([[1., 1., 0., 1.], [1., 0., 0., 0.]])
    mean_field.mo_energy = np.array([[-1., -.8, .2, .3], [-.9, .1, .3, .4]])
    mean_field.e_tot = -10.0
    mean_field.get_grad = Mock(return_value=np.array([4e-8, 3e-8, 0., 0., 0., 0.]))
    mean_field.spin_square = Mock(return_value=(2.2, np.sqrt(9.8)))
    mean_field.kernel = Mock(side_effect=AssertionError("SCF would start a new electronic solve"))
    module = f"{backend}.df" if density_fitting else backend
    gradient = np.arange(6.).reshape(2, 3)
    hessian = np.arange(36.).reshape(2, 2, 3, 3)
    mean_field.nuc_grad_method = Mock(return_value=_derivative(gradient, f"{module}.grad.uks"))
    mean_field.Hessian = Mock(return_value=_derivative(hessian, f"{module}.hessian.uks"))
    return mean_field


class UKSTripletEndpointTests(unittest.TestCase):
    def test_cpu_native_energy_and_derivatives(self):
        mean_field = _reference()
        result = analytic_uks_triplet_endpoint(mean_field, include_hessian=True)
        self.assertEqual(result.energy_hartree, -10.0)
        self.assertEqual(result.electron_counts, (3, 1))
        self.assertAlmostEqual(result.scf_residual_hartree, 5e-8, places=15)
        self.assertAlmostEqual(result.spin_squared, 2.2)
        self.assertAlmostEqual(result.effective_multiplicity, np.sqrt(9.8))
        self.assertAlmostEqual(result.spin_contamination, 0.2)
        self.assertEqual(result.backend, "pyscf")
        self.assertFalse(result.density_fitting)
        self.assertTrue(result.gradient_grid_response)
        self.assertFalse(result.hessian_grid_response)
        self.assertTrue(mean_field.nuc_grad_method.return_value.grid_response)
        self.assertFalse(mean_field.Hessian.return_value.grid_response)
        np.testing.assert_array_equal(result.gradient_hartree_per_bohr, np.arange(6.).reshape(2, 3))
        np.testing.assert_array_equal(result.hessian_hartree_per_bohr2, np.arange(36.).reshape(2, 2, 3, 3))
        self.assertFalse(result.gradient_hartree_per_bohr.flags.writeable)
        self.assertFalse(result.hessian_hartree_per_bohr2.flags.writeable)
        mean_field.kernel.assert_not_called()
        mean_field.get_grad.assert_called_once()
        self.assertIs(mean_field.get_grad.call_args.args[0], mean_field.mo_coeff)
        self.assertIs(mean_field.get_grad.call_args.args[1], mean_field.mo_occ)

    def test_gradient_only_and_fixed_quadrature_option(self):
        mean_field = _reference()
        result = analytic_uks_triplet_endpoint(mean_field, gradient_grid_response=False)
        self.assertFalse(result.gradient_grid_response)
        self.assertFalse(mean_field.nuc_grad_method.return_value.grid_response)
        self.assertIsNone(result.hessian_hartree_per_bohr2)
        self.assertIsNone(result.hessian_grid_response)
        mean_field.Hessian.assert_not_called()

    def test_gpu_native_density_fitted_derivatives(self):
        mean_field = _reference("gpu4pyscf", density_fitting=True)
        mean_field.mo_coeff = mean_field.mo_coeff.view(DeviceArray)
        for engine in (mean_field.nuc_grad_method.return_value, mean_field.Hessian.return_value):
            engine.kernel.return_value = engine.kernel.return_value.view(DeviceArray)
        cupy = SimpleNamespace(ndarray=DeviceArray, asarray=np.asarray, all=np.all, isfinite=np.isfinite)
        with patch.dict("sys.modules", {"cupy": cupy}):
            result = analytic_uks_triplet_endpoint(mean_field, include_hessian=True)
        self.assertEqual(result.backend, "gpu4pyscf")
        self.assertTrue(result.density_fitting)
        self.assertIs(type(result.gradient_hartree_per_bohr), np.ndarray)
        self.assertIs(type(result.hessian_hartree_per_bohr2), np.ndarray)
        self.assertIs(mean_field.nuc_grad_method.return_value.auxbasis_response, True)
        self.assertEqual(mean_field.Hessian.return_value.auxbasis_response, 2)
        mean_field.kernel.assert_not_called()

    def test_rejects_wrong_reference_and_spin(self):
        with self.assertRaisesRegex(TypeError, "must be a molecular.*UKS"):
            analytic_uks_triplet_endpoint(SimpleNamespace(xc="PBE0"))
        mean_field = _reference()
        mean_field.mol.spin = 0
        with self.assertRaisesRegex(ValueError, "mol.spin = 2"):
            analytic_uks_triplet_endpoint(mean_field)
        mean_field.nuc_grad_method.assert_not_called()

    def test_rejects_unconverged_or_nonstationary_reference(self):
        mean_field = _reference()
        mean_field.converged = False
        with self.assertRaisesRegex(ElectronicConvergenceError, "already be converged"):
            analytic_uks_triplet_endpoint(mean_field)
        mean_field.converged = True
        with self.assertRaisesRegex(ElectronicConvergenceError, "orbital residual"):
            analytic_uks_triplet_endpoint(mean_field, scf_residual_tolerance=1e-8)
        mean_field.nuc_grad_method.assert_not_called()

    def test_rejects_fractional_occupations_and_hf_only_xc(self):
        mean_field = _reference()
        mean_field.mo_occ[0, 0] = 0.5
        with self.assertRaisesRegex(ElectronicConvergenceError, "occupations must be 0 or 1"):
            analytic_uks_triplet_endpoint(mean_field)
        mean_field._numint._xc_type.return_value = "HF"
        with self.assertRaisesRegex(NotImplementedError, "requires LDA or GGA XC"):
            analytic_uks_triplet_endpoint(mean_field)
        mean_field.nuc_grad_method.assert_not_called()

    def test_rejects_missing_or_malformed_native_derivatives(self):
        mean_field = _reference()
        mean_field.nuc_grad_method.return_value = _derivative(np.zeros((2, 3)), "example.grad")
        with self.assertRaisesRegex(NotImplementedError, "native pyscf UKS gradient is unavailable"):
            analytic_uks_triplet_endpoint(mean_field)
        mean_field.nuc_grad_method.return_value.kernel.assert_not_called()
        mean_field.nuc_grad_method.return_value = _derivative(np.full((2, 3), np.nan), "pyscf.grad.uks")
        with self.assertRaisesRegex(ElectronicConvergenceError, "nuclear gradient contains nonfinite data"):
            analytic_uks_triplet_endpoint(mean_field)
