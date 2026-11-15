"""GPU adapter contract tests without CUDA or an electronic-structure run."""

from contextlib import nullcontext
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyscf_esd.cpu import ElectronicConvergenceError, Settings
from pyscf_esd.gpu import (
    GpuElectronicPoint,
    _dense_tda_retry,
    _fp64_host,
    _refine_hessian_response_block,
    _validated_tda,
    run_gpu_vibronic,
    solve_point,
)


class DeviceArray(np.ndarray):
    __cuda_array_interface__ = {}
    device = nullcontext()

    def get(self):
        return np.asarray(self)


def device_array(value):
    return np.asarray(value).view(DeviceArray)


class GpuBackendTests(unittest.TestCase):
    def test_rejects_unqualified_requests_before_building_a_molecule(self):
        with patch("pyscf_esd.gpu.build_molecule") as build:
            with self.assertRaisesRegex(ValueError, "requires PBE0"):
                solve_point(("O", "H", "H"), [], Settings("6-31+g*", "b3lyp"))
            with self.assertRaisesRegex(ValueError, "orbital seed"):
                solve_point(("O", "H", "H"), [], Settings("6-31+g*", "pbe0"), previous=Mock())
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                solve_point(("O", "H", "H"), [], Settings("6-31+g*", "pbe0"), hessian_grid_response=1)
            build.assert_not_called()

    def test_vg_uses_single_point_and_the_shared_analytic_preparation(self):
        settings = Settings("6-31+g*", "pbe0", nstates=2)
        symbols = ("O", "H", "H")
        coordinates = np.zeros((3, 3))
        with (
            patch("pyscf_esd.gpu.solve_point") as solve,
            patch("pyscf_esd.cpu_workflow._prepare_vg_center") as prepare,
        ):
            with self.assertRaisesRegex(ValueError, "higher reference root"):
                run_gpu_vibronic(symbols, coordinates, settings, target_index=1)
            solve.assert_not_called()
            result = run_gpu_vibronic(symbols, coordinates, settings, target_index=0)
            solve.assert_called_once_with(symbols, coordinates, settings, hessian_grid_response=False)
            self.assertEqual(prepare.call_args.args, (solve.return_value, 0))
            self.assertIs(result, prepare.return_value)

    def test_derivatives_keep_the_qualified_grid_and_state_settings(self):
        g0, g1, h0 = Mock(), Mock(), Mock()
        g0.kernel.return_value = np.zeros((3, 3))
        g1.kernel.return_value = device_array(np.zeros((3, 3)))
        h0.kernel.return_value = np.zeros((3, 3, 3, 3))
        mf = SimpleNamespace(
            mo_coeff=device_array(np.zeros((3, 3))), nuc_grad_method=lambda: g0, Hessian=lambda: h0,
        )
        settings = Settings("6-31+g*", "pbe0", cphf_max_cycle=37, cphf_conv_tol=1e-9)
        point = GpuElectronicPoint(
            mol=SimpleNamespace(natm=3), settings=settings, mo_coeff=np.zeros((3, 3)), nocc=1,
            amplitudes=np.zeros((2, 1, 2)), excitation_energies_hartree=np.array([0.2, 0.5]),
            transition_dipoles_au=np.zeros((2, 3)), ground_energy_hartree=-1.0,
            scf_residual=0.0, tda_residuals=np.zeros(2), attempts=(), _mean_field=mf,
            _tda=SimpleNamespace(Gradients=lambda: g1),
        )
        self.assertIs(type(point.ground_gradient()), np.ndarray)
        self.assertIs(type(point.excited_gradient(1)), np.ndarray)
        self.assertEqual(point.ground_hessian().shape, (3, 3, 3, 3))
        self.assertTrue(g0.grid_response)
        self.assertFalse(h0.grid_response)
        self.assertEqual(h0.max_cycle, 37)
        self.assertEqual(g1.cphf_max_cycle, 37)
        self.assertEqual(g1.cphf_conv_tol, 1e-9)
        g1.kernel.assert_called_once_with(state=2)

    def test_tda_recomputes_residual_with_the_device_response(self):
        cupy = SimpleNamespace(
            ndarray=DeviceArray, stack=lambda items: device_array(np.stack(items)),
            asarray=device_array, linalg=np.linalg,
        )
        operator = np.diag([0.2, 0.4, 0.6])
        response = Mock(side_effect=lambda vectors: vectors @ operator)
        td = SimpleNamespace(
            nstates=2, converged=device_array([True, True]), e=np.array([0.2, 0.4]),
            xy=[(x.reshape(1, 3) / np.sqrt(2), 0) for x in device_array(np.eye(3)[:2])],
            gen_vind=lambda: (response, None),
        )
        _, amplitudes, residuals = _validated_tda(td, cupy, 1, 3, 1e-8)
        self.assertIs(type(amplitudes), np.ndarray)
        np.testing.assert_allclose(residuals, 0.0)
        self.assertIsInstance(response.call_args.args[0], DeviceArray)
        td.e[0] = 0.1
        with self.assertRaisesRegex(ElectronicConvergenceError, "residual"):
            _validated_tda(td, cupy, 1, 3, 1e-8)
        with self.assertRaisesRegex(ElectronicConvergenceError, "FP64"):
            _fp64_host(np.ones(3, dtype=np.float32), (3,), "array")

    def test_dense_tda_recovery_keeps_the_same_device_operator_and_tda_format(self):
        def eigh(matrix):
            self.assertIsInstance(matrix, DeviceArray)
            return tuple(device_array(result) for result in np.linalg.eigh(matrix))

        cupy = SimpleNamespace(
            ndarray=DeviceArray, stack=lambda items: device_array(np.stack(items)),
            asarray=device_array, float64=np.float64, inf=np.inf, arange=np.arange, any=np.any,
            empty=lambda shape, dtype: device_array(np.empty(shape, dtype=dtype)),
            zeros=lambda shape, dtype: device_array(np.zeros(shape, dtype=dtype)),
            ones=lambda shape, dtype: device_array(np.ones(shape, dtype=dtype)),
            linalg=SimpleNamespace(norm=np.linalg.norm, eigh=eigh),
        )
        operator = np.diag(np.arange(1.0, 18.0))
        operator[0, 1] = operator[1, 0] = 0.25
        response = Mock(side_effect=lambda vectors: vectors @ operator)
        td = SimpleNamespace(nstates=2, converged=device_array([True, False]), gen_vind=lambda: (response, None))
        with self.assertRaisesRegex(ElectronicConvergenceError, "roots did not converge"):
            _validated_tda(td, cupy, 1, 17, 1e-9)
        _dense_tda_retry(td, cupy, 1, 17, 1e-9)
        roots, amplitudes, residuals = _validated_tda(td, cupy, 1, 17, 1e-9)
        np.testing.assert_allclose(roots, np.linalg.eigvalsh(operator)[:2])
        np.testing.assert_allclose(residuals, 0.0, atol=1e-9)
        vectors = amplitudes.reshape(2, 17)
        np.testing.assert_allclose(2 * vectors @ vectors.T, np.eye(2), atol=1e-14)
        self.assertIs(type(td.e), np.ndarray)
        self.assertTrue(all(isinstance(x, DeviceArray) and y == 0 for x, y in td.xy))
        self.assertTrue(np.all(td.converged))
        inputs = [call.args[0] for call in response.call_args_list]
        self.assertTrue(all(isinstance(array, DeviceArray) for array in inputs))
        np.testing.assert_array_equal(np.vstack(inputs[:2]), np.eye(17))

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the Krylov solver")
    def test_hessian_cphf_corrects_a_false_convergence_and_verifies_the_result(self):
        occupied = np.array([True, True, False, False])
        energy = np.array([-1.0, -0.4, 0.3, 0.8])
        gaps = energy[~occupied, None] - energy[occupied]
        orbitals = np.arange(24, dtype=float).reshape(3, 4, 2) / 100
        overlap = np.full_like(orbitals, 0.17)
        overlap[:, occupied] = -2 * orbitals[:, occupied]
        coupling = 0.08 * np.eye(8) + np.full((8, 8), 0.02)

        def response(values):
            return (values.reshape(-1, 8) @ coupling).reshape(values.shape)

        exact_potential = response(orbitals)
        fock = overlap * energy[occupied] - exact_potential
        fock[:, ~occupied] -= gaps * orbitals[:, ~occupied]
        erroneous = orbitals.copy()
        erroneous[:, ~occupied] += np.array([1.0, -0.4, 0.7])[:, None, None] * 1e-5
        args = (response, fock, overlap, energy, occupied, 3e-10, 50, 4000, None)

        corrected, first_order_energy = _refine_hessian_response_block(erroneous, *args)
        np.testing.assert_allclose(corrected, orbitals, atol=3e-10, rtol=0)
        expected = orbitals[:, occupied] * (energy[occupied, None] - energy[occupied])
        np.testing.assert_allclose(first_order_energy, expected, atol=3e-10, rtol=0)

        with patch("pyscf.lib.krylov", side_effect=lambda apply, rhs, **options: np.zeros_like(rhs)):
            with self.assertRaisesRegex(ElectronicConvergenceError, "CPHF residual .* exceeds tolerance"):
                _refine_hessian_response_block(erroneous, *args)
        with patch("pyscf.lib.krylov") as krylov:
            _refine_hessian_response_block(orbitals, *args)
            krylov.assert_not_called()


if __name__ == "__main__":
    unittest.main()
