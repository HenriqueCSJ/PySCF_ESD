"""Exercise analytic eigenpair derivatives without nuclear displacements."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.tda_hessian import (
    assemble_tda_hessian,
    prepare_tda_root_response,
    solve_tda_root_response,
)


class TDARootResponseTests(unittest.TestCase):
    def test_curvature_matches_closed_form_two_level_energy_in_rotated_frame(self):
        # The lower eigenvalue of A(R) is (a+d-sqrt((a-d)^2+4b^2))/2.
        # At b=0 and d-a=0.4: ω'=a' and ω''=a''-2 b' b'^T/(d-a).
        basis = np.array([[3.0, -4.0], [4.0, 3.0]]) / 5.0
        values = np.array([0.3, 0.7])
        omega = values[0]
        a_first = np.array([0.05, -0.08])
        b_first = np.array([0.12, 0.21])
        a_second = np.array([[0.4, -0.1], [-0.1, 0.6]])
        ground = np.array([[0.31, 0.04], [0.04, 0.7]])
        first = np.outer(a_first, basis[:, 0]) + np.outer(b_first, basis[:, 1])
        matrix = (basis * values) @ basis.T

        # Native PySCF amplitudes may have norm 1/sqrt(2). A_a q uses unit q.
        q = basis[:, 0] / np.sqrt(2)
        response = solve_tda_root_response(matrix, omega, q, (chunk for chunk in (first[:1], first[1:])))
        np.testing.assert_allclose(response.amplitude, basis[:, 0], atol=1e-15)
        np.testing.assert_allclose(response.excitation_gradient, a_first, atol=1e-15)
        np.testing.assert_allclose(response.amplitude_derivatives, -np.outer(b_first, basis[:, 1]) / 0.4)
        np.testing.assert_allclose(response.amplitude_derivatives @ response.amplitude, 0, atol=1e-15)
        np.testing.assert_allclose(
            assemble_tda_hessian(ground, a_second, response),
            ground + a_second - 2 * np.outer(b_first, b_first) / 0.4,
            atol=1e-14,
        )
        self.assertAlmostEqual(response.separation_hartree, 0.4)
        self.assertLess(response.root_residual, 1e-14)
        self.assertLess(response.max_response_residual, 1e-14)

        flipped = solve_tda_root_response(matrix, omega, -q, -first)
        np.testing.assert_allclose(flipped.amplitude_derivatives, -response.amplitude_derivatives)
        np.testing.assert_allclose(flipped.excitation_gradient, response.excitation_gradient)
        np.testing.assert_allclose(
            assemble_tda_hessian(ground, a_second, flipped),
            assemble_tda_hessian(ground, a_second, response),
        )

    @unittest.skipUnless(find_spec("scipy"), "SciPy is required for matrix-free TDA response")
    def test_iterative_resolvent_matches_dense_with_lower_and_upper_roots(self):
        rng = np.random.default_rng(13)
        basis = np.linalg.qr(rng.normal(size=(20, 20)))[0]
        values = np.linspace(0.15, 1.5, 20)
        matrix = (basis * values) @ basis.T
        root = 8
        amplitude = basis[:, root].reshape(4, 5)
        omega = values[root]
        nearest_gap = values[root] - values[root - 1]
        rhs = rng.normal(size=(3, 4, 5))

        def operator(vector):
            self.assertEqual(vector.shape, amplitude.shape)
            return (matrix @ vector.ravel()).reshape(amplitude.shape)

        iterative = prepare_tda_root_response(
            operator, omega, amplitude, diagonal=np.diag(matrix), root_separation_hartree=nearest_gap,
            response_atol=1e-12, response_rtol=1e-10,
        ).solve(rhs)
        dense = prepare_tda_root_response(matrix, omega, amplitude).solve(rhs)
        np.testing.assert_allclose(iterative.vectors, dense.vectors, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(iterative.vectors.reshape(3, -1) @ amplitude.ravel(), 0, atol=1e-13)
        contraction = rhs.reshape(3, -1) @ iterative.vectors.reshape(3, -1).T
        np.testing.assert_allclose(contraction, contraction.T, atol=1e-10, rtol=0)
        self.assertLess(float(np.max(iterative.residual_norms)), 1e-10)

        with self.assertRaisesRegex(RuntimeError, "MINRES failed to converge"):
            prepare_tda_root_response(
                operator, omega, amplitude, root_separation_hartree=nearest_gap, maxiter=1,
            ).solve(rhs)

    @unittest.skipUnless(find_spec("scipy"), "SciPy is required for matrix-free TDA response")
    def test_batched_lowest_root_resolvent_matches_dense_and_checks_residuals(self):
        rng = np.random.default_rng(215)
        shape = (3, 5)
        size = np.prod(shape)
        basis = np.linalg.qr(rng.normal(size=(size, size)))[0]
        values = np.linspace(0.2, 1.7, size)
        matrix = (basis * values) @ basis.T
        amplitude = basis[:, 0].reshape(shape)
        rhs = rng.normal(size=(7, *shape))
        rhs[-1] = 0
        calls = []

        def single(vector):
            calls.append("scalar")
            return (matrix @ vector.ravel()).reshape(shape)

        def batch(vectors):
            self.assertEqual(vectors.shape[1:], shape)
            calls.append(len(vectors))
            return (vectors.reshape(len(vectors), size) @ matrix.T).reshape(vectors.shape)

        atol, rtol = 1e-12, 1e-11
        batch_result = solve_tda_root_response(
            single, values[0], amplitude, rhs, diagonal=np.diag(matrix),
            root_separation_hartree=values[1] - values[0], operator_batch=batch, lowest_root=True,
            response_batch_size=3, response_atol=atol, response_rtol=rtol,
        )
        expected = solve_tda_root_response(matrix, values[0], amplitude, rhs)
        np.testing.assert_allclose(batch_result.amplitude_derivatives, expected.amplitude_derivatives, atol=1e-10)
        np.testing.assert_allclose(batch_result.excitation_gradient, expected.excitation_gradient, atol=1e-14)
        flat = batch_result.amplitude_derivatives.reshape(len(rhs), size)
        q = batch_result.amplitude.ravel()
        np.testing.assert_allclose(flat @ q, 0, atol=1e-14)
        error = flat @ (values[0] * np.eye(size) - matrix) - batch_result.projected_first_action
        direct = np.linalg.norm(error - (error @ q)[:, None] * q, axis=1)
        np.testing.assert_allclose(batch_result.max_response_residual, np.max(direct), atol=1e-14)
        self.assertTrue(np.all(direct <= atol + rtol * np.linalg.norm(batch_result.projected_first_action, axis=1)))
        self.assertEqual(calls.count("scalar"), 1)  # Initial eigenpair check only.
        self.assertEqual(max(calls[1:]), 3)

    @unittest.skipUnless(find_spec("scipy"), "SciPy is required for matrix-free TDA response")
    def test_batched_nonpositive_curvature_falls_back_per_row(self):
        diagonal = np.array([0.12, 0.41, 0.63])
        omega = diagonal[1]
        amplitude = np.array([0.0, 1.0, 0.0])
        rhs = np.array([[0.3, 0.0, 0.0], [0.0, 0.0, -0.2], [0.0, 0.8, 0.0]])
        calls = []

        def single(vector):
            calls.append("scalar")
            return vector * diagonal

        def batch(vectors):
            calls.append(len(vectors))
            return vectors * diagonal

        kwargs = {"root_separation_hartree": diagonal[2] - omega, "operator_batch": batch}
        expected = prepare_tda_root_response(np.diag(diagonal), omega, amplitude).solve(rhs)
        # A deliberately incorrect lowest-root assertion exercises curvature detection.
        trial = prepare_tda_root_response(single, omega, amplitude, lowest_root=True, **kwargs).solve(rhs)
        np.testing.assert_allclose(trial.vectors, expected.vectors, atol=1e-15)
        self.assertEqual(calls.count(2), 1)
        self.assertEqual(calls.count(1), 1)
        self.assertGreater(calls.count("scalar"), 1)

        calls.clear()
        nonlowest = prepare_tda_root_response(single, omega, amplitude, **kwargs).solve(rhs)
        np.testing.assert_allclose(nonlowest.vectors, expected.vectors, atol=1e-15)
        self.assertTrue(all(call == "scalar" for call in calls))

    @unittest.skipUnless(find_spec("scipy"), "SciPy is required for matrix-free TDA response")
    def test_unconverged_batched_rows_use_checked_scalar_fallback(self):
        diagonal = np.array([0.12, 0.41, 0.63])
        amplitude = np.array([1.0, 0.0, 0.0])
        root = prepare_tda_root_response(
            lambda vector: vector * diagonal, diagonal[0], amplitude, lowest_root=True,
            operator_batch=lambda vectors: vectors * diagonal,
            root_separation_hartree=diagonal[1] - diagonal[0], maxiter=1,
        )
        with self.assertRaisesRegex(RuntimeError, "MINRES failed to converge"):
            root.solve([[0.0, 0.2, 0.3]])

    def test_rejects_wrong_root_near_degeneracy_and_asymmetric_curvature(self):
        with self.assertRaisesRegex(ValueError, "eigenpair residual"):
            prepare_tda_root_response(np.diag([0.1, 0.5]), 0.1, [0, 1])
        with self.assertRaisesRegex(ValueError, "not isolated"):
            prepare_tda_root_response(np.diag([0.1, 0.1 + 1e-6]), 0.1, [1, 0])
        with self.assertRaisesRegex(ValueError, "must be symmetric"):
            prepare_tda_root_response([[0.1, 0.02], [0, 0.5]], 0.1, [1, 0])
        with self.assertRaisesRegex(ValueError, "nearest-root separation"):
            prepare_tda_root_response(lambda vector: np.diag([0.1, 0.5]) @ vector, 0.1, [1, 0])

        response = solve_tda_root_response([[0.1]], 0.1, [1], np.array([[0.15], [-0.15]]))
        np.testing.assert_array_equal(response.amplitude_derivatives, np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "second TDA derivative contraction asymmetry"):
            assemble_tda_hessian(np.eye(2), [[1, 0.02], [0, 1]], response)


if __name__ == "__main__":
    unittest.main()
