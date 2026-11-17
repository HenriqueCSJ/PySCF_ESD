"""Projected analytic response of an isolated real Hermitian TDA eigenpair.

A molecular backend supplies total derivatives of the TDA operator in a
common orthonormal orbital frame. This module solves the electronic response
and contracts those derivatives without building nuclear derivatives of the
full particle-hole matrix.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import inspect

import numpy as np
from numpy.typing import ArrayLike, NDArray


_Real = NDArray[np.float64]
_Operator = Callable[[_Real], ArrayLike]


def _real_array(value: ArrayLike, name: str) -> _Real:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class TDAProjectedSolution:
    """Response vectors and absolute projected equation residuals by RHS."""

    vectors: _Real
    residual_norms: _Real


@dataclass(frozen=True)
class ProjectedTDARoot:
    """An isolated unit-norm TDA root and reusable reduced resolvent."""

    amplitude: _Real
    omega_hartree: float
    root_residual: float
    separation_hartree: float
    _apply_shifted: _Operator = field(repr=False)
    _solve: _Operator = field(repr=False)
    _response_atol: float = field(repr=False)
    _response_rtol: float = field(repr=False)
    _solve_batch: Callable | None = field(default=None, repr=False)
    _response_batch_size: int = field(default=1, repr=False)

    def solve(self, rhs: ArrayLike) -> TDAProjectedSolution:
        """Solve P(ωI − A)P x = P b, with qᵀ x = 0, for each b in rhs.

        The input and output vectors have shape (n_rhs, *amplitude.shape).
        This solve also supports adjoint sources such as dipole operators.
        """
        right_hand_sides = _real_array(rhs, "response right-hand sides")
        if not right_hand_sides.ndim or right_hand_sides.shape[1:] != self.amplitude.shape:
            raise ValueError("response right-hand sides must have shape (n_rhs, *amplitude.shape)")
        q = self.amplitude.ravel()
        solutions = np.empty_like(right_hand_sides)
        residuals = np.empty(len(right_hand_sides))
        for start in range(0, len(right_hand_sides), self._response_batch_size):
            block = slice(start, start + self._response_batch_size)
            sources = right_hand_sides[block].reshape(-1, q.size)
            projected = _project_rows(sources, q)
            source_norms = np.linalg.norm(projected, axis=1)
            tolerances = self._response_atol + self._response_rtol * source_norms
            if self._solve_batch is not None:
                vectors, errors, accepted = self._solve_batch(projected, tolerances)
                solutions[block] = vectors.reshape(right_hand_sides[block].shape)
                residuals[block] = errors
                pending = np.flatnonzero(~accepted)
            else:
                pending = range(len(sources))
            for local in pending:
                source = projected[local]
                solution = np.zeros_like(q)
                if source_norms[local] != 0:
                    solution = _real_array(self._solve(source), "projected TDA solution")
                    solution -= q * np.dot(q, solution)
                error = self._apply_shifted(solution) - source
                error -= q * np.dot(q, error)
                residual = float(np.linalg.norm(error))
                tolerance = tolerances[local]
                index = start + local
                if not np.isfinite(residual) or residual > tolerance:
                    raise RuntimeError(
                        f"TDA response RHS {index} residual {residual:.3g} exceeds {tolerance:.3g}"
                    )
                solutions[index] = solution.reshape(self.amplitude.shape)
                residuals[index] = residual
        return TDAProjectedSolution(solutions, residuals)


def prepare_tda_root_response(
    operator: ArrayLike | _Operator,
    omega: float,
    amplitude: ArrayLike,
    *,
    diagonal: ArrayLike | None = None,
    root_separation_hartree: float | None = None,
    root_tolerance: float = 1e-8,
    root_gap_tolerance: float = 1e-5,
    response_atol: float = 1e-10,
    response_rtol: float = 1e-8,
    maxiter: int | None = None,
    operator_batch: _Operator | None = None,
    lowest_root: bool = False,
    response_batch_size: int = 8,
) -> ProjectedTDARoot:
    """Validate (ω, q) and prepare a reduced TDA resolvent at one geometry.

    Supply either a dense real symmetric A with flattened particle-hole
    indices or its action on one vector with the original amplitude shape.
    The amplitude is normalized to unit Euclidean norm, retaining its phase.
    A callable must implement a Hermitian action and supply the nearest-root
    separation from the same eigensolve. The optional diagonal of A provides
    a positive preconditioner for the matrix-free MINRES solve.
    If ``lowest_root`` is confirmed and ``operator_batch`` applies the same A
    to arrays of shape ``(n_rhs, *amplitude.shape)``, use independent CG
    recurrences with batched products. MINRES handles any unconverged rows.
    """
    q = _real_array(amplitude, "TDA amplitude")
    norm = np.linalg.norm(q)
    if not q.ndim or not np.isfinite(norm) or norm == 0:
        raise ValueError("TDA amplitude must be a nonzero array with finite norm")
    q = q / norm
    q.setflags(write=False)
    q_flat = q.ravel()
    omega = float(omega)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("target TDA excitation must be finite and positive")
    tolerances = (root_tolerance, root_gap_tolerance, response_atol, response_rtol)
    if any(not np.isfinite(tolerance) or tolerance <= 0 for tolerance in tolerances):
        raise ValueError("root and response tolerances must be finite and positive")
    if maxiter is not None and (not isinstance(maxiter, (int, np.integer)) or maxiter <= 0):
        raise ValueError("maxiter must be a positive integer")
    if operator_batch is not None and (not callable(operator_batch) or not callable(operator)):
        raise ValueError("operator_batch requires both a single-vector and a batched TDA operator callable")
    if not isinstance(lowest_root, (bool, np.bool_)):
        raise ValueError("lowest_root must be boolean")
    if (not isinstance(response_batch_size, (int, np.integer)) or isinstance(response_batch_size, (bool, np.bool_))
            or response_batch_size <= 0):
        raise ValueError("response_batch_size must be a positive integer")

    if callable(operator):
        def apply_shifted(vector):
            action = _real_array(operator(vector.reshape(q.shape)), "TDA operator action")
            if action.shape != q.shape:
                raise ValueError("TDA operator action must have the original amplitude shape")
            return omega * vector - action.ravel()
    else:
        matrix = _real_array(operator, "TDA matrix")
        if matrix.shape != (q.size, q.size):
            raise ValueError("TDA matrix must be square with one index per amplitude element")
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
            raise ValueError("dense TDA matrix must be symmetric")
        matrix = (matrix + matrix.T) * 0.5

        def apply_shifted(vector):
            return omega * vector - matrix @ vector

    root_residual = float(np.linalg.norm(apply_shifted(q_flat)))
    if not np.isfinite(root_residual) or root_residual > root_tolerance:
        raise ValueError(f"TDA eigenpair residual {root_residual:.3g} exceeds {root_tolerance:.3g}")

    if q.size == 1:
        separation = np.inf

        def solve(vector):
            return np.zeros_like(vector)
    elif callable(operator):
        if root_separation_hartree is None or not np.isfinite(root_separation_hartree):
            raise ValueError("matrix-free TDA response needs the finite nearest-root separation")
        separation = float(root_separation_hartree)
    else:
        complement = np.linalg.qr(q_flat[:, None], mode="complete")[0][:, 1:]
        eigenvalues, eigenvectors = np.linalg.eigh(complement.T @ matrix @ complement)
        denominators = omega - eigenvalues
        separation = float(np.min(np.abs(denominators)))
        basis = complement @ eigenvectors

        def solve(vector):
            return basis @ ((basis.T @ vector) / denominators)

    if separation < root_gap_tolerance:
        raise ValueError(f"target TDA root is not isolated (separation {separation:.3g} Hartree)")
    if root_residual >= separation * 0.1:
        raise ValueError("TDA eigenpair residual is too large relative to the root separation")
    solve_batch = None
    if callable(operator) and q.size > 1:
        inverse_diagonal = _response_inverse_diagonal(diagonal, q_flat, omega, root_gap_tolerance)
        solve = _prepare_minres(
            apply_shifted, q_flat, inverse_diagonal, response_atol, response_rtol, maxiter,
        )
        if lowest_root and operator_batch is not None:
            solve_batch = _prepare_batched_cg(operator_batch, q, omega, inverse_diagonal, maxiter)
    return ProjectedTDARoot(
        q, omega, root_residual, separation, apply_shifted, solve, response_atol, response_rtol,
        solve_batch, response_batch_size if solve_batch is not None else 1,
    )


def _project_rows(vectors, q):
    component = np.einsum("ij,j->i", vectors, q)
    return vectors - component[:, None] * q


def _response_inverse_diagonal(diagonal, q, omega, gap_tolerance):
    if diagonal is None:
        return None
    diagonal = _real_array(diagonal, "TDA operator diagonal").ravel()
    if diagonal.shape != q.shape:
        raise ValueError("TDA operator diagonal must have one value per amplitude element")
    return 1.0 / np.maximum(np.abs(omega - diagonal), gap_tolerance)


def _prepare_minres(apply_shifted, q, inverse_diagonal, atol, rtol, maxiter):
    from scipy.sparse.linalg import LinearOperator, minres

    def matvec(vector):
        component = np.dot(q, vector)
        response = apply_shifted(vector - q * component)
        # The rank-one shift makes the system invertible in the unused q direction.
        return response + q * (component - np.dot(q, response))

    shape = (q.size, q.size)
    shifted = LinearOperator(shape, matvec=matvec, dtype=float)
    preconditioner = None
    if inverse_diagonal is not None:
        preconditioner = LinearOperator(shape, matvec=lambda vector: inverse_diagonal * vector, dtype=float)

    # SciPy 1.10 and 1.11 use tol. Recent SciPy uses rtol for MINRES.
    tolerance_keyword = "rtol" if "rtol" in inspect.signature(minres).parameters else "tol"

    def solve(vector):
        threshold = max(np.finfo(float).eps, 0.1 * min(rtol, atol / np.linalg.norm(vector)))
        solution = np.zeros_like(vector)
        remainder = vector
        tolerance = atol + rtol * np.linalg.norm(vector)
        for _ in range(3):
            delta, info = minres(shifted, remainder, M=preconditioner, maxiter=maxiter,
                                 **{tolerance_keyword: threshold})
            if info != 0:
                raise RuntimeError(f"projected TDA MINRES failed to converge (status {info})")
            solution += delta
            # MINRES's preconditioned stopping test can pass before the
            # Euclidean residual used by the response diagnostics does.
            remainder = vector - matvec(solution)
            if np.linalg.norm(remainder) <= tolerance:
                break
        return solution

    return solve


def _prepare_batched_cg(operator, amplitude, omega, inverse_diagonal, maxiter):
    q = amplitude.ravel()
    iteration_limit = min(5 * q.size, 200) if maxiter is None else maxiter

    def apply_shifted(vectors):
        shaped = vectors.reshape(len(vectors), *amplitude.shape)
        applied = _real_array(operator(shaped), "batched TDA operator action")
        if applied.shape != shaped.shape:
            raise ValueError("batched TDA operator action must have shape (n_rhs, *amplitude.shape)")
        return applied.reshape(vectors.shape) - omega * vectors

    def precondition(residual):
        scaled = residual if inverse_diagonal is None else residual * inverse_diagonal
        return _project_rows(scaled, q)

    def row_inner(left, right):
        return np.einsum("ij,ij->i", left, right)

    def solve(sources, tolerances):
        # Work with the positive operator A − ω and the negative source.
        residual = -sources.copy()
        vectors = np.zeros_like(sources)
        directions = precondition(residual)
        products = row_inner(residual, directions)
        nonzero = np.any(sources != 0, axis=1)
        active = np.flatnonzero(nonzero)
        eligible = np.ones(len(residual), dtype=bool)
        for _ in range(iteration_limit):
            if not active.size:
                break
            step_directions = directions[active]
            applied = _project_rows(apply_shifted(step_directions), q)
            curvature = row_inner(step_directions, applied)
            current_products = products[active]
            positive = (curvature > 0) & np.isfinite(curvature) & (current_products > 0) & np.isfinite(current_products)
            eligible[active[~positive]] = False
            indices = active[positive]
            step = (products[indices] / curvature[positive])[:, None]
            vectors[indices] += step * step_directions[positive]
            residual[indices] -= step * applied[positive]

            # Verify against a fresh operator product below, allowing for
            # roundoff drift between the recurred and directly evaluated residual.
            unfinished = np.linalg.norm(residual[indices], axis=1) > 0.3 * tolerances[indices]
            active = indices[unfinished]
            conditioned = precondition(residual[active])
            next_products = row_inner(residual[active], conditioned)
            directions[active] = conditioned + (next_products / products[active])[:, None] * directions[active]
            products[active] = next_products

        vectors = _project_rows(vectors, q)
        errors = np.zeros(len(residual))
        checked = np.flatnonzero(nonzero & eligible)
        if checked.size:
            error = _project_rows(apply_shifted(vectors[checked]) + sources[checked], q)
            errors[checked] = np.linalg.norm(error, axis=1)
        accepted = eligible & np.isfinite(errors) & (errors <= tolerances)
        return vectors, errors, accepted

    return solve


@dataclass(frozen=True)
class TDARootResponse:
    """First nuclear derivatives of the validated unit-norm TDA eigenpair."""

    amplitude: _Real
    amplitude_derivatives: _Real
    excitation_gradient: _Real
    projected_first_action: _Real
    omega_hartree: float
    root_residual: float
    separation_hartree: float
    max_response_residual: float


def solve_tda_root_response(
    operator: ArrayLike | _Operator,
    omega: float,
    amplitude: ArrayLike,
    first_action: _Real | Iterable[ArrayLike],
    **response_options,
) -> TDARootResponse:
    """Solve for q_a and ω_a from A_a q in flattened Cartesian order.

    Supply (n_cart, *amplitude.shape) or an iterator of chunks with those
    trailing dimensions. Each row must be the derivative action on the same
    unit-norm q used in the response, including its phase.
    """
    root = prepare_tda_root_response(operator, omega, amplitude, **response_options)
    if isinstance(first_action, np.ndarray):
        first_action = (first_action,)
    gradients, projected_actions, responses = [], [], []
    max_residual = 0.0
    q = root.amplitude.ravel()
    for chunk in first_action:
        actions = _real_array(chunk, "first TDA derivative actions")
        solutions = root.solve(actions)
        flat_actions = actions.reshape(len(actions), q.size)
        gradient = flat_actions @ q
        gradients.append(gradient)
        projected_actions.append(flat_actions - gradient[:, None] * q[None, :])
        responses.append(solutions.vectors)
        max_residual = max(max_residual, float(np.max(solutions.residual_norms, initial=0.0)))
    if not gradients or not sum(map(len, gradients)):
        raise ValueError("at least one first TDA derivative action is required")
    return TDARootResponse(
        root.amplitude, np.concatenate(responses), np.concatenate(gradients), np.concatenate(projected_actions),
        root.omega_hartree, root.root_residual, root.separation_hartree, max_residual,
    )


def assemble_tda_hessian(
    ground_hessian: ArrayLike,
    second_contraction: ArrayLike,
    response: TDARootResponse,
    *,
    symmetry_tolerance: float = 1e-8,
) -> _Real:
    """Return E₀,ab + qᵀ A_ab q + 2 (A_a q)ᵀ q_b in Hartree/bohr².

    Both input matrices have shape (n_cart, n_cart). The ground-state Hessian
    and both derivatives of A must use the same geometry and Hamiltonian.
    The reduced resolvent makes the response contraction symmetric. Check
    the three contributions separately before averaging numerical asymmetry.
    """
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0:
        raise ValueError("symmetry_tolerance must be finite and positive")
    n_cart = response.excitation_gradient.size
    derivative = response.amplitude_derivatives.reshape(n_cart, -1)
    relaxation = 2.0 * (response.projected_first_action @ derivative.T)
    components = (
        ("ground Hessian", ground_hessian),
        ("second TDA derivative contraction", second_contraction),
        ("TDA eigenvector response contraction", relaxation),
    )
    result = np.zeros((n_cart, n_cart))
    for name, value in components:
        contribution = _real_array(value, name)
        if contribution.shape != result.shape:
            raise ValueError(f"{name} must have shape (n_cart, n_cart)")
        asymmetry = float(np.max(np.abs(contribution - contribution.T)))
        if asymmetry > symmetry_tolerance:
            raise ValueError(f"{name} asymmetry {asymmetry:.3g} exceeds {symmetry_tolerance:.3g}")
        result += 0.5 * (contribution + contribution.T)
    return _real_array(result, "total TDA Hessian")
