"""Checked analytic gradients for individual states of PySCF SA-CASSCF."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .cpu import ElectronicConvergenceError


@dataclass(frozen=True)
class SACASSCFResponseAttempt:
    max_cycles: int
    preconditioner_shift: float
    native_converged: bool
    residual_norm: float
    residual_limit: float


@dataclass(frozen=True)
class SACASSCFGradient:
    gradient_hartree_per_bohr: NDArray[np.float64]
    attempts: tuple[SACASSCFResponseAttempt, ...]


class SACASSCFResponseError(ElectronicConvergenceError):
    """An analytic state-gradient response failed its numerical qualification."""

    def __init__(self, message: str, attempts: tuple[SACASSCFResponseAttempt, ...] = ()):
        super().__init__(message)
        self.attempts = attempts


def checked_sacasscf_gradient(
    method, root: int, *, absolute_tolerance: float = 1e-12, relative_tolerance: float = 1e-7,
) -> SACASSCFGradient:
    """Evaluate a root-specific PySCF SA-CASSCF gradient with a checked Z-vector.

    Set the tolerances PySCF's Lagrange solver consumes. Reapply the same
    response operator to the returned multipliers and require the SciPy CG
    residual criterion alongside native convergence. Retries change the
    iteration budget or preconditioner of that same linear problem.
    """
    if any(not isfinite(value) or value <= 0 for value in (absolute_tolerance, relative_tolerance)):
        raise ValueError("SA-CASSCF response tolerances must be finite and positive")
    weights = getattr(method, "weights", ())
    if len(weights) < 2 or not callable(getattr(method, "nuc_grad_method", None)):
        raise ValueError("supply a PySCF state-averaged CASSCF method with at least two states")
    if isinstance(root, (bool, np.bool_)) or not isinstance(root, Integral) or not 0 <= root < len(weights):
        raise ValueError("root must be a zero-based state-averaged root index")
    if not method.converged:
        raise SACASSCFResponseError("converge the SA-CASSCF orbital and CI solution before its gradient")

    attempts = []
    gradient = method.nuc_grad_method()
    if not callable(getattr(gradient, "solve_lagrange", None)):
        raise SACASSCFResponseError("the backend does not expose an auditable SA-CASSCF response operator")
    original_cycles, original_shift = gradient.max_cycle, gradient.level_shift
    larger_budget = max(135, 3 * original_cycles)
    routes = (
        (original_cycles, original_shift), (larger_budget, original_shift),
        (larger_budget, max(1e-4, 10 * original_shift)),
    )
    for index, (max_cycles, shift) in enumerate(routes):
        if index:
            gradient = method.nuc_grad_method()
        solve = getattr(gradient, "solve_lagrange", None)
        gradient.conv_atol = absolute_tolerance
        gradient.conv_rtol = relative_tolerance
        gradient.max_cycle = max_cycles
        gradient.level_shift = shift

        def audit_response(*args, **kwargs):
            result = solve(*args, **kwargs)
            converged, multipliers, rhs, operator, _ = result
            rhs_norm = float(np.linalg.norm(rhs))
            residual = float(np.linalg.norm(rhs + operator(multipliers)))
            limit = max(absolute_tolerance, relative_tolerance * rhs_norm) if isfinite(rhs_norm) else np.nan
            attempts.append(SACASSCFResponseAttempt(max_cycles, shift, bool(converged), residual, limit))
            return result

        gradient.solve_lagrange = audit_response
        previous_attempts = len(attempts)
        forces = np.asarray(gradient.kernel(state=int(root)))
        if len(attempts) != previous_attempts + 1:
            raise SACASSCFResponseError("the gradient did not provide exactly one native Lagrange solve", tuple(attempts))
        attempt = attempts[-1]
        if (gradient.converged and attempt.native_converged and attempt.residual_norm <= attempt.residual_limit
                and forces.shape == (method.mol.natm, 3) and np.isrealobj(forces) and np.isfinite(forces).all()):
            forces = forces.copy()
            forces.flags.writeable = False
            return SACASSCFGradient(forces, tuple(attempts))

    last = attempts[-1]
    raise SACASSCFResponseError(
        f"SA-CASSCF root {root + 1} analytic gradient failed at residual "
        f"{last.residual_norm:.3g} (limit {last.residual_limit:.3g}); inspect the orbital and CI response",
        tuple(attempts),
    )
