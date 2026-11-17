"""Same-geometry orbital derivatives for a real closed-shell TDA Hessian.

The occupied and virtual blocks of the orbital coefficient transform relative
to the reference coefficients are symmetric. The occupied–virtual rotation is
set by RKS stationarity. An adjoint eliminates its mixed nuclear derivative
from the fixed-amplitude excitation curvature.
"""

from collections.abc import Callable
from dataclasses import dataclass
import inspect

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tda_hessian import _real_array


_Real = NDArray[np.float64]


def _matrix(value, shape, name):
    matrix = _real_array(value, name)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return matrix


def _inner(left, right):
    return np.einsum("ij,ij->", left, right)


@dataclass(frozen=True)
class OrbitalDerivative:
    """First derivatives in the smooth orbital frame at the reference point."""

    coefficients: _Real
    density: _Real
    fock: _Real
    fock_mo: _Real
    transform: _Real
    fock_in_reference_frame: _Real
    residual_norm: float


@dataclass(frozen=True)
class OrbitalCurvature:
    """Mixed derivatives with mixed occupied–virtual rotation set to zero."""

    coefficients: _Real
    density: _Real
    fock: _Real
    fock_mo: _Real
    stationarity_source: _Real


class OrbitalResponse:
    """Ground-state orbital response and curvature in a co-moving MO frame.

    ``apply_response`` maps a symmetric AO density perturbation to its AO
    Fock response at fixed geometry, with the same functional, DF and solvent
    as the reference. The adjoint uses self-adjointness of this density kernel.
    """

    def __init__(
        self, coefficients: ArrayLike, nocc: int, fock_ao: ArrayLike,
        apply_response: Callable[[_Real], ArrayLike], *, atol: float = 1e-10,
        rtol: float = 1e-9, max_cycle: int = 100, stationarity_tolerance: float = 1e-6,
    ):
        c = _real_array(coefficients, "reference coefficients")
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError("orbital response requires a complete square real MO basis")
        if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < c.shape[1]:
            raise ValueError("nocc must specify nonempty occupied and virtual spaces")
        if any(not np.isfinite(tol) or tol <= 0 for tol in (atol, rtol, stationarity_tolerance)):
            raise ValueError("response tolerances must be finite and positive")
        if not isinstance(max_cycle, (int, np.integer)) or max_cycle <= 0:
            raise ValueError("max_cycle must be a positive integer")
        self.coefficients = c
        self.nocc = nocc
        self.occupied, self.virtual = c[:, :nocc], c[:, nocc:]
        self.fock = _matrix(fock_ao, c.shape, "reference Fock matrix")
        self.fock_mo = c.T @ self.fock @ c
        self._response = apply_response
        self._atol, self._rtol, self._max_cycle = atol, rtol, max_cycle
        error = np.linalg.norm(self.fock_mo[nocc:, :nocc])
        if error > stationarity_tolerance:
            raise ValueError(f"RKS occupied–virtual Fock residual {error:.3g} exceeds {stationarity_tolerance:.3g}")
        diagonal = self.fock_mo.diagonal()
        self._gap = diagonal[nocc:, None] - diagonal[None, :nocc]

    def _density_derivative(self, occupied_coefficient_derivative):
        change = occupied_coefficient_derivative @ self.occupied.T
        return 2 * (change + change.T)

    def _density_response(self, density):
        return _matrix(self._response(density), self.fock.shape, "AO Fock response")

    def _stationarity_action(self, vo):
        nocc = self.nocc
        potential = self._density_response(self._density_derivative(self.virtual @ vo))
        return (self.fock_mo[nocc:, nocc:] @ vo - vo @ self.fock_mo[:nocc, :nocc]
                + self.virtual.T @ potential @ self.occupied)

    def solve(self, rhs: ArrayLike) -> tuple[_Real, float]:
        """Solve the linearized RKS stationarity equation for a vo source."""
        from scipy.sparse.linalg import LinearOperator, gmres

        source = _matrix(rhs, self._gap.shape, "orbital response source")
        norm = np.linalg.norm(source)
        if norm == 0:
            return np.zeros_like(source), 0.0
        size = source.size

        def matvec(vector):
            return self._stationarity_action(vector.reshape(source.shape)).ravel()

        matrix = LinearOperator((size, size), matvec=matvec, dtype=float)
        scales = np.maximum(np.abs(self._gap.ravel()), 1e-6)
        preconditioner = LinearOperator((size, size), matvec=lambda vector: vector / scales, dtype=float)
        # SciPy 1.10 and 1.11 use tol for the relative convergence tolerance.
        tolerance_keyword = "rtol" if "rtol" in inspect.signature(gmres).parameters else "tol"
        vector, info = gmres(
            matrix, source.ravel(), M=preconditioner, restart=min(40, size), maxiter=self._max_cycle,
            callback=lambda _: None, callback_type="legacy", atol=0.1 * self._atol,
            **{tolerance_keyword: 0.1 * self._rtol},
        )
        residual = float(np.linalg.norm(matvec(vector) - source.ravel()))
        tolerance = self._atol + self._rtol * norm
        if info < 0 or not np.isfinite(residual) or residual > tolerance:
            raise RuntimeError(f"RKS orbital response residual {residual:.3g} exceeds {tolerance:.3g} (GMRES {info})")
        return vector.reshape(source.shape), residual

    def first(self, overlap_x: ArrayLike, fixed_fock_x: ArrayLike, *, occupied_response=None) -> OrbitalDerivative:
        """Derive C, P and F for a fixed-AO-density nuclear Fock derivative."""
        c = self.coefficients
        nocc = self.nocc
        s_x = c.T @ _matrix(overlap_x, c.shape, "overlap derivative") @ c
        fixed_fock_x = _matrix(fixed_fock_x, c.shape, "fixed-density Fock derivative")
        u = -0.5 * s_x
        metric_density = self._density_derivative(c @ u[:, :nocc])
        fock_metric = fixed_fock_x + self._density_response(metric_density)
        fock_mo_metric = c.T @ fock_metric @ c + u.T @ self.fock_mo + self.fock_mo @ u
        rhs = -fock_mo_metric[nocc:, :nocc]
        if occupied_response is None:
            rotation, _ = self.solve(rhs)
        else:
            # Reuse the occupied response from the native ground Hessian.
            # Correct any remaining residual under the full moving Fock.
            guess = _matrix(occupied_response, (c.shape[1], nocc), "occupied orbital response")
            rotation = guess[nocc:] - u[nocc:, :nocc]
            defect = rhs - self._stationarity_action(rotation)
            if np.linalg.norm(defect) > self._atol + self._rtol * np.linalg.norm(rhs):
                correction, _ = self.solve(defect)
                rotation += correction
        u[nocc:, :nocc] += rotation
        u[:nocc, nocc:] -= rotation.T
        c_x = c @ u
        density = self._density_derivative(c_x[:, :nocc])
        fock = fock_metric + self._density_response(self._density_derivative(self.virtual @ rotation))
        fock_reference = c.T @ fock @ c
        fock_mo = fock_reference + u.T @ self.fock_mo + self.fock_mo @ u
        error = float(np.linalg.norm(fock_mo[nocc:, :nocc]))
        tolerance = self._atol + self._rtol * np.linalg.norm(fock_mo_metric[nocc:, :nocc])
        if error > tolerance:
            raise RuntimeError(f"first-order RKS stationarity residual {error:.3g} exceeds {tolerance:.3g}")
        return OrbitalDerivative(c_x, density, fock, fock_mo, u, fock_reference, error)

    def mixed(
        self, overlap_xy: ArrayLike, first_x: OrbitalDerivative, first_y: OrbitalDerivative,
        fixed_fock_xy: ArrayLike | None = None, *, include_density_response: bool = True,
    ) -> OrbitalCurvature:
        """Differentiate RKS stationarity before the mixed vo rotation.

        ``fixed_fock_xy`` contains the explicit nuclear mixed derivative,
        both explicit nuclear derivatives of the kernel acting on the other
        first-order density, and the density curvature of the Fock operator
        acting on both first-order densities. This method adds the remaining
        linear Fock response to the mixed density in the chosen gauge.
        When contracting a relaxed AO density separately, omit this input
        and set ``include_density_response=False``.
        """
        c = self.coefficients
        sxy = c.T @ _matrix(overlap_xy, c.shape, "mixed overlap derivative") @ c
        ux, uy = first_x.transform, first_y.transform
        # Eliminate S_x and S_y using U_x + U_x.T = -S_x (and likewise for y).
        products = ux.T @ uy + ux @ uy + uy @ ux
        uxy = -0.5 * (sxy - products - products.T)
        cxy = c @ uxy
        ox, oy = first_x.coefficients[:, :self.nocc], first_y.coefficients[:, :self.nocc]
        cross_density = 2 * (ox @ oy.T + oy @ ox.T)
        density = self._density_derivative(cxy[:, :self.nocc]) + cross_density
        fock = np.zeros_like(self.fock)
        if fixed_fock_xy is not None:
            fock += _matrix(fixed_fock_xy, c.shape, "mixed Fock derivative source")
        if include_density_response:
            fock += self._density_response(density)
        geometric = (uxy.T @ self.fock_mo + ux.T @ self.fock_mo @ uy
                     + ux.T @ first_y.fock_in_reference_frame + uy.T @ first_x.fock_in_reference_frame)
        fock_mo = geometric + geometric.T
        if fixed_fock_xy is not None or include_density_response:
            fock_mo += c.T @ fock @ c
        return OrbitalCurvature(cxy, density, fock, fock_mo, fock_mo[self.nocc:, :self.nocc])


def tda_transition_density(coefficients: ArrayLike, amplitude: ArrayLike) -> _Real:
    """Return the unsymmetrized AO density C_v q.T C_o.T for unit-norm q."""
    q = _real_array(amplitude, "TDA amplitude")
    c = _real_array(coefficients, "orbital coefficients")
    if q.ndim != 2 or c.ndim != 2 or c.shape[1] != sum(q.shape):
        raise ValueError("orbital coefficient columns must match TDA occupied and virtual dimensions")
    nocc = q.shape[0]
    return c[:, nocc:] @ q.T @ c[:, :nocc].T


def tda_transition_density_derivative(
    response: OrbitalResponse, amplitude: ArrayLike, derivative: OrbitalDerivative,
) -> _Real:
    """Differentiate the unsymmetrized unit-amplitude transition density."""
    nocc = response.nocc
    q = _matrix(amplitude, (nocc, response.virtual.shape[1]), "TDA amplitude")
    cx = derivative.coefficients
    return cx[:, nocc:] @ q.T @ response.occupied.T + response.virtual @ q.T @ cx[:, :nocc].T


def _gap_action(fock_mo, q):
    nocc = q.shape[0]
    return q @ fock_mo[nocc:, nocc:] - fock_mo[:nocc, :nocc] @ q


def tda_first_action(
    response: OrbitalResponse, amplitude: ArrayLike, first: OrbitalDerivative,
    transition_potential: ArrayLike, transition_potential_x: ArrayLike,
) -> _Real:
    """Apply the total directional derivative of the TDA operator to q.

    The potentials are K[T] and its total first derivative, including the
    action K[T_x] and the change of the kernel with geometry and ground-state
    density. These inputs omit the factor two in the PySCF TDA action.
    """
    nocc = response.nocc
    q = _matrix(amplitude, (nocc, response.virtual.shape[1]), "TDA amplitude")
    v = _matrix(transition_potential, response.fock.shape, "transition potential")
    vx = _matrix(transition_potential_x, response.fock.shape, "transition potential derivative")
    ox, v_orb_x = first.coefficients[:, :nocc], first.coefficients[:, nocc:]
    return _gap_action(first.fock_mo, q) + 2 * (
        ox.T @ v.T @ response.virtual + response.occupied.T @ v.T @ v_orb_x
        + response.occupied.T @ vx.T @ response.virtual
    )


def fixed_amplitude_fock_density(
    response: OrbitalResponse, amplitude: ArrayLike, adjoint: ArrayLike | None = None,
) -> _Real:
    """Return the AO density contracting F_xy in the excitation curvature.

    An optional orbital adjoint includes the stationarity contribution.
    """
    o, v = response.occupied, response.virtual
    q = _matrix(amplitude, (o.shape[1], v.shape[1]), "TDA amplitude")
    oq, vq = o @ q, v @ q.T
    difference = vq @ vq.T - oq @ oq.T
    if adjoint is not None:
        z = _matrix(adjoint, (v.shape[1], o.shape[1]), "orbital adjoint")
        relaxed = v @ z @ o.T
        difference -= 0.5 * (relaxed + relaxed.T)
    return difference


def tda_orbital_gradient(
    response: OrbitalResponse, amplitude: ArrayLike, transition_potential: ArrayLike,
    kernel_density_potential: ArrayLike,
) -> _Real:
    """Return the excitation derivative with respect to the vo rotation.

    The supplied transition potential is K[T]. The kernel density potential
    represents the map dP -> 2 <T, (dK/dP)[dP, T]> as an AO contraction.
    Solve the ground-state orbital adjoint for this vo source.
    """
    o, v = response.occupied, response.virtual
    q = _matrix(amplitude, (o.shape[1], v.shape[1]), "TDA amplitude")
    potential = _matrix(transition_potential, response.fock.shape, "transition potential")
    kernel = _matrix(kernel_density_potential, response.fock.shape, "kernel density potential")
    density_potential = response._density_response(fixed_amplitude_fock_density(response, q)) + kernel
    gradient = 2 * v.T @ (density_potential + density_potential.T) @ o
    gradient += 4 * ((v.T @ potential.T @ v) @ q.T - q.T @ (o.T @ potential.T @ o))
    fock_vo = response.fock_mo[response.nocc:, :response.nocc]
    gradient -= 2 * (q.T @ q @ fock_vo + fock_vo @ q @ q.T)
    return gradient


def tda_mixed_contraction(
    response: OrbitalResponse, amplitude: ArrayLike, mixed: OrbitalCurvature,
    first_x: OrbitalDerivative, first_y: OrbitalDerivative,
    transition_x: ArrayLike, transition_y: ArrayLike,
    transition_potential: ArrayLike,
    transition_potential_x: ArrayLike, transition_potential_y: ArrayLike,
    kernel_potential_x: ArrayLike, kernel_potential_y: ArrayLike,
    kernel_mixed_contraction: float,
) -> float:
    """Contract the mixed TDA derivative in the chosen orbital frame.

    ``transition_potential`` is K[T]. The first transition potentials are
    total derivatives, and ``kernel_potential_x`` and ``kernel_potential_y``
    are K_x[T] and K_y[T] at fixed T. The final input is <T, K_xy[T]>,
    including the ground-density response present in ``mixed.density``.
    Apply the stationarity adjoint separately to remove the unknown mixed
    vo rotation from the final Hessian.
    """
    nocc = response.nocc
    q = _matrix(amplitude, (nocc, response.virtual.shape[1]), "TDA amplitude")
    cxy, cx, cy = mixed.coefficients, first_x.coefficients, first_y.coefficients
    txy = (cxy[:, nocc:] @ q.T @ response.occupied.T + response.virtual @ q.T @ cxy[:, :nocc].T
           + cx[:, nocc:] @ q.T @ cy[:, :nocc].T + cy[:, nocc:] @ q.T @ cx[:, :nocc].T)
    tx, ty, v, vx, vy, kx, ky = (
        _matrix(matrix, response.fock.shape, name) for matrix, name in (
            (transition_x, "transition density x"), (transition_y, "transition density y"),
            (transition_potential, "transition potential"),
            (transition_potential_x, "transition potential derivative x"),
            (transition_potential_y, "transition potential derivative y"),
            (kernel_potential_x, "kernel derivative x"), (kernel_potential_y, "kernel derivative y"),
        )
    )
    return float(_inner(q, _gap_action(mixed.fock_mo, q)) + 4 * _inner(txy, v)
                 + 2 * (_inner(tx, vy + ky) + _inner(ty, vx + kx) + kernel_mixed_contraction))
