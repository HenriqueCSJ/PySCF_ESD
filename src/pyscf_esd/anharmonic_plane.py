"""Two-dimensional sine Galerkin Hamiltonians on a reflection-symmetric box."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray


_Samples = ArrayLike | Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class SineGalerkinResult:
    """Bound eigenstates in ascending energy, with exact x parity ±1.

    Coefficients have shape (n_x_basis, n_y_basis, n_levels). Sampled
    wavefunctions have shape (n_x_quad, n_y_quad, n_levels) and are normalized
    with the outer product of the quadrature weights. The next energy is
    available to diagnose a cutoff at a nearly degenerate level.
    """

    energies_hartree: NDArray[np.float64]
    parity: NDArray[np.int8]
    coefficients: NDArray[np.float64]
    wavefunctions: NDArray[np.float64]
    next_energy_hartree: float | None
    _grid: SineGalerkin2D = field(repr=False, compare=False)


class SineGalerkin2D:
    """Tensor sine basis with Dirichlet walls and constant unit masses.

    The box is [-x_limit, x_limit] × [y_min, y_max]. Coordinates have units
    of bohr times sqrt(electron mass), so the kinetic operator is
    ``-1/2 * (d²/dx² + d²/dy²)``. Each potential is even in x. Supply either
    samples with shape (len(x), len(y)) or a vectorized function evaluated
    on ``x[:, None], y[None, :]``. Gauss–Legendre quadrature integrates
    potential and operator matrix elements on the same box.
    """

    def __init__(
        self,
        x_limit: float,
        y_bounds: tuple[float, float],
        basis_shape: tuple[int, int],
        quadrature_shape: tuple[int, int] | None = None,
    ) -> None:
        self.x_bounds = (-float(x_limit), float(x_limit))
        self.y_bounds = tuple(map(float, y_bounds))
        if len(self.y_bounds) != 2 or any(
            not np.isfinite(bounds).all() or not np.isfinite(bounds[1] - bounds[0]) or bounds[0] >= bounds[1]
            for bounds in (self.x_bounds, self.y_bounds)
        ):
            raise ValueError("x_limit and y_bounds must define finite intervals of positive length")
        nx, ny = self.basis_shape = _dimensions(basis_shape, "basis_shape", minimum=(2, 1))
        if quadrature_shape is None:
            quadrature_shape = (max(24, 3 * nx), max(24, 3 * ny))
        nqx, nqy = _dimensions(quadrature_shape, "quadrature_shape", minimum=(nx, ny))
        self.x, self.x_weights, self._basis_x, kinetic_x = _sine_axis(self.x_bounds, nx, nqx)
        self.y, self.y_weights, self._basis_y, kinetic_y = _sine_axis(self.y_bounds, ny, nqy)
        self._kinetic = kinetic_x[:, None] + kinetic_y[None, :]
        self._sqrt_weights = np.sqrt(self.x_weights[:, None] * self.y_weights[None, :])
        self._y_products, self._y_pairs = _basis_products(self._basis_y, self.y_weights)
        self._x_products = tuple(_basis_products(self._basis_x[:, start::2], self.x_weights) for start in (0, 1))

    def solve(
        self,
        potential_hartree: _Samples,
        *,
        n_levels: int,
        parity_atol_hartree: float = 1e-9,
    ) -> SineGalerkinResult:
        """Solve even and odd blocks separately and retain the lowest levels.

        Potential energies share one energy zero across both electronic
        surfaces. The quadrature scale should be converged separately for
        potentials with appreciable structure at the quadrature spacing.
        """
        from scipy.linalg import eigh

        if (
            isinstance(n_levels, (bool, np.bool_)) or not isinstance(n_levels, Integral)
            or not 1 <= n_levels <= self._kinetic.size
        ):
            raise ValueError("n_levels must be an integer between 1 and the basis size")
        if not np.isfinite(parity_atol_hartree) or parity_atol_hartree < 0:
            raise ValueError("parity_atol_hartree must be finite and nonnegative")
        potential = self._sample(potential_hartree, "potential_hartree")
        defect = np.max(np.abs(potential - potential[::-1]))
        if defect > parity_atol_hartree:
            raise ValueError(f"potential must be even in x, but its reflection defect is {defect:.6g} Hartree")
        reference = float(np.min(potential))
        potential = potential - reference
        # Cache the first contraction across the y axis and expand each parity
        # block's symmetric pairs only after both quadrature contractions.
        y_contraction = (0.5 * potential + 0.5 * potential[::-1]) @ self._y_products
        blocks = []
        for start, (products, pairs) in enumerate(self._x_products):
            integrals = products.T @ y_contraction
            kinetic = self._kinetic[start::2].ravel()
            hamiltonian = integrals[pairs[:, None, :, None], self._y_pairs[None, :, None, :]]
            hamiltonian = hamiltonian.reshape(kinetic.size, kinetic.size)
            hamiltonian[np.diag_indices(kinetic.size)] += kinetic
            last = min(n_levels, kinetic.size - 1)
            energies, vectors = eigh(hamiltonian, subset_by_index=(0, last))
            blocks.append((energies, vectors))

        block_energies = np.concatenate([energies for energies, _ in blocks])
        order = np.argsort(block_energies, kind="stable")
        retained = order[:n_levels]
        next_energy = float(block_energies[order[n_levels]] + reference) if n_levels < len(order) else None
        coefficients = np.zeros((*self.basis_shape, n_levels))
        parity = np.ones(n_levels, dtype=np.int8)
        offset = 0
        for start, (energies, vectors) in enumerate(blocks):
            selected = (retained >= offset) & (retained < offset + len(energies))
            basis_vectors = vectors[:, retained[selected] - offset]
            block = coefficients[start::2]
            block[:, :, selected] = basis_vectors.reshape(*block.shape[:2], basis_vectors.shape[-1])
            parity[selected] = 1 - 2 * start
            offset += len(energies)
        wavefunctions = np.einsum("xi,ijn,yj->xyn", self._basis_x, coefficients, self._basis_y, optimize=True)
        energies = block_energies[retained] + reference
        for array in (energies, coefficients, parity, wavefunctions):
            array.flags.writeable = False
        return SineGalerkinResult(energies, parity, coefficients, wavefunctions, next_energy, self)

    def transition_matrix(
        self, initial: SineGalerkinResult, final: SineGalerkinResult, operator: _Samples,
    ) -> NDArray[np.float64]:
        """Return real matrix elements with rows initial and columns final."""
        initial_weighted = self._weighted_wavefunctions(initial)
        final_weighted = self._weighted_wavefunctions(final)
        samples = self._sample(operator, "operator").ravel()
        return initial_weighted.T @ (samples[:, None] * final_weighted)

    def expectation(self, state: SineGalerkinResult, operator: _Samples) -> NDArray[np.float64]:
        """Return the diagonal of the sampled operator on the given states."""
        weighted = self._weighted_wavefunctions(state)
        samples = self._sample(operator, "operator").ravel()
        return samples @ (weighted**2)

    def _sample(self, values: _Samples, name: str) -> NDArray[np.float64]:
        shape = (len(self.x), len(self.y))
        if callable(values):
            values = np.broadcast_to(np.asarray(values(self.x[:, None], self.y[None, :])), shape)
        data = np.asarray(values)
        if data.shape != shape or data.dtype.kind not in "fiu" or not np.isfinite(data).all():
            raise ValueError(f"{name} must contain finite real samples with shape {shape}")
        return data.astype(float, copy=False)

    def _weighted_wavefunctions(self, result: SineGalerkinResult) -> NDArray[np.float64]:
        if result._grid is not self:
            raise ValueError("eigenstates must come from the same sine Galerkin quadrature")
        return (result.wavefunctions * self._sqrt_weights[:, :, None]).reshape(-1, len(result.energies_hartree))


def _dimensions(values: tuple[int, int], name: str, minimum: tuple[int, int]) -> tuple[int, int]:
    if len(values) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower
        for value, lower in zip(values, minimum)
    ):
        raise ValueError(f"{name} must contain two integers with axis minima {minimum}")
    return int(values[0]), int(values[1])


def _sine_axis(
    bounds: tuple[float, float], n_basis: int, n_quad: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    left, right = bounds
    length = right - left
    nodes, weights = leggauss(n_quad)
    fraction = 0.5 * (nodes + 1.0)
    points = (left + 0.5 * length) + 0.5 * length * nodes
    weights *= 0.5 * length
    modes = np.arange(1, n_basis + 1)
    basis = np.sqrt(2.0 / length) * np.sin(np.pi * np.outer(fraction, modes))
    gram = basis.T @ (weights[:, None] * basis)
    if not np.allclose(gram, np.eye(n_basis), atol=1e-8, rtol=0):
        raise ValueError("quadrature_shape must resolve the sine basis to 1e-8 in overlap")
    kinetic = 0.5 * (np.pi * modes / length)**2
    for array in (points, weights, basis, kinetic):
        array.flags.writeable = False
    return points, weights, basis, kinetic


def _basis_products(
    basis: NDArray[np.float64], weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    n = basis.shape[1]
    left, right = np.triu_indices(n)
    lookup = np.empty((n, n), dtype=np.intp)
    lookup[left, right] = np.arange(len(left))
    lookup[right, left] = lookup[left, right]
    return weights[:, None] * basis[:, left] * basis[:, right], lookup
