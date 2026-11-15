"""Three-dimensional sine DVR on a box with reflection parity in the first axis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


_Real = NDArray[np.float64]
_Parity = Literal["even", "odd"]
_INV_SQRT_2 = 1 / np.sqrt(2.0)


@dataclass(frozen=True)
class DVRLevels:
    """Energies and independently evaluated residual norms in Hartree.

    ``vectors[i]`` contains the real, Euclidean-normalized DVR amplitudes
    of state ``i``. Divide by the square root of the grid cell volume to
    obtain samples of its continuum wavefunction.
    """

    energies: _Real
    vectors: _Real
    residuals: _Real


@dataclass(frozen=True)
class DVRSpectralMeasure:
    """Lanczos quadrature nodes (Hartree) and unnormalized spectral weights.

    ``weights`` sum to the parity-projected source norm squared. The nodes
    are Ritz values for quadrature of smooth functions of energy. Use an
    order-convergence comparison for the particular broadened observable.
    ``terminal_residual`` is the final Lanczos coupling in Hartree.
    ``max_reorthogonalization`` is the largest norm of a correction to the
    three-term recurrence in Hartree through this order.
    """

    energies: _Real
    weights: _Real
    order: int
    projected_norm_squared: float
    terminal_residual: float
    max_reorthogonalization: float


class SineDVR3D:
    """Tensor Dirichlet sine grid with unit masses and an even potential in x.

    Mass-weighted coordinates are in bohr times sqrt(electron mass). Each
    axis contains only the interior, equally spaced nodes. The discrete
    sine transform diagonalizes the kinetic energy with the analytic
    eigenvalues sum(k_i**2)/2. The sampled potential acts by pointwise
    multiplication. All energy inputs and outputs are in Hartree.
    """

    def __init__(
        self,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        shape: tuple[int, int, int],
        *,
        workers: int = 2,
    ) -> None:
        intervals = np.asarray(bounds, dtype=float)
        if intervals.shape != (3, 2) or not np.isfinite(intervals).all():
            raise ValueError("bounds must contain three finite intervals")
        lengths = intervals[:, 1] - intervals[:, 0]
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("bounds must define intervals of positive finite length")
        if intervals[0, 0] != -intervals[0, 1]:
            raise ValueError("the x interval must be symmetric about zero")
        if len(shape) != 3 or any(
            not _positive_integer(n) or n < minimum for n, minimum in zip(shape, (2, 1, 1))
        ):
            raise ValueError("shape must contain three integers with minima (2, 1, 1)")
        if not _positive_integer(workers):
            raise ValueError("workers must be a positive integer")

        self.shape = tuple(map(int, shape))
        self.bounds = tuple(map(tuple, intervals))
        self.workers = int(workers)
        self.spacing = tuple(length / (n + 1) for length, n in zip(lengths, self.shape))
        self.axes = tuple(
            _read_only(0.5 * lower + 0.5 * upper + step * (np.arange(n) - (n - 1) / 2))
            for (lower, upper), step, n in zip(self.bounds, self.spacing, self.shape)
        )
        kx, ky, kz = (0.5 * (np.pi * np.arange(1, n + 1) / length)**2 for n, length in zip(self.shape, lengths))
        self._kinetic = kx[:, None, None] + ky[None, :, None] + kz[None, None, :]
        self._paired_rows = self.shape[0] // 2

    def solve(
        self,
        potential: ArrayLike,
        *,
        parity: _Parity,
        n_levels: int,
        parity_atol_hartree: float = 1e-9,
        residual_atol_hartree: float = 1e-10,
    ) -> DVRLevels:
        """Solve the lowest states of one exact reflection-parity block.

        Requesting the complete eigensystem uses a dense solve for a block
        with at most 128 nodes. Partial eigensystems use matrix-free ARPACK.
        """
        from scipy.linalg import eigh
        from scipy.sparse.linalg import LinearOperator, eigsh

        sign = self._parity_sign(parity)
        reduced_shape = self._reduced_shape(sign)
        size = int(np.prod(reduced_shape))
        if not _positive_integer(n_levels) or n_levels > size:
            raise ValueError("n_levels must be a positive integer up to the parity block size")
        if not np.isfinite(residual_atol_hartree) or residual_atol_hartree <= 0:
            raise ValueError("residual_atol_hartree must be finite and positive")
        shifted, reference = self._shifted_potential(potential, parity_atol_hartree)
        action = self._reduced_action(shifted, sign)
        if n_levels == size:
            if size > 128:
                raise ValueError("a full eigensystem is available only for parity blocks of at most 128 nodes")
            dense = np.column_stack([action(column) for column in np.eye(size)])
            energies, states = eigh(dense)
        else:
            operator = LinearOperator((size, size), matvec=action, dtype=np.float64)
            energies, states = eigsh(
                operator,
                k=n_levels,
                which="SA",
                v0=np.random.default_rng(0).standard_normal(size),
                ncv=min(size, max(32, 4 * n_levels + 1)),
                tol=1e-12,
            )

        order = np.argsort(energies)
        energies = energies[order]
        vectors = np.stack([self._unpack(states[:, index].reshape(reduced_shape), sign) for index in order])
        residuals = np.array([
            np.linalg.norm((self._kinetic_action(vector) + (shifted - energy) * vector).ravel())
            for energy, vector in zip(energies, vectors)
        ])
        if not np.isfinite(residuals).all() or residuals.max() > residual_atol_hartree:
            raise RuntimeError(f"sine DVR eigenpair residual {residuals.max():.6g} Hartree exceeds tolerance")
        return DVRLevels(_read_only(energies + reference), _read_only(vectors), _read_only(residuals))

    def spectral_measure(
        self,
        potential: ArrayLike,
        initial_vector: ArrayLike,
        parity: _Parity,
        order: int,
        *,
        parity_atol_hartree: float = 1e-9,
    ) -> DVRSpectralMeasure:
        """Compute the spectral measure of a sampled source, such as μ times ψ."""
        return self.spectral_measures(
            potential, initial_vector, parity=parity, orders=(order,), parity_atol_hartree=parity_atol_hartree,
        )[0]

    def spectral_measures(
        self,
        potential: ArrayLike,
        initial_vector: ArrayLike,
        parity: _Parity,
        orders: tuple[int, ...],
        *,
        parity_atol_hartree: float = 1e-9,
    ) -> tuple[DVRSpectralMeasure, ...]:
        """Use prefixes of one fully reorthogonalized Lanczos run to compare orders.

        The source is projected into ``parity`` before its norm is measured.
        Requested orders beyond the cyclic subspace dimension return the
        same measure once the Lanczos recurrence breaks down at roundoff.
        An exactly zero projected source yields an empty measure.
        """
        from scipy.linalg import eigh_tridiagonal
        from scipy.linalg.blas import dnrm2

        sign = self._parity_sign(parity)
        orders = tuple(orders)
        if not orders or any(not _positive_integer(order) for order in orders):
            raise ValueError("orders must contain positive integers")
        shifted, reference = self._shifted_potential(potential, parity_atol_hartree)
        source = self._pack(self._samples(initial_vector, "initial_vector"), sign).ravel()
        norm = float(dnrm2(source))
        norm_squared = norm * norm
        if norm == 0:
            empty = _read_only(np.empty(0))
            return tuple(DVRSpectralMeasure(empty, empty, 0, 0.0, 0.0, 0.0) for _ in orders)
        if not np.isfinite(norm_squared) or norm_squared == 0:
            raise ValueError("the projected source squared norm must be representable in float64")

        norm_bound = float(self._kinetic.max() + shifted.max())
        diagonal, beta, reorthogonalization = _lanczos(
            self._reduced_action(shifted, sign), source / norm, min(max(orders), source.size), norm_bound,
        )

        measures = {}
        actual_orders = tuple(min(order, len(diagonal)) for order in orders)
        for actual_order in sorted(set(actual_orders)):
            nodes, states = eigh_tridiagonal(diagonal[:actual_order], beta[:actual_order - 1])
            measures[actual_order] = DVRSpectralMeasure(
                _read_only(nodes + reference),
                _read_only(norm_squared * states[0]**2),
                actual_order,
                norm_squared,
                float(beta[actual_order - 1]),
                float(reorthogonalization[actual_order - 1]),
            )
        return tuple(measures[actual_order] for actual_order in actual_orders)

    def _kinetic_action(self, vector: _Real) -> _Real:
        from scipy.fft import dstn

        coefficients = dstn(vector, type=1, norm="ortho", workers=self.workers)
        coefficients *= self._kinetic
        return dstn(coefficients, type=1, norm="ortho", workers=self.workers, overwrite_x=True)

    def _reduced_action(self, shifted: _Real, sign: int) -> Callable[[_Real], _Real]:
        reduced_shape = self._reduced_shape(sign)
        potential = shifted[:reduced_shape[0]]

        def action(vector: _Real) -> _Real:
            block = vector.reshape(reduced_shape)
            kinetic = self._kinetic_action(self._unpack(block, sign))
            return (self._pack(kinetic, sign) + potential * block).ravel()

        return action

    def _shifted_potential(self, potential: ArrayLike, parity_atol_hartree: float) -> tuple[_Real, float]:
        if not np.isfinite(parity_atol_hartree) or parity_atol_hartree < 0:
            raise ValueError("parity_atol_hartree must be finite and nonnegative")
        samples = self._samples(potential, "potential")
        defect = np.max(np.abs(samples - samples[::-1]))
        if defect > parity_atol_hartree:
            raise ValueError(f"potential must be even in x, but its reflection defect is {defect:.6g} Hartree")
        symmetric = 0.5 * samples + 0.5 * samples[::-1]
        reference = float(symmetric.min())
        return symmetric - reference, reference

    def _samples(self, values: ArrayLike, name: str) -> _Real:
        samples = np.asarray(values)
        if samples.shape != self.shape or samples.dtype.kind not in "fiu" or not np.isfinite(samples).all():
            raise ValueError(f"{name} must contain finite real samples with shape {self.shape}")
        return samples.astype(float, copy=False)

    @staticmethod
    def _parity_sign(parity: _Parity) -> int:
        if parity == "even":
            return 1
        if parity == "odd":
            return -1
        raise ValueError("parity must be 'even' or 'odd'")

    def _reduced_shape(self, sign: int) -> tuple[int, int, int]:
        return (self._paired_rows + int(sign == 1 and self.shape[0] % 2 == 1), *self.shape[1:])

    def _pack(self, full: _Real, sign: int) -> _Real:
        pairs = self._paired_rows
        reduced = np.empty(self._reduced_shape(sign))
        reduced[:pairs] = _INV_SQRT_2 * (full[:pairs] + sign * full[::-1][:pairs])
        if reduced.shape[0] > pairs:
            reduced[pairs] = full[pairs]
        return reduced

    def _unpack(self, reduced: _Real, sign: int) -> _Real:
        pairs = self._paired_rows
        full = np.zeros(self.shape)
        full[:pairs] = _INV_SQRT_2 * reduced[:pairs]
        full[-pairs:] = sign * full[:pairs][::-1]
        if reduced.shape[0] > pairs:
            full[pairs] = reduced[pairs]
        return full


def _lanczos(
    action: Callable[[_Real], _Real], start: _Real, order: int, norm_bound: float,
) -> tuple[_Real, _Real, _Real]:
    from scipy.linalg.blas import dgemv

    basis = np.empty((start.size, order), order="F")
    basis[:, 0] = start
    diagonal = np.empty(order)
    beta = np.empty(order)
    maximum_correction = np.empty(order)
    threshold = 64 * np.finfo(float).eps * norm_bound
    largest_correction = 0.0
    for index in range(order):
        current = basis[:, index]
        residual = action(current)
        diagonal[index] = np.dot(current, residual)
        residual -= diagonal[index] * current
        if index:
            residual -= beta[index - 1] * basis[:, index - 1]
        # Full two-pass reorthogonalization keeps later Ritz nodes from
        # representing repeated copies of previously converged states.
        previous = basis[:, :index + 1]
        for _ in range(2):
            overlaps = dgemv(1.0, previous, residual, trans=1)
            largest_correction = max(largest_correction, float(np.linalg.norm(overlaps)))
            residual = dgemv(-1.0, previous, overlaps, beta=1.0, y=residual, overwrite_y=1)
        maximum_correction[index] = largest_correction
        beta[index] = np.linalg.norm(residual)
        if not np.isfinite(beta[index]):
            raise RuntimeError("sine DVR Lanczos recurrence produced a nonfinite residual")
        if beta[index] <= threshold:
            return diagonal[:index + 1], beta[:index + 1], maximum_correction[:index + 1]
        if index + 1 < order:
            basis[:, index + 1] = residual / beta[index]
    return diagonal, beta, maximum_correction


def _positive_integer(value: int) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


def _read_only(array: _Real) -> _Real:
    array.flags.writeable = False
    return array
