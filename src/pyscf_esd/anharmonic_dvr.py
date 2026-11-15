"""One-dimensional Fourier and sine discrete variable representations."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAX_GRID_POINTS = 1024
_GRID_RTOL = 1e-8


@dataclass(frozen=True)
class DVRResult:
    """Energy levels and real grid coefficients on a uniform quadrature.

    The columns of ``coefficients`` are orthonormal with the Euclidean inner
    product. ``wavefunctions`` gives the sampled wavefunctions, normalized
    with quadrature weight ``spacing``. A matrix element of a grid-sampled
    function uses the coefficients directly, without another spacing factor.

    ``domain`` holds the endpoints of the domain. The upper endpoint is omitted
    on a periodic grid and both endpoints are omitted on a Dirichlet grid.
    ``all_energies_hartree`` contains every eigenvalue of the finite DVR
    Hamiltonian. Only the lowest levels have retained coefficients.
    """

    grid: NDArray[np.float64]
    boundary: Literal["dirichlet", "periodic"]
    spacing: float
    domain: tuple[float, float]
    kinetic_coefficient: float
    all_energies_hartree: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    @property
    def energies_hartree(self) -> NDArray[np.float64]:
        return self.all_energies_hartree[: self.coefficients.shape[1]]

    @property
    def wavefunctions(self) -> NDArray[np.float64]:
        return self.coefficients / np.sqrt(self.spacing)


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real vector")
    try:
        array = np.array(array, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real vector") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive real scalar")
    scalar = float(array)
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return scalar


def _kinetic_matrix(
    n_points: int, kinetic_scale: float, boundary: Literal["dirichlet", "periodic"]
) -> NDArray[np.float64]:
    if boundary == "periodic":
        # The Fourier kinetic operator is circulant in the coordinate grid.
        modes = 2 * np.fft.fftfreq(n_points)
        kernel = np.fft.ifft(kinetic_scale * modes**2).real
        points = np.arange(n_points)
        return kernel[np.subtract.outer(points, points)]

    points = np.arange(1, n_points + 1)
    modes = points / (n_points + 1)
    basis = np.sqrt(2 / (n_points + 1)) * np.sin(np.pi * np.outer(points, modes))
    return (basis * (kinetic_scale * modes**2)) @ basis.T


def solve_dvr(
    grid: ArrayLike,
    potential_hartree: ArrayLike,
    *,
    boundary: Literal["dirichlet", "periodic"],
    n_levels: int,
    kinetic_coefficient: float = 1.0,
    period: float | None = None,
) -> DVRResult:
    """Solve a constant-coefficient Hamiltonian on a sampled potential.

    The kinetic operator is ``-kinetic_coefficient / 2 * d²/dq²``. For a
    mass-weighted coordinate q in sqrt(electron mass) * bohr, the coefficient
    is 1. For a rotor with q in radians, it is ``1 / I``, with I in electron
    mass * bohr².

    Supply 3 to 1024 uniformly spaced, strictly increasing points. A periodic
    grid spans one cycle with the repeated endpoint excluded. Its period
    defaults to 2π. Dirichlet walls are located one spacing beyond the first
    and last supplied points.

    Supply one potential energy in Hartree per grid point. The eigenvalues
    keep the potential's energy reference, so use the same reference on both
    surfaces for transition energies. The dense eigensolver returns all
    eigenvalues, with coefficients retained for ``n_levels`` lowest states.
    Eigenvectors in degenerate subspaces may mix under a basis rotation.
    """
    if boundary not in ("dirichlet", "periodic"):
        raise ValueError("boundary must be dirichlet or periodic")
    coordinates = _real_vector(grid, "grid")
    n_points = coordinates.size
    if not 3 <= n_points <= _MAX_GRID_POINTS:
        raise ValueError(f"grid must contain 3 to {_MAX_GRID_POINTS} points")
    if (
        isinstance(n_levels, bool)
        or not isinstance(n_levels, Integral)
        or not 1 <= n_levels <= n_points
    ):
        raise ValueError("n_levels must be an integer from 1 to the number of grid points")
    potential = _real_vector(potential_hartree, "potential_hartree")
    if potential.shape != coordinates.shape:
        raise ValueError("potential_hartree must contain one energy per grid point")
    coefficient = _positive_scalar(kinetic_coefficient, "kinetic_coefficient")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        steps = np.diff(coordinates)
        spacing = float((coordinates[-1] - coordinates[0]) / (n_points - 1))
    if not np.isfinite(spacing) or spacing <= 0 or np.any(steps <= 0):
        raise ValueError("grid must be strictly increasing with finite spacing")
    if not np.allclose(steps, spacing, rtol=_GRID_RTOL, atol=0):
        raise ValueError("grid must be uniformly spaced")
    if boundary == "periodic":
        cycle = 2 * np.pi if period is None else _positive_scalar(period, "period")
        grid_period = n_points * spacing
        if not np.isclose(grid_period, cycle, rtol=_GRID_RTOL, atol=0):
            raise ValueError("periodic grid must cover one period without a repeated endpoint")
        domain = (float(coordinates[0]), float(coordinates[0]) + grid_period)
    else:
        if period is not None:
            raise ValueError("period is only defined for a periodic grid")
        domain = (float(coordinates[0]) - spacing, float(coordinates[-1]) + spacing)
    if (
        not all(np.isfinite(domain))
        or domain[1] <= coordinates[-1]
        or (boundary == "dirichlet" and domain[0] >= coordinates[0])
    ):
        raise ValueError("grid endpoints and spacing must define a finite domain")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kinetic_scale = np.square((np.sqrt(coefficient) / spacing) * (np.pi / np.sqrt(2)))
    if not np.isfinite(kinetic_scale) or kinetic_scale <= 0:
        raise ValueError("grid spacing and kinetic_coefficient give an unrepresentable kinetic energy")
    reference = potential.min()
    diagonal = np.diag_indices(n_points)
    with np.errstate(over="ignore", invalid="ignore"):
        hamiltonian = _kinetic_matrix(n_points, kinetic_scale, boundary)
        hamiltonian[diagonal] += potential - reference
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("grid potential or kinetic energy is too large for the DVR Hamiltonian")
    energies, coefficients = np.linalg.eigh(hamiltonian)
    with np.errstate(over="ignore", invalid="ignore"):
        energies += reference
    if not np.all(np.isfinite(energies)):
        raise ValueError("DVR energies are outside the finite floating-point range")
    retained = coefficients[:, :n_levels].copy()
    for array in (coordinates, energies, retained):
        array.flags.writeable = False
    return DVRResult(coordinates, boundary, spacing, domain, coefficient, energies, retained)
