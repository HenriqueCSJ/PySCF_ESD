"""Gaussian-broadened thermal harmonic and separable vibronic spectra."""

from dataclasses import dataclass
from math import isfinite, log, sqrt
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .anharmonic import (
    HarmonicSpectators,
    _common_dipole,
    _dipole_amplitudes,
    _initial_populations,
    _reject_split_manifold,
)
from .anharmonic_dvr import DVRResult
from .anharmonic_sampling import PhasedRotorDipole
from .harmonic import _KB_HARTREE_PER_K, _amplitude_array, _fc_parameters, _real_array
from .spectral_properties import HARTREE_TO_WAVENUMBER

_FloatVector = NDArray[np.float64]
_Process = Literal["absorption", "emission"]


@dataclass(frozen=True)
class ThermalSeparableSpectrum:
    """Signed energy grid and positive-photon, Gaussian-broadened profile.

    The profile is a density per cm⁻¹ of absolute thermal dipole strength
    weighted by E (absorption) or E³ (emission), with E in Hartree. Multiply
    by the relevant oscillator-strength or spontaneous-emission constant for
    physical units. Negative grid coordinates can hold Gaussian tails from
    transitions with positive photon energies.
    """

    wavenumber_cm_inverse: _FloatVector
    photon_weighted_density_per_cm_inverse: _FloatVector
    fwhm_cm_inverse: float
    step_cm_inverse: float
    temperature_kelvin: float
    process: _Process
    initial_population_coverage: float
    final_dipole_strength_coverage: float | None
    signed_dipole_strength_au2: float
    positive_photon_weighted_strength: float
    centroid_cm_inverse: float | None
    origin_00_cm_inverse: float
    tail_probability_bound: float
    auxiliary_sigma_cm_inverse: float
    negative_weight_grid_fraction: float
    dipole_strength_density_per_cm_inverse: _FloatVector | None = None


@dataclass(frozen=True)
class _LinearDipole:
    """Affine Cartesian dipole multiplying the Bargmann overlap."""

    constant: _FloatVector
    slope: _FloatVector
    strength: float


@dataclass(frozen=True)
class _ConstantQuadratic:
    inverse_minus: _FloatVector
    cross_covariance: _FloatVector
    log_determinant: float


@dataclass(frozen=True)
class _HarmonicCharacteristic:
    """Gaussian evaluation of the squared-overlap thermal generating function."""

    energy: _FloatVector
    log_activity: _FloatVector
    linear: _FloatVector
    quadratic: _FloatVector
    log_prefactor: float
    zero_point: float
    dipole: _LinearDipole | None = None
    constant_quadratic: _ConstantQuadratic | None = None

    def _gaussian_moments(self, matrix, half_activity):
        source = half_activity * self.linear
        rhs = source[..., None]
        if self.dipole is not None:
            slope = half_activity[..., :, None] * self.dipole.slope
            rhs = np.concatenate((rhs, slope), axis=-1)
        factors = self.constant_quadratic
        if factors is None:
            identity = np.eye(len(self.energy))
            solved = np.linalg.solve(identity - matrix, rhs)
        else:
            solved = factors.inverse_minus @ rhs
        mean = solved[..., 0]
        exponent = np.einsum("...i,...i->...", source, mean)
        if self.dipole is None:
            return exponent, 1.0

        # Cross covariance of the two Bargmann copies is (I - M²)^(-1).
        # Half the sum of the shifted solves avoids forming M² explicitly.
        if factors is None:
            cross = 0.5 * (solved[..., 1:] + np.linalg.solve(identity + matrix, slope))
        else:
            cross = factors.cross_covariance @ slope
        dipole_mean = self.dipole.constant + np.einsum("...iv,...i->...v", slope, mean)
        polynomial = np.einsum("...v,...v->...", dipole_mean, dipole_mean)
        polynomial += np.einsum("...iv,...iv->...", slope, cross)
        return exponent, polynomial / self.dipole.strength

    def transform(self, time: _FloatVector) -> NDArray[np.complex128]:
        """Return E[exp(-i*time*energy)] in the signed photon convention."""
        if not self.energy.size:
            return np.ones(time.shape, dtype=complex)

        result = np.empty(time.shape, dtype=complex)
        # Bound the transient allocation by the number of matrix elements.
        elements = self.quadratic.size if self.constant_quadratic is None else 4 * self.energy.size
        batch = max(1, 2**19 // elements)
        for start in range(0, len(time), batch):
            local_time = time[start:start + batch]
            half_activity = np.exp(
                0.5 * (self.log_activity[None, :] - 1j * local_time[:, None] * self.energy[None, :])
            )
            if self.constant_quadratic is None:
                matrix = half_activity[:, :, None] * self.quadratic * half_activity[:, None, :]
                eigenvalues = np.linalg.eigvals(matrix)
                # Each eigenvalue stays in the unit disk. Sum principal logs
                # to continue the determinant square root without phase jumps.
                determinant = np.sum(np.log1p(eigenvalues) + np.log1p(-eigenvalues), axis=1)
            else:
                matrix = None
                determinant = self.constant_quadratic.log_determinant
            exponent, polynomial = self._gaussian_moments(matrix, half_activity)
            result[start:start + batch] = polynomial * np.exp(
                self.log_prefactor + exponent - determinant / 2 - 1j * local_time * self.zero_point
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("the harmonic characteristic function exceeds the numerical range")
        return result

    def log_moment(self, parameter: float) -> float:
        """Return log E[exp(parameter*energy)] or infinity outside its convergence domain."""
        if not self.energy.size:
            return 0.0
        with np.errstate(over="ignore", under="ignore"):
            half_activity = np.exp(0.5 * (self.log_activity + parameter * self.energy))
        if not np.all(np.isfinite(half_activity)):
            return np.inf
        if self.constant_quadratic is None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                matrix = half_activity[:, None] * self.quadratic * half_activity[None, :]
            if not np.all(np.isfinite(matrix)):
                return np.inf
            eigenvalues = np.linalg.eigvalsh(matrix)
            if np.max(np.abs(eigenvalues)) >= 1:
                return np.inf
            determinant = np.log1p(-eigenvalues**2).sum()
        else:
            matrix = None
            determinant = self.constant_quadratic.log_determinant
        exponent, polynomial = self._gaussian_moments(matrix, half_activity)
        if not isfinite(polynomial) or polynomial <= 0:
            return np.inf
        return float(self.log_prefactor + exponent - determinant / 2 + parameter * self.zero_point + log(polynomial))


def _harmonic_characteristic(
    spectators: HarmonicSpectators | None,
    thermal_energy: float,
    sign: int,
    *,
    dipole: _FloatVector | None = None,
    derivatives: ArrayLike | None = None,
) -> _HarmonicCharacteristic:
    empty = _HarmonicCharacteristic(np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0)), 0.0, 0.0)
    if spectators is None:
        return empty
    if not isinstance(spectators, HarmonicSpectators):
        raise ValueError("spectators must be a HarmonicSpectators instance")
    omega_i = _real_array(spectators.initial_frequencies_hartree, "initial spectator frequencies")
    omega_f = _real_array(spectators.final_frequencies_hartree, "final spectator frequencies")
    if omega_i.ndim != 1 or omega_i.shape != omega_f.shape:
        raise ValueError("initial and final spectator frequencies must be equal-length vectors")
    if np.any(omega_i <= 0) or np.any(omega_f <= 0):
        raise ValueError("spectator frequencies must be strictly positive")
    modes = omega_i.size
    if derivatives is not None:
        derivatives = _real_array(derivatives, "dipole_derivatives_au")
        if dipole is None or derivatives.shape != (modes, dipole.size):
            raise ValueError("dipole derivatives must match the harmonic modes and dipole components")
        if not np.any(derivatives):
            derivatives = None
    rotation = np.eye(modes) if spectators.rotation is None else _real_array(spectators.rotation, "spectator rotation")
    if rotation.shape != (modes, modes) or not np.allclose(rotation.T @ rotation, np.eye(modes), atol=1e-8, rtol=0):
        raise ValueError("spectator rotation must be an orthogonal square Duschinsky matrix")
    shift = np.zeros(modes) if spectators.shift_au is None else _real_array(spectators.shift_au, "spectator shift_au")
    if shift.shape != (modes,):
        raise ValueError("spectator shift_au must have one entry per spectator mode")
    same_curvature = np.array_equal(omega_f[:, None] * rotation, rotation * omega_i)
    if not modes or (derivatives is None and not np.any(shift) and same_curvature):
        # Identical harmonic Hamiltonians, including permutations and
        # rotations within an exactly degenerate eigenspace, preserve energy.
        return empty

    # The Bargmann overlap generating function is
    # a_00 exp(l.T z + z.T Q z/2). Contracting its two copies against
    # the Gaussian with cross covariance R sums |a_nm|² product(R_j^n_j).
    # This is the Condon generating function of Huh and Berger (Eq. 17,
    # https://arxiv.org/abs/1111.3052), with the sign convention Q = -W.
    origin, linear, quadratic = _fc_parameters(omega_i, omega_f, rotation, shift)
    energy = sign * HARTREE_TO_WAVENUMBER * np.concatenate((-omega_i, omega_f))
    zero_point = 0.0 if same_curvature else float(0.5 * energy.sum())
    if same_curvature:
        # With matched curvature the only quadratic entries couple initial and
        # final modes of identical energy. Use their exact values so activity
        # factors cancel the complex or real energy tilt in every entry.
        quadratic[:modes, :modes] = quadratic[modes:, modes:] = 0
        quadratic[:modes, modes:] = rotation.T
        quadratic[modes:, :modes] = rotation
    log_prefactor = 2 * log(origin)
    occupation = np.zeros(modes)
    if thermal_energy > 0:
        with np.errstate(over="ignore", under="ignore"):
            beta_omega = omega_i / thermal_energy
            partition = -np.expm1(-beta_omega)
            log_prefactor += float(np.log(partition).sum())
            if derivatives is not None:
                occupation = np.exp(-beta_omega) / partition
        log_activity = np.concatenate((-beta_omega, np.zeros(modes)))
    dipole_factor = None
    if derivatives is not None:
        inverse_scale = 1.0 / np.sqrt(2 * omega_i[:, None])
        scaled = derivatives * inverse_scale
        strength = float(dipole @ dipole + (2 * occupation + 1) @ np.sum(scaled**2, axis=1))
        if not isfinite(strength) or strength <= 0:
            raise ValueError("thermal dipole closure exceeds the numerical range")
        # Q_i inserts (z_i + ∂/∂z_i) / sqrt(2ω_i) on the initial Fock index.
        constant = dipole + linear[:modes] @ scaled
        slope = (np.eye(2 * modes)[:modes] + quadratic[:modes]).T @ scaled
        dipole_factor = _LinearDipole(constant, slope if thermal_energy > 0 else slope[modes:], strength)
    if thermal_energy == 0:
        # Only initial spectator occupation zero survives the thermal limit.
        energy = energy[modes:]
        log_activity = np.zeros(modes)
        linear = linear[modes:]
        quadratic = quadratic[modes:, modes:]
    factors = None
    if same_curvature:
        half_activity = np.exp(0.5 * log_activity)
        matrix = half_activity[:, None] * quadratic * half_activity[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        cross_denominator = 1 - eigenvalues**2
        if np.any(cross_denominator <= 0):
            raise ValueError("thermal harmonic covariance exceeds the numerical range")
        factors = _ConstantQuadratic(
            (eigenvectors / (1 - eigenvalues)) @ eigenvectors.T,
            (eigenvectors / cross_denominator) @ eigenvectors.T,
            float(np.log(cross_denominator).sum()),
        )
    model = _HarmonicCharacteristic(
        energy, log_activity, linear, quadratic, log_prefactor, zero_point, dipole_factor, factors,
    )
    log_normalizer = model.log_moment(0.0)
    if not isfinite(log_normalizer) or abs(log_normalizer) > 1e-8:
        raise ValueError("the harmonic characteristic function fails thermal dipole closure at zero time")
    return model


def _signed_tail_limits(
    energy: _FloatVector,
    strength: _FloatVector,
    harmonic: _HarmonicCharacteristic,
    tolerance: float,
    sigma: float,
) -> tuple[float, float]:
    """Two-sided Chernoff limits for the normalized raw dipole measure."""
    log_weights = np.log(strength) - log(float(strength.sum()))
    # Center before the log-sum-exp to avoid cancellation at the electronic gap.
    center = float(strength @ energy / strength.sum() + harmonic.zero_point)
    centered_active = energy + harmonic.zero_point - center
    parameters = np.geomspace(1e-5, 5, 72) / np.max(np.abs(harmonic.energy))
    log_tail = log(tolerance) - log(2)
    limits = []
    for direction in (-1, 1):
        candidates = []
        for rate in parameters:
            parameter = direction * rate
            log_harmonic = harmonic.log_moment(parameter) - parameter * harmonic.zero_point
            if not isfinite(log_harmonic):
                break
            log_active = np.logaddexp.reduce(log_weights + parameter * centered_active)
            log_smoothed = log_active + log_harmonic + 0.5 * (rate * sigma)**2
            candidates.append((log_smoothed - log_tail) / rate)
        if not candidates:
            raise ValueError("could not bound the harmonic energy tails at the requested temperature")
        limits.append(center + direction * min(candidates))
    return float(limits[0]), float(limits[1])


def _finite_tail_limits(energy: _FloatVector, weight: _FloatVector, tolerance: float) -> tuple[float, float]:
    """Bound the strength outside a finite window by the given fraction."""
    if not len(energy):
        return 0.0, 0.0
    order = np.argsort(energy)
    sorted_energy = energy[order]
    sorted_weight = weight[order]
    half_tolerance = tolerance * float(weight.sum()) / 2
    first = np.searchsorted(np.cumsum(sorted_weight), half_tolerance, side="right")
    last = np.searchsorted(np.cumsum(sorted_weight[::-1]), half_tolerance, side="right")
    return float(sorted_energy[first]), float(sorted_energy[len(energy) - 1 - last])


def _active_weights(
    initial: DVRResult,
    final: DVRResult,
    dipole: _FloatVector,
    thermal_energy: float,
    sign: int,
) -> tuple[_FloatVector, _FloatVector, float, float | None]:
    initial_population = _initial_populations(initial, thermal_energy)
    _reject_split_manifold(final, "final")
    count = len(initial_population)
    amplitudes, closure = _dipole_amplitudes(
        initial.coefficients[:, :count], final.coefficients, dipole, initial_population
    )
    weight = np.einsum("ijx,ijx->ij", amplitudes, amplitudes) * initial_population[:, None]
    energy = final.energies_hartree[None, :] - initial.energies_hartree[:count, None]
    keep = weight > 0
    return (
        energy[keep] * (sign * HARTREE_TO_WAVENUMBER),
        weight[keep],
        min(float(initial_population.sum()), 1.0),
        closure,
    )


def _fft_grid(
    limits: tuple[float, float],
    step: float,
    sigma: float,
    max_grid_points: int,
) -> _FloatVector:
    # Allow the entire physical Gaussian to decay at the spectral boundaries.
    pad = 10 * sigma + 2 * step
    lower = int(np.floor((limits[0] - pad) / step))
    upper = int(np.ceil((limits[1] + pad) / step))
    count = 2 ** ((upper - lower).bit_length())
    if count > max_grid_points:
        raise ValueError(f"the requested spectrum requires {count} grid points, exceeding max_grid_points")
    first = (lower + upper - count) // 2
    return (first + np.arange(count)) * step


def _binned_active_transform(
    energy: _FloatVector,
    strength: _FloatVector,
    first: float,
    step: float,
    count: int,
) -> NDArray[np.complex128]:
    if energy.size == 1:
        time = 2 * np.pi * np.fft.rfftfreq(count, d=step)
        return strength[0] * np.exp(-1j * time * (energy[0] - first))
    # Linear deposition preserves each line's strength and first moment
    # before wraparound on the periodic FFT grid.
    positions = (energy - first) / step
    left = np.floor(positions)
    fraction = positions - left
    bins = np.remainder(left, count).astype(np.intp)
    histogram = np.bincount(bins, weights=strength * (1 - fraction), minlength=count)
    histogram += np.bincount((bins + 1) % count, weights=strength * fraction, minlength=count)
    return np.fft.rfft(histogram)


def _smoothed_weighted_bins(
    transform: NDArray[np.complex128],
    frequencies: _FloatVector,
    axis: _FloatVector,
    auxiliary_sigma: float,
    exponent: int,
) -> _FloatVector:
    # If f = G_sigma * dS, (E + sigma² d/dE) f = G_sigma * (E dS).
    # Applying the operator p times makes the smooth profile equivalent
    # to Gaussian convolution after weighting the original energies by E^p.
    count = len(axis)
    variance = auxiliary_sigma**2
    derivative = 1j * frequencies * variance
    smooth = transform * np.exp(-0.5 * (frequencies * auxiliary_sigma)**2)
    zeroth = np.fft.irfft(smooth, n=count)
    first = np.fft.irfft(derivative * smooth, n=count)
    if exponent == 1:
        return (axis * zeroth + first) / HARTREE_TO_WAVENUMBER
    second = np.fft.irfft(derivative**2 * smooth, n=count)
    third = np.fft.irfft(derivative**3 * smooth, n=count)
    return (
        (axis**3 + 3 * axis * variance) * zeroth
        + (3 * axis**2 + 3 * variance) * first
        + 3 * axis * second
        + third
    ) / HARTREE_TO_WAVENUMBER**3


def _thermal_settings(process, temperature_kelvin, fwhm_cm_inverse, step_cm_inverse, tail_tolerance, max_grid_points):
    if process not in ("absorption", "emission"):
        raise ValueError("process must be absorption or emission")
    if not isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    if not isfinite(fwhm_cm_inverse) or fwhm_cm_inverse <= 0:
        raise ValueError("fwhm_cm_inverse must be finite and positive")
    if not isfinite(tail_tolerance) or not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie strictly between zero and one")
    if isinstance(max_grid_points, bool) or not isinstance(max_grid_points, Integral) or max_grid_points < 2:
        raise ValueError("max_grid_points must be an integer of at least two")

    sigma = fwhm_cm_inverse / sqrt(8 * log(2))
    step = fwhm_cm_inverse / 80 if step_cm_inverse is None else step_cm_inverse
    if not isfinite(step) or not 0 < 2 * step < sigma:
        raise ValueError("step_cm_inverse must be positive and smaller than half the physical Gaussian sigma")
    sign = -1 if process == "emission" else 1
    return sign, _KB_HARTREE_PER_K * temperature_kelvin, sigma, step


def thermal_separable_spectrum(
    initial: DVRResult,
    final: DVRResult,
    *,
    transition_dipole_au: ArrayLike | PhasedRotorDipole,
    process: _Process,
    temperature_kelvin: float,
    fwhm_cm_inverse: float,
    spectators: HarmonicSpectators | None = None,
    step_cm_inverse: float | None = None,
    tail_tolerance: float = 1e-12,
    max_grid_points: int = 2**20,
) -> ThermalSeparableSpectrum:
    """Evaluate a separable model without enumerating harmonic transitions.

    The two DVRs must share one coordinate grid and electronic energy
    reference. Dipoles are real signed Cartesian samples on that grid, or a
    constant Cartesian vector. If either rotor surface is antiperiodic, supply
    a PhasedRotorDipole with the product of both nuclear closure signs. For
    emission supply the emitting DVR first and reverse the spectator transform.
    Harmonic spectators use all Fock states, irrespective of the quanta cutoffs
    in `HarmonicSpectators`.

    The active DVR populations use the full finite-grid partition function.
    Both reported coverages concern the retained DVR coefficients. Harmonic
    partition and final-state closure are analytical. An energy-domain
    Chernoff bound controls aliasing of the normalized signed dipole measure.

    Multiple active lines are linearly deposited. For a harmonic bath, a narrow auxiliary Gaussian
    of twice the mesh step isolates the positive-photon part of the spectrum.
    Its variance is subtracted from the remaining Gaussian broadening.
    Reduce `step_cm_inverse` to check accuracy, particularly for transitions
    close to zero photon energy. This method returns a profile without
    individual line assignments.
    """
    sign, thermal_energy, sigma, step = _thermal_settings(
        process, temperature_kelvin, fwhm_cm_inverse, step_cm_inverse, tail_tolerance, max_grid_points,
    )
    dipole = _common_dipole(initial, final, transition_dipole_au)
    harmonic = _harmonic_characteristic(spectators, thermal_energy, sign)
    energy, strength, population, closure = _active_weights(initial, final, dipole, thermal_energy, sign)
    origin = sign * HARTREE_TO_WAVENUMBER * (final.energies_hartree[0] - initial.energies_hartree[0])
    origin += harmonic.zero_point
    return _thermal_profile(
        energy, strength, harmonic, origin, population, closure, process=process,
        temperature_kelvin=temperature_kelvin, fwhm_cm_inverse=fwhm_cm_inverse, sigma=sigma, step=step,
        tail_tolerance=tail_tolerance, max_grid_points=max_grid_points,
    )


def _thermal_profile(
    energy, strength, harmonic, origin, population, closure, *, process, temperature_kelvin, fwhm_cm_inverse,
    sigma, step, tail_tolerance, max_grid_points, include_dipole_strength_density=False,
):
    auxiliary_sigma = 2 * step
    signed_strength = float(strength.sum())
    exponent = 3 if process == "emission" else 1
    negative_fraction = 0.0
    dipole_transform = None
    if not harmonic.energy.size or signed_strength == 0:
        positive = energy > 0
        energy = energy[positive]
        positive_strength = strength[positive]
        weight = positive_strength * (energy / HARTREE_TO_WAVENUMBER)**exponent
        total = float(weight.sum())
        centroid = None if total == 0 else float(weight @ energy / total)
        limits = (origin, origin) if total == 0 else _finite_tail_limits(energy, weight, tail_tolerance)
        if include_dipole_strength_density and np.any(positive_strength):
            dipole_limits = _finite_tail_limits(energy, positive_strength, tail_tolerance)
            limits = min(limits[0], dipole_limits[0]), max(limits[1], dipole_limits[1])
        axis = _fft_grid(limits, step, sigma, int(max_grid_points))
        transform = _binned_active_transform(energy, weight, axis[0], step, len(axis))
        if include_dipole_strength_density:
            dipole_transform = _binned_active_transform(energy, positive_strength, axis[0], step, len(axis))
        auxiliary_sigma = 0.0
    else:
        limits = _signed_tail_limits(energy, strength, harmonic, tail_tolerance, auxiliary_sigma)
        axis = _fft_grid(limits, step, sigma, int(max_grid_points))
        frequencies = 2 * np.pi * np.fft.rfftfreq(len(axis), d=step)
        transform = _binned_active_transform(energy, strength, axis[0], step, len(axis))
        # A spectator shift can change the sign of a photon energy. Combine
        # signed contributions before applying the positive-energy cutoff.
        transform *= harmonic.transform(frequencies)
        if include_dipole_strength_density:
            # Use the same auxiliary energy resolution and the same positive
            # photon cutoff as the photon-weighted profile.
            smooth = transform * np.exp(-0.5 * (frequencies * auxiliary_sigma)**2)
            dipole_bins = np.maximum(np.fft.irfft(smooth, n=len(axis)), 0)
            dipole_bins[axis <= 0] = 0
            dipole_transform = np.fft.rfft(dipole_bins)
        weight = _smoothed_weighted_bins(transform, frequencies, axis, auxiliary_sigma, exponent)
        weight[axis <= 0] = 0
        negative_fraction = -float(weight[weight < 0].sum()) / max(
            float(weight[weight > 0].sum()), np.finfo(float).tiny
        )
        weight = np.maximum(weight, 0)
        total = float(weight.sum())
        centroid = None if total == 0 else float(weight @ axis / total)
        transform = np.fft.rfft(weight)
    frequencies = 2 * np.pi * np.fft.rfftfreq(len(axis), d=step)
    gaussian = np.exp(-0.5 * frequencies**2 * (sigma**2 - auxiliary_sigma**2))
    # Convert bin weights to a density per cm⁻¹, preserving integrated strength.
    density = np.fft.irfft(transform * gaussian, n=len(axis)) / step
    dipole_density = None if dipole_transform is None else np.maximum(
        np.fft.irfft(dipole_transform * gaussian, n=len(axis)) / step, 0,
    )
    return ThermalSeparableSpectrum(
        axis, np.maximum(density, 0), fwhm_cm_inverse, step, temperature_kelvin, process,
        population, closure, signed_strength, total, centroid, origin, tail_tolerance, auxiliary_sigma, negative_fraction,
        dipole_density,
    )


def thermal_harmonic_spectrum(
    initial_frequencies_hartree: ArrayLike,
    final_frequencies_hartree: ArrayLike,
    *,
    gap_00_hartree: float,
    transition_dipole_au: ArrayLike,
    dipole_derivatives_au: ArrayLike | None = None,
    rotation: ArrayLike | None = None,
    shift_au: ArrayLike | None = None,
    process: _Process,
    temperature_kelvin: float,
    fwhm_cm_inverse: float,
    step_cm_inverse: float | None = None,
    tail_tolerance: float = 1e-12,
    max_grid_points: int = 2**20,
    include_dipole_strength_density: bool = False,
) -> ThermalSeparableSpectrum:
    """Evaluate an all-quanta FC/HT harmonic profile on a photon-energy grid.

    `gap_00_hartree` is the signed final-minus-initial energy difference
    including zero-point energy. For emission the signed gap is negative.
    Final mass-weighted coordinates obey Q_f = rotation @ Q_i + shift_au,
    with Q in sqrt(electron mass) * bohr. The real or complex Cartesian dipole and its
    optional (n_modes, 3) gradient are taken at Q_i=0. Franck–Condon and
    Herzberg–Teller contributions interfere at the amplitude level.

    The characteristic transform and thermal closure include all harmonic
    states. The energy mesh controls the auxiliary Gaussian cutoff at zero
    photon energy. The returned profile includes the positive-energy dipole
    weight E for absorption and E³ for emission, with E in Hartree. Request
    the separate dipole-strength density for comparisons to experimental
    spectra reduced by the photon-energy factor before broadening.
    """
    sign, thermal_energy, sigma, step = _thermal_settings(
        process, temperature_kelvin, fwhm_cm_inverse, step_cm_inverse, tail_tolerance, max_grid_points,
    )
    if not isfinite(gap_00_hartree):
        raise ValueError("gap_00_hartree must be finite")
    dipole = _amplitude_array(transition_dipole_au, "transition_dipole_au")
    if dipole.shape != (3,):
        raise ValueError("transition_dipole_au must have three Cartesian components")
    derivatives = None
    if dipole_derivatives_au is not None:
        derivatives = _amplitude_array(dipole_derivatives_au, "dipole_derivatives_au")
        if derivatives.shape != (np.size(initial_frequencies_hartree), 3):
            raise ValueError("dipole_derivatives_au must have shape (n_modes, 3)")
    if np.iscomplexobj(dipole) or np.iscomplexobj(derivatives):
        # FC and coordinate overlaps are real. The squared complex moment
        # is the sum of independent real and imaginary Cartesian strengths.
        # The Fourier phase in each generating function is left intact.
        dipole = np.concatenate((dipole.real, dipole.imag))
        if derivatives is not None:
            derivatives = np.concatenate((derivatives.real, derivatives.imag), axis=1)
    bath = HarmonicSpectators(initial_frequencies_hartree, final_frequencies_hartree, 0, 0, rotation, shift_au)
    harmonic = _harmonic_characteristic(bath, thermal_energy, sign, dipole=dipole, derivatives=derivatives)
    origin = sign * HARTREE_TO_WAVENUMBER * gap_00_hartree
    strength = float(dipole @ dipole) if harmonic.dipole is None else harmonic.dipole.strength
    if not isfinite(strength):
        raise ValueError("thermal dipole closure exceeds the numerical range")
    return _thermal_profile(
        np.array([origin - harmonic.zero_point]), np.array([strength]), harmonic, origin, 1.0,
        1.0 if strength > 0 else None, process=process, temperature_kelvin=temperature_kelvin,
        fwhm_cm_inverse=fwhm_cm_inverse, sigma=sigma, step=step, tail_tolerance=tail_tolerance,
        max_grid_points=max_grid_points, include_dipole_strength_density=include_dipole_strength_density,
    )
