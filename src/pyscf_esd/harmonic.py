"""Assigned harmonic Franck–Condon and linear Herzberg–Teller sticks."""

from dataclasses import dataclass
from math import comb, isfinite
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic_moments import HarmonicPhotonMoment, harmonic_photon_moments

_KB_HARTREE_PER_K = 3.166811563e-6


@dataclass(frozen=True)
class HarmonicSpectrum:
    """Assigned sticks and basis coverage from a harmonic calculation.

    ``transition_dipole_au`` holds the Cartesian x, y, z amplitudes for
    each returned stick, in the same order as the energies and quanta.
    Amplitudes are complex when the dipole or its derivatives are complex.
    The Condon contribution, when decomposed, is the amplitude from the
    constant dipole in the initial surface's coordinate expansion.
    ``positive_photon_moments`` maps photon-energy powers to retained moments
    and omitted-moment bounds, independently of the dipole closure.
    """

    photon_energy_hartree: NDArray[np.float64]
    initial_quanta: NDArray[np.intp]
    final_quanta: NDArray[np.intp]
    population: NDArray[np.float64]
    dipole_strength_au2: NDArray[np.float64]
    initial_population_coverage: float
    final_dipole_strength_coverage: float | None
    transition_dipole_au: NDArray[np.float64] | NDArray[np.complex128] | None = None
    condon_transition_dipole_au: NDArray[np.float64] | NDArray[np.complex128] | None = None
    positive_photon_moments: dict[int, HarmonicPhotonMoment] | None = None

    @property
    def thermal_strength_au2(self) -> NDArray[np.float64]:
        return self.population * self.dipole_strength_au2


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _amplitude_array(value: ArrayLike, name: str) -> NDArray[np.float64] | NDArray[np.complex128]:
    array = np.asarray(value)
    array = np.asarray(array, dtype=complex if np.iscomplexobj(array) else float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _squared_magnitude(values):
    if np.iscomplexobj(values):
        return values.real**2 + values.imag**2
    return values**2


def _fock_states(n_modes: int, max_quanta: int) -> NDArray[np.intp]:
    # Build each layer from the preceding one. Every state's parents then
    # precede it, including when the number of modes is larger than the cutoff.
    states = [(0,) * n_modes]
    previous_layer = [(states[0], 0)]
    for _ in range(max_quanta):
        next_layer = []
        for state, first_mode in previous_layer:
            for mode in range(first_mode, n_modes):
                child = state[:mode] + (state[mode] + 1,) + state[mode + 1 :]
                states.append(child)
                next_layer.append((child, mode))
        previous_layer = next_layer
    return np.array(states, dtype=np.intp)


def _fc_parameters(
    omega_i: NDArray[np.float64],
    omega_f: NDArray[np.float64],
    rotation: NDArray[np.float64],
    shift: NDArray[np.float64],
):
    """Combine the two ground-state Gaussians in the initial coordinates."""
    modes = omega_i.size
    precision = np.diag(omega_i) + rotation.T @ (omega_f[:, None] * rotation)
    cholesky = np.linalg.cholesky(precision)
    mean = -np.linalg.solve(precision, rotation.T @ (omega_f * shift))
    final_mean = rotation @ mean + shift
    log_overlap = (
        modes / 2 * np.log(2)
        + np.sum(np.log(omega_i) + np.log(omega_f)) / 4
        - np.log(np.diag(cholesky)).sum()
        - (omega_i @ mean**2 + omega_f @ final_mean**2) / 2
    )
    if log_overlap < np.log(np.finfo(float).tiny):
        raise ValueError("ground-state overlap underflows in the harmonic recurrence")

    # The generating variables t are ordered as initial then final modes.
    # Multiply the monomial coefficient of exp(l.T @ t + t.T @ Q @ t / 2)
    # by the ground-state overlap and sqrt(product(n_j!)) to obtain the FC amplitude.
    scale_i = np.sqrt(2 * omega_i)
    scale_f = np.sqrt(2 * omega_f)
    sources = np.vstack((np.diag(scale_i), scale_f[:, None] * rotation))
    linear = np.concatenate((scale_i * mean, scale_f * final_mean))
    transformed_sources = np.linalg.solve(cholesky, sources.T)
    quadratic = transformed_sources.T @ transformed_sources - np.eye(2 * modes)
    return np.exp(log_overlap), linear, quadratic


def _fc_overlaps(
    omega_i: NDArray[np.float64],
    omega_f: NDArray[np.float64],
    rotation: NDArray[np.float64],
    shift: NDArray[np.float64],
    initial_states: NDArray[np.intp],
    final_states: NDArray[np.intp],
) -> NDArray[np.float64]:
    modes = omega_i.size
    origin, linear, quadratic = _fc_parameters(omega_i, omega_f, rotation, shift)
    overlaps = np.empty((len(initial_states), len(final_states)))
    overlaps[0, 0] = origin
    initial_indices = {tuple(state): idx for idx, state in enumerate(initial_states)}
    final_indices = {tuple(state): idx for idx, state in enumerate(final_states)}
    for i, initial in enumerate(initial_states):
        for f, final in enumerate(final_states):
            state = np.concatenate((initial, final))
            occupied = state.nonzero()[0]
            if occupied.size == 0:
                continue
            mode = occupied[0]
            state[mode] -= 1
            value = linear[mode] * overlaps[
                initial_indices[tuple(state[:modes])], final_indices[tuple(state[modes:])]
            ]
            for other in state.nonzero()[0]:
                factor = quadratic[mode, other] * np.sqrt(state[other])
                state[other] -= 1
                value += factor * overlaps[
                    initial_indices[tuple(state[:modes])], final_indices[tuple(state[modes:])]
                ]
                state[other] += 1
            overlaps[i, f] = value / np.sqrt(state[mode] + 1)
    return overlaps


def _ground_amplitudes(omega_i, omega_f, transform, shift, states, dipole, derivatives):
    """Contract linear HT into the final-state recurrence at zero kelvin."""
    modes = omega_i.size
    origin, linear, quadratic = _fc_parameters(omega_i, omega_f, transform, shift)
    overlaps = np.empty(len(states))
    overlaps[0] = origin
    total = float(np.vdot(dipole, dipole).real)
    if derivatives is None:
        constant = dipole
        slope = None
    else:
        # Q_i on the initial vacuum creates only one quantum in mode i.
        # Differentiate the generating function once in that mode at t_i=0.
        scaled = derivatives / np.sqrt(2 * omega_i[:, None])
        constant = dipole + linear[:modes] @ scaled
        slope = quadratic[modes:, :modes] @ scaled
        total += float(np.sum(_squared_magnitude(scaled)))
    final_linear = linear[modes:]
    final_quadratic = quadratic[modes:, modes:]
    indices = {tuple(state): index for index, state in enumerate(states)}
    amplitudes = np.empty((len(states), 3), dtype=dipole.dtype)
    amplitudes[0] = origin * constant
    for index, state in enumerate(states[1:], start=1):
        work = state.copy()
        mode = int(np.flatnonzero(work)[0])
        work[mode] -= 1
        value = final_linear[mode] * overlaps[indices[tuple(work)]]
        for other in np.flatnonzero(work):
            factor = final_quadratic[mode, other] * np.sqrt(work[other])
            work[other] -= 1
            value += factor * overlaps[indices[tuple(work)]]
            work[other] += 1
        overlaps[index] = value / np.sqrt(state[mode])
        amplitudes[index] = overlaps[index] * constant
        if slope is not None:
            work[mode] += 1
            for other in np.flatnonzero(state):
                work[other] -= 1
                amplitudes[index] += slope[other] * np.sqrt(state[other]) * overlaps[indices[tuple(work)]]
                work[other] += 1
    return amplitudes[None, :, :], np.array([total]), overlaps[None, :]


def harmonic_spectrum(
    initial_frequencies_hartree: ArrayLike,
    final_frequencies_hartree: ArrayLike,
    *,
    gap_00_hartree: float,
    transition_dipole_au: ArrayLike,
    initial_max_quanta: int,
    final_max_quanta: int,
    rotation: ArrayLike | None = None,
    shift_au: ArrayLike | None = None,
    dipole_derivatives_au: ArrayLike | None = None,
    temperature_kelvin: float = 0.0,
    process: Literal["absorption", "emission"] = "absorption",
    max_overlaps: int = 250_000,
) -> HarmonicSpectrum:
    """Compute line strengths for a pair of stable harmonic surfaces.

    Frequencies are vibrational energy quanta. The signed gap is the final
    electronic minima energy minus the initial, plus the same difference in
    vibrational zero-point energy.
    Final mass-weighted coordinates obey Q_f = rotation @ Q_i + shift_au,
    with Q in sqrt(electron mass) * bohr. The Cartesian dipole is given at
    Q_i=0. Its optional (n_modes, 3) first derivatives are with respect to
    Q_i, in atomic units. The dipole and derivatives may be real or complex
    and use the same electronic phase convention. Condon and linear HT
    amplitudes add before taking the squared Cartesian norm. Cutoffs bound
    total quanta on each surface.

    Results include all positive-photon-energy pairs with populated initial
    states, including zero-strength pairs. Strengths exclude photon-energy
    factors, radiative prefactors, and broadening. Coverage diagnostics use
    all retained pairs, even those omitted by the photon-energy filter.
    Separate positive-photon diagnostics bound omitted M1 for absorption
    and omitted M3 and M4 for emission.
    """
    omega_i = _real_array(initial_frequencies_hartree, "initial frequencies")
    omega_f = _real_array(final_frequencies_hartree, "final frequencies")
    if omega_i.ndim != 1 or omega_f.shape != omega_i.shape:
        raise ValueError("initial and final frequencies must be equal-length vectors")
    if np.any(omega_i <= 0) or np.any(omega_f <= 0):
        raise ValueError("harmonic frequencies must be strictly positive")
    if not isfinite(gap_00_hartree):
        raise ValueError("gap_00_hartree must be finite")
    if not isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    if process not in ("absorption", "emission"):
        raise ValueError("process must be absorption or emission")
    for name, value, minimum in (
        ("initial_max_quanta", initial_max_quanta, 0),
        ("final_max_quanta", final_max_quanta, 0),
        ("max_overlaps", max_overlaps, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    modes = omega_i.size
    transform = np.eye(modes) if rotation is None else _real_array(rotation, "rotation")
    if transform.shape != (modes, modes) or not np.allclose(
        transform.T @ transform, np.eye(modes), atol=1e-8, rtol=0
    ):
        raise ValueError("rotation must be an orthogonal square Duschinsky matrix")
    shift = np.zeros(modes) if shift_au is None else _real_array(shift_au, "shift_au")
    if shift.shape != (modes,):
        raise ValueError("shift_au must have one entry per mode")
    dipole = _amplitude_array(transition_dipole_au, "transition_dipole_au")
    if dipole.shape != (3,):
        raise ValueError("transition_dipole_au must have three Cartesian components")
    derivatives = None
    if dipole_derivatives_au is not None:
        derivatives = _amplitude_array(dipole_derivatives_au, "dipole_derivatives_au")
        if derivatives.shape != (modes, 3):
            raise ValueError("dipole_derivatives_au must have shape (n_modes, 3)")
        dipole = dipole.astype(np.result_type(dipole, derivatives), copy=False)

    initial_cutoff = int(initial_max_quanta) if temperature_kelvin > 0 and modes else 0
    final_cutoff = int(final_max_quanta) if modes else 0
    initial_count = comb(modes + initial_cutoff, modes)
    # The linear dipole can raise the initial state by one quantum, so
    # the overlap basis needs a layer beyond the populated basis.
    overlap_cutoff = initial_cutoff + int(derivatives is not None and modes > 0 and temperature_kelvin > 0)
    if comb(modes + overlap_cutoff, modes) * comb(modes + final_cutoff, modes) > max_overlaps:
        raise ValueError("the requested basis exceeds max_overlaps")
    initial_states = _fock_states(modes, overlap_cutoff)
    final_states = _fock_states(modes, final_cutoff)
    retained_initial = initial_states[:initial_count]
    initial_energies = retained_initial @ omega_i
    mean_occupation = np.zeros_like(omega_i)
    if temperature_kelvin == 0.0:
        populations = np.ones(1)
        amplitudes, total_strength, overlaps = _ground_amplitudes(
            omega_i, omega_f, transform, shift, final_states, dipole, derivatives,
        )
    else:
        overlaps = _fc_overlaps(omega_i, omega_f, transform, shift, initial_states, final_states)
        amplitudes = overlaps[:initial_count, :, None] * dipole
        total_strength = np.full(initial_count, np.vdot(dipole, dipole).real)
        thermal_energy = _KB_HARTREE_PER_K * temperature_kelvin
        if thermal_energy == 0:
            populations = (initial_energies == 0).astype(float)
        else:
            with np.errstate(over="ignore", under="ignore"):
                # Use the untruncated oscillator partition so omitted hot
                # states reduce the reported population coverage.
                beta_omega = omega_i / thermal_energy
                inverse_partition = -np.expm1(-beta_omega)
                mean_occupation = np.exp(-beta_omega) / inverse_partition
                log_inverse_partition = np.log(inverse_partition).sum()
                populations = np.exp(log_inverse_partition - initial_energies / thermal_energy)

    if derivatives is not None and temperature_kelvin > 0:
        initial_indices = {tuple(state): idx for idx, state in enumerate(initial_states)}
        scaled_derivatives = derivatives / np.sqrt(2 * omega_i[:, None])
        # Complete final-state closure equals the initial expectation of |mu(Q)|².
        # Number states have zero mean Q and variance (2*n + 1)/(2*omega).
        total_strength += (2 * retained_initial + 1) @ np.sum(_squared_magnitude(scaled_derivatives), axis=1)
        for i, state in enumerate(retained_initial):
            neighbor = state.copy()
            for mode, quanta in enumerate(state):
                neighbor[mode] += 1
                coordinate_overlap = np.sqrt(quanta + 1) * overlaps[initial_indices[tuple(neighbor)]]
                if quanta:
                    neighbor[mode] -= 2
                    coordinate_overlap += np.sqrt(quanta) * overlaps[initial_indices[tuple(neighbor)]]
                neighbor[mode] = quanta
                amplitudes[i] += coordinate_overlap[:, None] * scaled_derivatives[mode]

    strengths = np.sum(_squared_magnitude(amplitudes), axis=-1)
    if not np.all(np.isfinite(strengths)) or not np.all(np.isfinite(total_strength)):
        raise ValueError("harmonic dipole strengths exceed the numerical range")
    denominator = float(populations @ total_strength)
    coverage = float(populations @ strengths.sum(axis=1) / denominator) if denominator else None
    photon_energies = gap_00_hartree + (final_states @ omega_f)[None, :] - initial_energies[:, None]
    if process == "emission":
        photon_energies *= -1
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        photon_moments = harmonic_photon_moments(
            omega_i, omega_f, transform, shift, gap_00_hartree, dipole, derivatives, process,
            retained_initial, initial_energies, populations, final_cutoff, strengths, total_strength,
            photon_energies, mean_occupation,
        )
    i, f = np.nonzero((photon_energies > 0) & (populations[:, None] > 0))
    order = np.argsort(photon_energies[i, f], kind="stable")
    i, f = i[order], f[order]
    return HarmonicSpectrum(
        photon_energy_hartree=photon_energies[i, f],
        initial_quanta=retained_initial[i],
        final_quanta=final_states[f],
        population=populations[i],
        dipole_strength_au2=strengths[i, f],
        initial_population_coverage=float(populations.sum()),
        final_dipole_strength_coverage=coverage,
        transition_dipole_au=amplitudes[i, f],
        condon_transition_dipole_au=overlaps[i, f, None] * dipole if derivatives is not None else None,
        positive_photon_moments=photon_moments,
    )
