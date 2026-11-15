"""Assigned sticks for a separable one-coordinate anharmonic model."""

from dataclasses import dataclass
from math import comb, isfinite
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .anharmonic_dvr import DVRResult
from .harmonic import _KB_HARTREE_PER_K, _fc_overlaps, _fock_states, _real_array


@dataclass(frozen=True)
class HarmonicSpectators:
    """Harmonic modes separable from the active coordinate on both surfaces.

    Frequencies are vibrational quanta in Hartree. Final mass-weighted spectator
    coordinates obey Q_f = rotation @ Q_i + shift_au, where Q has units
    sqrt(electron mass) * bohr. Each cutoff bounds the total spectator quanta.
    """

    initial_frequencies_hartree: ArrayLike
    final_frequencies_hartree: ArrayLike
    initial_max_quanta: int
    final_max_quanta: int
    rotation: ArrayLike | None = None
    shift_au: ArrayLike | None = None


@dataclass(frozen=True)
class AnharmonicSpectrum:
    """Assigned sticks with zero-based DVR levels and spectator occupations."""

    photon_energy_hartree: NDArray[np.float64]
    initial_level: NDArray[np.intp]
    final_level: NDArray[np.intp]
    initial_spectator_quanta: NDArray[np.intp]
    final_spectator_quanta: NDArray[np.intp]
    population: NDArray[np.float64]
    dipole_strength_au2: NDArray[np.float64]
    initial_population_coverage: float
    final_dipole_strength_coverage: float | None

    @property
    def thermal_strength_au2(self) -> NDArray[np.float64]:
        return self.population * self.dipole_strength_au2


@dataclass(frozen=True)
class _SpectatorFactors:
    initial_quanta: NDArray[np.intp]
    final_quanta: NDArray[np.intp]
    population: NDArray[np.float64]
    energy_difference: NDArray[np.float64]
    strength: NDArray[np.float64]
    closure: float


def _spectator_factors(
    spectators: HarmonicSpectators | None,
    thermal_energy: float,
    max_pairs: int,
) -> _SpectatorFactors:
    if max_pairs < 1:
        raise ValueError("the requested basis exceeds max_transitions")
    if spectators is None:
        empty_quanta = np.empty((1, 0), dtype=np.intp)
        return _SpectatorFactors(empty_quanta, empty_quanta, np.ones(1),
                                 np.zeros((1, 1)), np.ones((1, 1)), 1.0)

    omega_i = _real_array(spectators.initial_frequencies_hartree, "initial spectator frequencies")
    omega_f = _real_array(spectators.final_frequencies_hartree, "final spectator frequencies")
    if omega_i.ndim != 1 or omega_f.shape != omega_i.shape:
        raise ValueError("initial and final spectator frequencies must be equal-length vectors")
    if np.any(omega_i <= 0) or np.any(omega_f <= 0):
        raise ValueError("spectator frequencies must be strictly positive")
    for name, cutoff in (("initial_max_quanta", spectators.initial_max_quanta),
                         ("final_max_quanta", spectators.final_max_quanta)):
        if isinstance(cutoff, bool) or not isinstance(cutoff, Integral) or cutoff < 0:
            raise ValueError(f"spectator {name} must be a nonnegative integer")

    modes = omega_i.size
    rotation = np.eye(modes) if spectators.rotation is None else _real_array(
        spectators.rotation, "spectator rotation"
    )
    if rotation.shape != (modes, modes) or not np.allclose(
        rotation.T @ rotation, np.eye(modes), atol=1e-8, rtol=0
    ):
        raise ValueError("spectator rotation must be an orthogonal square Duschinsky matrix")
    shift = np.zeros(modes) if spectators.shift_au is None else _real_array(
        spectators.shift_au, "spectator shift_au"
    )
    if shift.shape != (modes,):
        raise ValueError("spectator shift_au must have one entry per spectator mode")

    initial_cutoff = int(spectators.initial_max_quanta) if thermal_energy and modes else 0
    final_cutoff = int(spectators.final_max_quanta) if modes else 0
    if comb(modes + initial_cutoff, modes) * comb(modes + final_cutoff, modes) > max_pairs:
        raise ValueError("the requested basis exceeds max_transitions")
    initial_quanta = _fock_states(modes, initial_cutoff)
    final_quanta = _fock_states(modes, final_cutoff)
    overlap = _fc_overlaps(omega_i, omega_f, rotation, shift, initial_quanta, final_quanta)
    strength = overlap**2
    retained_strength = strength.sum(axis=1)
    if not np.all(np.isfinite(retained_strength)) or np.any(retained_strength > 1 + 1e-8):
        raise ValueError("spectator Franck–Condon strengths violate the closure bound")

    initial_energy = initial_quanta @ omega_i
    final_energy = final_quanta @ omega_f
    zpe_difference = 0.5 * (omega_f - omega_i).sum()
    energy_difference = zpe_difference + final_energy[None, :] - initial_energy[:, None]
    if thermal_energy == 0:
        weights = population = np.ones(1)
    else:
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            log_weights = -initial_energy / thermal_energy
            weights = np.exp(log_weights)
            log_inverse_partition = np.log(-np.expm1(-omega_i / thermal_energy)).sum()
            population = np.exp(log_weights + log_inverse_partition)
    closure = float(weights @ retained_strength / weights.sum())
    return _SpectatorFactors(initial_quanta, final_quanta, population, energy_difference,
                             strength, min(closure, 1.0))


def _degeneracy_tolerance(energies: NDArray[np.float64]) -> float:
    # Eigensolver roundoff is controlled by the scale of the full matrix,
    # which can be appreciably larger than the retained low-lying energies.
    energy_scale = max(abs(energies[0]), abs(energies[-1]))
    return max(1e-12, 32 * np.finfo(float).eps * energy_scale)


def _reject_split_manifold(surface: DVRResult, name: str) -> None:
    all_energies = surface.all_energies_hartree
    retained_count = len(surface.energies_hartree)
    if retained_count < len(all_energies):
        last, next_energy = all_energies[retained_count - 1 : retained_count + 1]
        if next_energy - last <= _degeneracy_tolerance(all_energies):
            raise ValueError(f"{name} DVR cutoff splits a degenerate manifold")


def _initial_populations(initial: DVRResult, thermal_energy: float) -> NDArray[np.float64]:
    all_energies = initial.all_energies_hartree
    retained_count = len(initial.energies_hartree)
    with np.errstate(over="ignore"):
        excitation_energies = all_energies - all_energies[0]
    if thermal_energy == 0:
        ground_count = np.count_nonzero(excitation_energies <= _degeneracy_tolerance(all_energies))
        if ground_count > retained_count:
            raise ValueError("initial DVR cutoff splits the degenerate ground manifold")
        return np.full(ground_count, 1.0 / ground_count)

    _reject_split_manifold(initial, "initial")
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-excitation_energies / thermal_energy)
    return weights[:retained_count] / weights.sum()


def _dipole_strengths(
    initial: NDArray[np.float64],
    final: NDArray[np.float64],
    dipole: NDArray[np.float64],
    populations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float | None]:
    for coefficients in (initial, final):
        gram = coefficients.T @ coefficients
        if not np.allclose(gram, np.eye(coefficients.shape[1]), atol=1e-8, rtol=0):
            raise ValueError("DVR coefficients must be orthonormal in the common quadrature metric")

    # The square root of the quadrature weight is already included in C.
    # Each component is C_f.T @ diag(mu) @ C_i.
    strength = np.zeros((initial.shape[1], final.shape[1]))
    total_strength = np.zeros(initial.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        for component in dipole.T:
            weighted_initial = component[:, None] * initial
            strength += (weighted_initial.T @ final)**2
            total_strength += np.einsum("ni,ni->i", weighted_initial, weighted_initial)

        denominator = float(populations @ total_strength)
        numerator = float(populations @ strength.sum(axis=1))
    if not np.all(np.isfinite(total_strength)) or not isfinite(denominator) or not isfinite(numerator):
        raise ValueError("DVR dipole strengths exceed the numerical range")
    if denominator == 0:
        return strength, None
    closure = numerator / denominator
    if closure > 1 + 1e-8:
        raise ValueError("DVR dipole strengths violate the closure bound")
    return strength, min(closure, 1.0)


def anharmonic_spectrum(
    initial: DVRResult,
    final: DVRResult,
    *,
    transition_dipole_au: ArrayLike,
    temperature_kelvin: float = 0.0,
    process: Literal["absorption", "emission"] = "absorption",
    spectators: HarmonicSpectators | None = None,
    max_transitions: int = 250_000,
) -> AnharmonicSpectrum:
    """Combine DVR levels on the same coordinate grid and energy reference.

    The potentials used to obtain `initial` and `final` share one electronic
    energy zero. The final minus initial energy includes the active eigenvalue
    difference and (if present) spectator excitation and zero-point differences.
    Pass the emitting surface as `initial` with `process="emission"` to reverse
    the sign. Only strictly positive photon energies are returned. Real dipoles
    use a common Cartesian frame and a consistent electronic phase throughout
    the scan. Supply a constant (3,) vector or samples of shape (n_grid, 3).
    This model permits rotations within the harmonic spectator coordinates.

    Populations use the full finite-grid DVR partition function and the exact
    harmonic spectator partition function. At zero temperature the initial
    ground manifold has equal populations. Degeneracy is checked at a tolerance
    of 1e-12 Hartree or 32 floating-point epsilons times the largest absolute
    energy on each finite grid, whichever is larger. Cutoffs cannot split the
    zero-temperature initial ground manifold, any thermal initial boundary
    manifold, or a final boundary manifold. Individual strengths within
    degenerate blocks depend on the chosen eigenvectors.

    Dipole-strength coverage compares retained final levels with dipole closure
    for the retained, populated initial levels before filtering by photon energy.
    It is None if the total dipole strength vanishes. Initial coverage and final
    active-mode closure refer to the given finite grid. Strengths omit
    photon-energy prefactors and broadening. `max_transitions` bounds the number
    of active-times-spectator pairs considered before the photon-energy filter.
    """
    if not isinstance(initial, DVRResult) or not isinstance(final, DVRResult):
        raise ValueError("initial and final must be DVRResult instances")
    if (initial.boundary != final.boundary or initial.domain != final.domain
            or not np.array_equal(initial.grid, final.grid)):
        raise ValueError("initial and final DVR states must use the same grid, domain and boundary")
    if not isfinite(temperature_kelvin) or temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    if process not in ("absorption", "emission"):
        raise ValueError("process must be absorption or emission")
    if spectators is not None and not isinstance(spectators, HarmonicSpectators):
        raise ValueError("spectators must be a HarmonicSpectators instance")
    if (isinstance(max_transitions, bool) or not isinstance(max_transitions, Integral)
            or max_transitions < 1):
        raise ValueError("max_transitions must be a positive integer")

    dipole = _real_array(transition_dipole_au, "transition_dipole_au")
    if dipole.shape == (3,):
        dipole = np.broadcast_to(dipole, (len(initial.grid), 3))
    if dipole.shape != (len(initial.grid), 3):
        raise ValueError("transition_dipole_au must have shape (3,) or (n_grid, 3)")

    thermal_energy = _KB_HARTREE_PER_K * temperature_kelvin
    initial_population = _initial_populations(initial, thermal_energy)
    _reject_split_manifold(final, "final")
    n_initial = len(initial_population)
    active_pair_count = n_initial * len(final.energies_hartree)
    bath = _spectator_factors(spectators, thermal_energy, int(max_transitions) // active_pair_count)
    active_strength, active_closure = _dipole_strengths(
        initial.coefficients[:, :n_initial], final.coefficients, dipole, initial_population
    )

    active_difference = final.energies_hartree[None, :] - initial.energies_hartree[:n_initial, None]
    photon_energy = active_difference[:, :, None, None] + bath.energy_difference[None, None, :, :]
    if process == "emission":
        photon_energy *= -1
    if not np.all(np.isfinite(photon_energy)):
        raise ValueError("photon energies exceed the numerical range")
    population = initial_population[:, None] * bath.population[None, :]
    i, f, bi, bf = np.nonzero((photon_energy > 0) & (population[:, None, :, None] > 0))
    order = np.argsort(photon_energy[i, f, bi, bf], kind="stable")
    i, f, bi, bf = i[order], f[order], bi[order], bf[order]
    return AnharmonicSpectrum(
        photon_energy_hartree=photon_energy[i, f, bi, bf],
        initial_level=i,
        final_level=f,
        initial_spectator_quanta=bath.initial_quanta[bi],
        final_spectator_quanta=bath.final_quanta[bf],
        population=population[i, bi],
        dipole_strength_au2=active_strength[i, f] * bath.strength[bi, bf],
        initial_population_coverage=min(float(initial_population.sum() * bath.population.sum()), 1.0),
        final_dipole_strength_coverage=None if active_closure is None else active_closure * bath.closure,
    )
