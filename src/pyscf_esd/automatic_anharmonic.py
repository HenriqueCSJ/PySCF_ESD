"""Automatic local anharmonic screening and vibrational configuration interaction.

Energy cuts use dimensionless mass-weighted normal coordinates, x = sqrt(omega) Q.
Analytic Hessians fix the quadratic potentials. Regression uses direct energies
to estimate cubic and quartic corrections in one-mode and selected mixed cuts.
Independent mixed points check the truncated potential where it was sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush, nsmallest
from itertools import combinations, product
from math import comb
from numbers import Integral
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .anharmonic import _degeneracy_tolerance
from .cpu import ElectronicConvergenceError, ReferenceStabilityUnresolvedError, UnstableReferenceError
from .harmonic import HarmonicSpectrum, _KB_HARTREE_PER_K, _fc_overlaps, _fock_states
from .pyscf_adapter import AdiabaticHessianParameters
from .state_tracking import StateTrackingError

if TYPE_CHECKING:
    from .automatic_1d import DirectOneMode


_Real = NDArray[np.float64]
_StateList = NDArray[np.intp]
_HARTREE_TO_WAVENUMBER = 219474.6313705
_POTENTIAL_EPSILON = 3.0 / _HARTREE_TO_WAVENUMBER
_LEVEL_TOLERANCE = 2.0 / _HARTREE_TO_WAVENUMBER
_MIN_STATE_OVERLAP_SQUARED = 0.998
_MAX_PROBED_PAIRS = 15
_MAX_PROBED_TRIPLES = 4
_MAX_VCI_STATES = 1100
_MAX_VCI_PADDING = 7
_AXIS_RADII = (0.8, 1.8, 2.6, 3.5)
_AXIS_POINTS = np.array(sorted(x * sign for x in _AXIS_RADII for sign in (-1, 1)))
_AXIS_TEST = np.isin(np.abs(_AXIS_POINTS), (0.8, 3.5))
_PAIR_POWERS = ((2, 1), (1, 2), (3, 1), (2, 2), (1, 3))
_TRIPLE_POWERS = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))


class _ElectronicSurfaces(Protocol):
    ground_coordinates_bohr: _Real
    excited_coordinates_bohr: _Real
    model: AdiabaticHessianParameters
    transition_dipole_au: _Real
    dipole_derivatives_au: _Real | None

    def evaluate(self, coordinates_bohr: _Real) -> tuple[float, float, _Real]: ...


class AnharmonicUnavailableError(ValueError):
    """Automatic screening or a requested corrected spectrum is inconclusive."""


@dataclass(frozen=True)
class _PotentialTerm:
    modes: tuple[int, ...]
    powers: tuple[int, ...]
    coefficient_hartree: float

    def value(self, coordinates: _Real) -> float:
        return self.coefficient_hartree * float(np.prod(coordinates[list(self.modes)] ** self.powers))


@dataclass(frozen=True)
class _LocalSurface:
    omega: _Real
    terms: tuple[_PotentialTerm, ...]


@dataclass(frozen=True)
class _VibrationalStates:
    harmonic_basis: _StateList
    labels: _StateList
    energies_hartree: _Real
    eigenvectors: _Real
    assignment_overlaps_squared: _Real | None = None
    population: _Real | None = None


@dataclass(frozen=True)
class AutomaticAnharmonic:
    """Prepared local models and evidence from direct electronic energy cuts."""

    electronic: _ElectronicSurfaces
    diagnostics: dict
    _surfaces: tuple[_LocalSurface, _LocalSurface] | None
    _direct_one_mode: DirectOneMode | None = None

    def spectrum(
        self,
        *,
        process: str = "absorption",
        temperature_kelvin: float = 0.0,
        initial_max_quanta: int = 0,
        final_max_quanta: int = 5,
        max_overlaps: int = 250_000,
    ) -> HarmonicSpectrum:
        """Compute sticks and report VCI convergence for the requested state range.

        Quanta identify the harmonic states with the best joint match to the
        anharmonic states. Near resonances they are assignment labels, while the
        amplitudes use the complete retained VCI eigenvectors.
        """
        for name, value, minimum in (
            ("initial_max_quanta", initial_max_quanta, 0),
            ("final_max_quanta", final_max_quanta, 0),
            ("max_overlaps", max_overlaps, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if process not in ("absorption", "emission"):
            raise ValueError("process must be absorption or emission")
        if not np.isfinite(temperature_kelvin) or temperature_kelvin < 0:
            raise ValueError("temperature_kelvin must be finite and nonnegative")
        if self._direct_one_mode is not None:
            return self._direct_one_mode.spectrum(
                process=process, temperature=temperature_kelvin,
                initial_cutoff=initial_max_quanta, final_cutoff=final_max_quanta,
                max_overlaps=max_overlaps,
            )
        if self._surfaces is None:
            raise AnharmonicUnavailableError("; ".join(self.diagnostics["reasons"]))

        electronic = self.electronic
        model = electronic.model
        mu = electronic.transition_dipole_au
        derivatives = electronic.dipole_derivatives_au
        if self.diagnostics["status"] == "harmonic_screened":
            return model.spectrum(
                transition_dipole_au=mu, dipole_derivatives_au=derivatives,
                process=process, temperature_kelvin=temperature_kelvin,
                initial_max_quanta=initial_max_quanta, final_max_quanta=final_max_quanta,
                max_overlaps=max_overlaps,
            )

        initial_surface, final_surface = self._surfaces
        rotation, shift, gap = model.rotation, model.shift_au, model.gap_00_hartree
        if process == "emission":
            initial_surface, final_surface = final_surface, initial_surface
            rotation = model.rotation.T
            shift = -(rotation @ model.shift_au)
            gap = -gap
            if derivatives is not None:
                mu = mu + shift @ derivatives
                derivatives = model.rotation @ derivatives

        initial_cutoff = int(initial_max_quanta) if temperature_kelvin else 0
        thermal_energy = _KB_HARTREE_PER_K * temperature_kelvin
        initial, initial_checks = _local_states(initial_surface, initial_cutoff, "initial", thermal_energy=thermal_energy)
        final, final_checks = _local_states(final_surface, int(final_max_quanta), "final")
        basis_order = int(initial.harmonic_basis[-1].sum()) + int(derivatives is not None)
        modes = initial_surface.omega.size
        overlap_count = comb(modes + basis_order, modes) * len(final.harmonic_basis)
        if overlap_count > max_overlaps:
            raise AnharmonicUnavailableError(
                f"the converged vibrational bases require {overlap_count:,} overlaps, exceeding max_overlaps"
            )
        overlap_initial = _fock_states(modes, basis_order)
        overlaps = _fc_overlaps(
            initial_surface.omega, final_surface.omega, rotation, shift,
            overlap_initial, final.harmonic_basis,
        )
        source = _dipole_source(overlap_initial, initial, initial_surface.omega, mu, derivatives)
        amplitudes = np.stack([
            (source[..., axis].T @ overlaps) @ final.eigenvectors for axis in range(3)
        ], axis=-1)
        strengths = np.sum(amplitudes**2, axis=-1)
        populations = initial.population
        if populations is None:
            populations = np.zeros(len(initial.labels))
            populations[0] = 1.0
        total_strength = float(populations @ np.sum(source**2, axis=(0, 2)))
        coverage = float(populations @ strengths.sum(axis=1) / total_strength) if total_strength else None

        photon_energies = gap + final.energies_hartree[None, :] - initial.energies_hartree[:, None]
        if process == "emission":
            photon_energies *= -1
        i, f = np.nonzero((photon_energies > 0) & (populations[:, None] > 0))
        order = np.argsort(photon_energies[i, f], kind="stable")
        i, f = i[order], f[order]
        if coverage is not None:
            coverage = min(coverage, 1.0)
        self.diagnostics.setdefault("spectrum", {})[process] = {
            "initial_vci": initial_checks,
            "final_vci": final_checks,
            "overlap_count": overlap_count,
            "thermal_partition": initial_checks.get("thermal_partition"),
        }
        return HarmonicSpectrum(
            photon_energy_hartree=photon_energies[i, f],
            initial_quanta=initial.labels[i],
            final_quanta=final.labels[f],
            population=populations[i],
            dipole_strength_au2=strengths[i, f],
            initial_population_coverage=float(populations.sum()),
            final_dipole_strength_coverage=coverage,
            transition_dipole_au=amplitudes[i, f],
        )


def prepare_anharmonic(electronic: _ElectronicSurfaces) -> AutomaticAnharmonic:
    """Screen direct cuts about the two minima and prepare any local correction.

    The fixed screening policy has no experimental inputs. Alternate verified
    minima, untracked electronic states and inadequate held-out fits prevent a
    local result. The mixed probe budget bounds work independently of molecule
    size. It does not establish the absence of unprobed couplings.
    """
    model = electronic.model
    diagnostics: dict = {"status": "unsupported", "reasons": [], "surfaces": {}}
    diagnostics["policy"] = "local_cubic_quartic_v1"
    diagnostics["energy_evaluations"] = 0
    diagnostics["known_scope"] = (
        "local Cartesian normal-mode cuts through both minima, selected pair and triple couplings, "
        "selected multimode checks, linear Herzberg–Teller"
    )
    for state in ("ground", "excited"):
        if len(getattr(electronic, f"alternate_{state}_coordinates_bohr", ())):
            diagnostics["reasons"].append(
                f"a second {state}-state minimum requires a nonlocal nuclear model"
            )
    if diagnostics["reasons"]:
        return AutomaticAnharmonic(electronic, diagnostics, None)

    if model.ground_frequencies_hartree.size == 1:
        from .automatic_1d import DirectOneMode

        diagnostics.update({
            "policy": "common_coordinate_dvr_v1",
            "solver": "adaptive_open_dvr",
            "dipole_model": "direct_coordinate_dependent",
            "known_scope": "mass-weighted Cartesian common axis with independently sampled surfaces and dipoles",
        })
        try:
            direct = DirectOneMode(electronic, diagnostics)
        except AnharmonicUnavailableError as error:
            diagnostics["reasons"].append(str(error))
            return AutomaticAnharmonic(electronic, diagnostics, None)
        diagnostics["status"] = "corrected"
        return AutomaticAnharmonic(electronic, diagnostics, None, direct)

    displacement = model.shift_au
    surface_inputs = (
        ("ground", electronic.ground_coordinates_bohr, model.ground_frequencies_hartree,
         model.ground_cartesian_mode_derivatives_au, -(model.rotation.T @ displacement)),
        ("excited", electronic.excited_coordinates_bohr, model.excited_frequencies_hartree,
         model.excited_cartesian_mode_derivatives_au, displacement),
    )
    surfaces = []
    for index, (name, center, omega, cartesian_modes, projected_displacement) in enumerate(surface_inputs):
        report: dict = {}
        diagnostics["surfaces"][name] = report
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0):
            diagnostics["reasons"].append(f"{name} minimum has a nonpositive or nonfinite frequency")
            return AutomaticAnharmonic(electronic, diagnostics, None)
        if not omega.size:
            surfaces.append(_LocalSurface(omega, ()))
            continue

        def evaluate(displacement_bohr: _Real) -> float:
            diagnostics["energy_evaluations"] += 1
            energy = float(electronic.evaluate(center + displacement_bohr)[index])
            if not np.isfinite(energy):
                raise AnharmonicUnavailableError("a direct energy was nonfinite")
            return energy

        try:
            minimum = getattr(electronic, f"{name}_energy_hartree", None)
            if minimum is None:
                minimum = evaluate(np.zeros_like(center))
            surfaces.append(_fit_surface(
                evaluate, float(minimum), omega, cartesian_modes, projected_displacement, report,
            ))
        except (ReferenceStabilityUnresolvedError, UnstableReferenceError):
            raise
        except (AnharmonicUnavailableError, ElectronicConvergenceError, StateTrackingError) as error:
            diagnostics["reasons"].append(f"{name} surface: {error}")
            return AutomaticAnharmonic(electronic, diagnostics, None)

    diagnostics["status"] = "corrected" if any(s.terms for s in surfaces) else "harmonic_screened"
    return AutomaticAnharmonic(electronic, diagnostics, (surfaces[0], surfaces[1]))


def _fit_surface(evaluate, minimum, omega, cartesian_modes, projected_displacement, report):
    # Use dimensionless x = sqrt(omega) Q so the sampled harmonic cut
    # is omega*x²/2. The fit adds only terms above quadratic order.
    modes = omega.size
    scaled_modes = cartesian_modes / np.sqrt(omega)[:, None, None]
    displacement = np.sqrt(omega) * projected_displacement
    samples = {}

    def at(indices, coordinates):
        key = tuple(zip(indices, coordinates))
        if key not in samples:
            cartesian = np.tensordot(coordinates, scaled_modes[list(indices)], axes=1)
            samples[key] = evaluate(cartesian) - minimum
        return samples[key]

    axis_reports: list[dict] = []
    pair_reports: list[dict] = []
    triple_reports: list[dict] = []
    mixed_audit: list[dict] = []
    report.update(axis_cuts=axis_reports, pair_probes=pair_reports, triple_probes=triple_reports,
                  multimode_audit=mixed_audit)
    axis_terms: list[_PotentialTerm] = []
    for mode, frequency in enumerate(omega):
        values = np.array([at((mode,), (x,)) for x in _AXIS_POINTS])
        correction = values - 0.5 * frequency * _AXIS_POINTS**2
        design = _AXIS_POINTS[:, None] ** np.array([3, 4])
        weight = 1 / (1 + _AXIS_POINTS**2)
        train = ~_AXIS_TEST
        coefficients = np.linalg.lstsq(
            design[train] * weight[train, None], correction[train] * weight[train], rcond=None,
        )[0]
        residuals = correction - design @ coefficients
        fit_error = float(np.max(np.abs(residuals[_AXIS_TEST])))
        harmonic_tolerance = max(_POTENTIAL_EPSILON, 0.005 * frequency)
        needs_correction = np.max(np.abs(correction)) > harmonic_tolerance
        fit_tolerance = max(2 * _POTENTIAL_EPSILON, 0.035 * frequency)
        axis_reports.append({
            "mode": mode, "frequency_hartree": float(frequency),
            "max_harmonic_residual_hartree": float(np.max(np.abs(correction))),
            "max_holdout_residual_hartree": fit_error,
            "holdout_tolerance_hartree": float(fit_tolerance),
            "cubic_hartree": float(coefficients[0]), "quartic_hartree": float(coefficients[1]),
            "corrected": bool(needs_correction),
        })
        if np.min(values) < -harmonic_tolerance or any(
            np.min(np.diff(side)) < -harmonic_tolerance for side in (values[:4][::-1], values[4:])
        ):
            raise AnharmonicUnavailableError(f"mode {mode} samples indicate nonlocal motion or another minimum")
        if needs_correction and fit_error > fit_tolerance:
            raise AnharmonicUnavailableError(f"mode {mode} fails the held-out local potential check")
        if needs_correction:
            axis_terms.extend(_PotentialTerm((mode,), (p,), float(c)) for p, c in zip((3, 4), coefficients))

    pairs = tuple(combinations(range(modes), 2))
    selected_pairs = _prioritize_pairs(pairs, omega, displacement)
    report["possible_mode_pairs"] = len(pairs)
    report["planned_probed_mode_pairs"] = len(selected_pairs)
    triples = _prioritize_triples(omega)
    report["possible_mode_triples"] = comb(modes, 3)
    report["planned_probed_mode_triples"] = len(triples)
    terms = list(axis_terms)
    for subset in selected_pairs:
        # Two distinct radii separate x**3*y from x*y**3 even when their
        # contributions cancel on the equal-amplitude directions.
        terms.extend(_mixed_terms(
            at, omega, subset, _PAIR_POWERS, ((1.8, 1.8), (0.8, 1.8)),
            ((2.6, 0.8), (0.8, 2.6)), pair_reports,
        ))
    for subset in triples:
        terms.extend(_mixed_terms(
            at, omega, subset, _TRIPLE_POWERS, ((1.8, 1.8, 1.8),),
            ((0.8, 1.8, 1.8), (1.8, 0.8, 1.8), (1.8, 1.8, 0.8)), triple_reports,
        ))
    for point in _mixed_check_points(displacement):
        expected = 0.5 * omega @ point**2 + sum(term.value(point) for term in terms)
        scale = float(np.linalg.norm(omega * point**2))
        tolerance = max(3 * _POTENTIAL_EPSILON, 0.035 * scale)
        actual = at(tuple(range(modes)), point)
        defect = float(abs(expected - actual))
        mixed_audit.append({
            "dimensionless_coordinates": point.tolist(),
            "abs_residual_hartree": defect,
            "tolerance_hartree": tolerance,
        })
        if actual < -3 * _POTENTIAL_EPSILON:
            raise AnharmonicUnavailableError("a multimode check samples a lower-energy region")
        if defect > tolerance:
            raise AnharmonicUnavailableError("a multimode check exceeds the sampled local model's tolerance")
    return _LocalSurface(omega, tuple(terms))


def _mixed_terms(at, omega, modes, powers, fit_radii, validation_radii, reports):
    # Isolate the many-mode potential increment via the n-mode expansion of
    # direct energies. Lower-dimensional cuts are shared across increments.
    support = len(modes)
    projections = [
        (indices, (-1)**(support - len(indices)))
        for size in range(1, support)
        for indices in combinations(range(support), size)
    ]
    scale = float(np.prod(omega[list(modes)])**(1 / support))
    cutoff = max(_POTENTIAL_EPSILON, 0.005 * scale)

    def increment(coordinates):
        value = at(modes, coordinates)
        if value < -cutoff:
            raise AnharmonicUnavailableError(f"coupled modes {modes} sample a lower-energy region")
        for projection, sign in projections:
            subset = tuple(modes[i] for i in projection)
            point = tuple(coordinates[i] for i in projection)
            value += sign * at(subset, point)
        return value

    def signed_points(radii):
        return [
            tuple(sign * radius for sign, radius in zip(signs, point))
            for point in radii for signs in product((-1, 1), repeat=support)
        ]

    points = signed_points(fit_radii)
    sampled = [increment(point) for point in points]
    max_residual = max(map(abs, sampled))
    needs_correction = max_residual > cutoff
    record = {
        "modes": list(modes), "max_lower_dimensional_residual_hartree": float(max_residual),
        "corrected": bool(needs_correction),
    }
    reports.append(record)
    if not needs_correction:
        return ()
    coefficients = np.linalg.lstsq(_mixed_design(points, powers), sampled, rcond=None)[0]
    validation = signed_points(validation_radii)
    held_out = np.array([increment(point) for point in validation])
    fit_error = float(np.max(np.abs(_mixed_design(validation, powers) @ coefficients - held_out)))
    tolerance = max(2 * _POTENTIAL_EPSILON, 0.035 * scale)
    record["max_holdout_residual_hartree"] = fit_error
    record["holdout_tolerance_hartree"] = float(tolerance)
    record["coefficients_hartree"] = coefficients.tolist()
    if fit_error > tolerance:
        raise AnharmonicUnavailableError(f"coupled modes {modes} fail the held-out local potential check")
    return tuple(_PotentialTerm(modes, p, float(c)) for p, c in zip(powers, coefficients))


def _prioritize_pairs(pairs, omega, displacement):
    if len(pairs) <= _MAX_PROBED_PAIRS:
        return pairs
    # Cover softness, displacement, and possible 1:2 resonances independently.
    def soft(pair):
        i, j = pair
        return omega[i] * omega[j]

    def displaced(pair):
        i, j = pair
        return -max(abs(displacement[i]), abs(displacement[j])) / soft(pair)**0.25

    def resonant(pair):
        i, j = pair
        return abs(1 - omega[j] / (2 * omega[i]))

    rankings = [sorted(pairs, key=key) for key in (soft, displaced, resonant)]
    chosen = {}
    for bucket in zip(*rankings):
        for pair in bucket:
            chosen[pair] = None
            if len(chosen) == _MAX_PROBED_PAIRS:
                return tuple(chosen)
    return tuple(chosen)


def _prioritize_triples(omega):
    triples = combinations(range(omega.size), 3)

    def detuning(modes):
        first, second, third = omega[list(modes)]
        return abs(first + second - third) / np.sqrt((first + second) * third)

    return tuple(nsmallest(_MAX_PROBED_TRIPLES, triples, key=detuning))


def _mixed_design(points, powers):
    coordinates = np.asarray(points)
    return np.prod(coordinates[:, None, :] ** np.asarray(powers)[None, :, :], axis=-1)


def _mixed_check_points(displacement):
    points = []
    modes = displacement.size
    if modes >= 3:
        rng = np.random.default_rng(1407)
        for row in rng.choice((-1.0, 1.0), size=(3, modes)):
            point = (2.2 / np.sqrt(modes)) * row
            points.extend((point, -point))
    norm = np.linalg.norm(displacement)
    if np.count_nonzero(np.abs(displacement) > 0.1) > 1 and norm > 0.6:
        points.append(displacement * min(1.0, 2.2 / norm))
    return points


@lru_cache(maxsize=16)
def _projected_powers(order, inner_radius, outer_radius):
    # The Jacobi matrix gives Gauss–Hermite nodes and normalized oscillator
    # evaluations. Complete harmonic confinement outside the sampled region
    # avoids an unphysical divergent cubic or negative quartic extrapolation.
    off_diagonal = np.sqrt(np.arange(1.0, 112.0) / 2)
    jacobi = np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
    points, vectors = np.linalg.eigh(jacobi)
    vectors = vectors[:order + 1]
    transition = np.clip((np.abs(points) - inner_radius) / (outer_radius - inner_radius), 0, 1)
    window = 0.5 + 0.5 * np.cos(np.pi * transition)
    return {power: (vectors * (window * points**power)) @ vectors.T for power in (1, 2, 3, 4)}


def _vci_states(surface, cutoff, padding, inner_radius, outer_radius, *, thermal_order=False):
    from scipy.optimize import linear_sum_assignment

    order = cutoff + padding
    basis = _fock_states(surface.omega.size, order)
    retained = comb(surface.omega.size + cutoff, surface.omega.size)
    # The AH 0–0 reference already includes harmonic ZPE. Store the VCI
    # eigenvalues relative to the harmonic zero-point level on each surface.
    hamiltonian = np.diag(basis @ surface.omega)
    powers = _projected_powers(order, inner_radius, outer_radius)
    for term in surface.terms:
        spectators = [i for i in range(surface.omega.size) if i not in term.modes]
        blocks = {}
        for index, state in enumerate(basis[:, spectators]):
            blocks.setdefault(tuple(state), []).append(index)
        for indices in blocks.values():
            block = np.full((len(indices), len(indices)), term.coefficient_hartree)
            for mode, exponent in zip(term.modes, term.powers):
                active_quanta = basis[indices, mode]
                block *= powers[exponent][np.ix_(active_quanta, active_quanta)]
            hamiltonian[np.ix_(indices, indices)] += block
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    # The thermal trace requires a complete lowest-energy prefix. Assignment
    # to oscillator labels describes those levels and does not select them.
    if thermal_order:
        limit = energies[retained - 1] + _degeneracy_tolerance(energies)
        retained = int(np.searchsorted(energies, limit, side="right"))
        columns = np.arange(retained)
        _, assigned = linear_sum_assignment(eigenvectors[1:, 1:retained].T**2, maximize=True)
        label_rows = np.r_[0, assigned + 1]
    else:
        _, assigned = linear_sum_assignment(eigenvectors[1:retained, 1:]**2, maximize=True)
        columns = np.r_[0, assigned + 1]
        label_rows = np.arange(retained)
    assigned_vectors = eigenvectors[:, columns]
    assignment = assigned_vectors[label_rows, np.arange(retained)]
    phase = np.where(assignment < 0, -1, 1)
    return _VibrationalStates(basis, basis[label_rows], energies[columns], assigned_vectors * phase, assignment**2)


def _compare_states(first, second):
    if len(first.energies_hartree) != len(second.energies_hartree):
        return np.inf, 0.0
    differences = abs(first.energies_hartree - second.energies_hartree)
    retained = len(first.harmonic_basis)
    common = second.eigenvectors[:retained]
    overlaps = first.eigenvectors.T @ common
    squared = np.diag(overlaps)**2
    # A rotation among degenerate states does not invalidate their subspace.
    energy_separation = abs(first.energies_hartree[:, None] - first.energies_hartree)
    second_separation = abs(second.energies_hartree[:, None] - second.energies_hartree)
    degenerate = (energy_separation < _LEVEL_TOLERANCE / 10) & (second_separation < _LEVEL_TOLERANCE / 10)
    for level in np.flatnonzero(degenerate.sum(axis=1) > 1):
        members = np.flatnonzero(degenerate[level])
        singular = np.linalg.svd(overlaps[np.ix_(members, members)], compute_uv=False)
        squared[level] = singular[-1]**2
    return float(np.max(differences)), float(np.min(squared))


def _local_states(surface, cutoff, label, *, thermal_energy=0.0):
    modes = surface.omega.size
    retained = comb(modes + cutoff, modes)
    if retained > _MAX_VCI_STATES:
        raise AnharmonicUnavailableError(f"the {label} state range exceeds the bounded VCI basis")
    if not surface.terms:
        basis = _fock_states(modes, cutoff)
        energies = basis @ surface.omega
        population = None
        report = {"basis_count": retained, "harmonic": True}
        if thermal_energy:
            with np.errstate(over="ignore", under="ignore"):
                log_inverse_partition = np.log(-np.expm1(-surface.omega / thermal_energy)).sum()
                population = np.exp(log_inverse_partition - energies / thermal_energy)
            report["thermal_partition"] = {"method": "exact_harmonic_partition",
                                           "omitted_population_upper_bound": max(0., 1. - float(population.sum()))}
        return _VibrationalStates(basis, basis, energies, np.eye(retained), population=population), report
    min_padding = 4 if comb(modes + cutoff + 4, modes) <= _MAX_VCI_STATES else 3
    if comb(modes + cutoff + min_padding, modes) > _MAX_VCI_STATES:
        raise AnharmonicUnavailableError(f"the {label} state range leaves insufficient room for a VCI basis check")
    thermal_order = bool(thermal_energy)
    previous = _vci_states(surface, cutoff, min_padding - 1, 3.5, 4.5, thermal_order=thermal_order)
    for padding in range(min_padding, _MAX_VCI_PADDING + 1):
        if comb(modes + cutoff + padding, modes) > _MAX_VCI_STATES:
            break
        result = _vci_states(surface, cutoff, padding, 3.5, 4.5, thermal_order=thermal_order)
        energy_error, state_overlap = _compare_states(previous, result)
        thermal_basis_error = _thermal_distribution_error(previous, result, thermal_energy)
        previous = result
        if (energy_error <= _LEVEL_TOLERANCE and state_overlap >= _MIN_STATE_OVERLAP_SQUARED
                and thermal_basis_error <= 1 - _MIN_STATE_OVERLAP_SQUARED):
            # Check whether the spectrum is sensitive to how the local model
            # is joined to its confining harmonic tails.
            alternative = _vci_states(surface, cutoff, padding, 3.2, 4.2, thermal_order=thermal_order)
            tail_error, tail_overlap = _compare_states(result, alternative)
            thermal_tail_error = _thermal_distribution_error(result, alternative, thermal_energy)
            if (tail_error > _LEVEL_TOLERANCE or tail_overlap < _MIN_STATE_OVERLAP_SQUARED
                    or thermal_tail_error > 1 - _MIN_STATE_OVERLAP_SQUARED):
                raise AnharmonicUnavailableError(f"the {label} VCI levels depend on unsampled potential tails")
            report = {
                "basis_count": len(result.harmonic_basis),
                "states_retained": len(result.labels), "max_basis_shift_hartree": energy_error,
                "min_basis_state_overlap_squared": state_overlap, "max_tail_shift_hartree": tail_error,
                "min_tail_state_overlap_squared": tail_overlap,
                "min_assignment_overlap_squared": float(result.assignment_overlaps_squared.min()),
            }
            if thermal_energy:
                population, thermal_report = _bounded_thermal_populations(
                    surface, result, alternative, thermal_energy,
                )
                result = _VibrationalStates(
                    result.harmonic_basis, result.labels, result.energies_hartree, result.eigenvectors,
                    result.assignment_overlaps_squared, population,
                )
                report["thermal_partition"] = {
                    **thermal_report, "basis_population_l1_change": thermal_basis_error,
                    "alternate_taper_population_l1_change": thermal_tail_error,
                }
            return result, report
    raise AnharmonicUnavailableError(f"the {label} VCI states did not converge within the bounded basis")


def _dipole_source(basis, states, omega, dipole, derivatives):
    size = len(states.harmonic_basis)
    source = np.zeros((len(basis), len(states.labels), 3))
    source[:size] = states.eigenvectors[..., None] * dipole
    if derivatives is None:
        return source
    indices = {tuple(row): i for i, row in enumerate(basis)}
    for i, state in enumerate(states.harmonic_basis):
        neighbor = state.copy()
        for mode, quanta in enumerate(state):
            prefactor = states.eigenvectors[i, :, None] * (derivatives[mode] / np.sqrt(2 * omega[mode]))
            neighbor[mode] += 1
            source[indices[tuple(neighbor)]] += np.sqrt(quanta + 1) * prefactor
            if quanta:
                neighbor[mode] -= 2
                source[indices[tuple(neighbor)]] += np.sqrt(quanta) * prefactor
            neighbor[mode] = quanta
    return source


def _relative_boltzmann(energies, thermal_energy):
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-(energies - energies[0]) / thermal_energy)


def _thermal_distribution_error(first, second, thermal_energy):
    if not thermal_energy:
        return 0.0
    if len(first.energies_hartree) != len(second.energies_hartree):
        return np.inf
    left = _relative_boltzmann(first.energies_hartree, thermal_energy)
    right = _relative_boltzmann(second.energies_hartree, thermal_energy)
    return float(np.linalg.norm(left / left.sum() - right / right.sum(), ord=1))


def _negative_window_bound(surface, inner_radius, outer_radius):
    """Bound the most negative correction for the windowed local potential."""
    radii = np.linspace(inner_radius, outer_radius, 33)
    weights = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, len(radii)))
    # On each interval the taper decreases and the absolute power increases.
    powers = {
        p: max(inner_radius**p, float(np.max(weights[:-1] * radii[1:]**p)))
        for p in (1, 2, 3, 4)
    }
    return sum(
        abs(term.coefficient_hartree) * float(np.prod([powers[p] for p in term.powers]))
        for term in surface.terms if term.coefficient_hartree < 0 or any(p % 2 for p in term.powers)
    )


def _harmonic_log_rank_tail(frequencies, retained, thermal_energy, reference):
    """Sum the tail of the energy-ranked harmonic partition in log space."""
    omega = np.sort(frequencies)
    # Represent quanta by a nondecreasing sequence of mode indices. Unpopped
    # nodes partition all omitted states into disjoint harmonic subtrees.
    frontier = [(0.0, 0)]
    for _ in range(retained):
        energy, first_mode = heappop(frontier)
        for mode in range(first_mode, len(omega)):
            heappush(frontier, (energy + omega[mode], mode))
    energies, first_modes = zip(*frontier, strict=True)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        log_mode_partitions = -np.log(-np.expm1(-omega / thermal_energy))
        log_suffix_partitions = np.cumsum(log_mode_partitions[::-1])[::-1]
        terms = (reference - np.asarray(energies)) / thermal_energy + log_suffix_partitions[list(first_modes)]
        return float(np.logaddexp.reduce(terms))


def _bounded_thermal_populations(surface, states, alternative, thermal_energy):
    """Min–max tail bound for the tapered operator and converged Ritz prefix."""
    omitted = []
    envelopes = []
    for model, inner, outer in ((states, 3.5, 4.5), (alternative, 3.2, 4.2)):
        bound = _negative_window_bound(surface, inner, outer)
        energies = model.energies_hartree
        # If V >= -B, each ordered eigenvalue is at least its harmonic
        # counterpart minus B, on the same harmonic zero-point reference.
        log_tail = _harmonic_log_rank_tail(surface.omega, len(energies), thermal_energy, energies[0] + bound)
        log_retained = np.log(_relative_boltzmann(energies, thermal_energy).sum())
        log_coverage = log_retained - np.logaddexp(log_retained, log_tail)
        omitted.append(float(-np.expm1(log_coverage)))
        envelopes.append(bound)
    weights = _relative_boltzmann(states.energies_hartree, thermal_energy)
    worst_tail = max(omitted)
    population = (1. - worst_tail) * weights / weights.sum()
    return population, {
        "method": "windowed_vci_minmax_bound",
        "scope": "tapered local Hamiltonians with numerically converged retained levels",
        "states_retained": len(weights),
        "omitted_population_upper_bound": worst_tail,
        "potential_negative_bound_hartree": envelopes[0],
        "alternate_taper_negative_bound_hartree": envelopes[1],
        "normalization": "upper_partition_bound",
    }
