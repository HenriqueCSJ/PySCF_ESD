"""Absolute electric-dipole observables and moments of retained vibronic sticks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .harmonic import _real_array


# CODATA 2022. Wavenumber conversion is twice the Rydberg constant in cm^-1.
HARTREE_TO_WAVENUMBER = 219474.63136314
HARTREE_TO_EV = 27.211386245981
_HARTREE_TO_JOULE = 4.3597447222060e-18
_FINE_STRUCTURE_CONSTANT = 7.2973525643e-3
_ATOMIC_TIME_S = 2.4188843265864e-17
_EINSTEIN_A_FACTOR = 4 * _FINE_STRUCTURE_CONSTANT**3 / (3 * _ATOMIC_TIME_S)

_Process = Literal["absorption", "emission"]
_QuantaScheme = Literal["harmonic_quanta", "vci_dominant_harmonic_labels", "dvr_levels"]
_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.intp]


class _StickSpectrum(Protocol):
    photon_energy_hartree: _FloatArray
    population: _FloatArray
    dipole_strength_au2: _FloatArray
    initial_population_coverage: float
    final_dipole_strength_coverage: float | None


@dataclass(frozen=True)
class SpectralLineProperties:
    """Input-order line arrays. Direction-specific fields are None for the other process.

    ``oscillator_strength`` and ``einstein_a_s_inverse`` describe individual
    transitions, independent of initial-state populations. The population-
    weighted emission rate counts photons. Radiant power weights each photon
    by its energy in joules and is per molecule in the emitting electronic state.
    """

    photon_weighted_strength: _FloatArray
    oscillator_strength: _FloatArray | None = None
    population_weighted_oscillator_strength: _FloatArray | None = None
    einstein_a_s_inverse: _FloatArray | None = None
    population_weighted_einstein_a_s_inverse: _FloatArray | None = None
    population_weighted_radiant_power_w: _FloatArray | None = None


def spectral_line_properties(spectrum: _StickSpectrum, process: _Process) -> SpectralLineProperties:
    """Convert electric-dipole sticks to length-gauge f values or vacuum Einstein A.

    For photon energy E in Hartree and Cartesian dipole strength S in squared
    atomic units, f = 2 E S / 3 and A = 4 alpha^3 E^3 S / (3 t_au).
    These expressions use a single modeled molecular transition between states
    and a sum over the three Cartesian dipole components. Populations are used
    as supplied by the vibrational model, without renormalizing retained states.
    """
    if process not in ("absorption", "emission"):
        raise ValueError("process must be absorption or emission")
    energy = _real_array(spectrum.photon_energy_hartree, "photon_energy_hartree")
    population = _real_array(spectrum.population, "population")
    strength = _real_array(spectrum.dipole_strength_au2, "dipole_strength_au2")
    if energy.ndim != 1 or population.shape != energy.shape or strength.shape != energy.shape:
        raise ValueError("line energies, populations and dipole strengths must be matching vectors")
    if np.any(energy <= 0) or np.any(strength < 0):
        raise ValueError("expected positive photon energies and nonnegative dipole strengths")
    if np.any((population < 0) | (population > 1)):
        raise ValueError("line populations must be between zero and one")

    with np.errstate(over="ignore", invalid="ignore"):
        energy_weighted_strength = energy**(1 if process == "absorption" else 3) * strength
        photon_weighted = population * energy_weighted_strength
        if process == "absorption":
            oscillator = (2 / 3) * energy_weighted_strength
            weighted_oscillator = (2 / 3) * photon_weighted
            if np.isfinite(oscillator).all() and np.isfinite(weighted_oscillator).all():
                return SpectralLineProperties(photon_weighted, oscillator, weighted_oscillator)
        else:
            rate = _EINSTEIN_A_FACTOR * energy_weighted_strength
            weighted_rate = _EINSTEIN_A_FACTOR * photon_weighted
            power = weighted_rate * (energy * _HARTREE_TO_JOULE)
            if np.isfinite(rate).all() and np.isfinite(weighted_rate).all() and np.isfinite(power).all():
                return SpectralLineProperties(
                    photon_weighted, einstein_a_s_inverse=rate,
                    population_weighted_einstein_a_s_inverse=weighted_rate,
                    population_weighted_radiant_power_w=power,
                )
    raise ValueError("line properties exceed the numerical range")


def _finite(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError("spectral properties exceed the numerical range")
    return value


def _energy_units(hartree: float) -> dict[str, float]:
    return {
        "hartree": _finite(hartree),
        "ev": _finite(hartree * HARTREE_TO_EV),
        "cm_inverse": _finite(hartree * HARTREE_TO_WAVENUMBER),
    }


def _assignments(spectrum: _StickSpectrum, scheme: _QuantaScheme | None):
    initial = getattr(spectrum, "initial_quanta", None)
    final = getattr(spectrum, "final_quanta", None)
    if initial is not None and final is not None:
        return scheme or "quanta_labels", initial, final
    initial_level = getattr(spectrum, "initial_level", None)
    final_level = getattr(spectrum, "final_level", None)
    if initial_level is not None and final_level is not None:
        return (
            "dvr_level_and_spectator_quanta",
            np.column_stack((initial_level, spectrum.initial_spectator_quanta)),
            np.column_stack((final_level, spectrum.final_spectator_quanta)),
        )
    return None, None, None


def _assignment(label: _IntArray, scheme: str) -> dict:
    if scheme == "dvr_levels":
        return {"dvr_level": int(label.item())}
    if scheme == "dvr_level_and_spectator_quanta":
        return {"dvr_level": int(label[0]), "spectator_quanta": label[1:].tolist()}
    return {"quanta": label.tolist()}


def _lifetime_s(rate_s_inverse: float) -> float | None:
    if rate_s_inverse <= 0:
        return None
    lifetime = 1.0 / rate_s_inverse
    return lifetime if isfinite(lifetime) else None


def _emitting_states(initial: _IntArray, scheme: str, population: _FloatArray, rate: _FloatArray) -> dict:
    # Each line carries its initial population. Group by state to count that
    # population once and sum the intrinsic rates over its decay channels.
    labels, first, group = np.unique(initial, axis=0, return_index=True, return_inverse=True)
    populations = population[first]
    group = group.ravel()
    if not np.allclose(population, populations[group], rtol=1e-12, atol=0):
        return {"emitting_state_rates_status": "ambiguous_initial_labels"}
    rates = np.bincount(group, weights=rate, minlength=len(labels))
    return {
        "emitting_state_rates_status": "retained_lines",
        "represented_initial_population": _finite(populations.sum()),
        "emitting_vibrational_states": [
            {
                "initial_assignment": _assignment(label, scheme),
                "relative_population": float(state_population),
                "vacuum_radiative_rate_s_inverse": _finite(state_rate),
                "vacuum_radiative_lifetime_s": _lifetime_s(float(state_rate)),
            }
            for label, state_population, state_rate in zip(labels, populations, rates, strict=True)
        ],
    }


def _spectrum_summary(
    spectrum: _StickSpectrum, process: _Process, status: str, max_dominant_lines: int,
    assignment_scheme: _QuantaScheme | None,
) -> tuple[dict, float | None]:
    lines = spectral_line_properties(spectrum, process)
    energy = np.asarray(spectrum.photon_energy_hartree, dtype=float)
    population = np.asarray(spectrum.population, dtype=float)
    strength = np.asarray(spectrum.dipole_strength_au2, dtype=float)
    weights = lines.photon_weighted_strength
    total = _finite(weights.sum())
    positive = np.flatnonzero(weights > 0)
    scheme, initial, final = _assignments(spectrum, assignment_scheme)
    coverage = spectrum.final_dipole_strength_coverage
    summary = {
        "coverage_status": status,
        "initial_population_coverage": _finite(spectrum.initial_population_coverage),
        "final_dipole_strength_coverage": None if coverage is None else _finite(coverage),
        "line_count": len(energy),
        "nonzero_weight_line_count": len(positive),
        "population_weighted_dipole_strength_sum_au2": _finite(np.dot(population, strength)),
        "assignment_scheme": scheme,
        "centroid_energy": None,
        "energy_standard_deviation": None,
        "dominant_lines": [],
    }
    if scheme in ("harmonic_quanta", "dvr_levels", "dvr_level_and_spectator_quanta"):
        origins = np.flatnonzero(np.all(initial == 0, axis=1) & np.all(final == 0, axis=1))
        if origins.size == 1:
            origin = origins.item()
            summary["origin_00_line"] = {
                "stick_index": int(origin),
                "photon_energy": _energy_units(energy[origin]),
                "dipole_strength_au2": float(strength[origin]),
            }
    if process == "absorption":
        absolute_weights = lines.population_weighted_oscillator_strength
        intrinsic_weights = lines.oscillator_strength
        summary.update(
            moment_weight="population_weighted_length_gauge_oscillator_strength",
            population_weighted_oscillator_strength_sum=_finite(absolute_weights.sum()),
        )
    else:
        absolute_weights = lines.population_weighted_einstein_a_s_inverse
        intrinsic_weights = lines.einstein_a_s_inverse
        total_rate = _finite(absolute_weights.sum())
        summary.update(
            moment_weight="population_weighted_vacuum_spontaneous_photon_rate",
            vacuum_radiative_rate_s_inverse=total_rate,
            vacuum_radiative_lifetime_s=_lifetime_s(total_rate),
            vacuum_radiant_power_w_per_excited_molecule=_finite(lines.population_weighted_radiant_power_w.sum()),
        )
        if initial is not None:
            summary.update(_emitting_states(initial, scheme, population, intrinsic_weights))
    if total == 0:
        return summary, None

    fractions = weights[positive] / total
    line_energies = energy[positive]
    centroid = _finite(np.dot(fractions, line_energies))
    spread = sqrt(_finite(np.dot(fractions, (line_energies - centroid)**2)))
    strongest = float(energy[np.argmax(weights)])
    summary.update(centroid_energy=_energy_units(centroid), energy_standard_deviation=_energy_units(spread))
    ranked = positive[np.argsort(-weights[positive], kind="stable")[:max_dominant_lines]]
    dipoles = getattr(spectrum, "transition_dipole_au", None)
    for index in ranked:
        line = {
            "stick_index": int(index),
            "photon_energy": _energy_units(energy[index]),
            "weight_fraction": float(weights[index] / total),
            "initial_population": float(population[index]),
            "dipole_strength_au2": float(strength[index]),
        }
        if scheme is not None:
            line.update(initial_assignment=_assignment(initial[index], scheme),
                        final_assignment=_assignment(final[index], scheme))
        if dipoles is not None:
            moment = dipoles[index]
            if np.iscomplexobj(moment):
                line["transition_dipole_au"] = {
                    "real": [_finite(component) for component in moment.real],
                    "imaginary": [_finite(component) for component in moment.imag],
                }
            else:
                line["transition_dipole_au"] = [_finite(component) for component in moment]
        if process == "absorption":
            line.update(oscillator_strength=float(intrinsic_weights[index]),
                        population_weighted_oscillator_strength=float(absolute_weights[index]))
        else:
            line.update(einstein_a_s_inverse=float(intrinsic_weights[index]),
                        population_weighted_photon_rate_s_inverse=float(absolute_weights[index]))
        summary["dominant_lines"].append(line)

    if scheme in ("harmonic_quanta", "dvr_level_and_spectator_quanta"):
        offset = int(scheme == "dvr_level_and_spectator_quanta")
        quanta = final[positive, offset:]
        summary["final_mode_statistics"] = {
            "mode_basis": "harmonic_spectators" if offset else "final_surface_harmonic_modes",
            "weighting_note": "Lines can excite several modes and contribute to each of those mode fractions.",
            "modes": [
                {
                    "mode": mode,
                    "weighted_mean_final_quanta": float(mean),
                    "weight_fraction_with_nonzero_final_quanta": float(fraction),
                }
                for mode, (mean, fraction) in enumerate(
                    zip(fractions @ quanta, fractions @ (quanta > 0), strict=True), start=1,
                )
            ],
        }
    return summary, strongest


def summarize_spectral_properties(
    spectra: Mapping[str, _StickSpectrum],
    *,
    spectral_status: Mapping[str, str] | None = None,
    temperature_kelvin: float | None = None,
    max_dominant_lines: int = 5,
    assignment_scheme: _QuantaScheme | None = None,
) -> dict:
    """Summarize retained, positive-photon-energy sticks in a JSON-ready report.

    Supply ``spectral_status`` from the workflow to carry basis convergence
    through each result. Each emission expectation is per molecule in the
    excited electronic state, using the supplied initial vibrational populations.
    The lifetime is the reciprocal of the retained rate, with a zero rate
    represented by None. Specify ``assignment_scheme="harmonic_quanta"`` to
    summarize the final harmonic occupations in one-based model mode order.
    VCI dominant labels and DVR levels can identify lines without implying exact
    harmonic mode occupations.
    """
    if (isinstance(max_dominant_lines, bool) or not isinstance(max_dominant_lines, Integral)
            or max_dominant_lines < 0):
        raise ValueError("max_dominant_lines must be a nonnegative integer")
    if assignment_scheme not in (None, "harmonic_quanta", "vci_dominant_harmonic_labels", "dvr_levels"):
        raise ValueError("unknown assignment_scheme")
    if temperature_kelvin is not None and (not isfinite(temperature_kelvin) or temperature_kelvin < 0):
        raise ValueError("temperature_kelvin must be finite and nonnegative")
    statuses = {} if spectral_status is None else dict(spectral_status)
    for direction in spectra.keys() | statuses.keys():
        if direction not in ("absorption", "emission"):
            raise ValueError("spectrum keys must be absorption or emission")
        if statuses.get(direction, "unassessed") not in ("complete", "incomplete", "unassessed"):
            raise ValueError("spectral_status values must be complete, incomplete or unassessed")
    report = {
        "schema_version": 1,
        "temperature_kelvin": None if temperature_kelvin is None else float(temperature_kelvin),
        "line_scope": (
            "All sums and moments use retained positive-photon-energy electric-dipole lines. "
            "Initial vibrational populations are used without renormalization. "
            "Basis coverage measures population and dipole-strength closure. "
            "Energy-weighted sums and moments can require separate convergence checks."
        ),
        "radiative_lifetime_definition": (
            "The vacuum radiative lifetime is the reciprocal of the radiative rate. "
            "For a mixture it is the inverse initial population-averaged rate. "
            "Observed fluorescence lifetimes also depend on competing processes and population kinetics."
        ),
        "spectra": {},
        "paired": None,
    }
    strongest = {}
    for process in ("absorption", "emission"):
        coverage_status = statuses.get(process, "unassessed")
        if process in spectra:
            report["spectra"][process], strongest[process] = _spectrum_summary(
                spectra[process], process, coverage_status, int(max_dominant_lines), assignment_scheme,
            )
        elif process in statuses:
            report["spectra"][process] = {"coverage_status": coverage_status, "line_count": None}

    if "absorption" in spectra and "emission" in spectra:
        absorption, emission = (report["spectra"][process] for process in ("absorption", "emission"))
        paired_statuses = {absorption["coverage_status"], emission["coverage_status"]}
        paired_status = "incomplete" if "incomplete" in paired_statuses else "unassessed"
        if paired_statuses == {"complete"}:
            paired_status = "complete"
        centroid_a, centroid_e = absorption["centroid_energy"], emission["centroid_energy"]
        strongest_a, strongest_e = strongest["absorption"], strongest["emission"]
        report["paired"] = {
            "coverage_status": paired_status,
            "absorption_minus_emission_centroid_energy": (
                None if centroid_a is None or centroid_e is None
                else _energy_units(centroid_a["hartree"] - centroid_e["hartree"])
            ),
            "absorption_minus_emission_strongest_stick_energy": (
                None if strongest_a is None or strongest_e is None
                else _energy_units(strongest_a - strongest_e)
            ),
        }
    return report
