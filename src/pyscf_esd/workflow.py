"""The public analytic-first workflow for closed-shell singlet spectroscopy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Integral, Real
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .cpu import ElectronicConvergenceError, Settings
from .harmonic import HarmonicSpectrum
from .state_tracking import StateTrackingError


DEFAULT_SETTINGS = Settings("def2-tzvpd", "pbe0")
DEFAULT_MAX_QUANTA = 32


class WorkflowIncomplete(ValueError):
    """The requested model could not qualify a predicted spectrum."""

    def __init__(self, message: str, diagnostics: dict, partial_spectra: dict[str, HarmonicSpectrum] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.partial_spectra = {} if partial_spectra is None else partial_spectra


@dataclass(frozen=True)
class VibronicResult:
    """Predicted line spectra and diagnostics for the chosen model."""

    model_requested: str
    model_used: str
    spectra: dict[str, HarmonicSpectrum]
    diagnostics: dict


def _adaptive_spectrum(
    solver, *, process: str, temperature: float, coverage: float,
    max_quanta: int, max_overlaps: int,
) -> tuple[HarmonicSpectrum, list[dict]]:
    initial = 0
    final = min(2, max_quanta)
    attempts = []
    last_valid = {}
    while True:
        try:
            spectrum = solver(
                process=process, temperature_kelvin=temperature,
                initial_max_quanta=initial, final_max_quanta=final, max_overlaps=max_overlaps,
            )
        except WorkflowIncomplete as error:
            raise WorkflowIncomplete(
                str(error), {"process": process, "basis_attempts": attempts, **error.diagnostics}, last_valid,
            ) from error
        except ValueError as error:
            if "max_overlaps" not in str(error):
                raise
            raise WorkflowIncomplete(
                f"{process} exhausted the harmonic overlap budget before reaching {coverage:.1%} coverage",
                {"process": process, "basis_attempts": attempts, "max_overlaps": max_overlaps,
                 "target_coverage": coverage},
                last_valid,
            ) from error
        population = spectrum.initial_population_coverage
        strength = spectrum.final_dipole_strength_coverage
        attempts.append({
            "initial_max_quanta": initial, "final_max_quanta": final,
            "initial_population_coverage": population, "final_dipole_strength_coverage": strength,
        })
        if not np.isfinite(population) or (strength is not None and not np.isfinite(strength)):
            raise WorkflowIncomplete(f"{process} has nonfinite coverage", {"basis_attempts": attempts}, last_valid)
        last_valid = {process: spectrum}
        if population >= coverage and (strength is None or strength >= coverage):
            return spectrum, attempts
        if population < coverage and initial < max_quanta:
            initial += 1
        elif population >= coverage and final < max_quanta:
            final += 1
        else:
            raise WorkflowIncomplete(
                f"{process} reached the vibrational ceiling before obtaining {coverage:.1%} coverage",
                {"process": process, "basis_attempts": attempts, "max_quanta": max_quanta,
                 "target_coverage": coverage},
                last_valid,
            )


def run_vibronic(
    symbols: tuple[str, ...],
    coordinates_bohr: ArrayLike,
    settings: Settings | None = None,
    *,
    target_index: int = 0,
    model: Literal["auto", "ah", "vg"] = "auto",
    process: Literal["both", "absorption", "emission"] = "both",
    temperature_kelvin: float = 0.0,
    coverage: float = 0.995,
    max_quanta: int = DEFAULT_MAX_QUANTA,
    max_overlaps: int = 250_000,
    progress: Callable[[str], None] | None = None,
) -> VibronicResult:
    """Predict singlet absorption and/or relaxed-state emission from one geometry.

    Coordinates are Cartesian bohr in atom order. The zero-based target labels
    the root at the relaxed ground geometry. The default requests adiabatic
    Hessians, Duschinsky rotation and Herzberg–Teller amplitudes, then screens
    direct electronic cuts for anharmonic effects. A requested model that
    cannot be qualified raises WorkflowIncomplete with available diagnostics.
    ``model="ah"`` requests the harmonic calculation directly and ``model="vg"``
    explicitly requests the vertical-gradient approximation.
    """
    if model not in ("auto", "ah", "vg"):
        raise ValueError("model must be auto, ah or vg")
    if process not in ("both", "absorption", "emission"):
        raise ValueError("process must be both, absorption or emission")
    if (isinstance(temperature_kelvin, bool) or not isinstance(temperature_kelvin, Real)
            or not isfinite(temperature_kelvin) or temperature_kelvin < 0):
        raise ValueError("temperature_kelvin must be nonnegative and finite")
    if isinstance(coverage, bool) or not isinstance(coverage, Real) or not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    for name, value, minimum in (("max_quanta", max_quanta, 0), ("max_overlaps", max_overlaps, 1),
                                  ("target_index", target_index, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(symbols, str):
        raise ValueError("provide one element symbol per atom")
    symbols = tuple(symbols)
    geometry = np.asarray(coordinates_bohr, dtype=float)
    if len(symbols) < 2 or geometry.shape != (len(symbols), 3) or not np.isfinite(geometry).all():
        raise ValueError("provide at least two atoms and one finite Cartesian point per atom")
    if settings is None:
        settings = replace(DEFAULT_SETTINGS, nstates=max(DEFAULT_SETTINGS.nstates, target_index + 2))
    elif not isinstance(settings, Settings):
        raise ValueError("settings must be a pyscf_esd.cpu.Settings instance")
    if model == "vg" and target_index >= settings.nstates - 1:
        raise ValueError("request the target root and at least one higher reference root")

    from .electronic_workflow import optimize_ground, prepare_electronic

    diagnostics = {
        "settings": asdict(settings), "reference_target_index": int(target_index),
        "temperature_kelvin": float(temperature_kelvin), "target_coverage": float(coverage),
        "basis_attempts": {},
    }
    if model == "vg":
        from .cpu_workflow import run_cpu_vibronic

        if progress is not None:
            progress("Optimizing S0 and evaluating the vertical-gradient model")
        try:
            ground = optimize_ground(symbols, geometry, settings)
            vg = run_cpu_vibronic(symbols, ground, settings, target_index=target_index)
        except (ElectronicConvergenceError, StateTrackingError) as error:
            raise WorkflowIncomplete(str(error), {**diagnostics, "stage": "electronic_vg"}) from error
        diagnostics["ground_coordinates_bohr"] = np.asarray(ground).tolist()
        diagnostics["gap_00_hartree"] = vg.model.gap_00_hartree
        diagnostics["vibronic_terms"] = {"duschinsky": False, "herzberg_teller": False}
        solver = vg.spectrum
        used = "vg"
    else:
        if progress is not None:
            progress("Optimizing S0 and the tracked singlet and checking electronic derivatives")
        try:
            electronic = prepare_electronic(
                symbols, geometry, settings, target_index=target_index,
                direct_dipole=model == "auto" and len(symbols) == 2,
            )
        except (ElectronicConvergenceError, StateTrackingError) as error:
            raise WorkflowIncomplete(str(error), {**diagnostics, "stage": "electronic_ah"}) from error
        diagnostics.update(
            electronic=electronic.diagnostics,
            ground_coordinates_bohr=electronic.ground_coordinates_bohr.tolist(),
            excited_coordinates_bohr=electronic.excited_coordinates_bohr.tolist(),
            harmonic_gap_00_hartree=electronic.model.gap_00_hartree,
            vibronic_terms={"duschinsky": True, "herzberg_teller": electronic.dipole_derivatives_au is not None},
        )

        def ah_spectrum(**options):
            return electronic.model.spectrum(
                transition_dipole_au=electronic.transition_dipole_au,
                dipole_derivatives_au=electronic.dipole_derivatives_au, **options,
            )

        solver = ah_spectrum
        used = "ah_ht"
        if model == "auto":
            from .automatic_anharmonic import AnharmonicUnavailableError, prepare_anharmonic

            if progress is not None:
                progress("Screening direct electronic cuts for anharmonicity")
            try:
                anharmonic = prepare_anharmonic(electronic)
            except AnharmonicUnavailableError as error:
                raise WorkflowIncomplete(str(error), diagnostics) from error
            diagnostics["anharmonic"] = anharmonic.diagnostics
            status = anharmonic.diagnostics["status"]
            if status == "unsupported":
                reasons = " | ".join(anharmonic.diagnostics.get("reasons", ()))
                raise WorkflowIncomplete(
                    reasons or "the sampled potential exceeds the automatic anharmonic model",
                    diagnostics,
                )
            if status == "corrected":
                if temperature_kelvin > 0:
                    raise WorkflowIncomplete(
                        "The anharmonic thermal partition tail is not independently bounded. "
                        "Use 0 K or request harmonic AH explicitly",
                        diagnostics,
                    )

                def solver(**options):
                    try:
                        return anharmonic.spectrum(**options)
                    except AnharmonicUnavailableError as error:
                        raise WorkflowIncomplete(str(error), {}) from error

                used = "anharmonic_ht"
                if anharmonic.diagnostics.get("solver") == "adaptive_open_dvr":
                    used = "anharmonic_direct_diatomic"
                    diagnostics["vibronic_terms"] = {
                        "common_diatomic_coordinate": True,
                        "dipole_model": "direct_coordinate_dependent",
                    }
            elif status != "harmonic_screened":
                raise RuntimeError(f"unknown anharmonic screening status {status!r}")

    spectra = {}
    origins = {}
    failures = {}
    spectral_status = {}
    diagnostics["model_used"] = used
    for direction in ("absorption", "emission") if process == "both" else (process,):
        if progress is not None:
            progress(f"Converging the {direction} spectrum")
        try:
            spectrum, attempts = _adaptive_spectrum(
                solver, process=direction, temperature=temperature_kelvin, coverage=coverage,
                max_quanta=max_quanta, max_overlaps=max_overlaps,
            )
        except WorkflowIncomplete as error:
            failures[direction] = {"reason": str(error), **error.diagnostics}
            spectral_status[direction] = "incomplete"
            if direction not in error.partial_spectra:
                continue
            spectrum = error.partial_spectra[direction]
            attempts = error.diagnostics.get("basis_attempts", [])
        else:
            spectral_status[direction] = "complete"
        spectra[direction] = spectrum
        diagnostics["basis_attempts"][direction] = attempts
        zero_zero = (np.all(spectrum.initial_quanta == 0, axis=1)
                     & np.all(spectrum.final_quanta == 0, axis=1))
        if np.count_nonzero(zero_zero) == 1:
            origins[direction] = float(spectrum.photon_energy_hartree[zero_zero].item())
    diagnostics["predicted_00_lines_hartree"] = origins
    diagnostics["spectral_status"] = spectral_status
    if failures:
        diagnostics["spectrum_failures"] = failures
        summary = " | ".join(f"{direction} — {error['reason']}" for direction, error in failures.items())
        raise WorkflowIncomplete(summary, diagnostics, spectra)
    return VibronicResult(model, used, spectra, diagnostics)
