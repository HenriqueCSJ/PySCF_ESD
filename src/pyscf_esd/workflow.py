"""The public analytic-first workflow for closed-shell singlet spectroscopy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Integral, Real
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .anharmonic import AnharmonicSpectrum
from .cpu import ElectronicConvergenceError, ReferenceStabilityUnresolvedError, Settings, UnstableReferenceError
from .harmonic import HarmonicSpectrum
from .pyscf_adapter import AdiabaticHessianParameters
from .state_tracking import StateTrackingError
from .vibronic_properties import condon_ht_properties, harmonic_deformation_check, harmonic_model_properties


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
    spectra: dict[str, HarmonicSpectrum | AnharmonicSpectrum]
    diagnostics: dict

    @property
    def physical_properties(self) -> dict:
        """Quantitative properties of the retained spectrum and its input model."""
        from .spectral_properties import summarize_spectral_properties

        assignment = {
            "ah_fc": "harmonic_quanta",
            "ah_ht": "harmonic_quanta",
            "vg": "harmonic_quanta",
            "anharmonic_ht": "vci_dominant_harmonic_labels",
            "anharmonic_direct_diatomic": "dvr_levels",
        }.get(self.model_used)
        properties = summarize_spectral_properties(
            self.spectra, spectral_status=self.diagnostics.get("spectral_status"),
            temperature_kelvin=self.diagnostics.get("temperature_kelvin"),
            assignment_scheme=assignment,
        )
        properties["model_used"] = self.model_used
        for key in ("reference_target_index", "atom_symbols", "electronic_method", "electronic_validation"):
            if key in self.diagnostics:
                properties[key] = self.diagnostics[key]
        for process, spectrum in self.spectra.items():
            decomposition = condon_ht_properties(spectrum, process)
            if decomposition is not None:
                properties["spectra"][process]["condon_herzberg_teller"] = decomposition
        electronic = self.diagnostics.get("electronic", {})
        for name in ("electronic_relaxation", "harmonic_reference"):
            source = electronic if name in electronic else self.diagnostics
            if name in source:
                properties[name] = source[name]
        if "electronic_relaxation" in properties and "harmonic_reference" in properties:
            properties["harmonic_cross_minimum_check"] = harmonic_deformation_check(
                properties["electronic_relaxation"], properties["harmonic_reference"],
            )
        return properties


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


def _prepare_reference(prepare, settings: Settings, fallback_xc: Sequence[str], attempts: list[dict], progress):
    """Restart preparation for an allowed stability recovery on one Hamiltonian at a time."""
    functionals = tuple(dict.fromkeys((settings.xc, *fallback_xc)))
    for index, xc in enumerate(functionals):
        candidate = settings if index == 0 else replace(settings, xc=xc, stability_checks="all")
        while True:
            try:
                result = prepare(candidate)
            except UnstableReferenceError as error:
                attempts.append({
                    "xc": xc, "stability_checks": candidate.stability_checks, "status": "unstable",
                    "channel": error.channel, "geometry_stage": error.stage, "reason": str(error),
                })
                if error.channel == "internal" and candidate.stability_checks == "minima":
                    if progress is not None:
                        progress(f"Restarting {xc} with restricted orbital-stability recovery at each point")
                    candidate = replace(candidate, stability_checks="all")
                    continue
                if error.channel == "restricted_to_unrestricted" and index + 1 < len(functionals):
                    if progress is not None:
                        progress(f"Restarting the electronic workflow with alternate XC {functionals[index + 1]}")
                    break
                raise
            except ReferenceStabilityUnresolvedError as error:
                attempts.append({
                    "xc": xc, "stability_checks": candidate.stability_checks, "status": "stability_unresolved",
                    "channel": error.channel, "geometry_stage": error.stage, "reason": str(error),
                })
                raise
            except WorkflowIncomplete:
                attempts.append({
                    "xc": xc, "stability_checks": candidate.stability_checks, "status": "incomplete",
                })
                raise
            except (ElectronicConvergenceError, StateTrackingError) as error:
                attempts.append({
                    "xc": xc, "stability_checks": candidate.stability_checks, "status": "electronic_failure",
                    "failure_kind": "state_tracking" if isinstance(error, StateTrackingError) else "convergence",
                    "reason": str(error),
                })
                raise
            else:
                attempts.append({
                    "xc": xc, "stability_checks": candidate.stability_checks, "status": "prepared",
                })
                return result


def _validate_spectral_options(process, temperature_kelvin, coverage, max_quanta, max_overlaps):
    if process not in ("both", "absorption", "emission"):
        raise ValueError("process must be both, absorption or emission")
    if (isinstance(temperature_kelvin, bool) or not isinstance(temperature_kelvin, Real)
            or not isfinite(temperature_kelvin) or temperature_kelvin < 0):
        raise ValueError("temperature_kelvin must be nonnegative and finite")
    if isinstance(coverage, bool) or not isinstance(coverage, Real) or not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    for name, value, minimum in (("max_quanta", max_quanta, 0), ("max_overlaps", max_overlaps, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def run_prepared_ah(
    parameters: AdiabaticHessianParameters,
    transition_dipole_au: ArrayLike,
    *,
    electronic_method: str,
    dipole_derivatives_au: ArrayLike | None = None,
    process: Literal["both", "absorption", "emission"] = "both",
    temperature_kelvin: float = 0.0,
    coverage: float = 0.995,
    max_quanta: int = DEFAULT_MAX_QUANTA,
    max_overlaps: int = 250_000,
    progress: Callable[[str], None] | None = None,
) -> VibronicResult:
    """Run adaptive AH spectroscopy from explicitly prepared electronic data.

    Supply an AH model, normally built with `adiabatic_hessian`, and a
    length-gauge transition dipole at the ground minimum. Optional linear
    derivatives use its ground normal coordinates and the same phase. The
    caller qualifies the electronic surfaces, derivatives and state identity.
    Absorption and emission share the input gap and the ordinary workflow
    coverage checks. There are no electronic calculations or method retries.
    """
    _validate_spectral_options(process, temperature_kelvin, coverage, max_quanta, max_overlaps)
    if not isinstance(parameters, AdiabaticHessianParameters):
        raise ValueError("supply AdiabaticHessianParameters from the prepared electronic surfaces")
    if not isinstance(electronic_method, str) or not electronic_method.strip():
        raise ValueError("electronic_method must identify the method used to prepare both surfaces")

    def solver(**options):
        return parameters.spectrum(
            transition_dipole_au=transition_dipole_au, dipole_derivatives_au=dipole_derivatives_au, **options,
        )

    diagnostics = {
        "electronic_method": electronic_method.strip(),
        "electronic_validation": "caller_supplied",
        "temperature_kelvin": float(temperature_kelvin),
        "target_coverage": float(coverage),
        "basis_attempts": {},
        "harmonic_gap_00_hartree": parameters.gap_00_hartree,
        "harmonic_reference": harmonic_model_properties(
            parameters, dipole_derivatives_au=dipole_derivatives_au,
            transition_dipole_at_ground_au=transition_dipole_au,
        ),
        "vibronic_terms": {"duschinsky": True, "herzberg_teller": dipole_derivatives_au is not None},
    }
    used = "ah_fc" if dipole_derivatives_au is None else "ah_ht"
    return _qualify_spectra(
        solver, "ah", used, diagnostics, process=process, temperature_kelvin=temperature_kelvin,
        coverage=coverage, max_quanta=max_quanta, max_overlaps=max_overlaps, progress=progress,
    )


def run_vibronic(
    symbols: tuple[str, ...],
    coordinates_bohr: ArrayLike,
    settings: Settings | None = None,
    *,
    target_index: int = 0,
    model: Literal["auto", "ah", "vg"] = "auto",
    backend: Literal["cpu", "gpu"] = "cpu",
    solvent: str | None = None,
    process: Literal["both", "absorption", "emission"] = "both",
    temperature_kelvin: float = 0.0,
    coverage: float = 0.995,
    max_quanta: int = DEFAULT_MAX_QUANTA,
    max_overlaps: int = 250_000,
    fallback_xc: Sequence[str] = (),
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
    The GPU backend uses analytic DF LR-TDA derivatives. Its optional solvent
    name selects equilibrium C-PCM. An unsupported analytic response fails
    with the reason recorded in the run error.
    An internal orbital instability can retry the same restricted model.
    Supply ordered ``fallback_xc`` alternatives to permit a fresh electronic
    preparation after a restricted-to-unrestricted instability. The target
    index is defined at the ground minimum of each attempted functional.
    """
    if model not in ("auto", "ah", "vg"):
        raise ValueError("model must be auto, ah or vg")
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    if solvent is not None and (not isinstance(solvent, str) or not solvent.strip() or backend != "gpu"):
        raise ValueError("a named equilibrium C-PCM solvent requires backend='gpu'")
    _validate_spectral_options(process, temperature_kelvin, coverage, max_quanta, max_overlaps)
    if isinstance(target_index, bool) or not isinstance(target_index, Integral) or target_index < 0:
        raise ValueError("target_index must be an integer >= 0")
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
    if isinstance(fallback_xc, str):
        raise ValueError("fallback_xc must be a sequence of XC names")
    fallback_xc = tuple(fallback_xc)
    if any(not isinstance(xc, str) or not xc.strip() for xc in fallback_xc):
        raise ValueError("fallback_xc must contain nonempty XC names")
    if model == "vg" and target_index >= settings.nstates - 1:
        raise ValueError("request the target root and at least one higher reference root")

    requested = asdict(settings)
    attempts = []
    try:
        result = _prepare_reference(
            lambda candidate: _run_vibronic_model(
                symbols, geometry, candidate, target_index=target_index, model=model, backend=backend,
                solvent=solvent, process=process,
                temperature_kelvin=temperature_kelvin, coverage=coverage, max_quanta=max_quanta,
                max_overlaps=max_overlaps, progress=progress,
            ),
            settings, fallback_xc, attempts, progress,
        )
    except WorkflowIncomplete as error:
        error.diagnostics.update(settings_requested=requested, reference_attempts=attempts,
                                 electronic_backend=backend, solvent=solvent)
        raise
    except (ElectronicConvergenceError, StateTrackingError) as error:
        stage = "electronic_vg" if model == "vg" else "electronic_ah"
        raise WorkflowIncomplete(str(error), {
            "settings_requested": requested, "reference_attempts": attempts, "stage": stage,
            "electronic_backend": backend, "solvent": solvent,
        }) from error
    result.diagnostics.update(settings_requested=requested, reference_attempts=attempts)
    return result


def _run_vibronic_model(
    symbols, geometry, settings, *, target_index, model, backend, solvent, process, temperature_kelvin,
    coverage, max_quanta, max_overlaps, progress,
) -> VibronicResult:
    from .electronic_workflow import optimize_ground, prepare_electronic

    diagnostics = {
        "settings": asdict(settings), "reference_target_index": int(target_index),
        "electronic_backend": backend, "solvent": solvent,
        "atom_symbols": list(symbols),
        "temperature_kelvin": float(temperature_kelvin), "target_coverage": float(coverage),
        "basis_attempts": {},
    }
    if model == "vg":
        if progress is not None:
            progress("Optimizing S0 and evaluating the vertical-gradient model")
        if backend == "gpu":
            from .gpu_adiabatic import prepare_gpu_vg

            ground, vg = prepare_gpu_vg(symbols, geometry, settings, target_index=target_index, solvent=solvent)
        else:
            from .cpu_workflow import run_cpu_vibronic

            ground = optimize_ground(symbols, geometry, settings)
            vg = run_cpu_vibronic(symbols, ground, settings, target_index=target_index)
        diagnostics["ground_coordinates_bohr"] = np.asarray(ground).tolist()
        diagnostics["gap_00_hartree"] = vg.model.gap_00_hartree
        diagnostics["harmonic_reference"] = harmonic_model_properties(vg.model)
        diagnostics["vibronic_terms"] = {"duschinsky": False, "herzberg_teller": False}
        solver = vg.spectrum
        used = "vg"
    else:
        if progress is not None:
            progress("Optimizing S0 and the tracked singlet and checking electronic derivatives")
        direct_dipole = model == "auto" and len(symbols) == 2
        if backend == "gpu":
            from .gpu_adiabatic import prepare_gpu_electronic

            electronic = prepare_gpu_electronic(
                symbols, geometry, settings, target_index=target_index, direct_dipole=direct_dipole, solvent=solvent,
            )
        else:
            electronic = prepare_electronic(
                symbols, geometry, settings, target_index=target_index, direct_dipole=direct_dipole,
            )
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
        used = "ah_fc" if electronic.dipole_derivatives_au is None else "ah_ht"
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

    return _qualify_spectra(
        solver, model, used, diagnostics, process=process, temperature_kelvin=temperature_kelvin,
        coverage=coverage, max_quanta=max_quanta, max_overlaps=max_overlaps, progress=progress,
    )


def _qualify_spectra(
    solver, model, used, diagnostics, *, process, temperature_kelvin, coverage, max_quanta, max_overlaps, progress,
):
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
