"""Run the standard PySCF ESD pipeline from a single XYZ file."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .workflow import DEFAULT_MAX_QUANTA, DEFAULT_SETTINGS, VibronicResult, WorkflowIncomplete, run_vibronic


_ANGSTROM_PER_BOHR = 0.529177210903
_CM_PER_HARTREE = 219474.63136320


def _package_version():
    try:
        return version("pyscf-esd")
    except PackageNotFoundError:
        return "source"


def read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read one Angstrom XYZ geometry, preserving the supplied atom order."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ requires an atom count and comment line")
    try:
        count = int(lines[0])
    except ValueError as error:
        raise ValueError("the first XYZ line must be an atom count") from error
    atom_lines = lines[2:count+2]
    if count < 2 or len(atom_lines) != count or any(line.strip() for line in lines[count+2:]):
        raise ValueError("XYZ must contain one complete geometry with at least two atoms")
    symbols, coordinates = [], []
    for index, line in enumerate(atom_lines, start=1):
        row = line.split()
        if len(row) != 4:
            raise ValueError(f"XYZ atom {index} must contain a symbol and three coordinates")
        symbols.append(row[0])
        try:
            point = [float(item) for item in row[1:]]
        except ValueError as error:
            raise ValueError(f"XYZ atom {index} has invalid coordinates") from error
        if not all(isfinite(value) for value in point):
            raise ValueError(f"XYZ atom {index} has nonfinite coordinates")
        coordinates.append(point)
    return tuple(symbols), np.asarray(coordinates) / _ANGSTROM_PER_BOHR


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} in the run diagnostics")


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_value) + "\n",
                    encoding="utf-8")


def _write_sticks(path: Path, spectrum, process: str) -> tuple[np.ndarray, np.ndarray]:
    energy = spectrum.photon_energy_hartree
    frequency = energy * _CM_PER_HARTREE
    weighted = energy**(1 if process == "absorption" else 3) * spectrum.thermal_strength_au2
    first = getattr(spectrum, "initial_quanta", None)
    last = getattr(spectrum, "final_quanta", None)
    dipoles = spectrum.transition_dipole_au
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "wavenumber_cm^-1", "energy_hartree", "population", "dipole_strength_au^2",
            "photon_weighted_strength_relative_units", "initial_assignment_quanta", "final_assignment_quanta",
            "transition_dipole_x_au", "transition_dipole_y_au", "transition_dipole_z_au",
        ))
        for index, (wn, er, population, strength, value) in enumerate(
            zip(frequency, energy, spectrum.population, spectrum.dipole_strength_au2, weighted, strict=True),
        ):
            writer.writerow((
                wn, er, population, strength, value,
                "" if first is None else " ".join(str(number) for number in first[index]),
                "" if last is None else " ".join(str(number) for number in last[index]),
                *(("", "", "") if dipoles is None else dipoles[index]),
            ))
    return frequency, weighted


def _write_profile(path: Path, positions: np.ndarray, strengths: np.ndarray, sigma: float) -> bool:
    supported = strengths > 0
    if not np.any(supported):
        return False
    positions, strengths = positions[supported], strengths[supported]
    lower = max(0.0, float(positions.min()) - 6 * sigma)
    upper = float(positions.max()) + 6 * sigma
    segments = (upper - lower) / (sigma / 5)
    if not isfinite(segments) or segments > 200_000:
        raise ValueError("profile grid is too large; increase --sigma or pass --sigma 0 for sticks only")
    points = np.linspace(lower, upper, int(np.ceil(segments)) + 1)
    profile = np.zeros_like(points)
    normal = 1 / (np.sqrt(2 * np.pi) * sigma)
    start = np.searchsorted(points, positions - 6 * sigma)
    stop = np.searchsorted(points, positions + 6 * sigma, side="right")
    for center, strength, lo, hi in zip(positions, strengths, start, stop, strict=True):
        profile[lo:hi] += strength * normal * np.exp(-0.5 * ((points[lo:hi] - center) / sigma)**2)
    np.savetxt(path, np.column_stack((points, profile)), delimiter=",",
               header="wavenumber_cm^-1,photon_weighted_density_relative_units_per_cm^-1", comments="")
    return True


def _publish(output: Path, payload: dict, result: VibronicResult | None, sigma: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pyscf-esd-", dir=output.parent) as temporary:
        staged = Path(temporary)
        files = {}
        if result is not None:
            for process, spectrum in result.spectra.items():
                status = result.diagnostics["spectral_status"][process]
                stem = process if status == "complete" else f"{process}_partial"
                filename = f"{stem}_sticks.csv"
                frequency, strength = _write_sticks(staged / filename, spectrum, process)
                entry = {"sticks": filename, "count": len(frequency), "coverage_status": status}
                filename = f"{stem}_profile.csv"
                if sigma > 0 and _write_profile(staged / filename, frequency, strength, sigma):
                    entry["profile"] = filename
                files[process] = entry
        _write_json(staged / "diagnostics.json", {**payload, "spectra": files})
        # Reserve the final directory without replacing an existing run. Write
        # diagnostics last so an interrupted publication has no success record.
        output.mkdir()
        for file in sorted(staged.iterdir(), key=lambda item: item.name == "diagnostics.json"):
            file.rename(output / file.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    parser.add_argument("xyz", type=Path, help="single XYZ molecular geometry in Angstrom")
    parser.add_argument("--output", type=Path, help="new result directory (default: XYZ name with .esd)")
    parser.add_argument("--model", choices=("auto", "ah", "vg"), default="auto",
                        help="automatic anharmonic screening, explicit harmonic AH, or explicit VG")
    parser.add_argument("--process", choices=("both", "absorption", "emission"), default="both")
    parser.add_argument("--state", type=int, default=1, help="singlet root number at relaxed S0 (1 = S1)")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--basis", default=DEFAULT_SETTINGS.basis)
    parser.add_argument("--xc", default=DEFAULT_SETTINGS.xc)
    parser.add_argument("--temperature", type=float, default=0.0, metavar="K")
    parser.add_argument("--sigma", type=float, default=50.0, metavar="CM^-1",
                        help="Gaussian standard deviation (0 writes sticks only)")
    parser.add_argument("--max-quanta", type=int, default=DEFAULT_MAX_QUANTA)
    parser.add_argument("--max-overlaps", type=int, default=250_000)
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.xyz.with_suffix(".esd")
    if output.exists():
        parser.error(f"result destination already exists: {output}")
    if not isfinite(args.sigma) or args.sigma < 0:
        parser.error("--sigma must be nonnegative and finite")
    if args.state < 1:
        parser.error("--state must be at least 1")

    payload = {"schema_version": 1, "pyscf_esd_version": _package_version(), "input_xyz": str(args.xyz.resolve()),
               "model_requested": args.model, "gaussian_sigma_cm^-1": args.sigma}
    try:
        symbols, geometry = read_xyz(args.xyz)
        from pyscf import lib

        if not os.environ.get("OMP_NUM_THREADS"):
            lib.num_threads(min(os.cpu_count() or 1, 4 if len(symbols) <= 12 else 8))
        settings = replace(DEFAULT_SETTINGS, basis=args.basis, xc=args.xc, charge=args.charge,
                           nstates=max(DEFAULT_SETTINGS.nstates, args.state+2))
        payload["openmp_threads"] = lib.num_threads()
        try:
            result = run_vibronic(
                symbols, geometry, settings, target_index=args.state-1, model=args.model, process=args.process,
                temperature_kelvin=args.temperature, max_quanta=args.max_quanta, max_overlaps=args.max_overlaps,
                progress=lambda message: print(f"pyscf-esd: {message}", file=sys.stderr, flush=True),
            )
        except WorkflowIncomplete as error:
            partial = None
            if error.partial_spectra:
                partial = VibronicResult(args.model, error.diagnostics["model_used"],
                                         error.partial_spectra, error.diagnostics)
            _publish(output, {**payload, "status": "incomplete", "reason": str(error),
                              "diagnostics": error.diagnostics}, partial, args.sigma)
            print(f"pyscf-esd: {error}. Diagnostics: {output / 'diagnostics.json'}", file=sys.stderr)
            return 2
        _publish(output, {**payload, "status": "complete", "model_used": result.model_used,
                          "diagnostics": result.diagnostics}, result, args.sigma)
    except ImportError as error:
        print(f"pyscf-esd: {error}. Install with python -m pip install '.[workflow]' on Linux or WSL.",
              file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError) as error:
        print(f"pyscf-esd: {error}", file=sys.stderr)
        return 2
    print(f"{result.model_used}: {output / 'diagnostics.json'}")
    return 0
