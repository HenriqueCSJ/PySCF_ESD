"""Compare analytic-only CPU and RTX 4070 water VG spectroscopy."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

TEMPERATURE_K = 298.15
INITIAL_QUANTA = 3
FINAL_QUANTA = 12


def comparison(cpu, gpu, reference):
    import numpy as np
    from pyscf import gto
    from pyscf.data.nist import HARTREE2WAVENUMBER
    from pyscf_esd.state_tracking import orbital_overlap_blocks, track_state

    a, b = cpu.center, gpu.center
    cross = gto.intor_cross("int1e_ovlp", a.mol, b.mol)
    occupation = 2 * (np.arange(a.mo_coeff.shape[1]) < a.nocc)
    occupied, virtual = orbital_overlap_blocks(a.mo_coeff, b.mo_coeff, cross, occupation, occupation)
    links = [track_state(amplitude, b.amplitudes, occupied, virtual) for amplitude in a.amplitudes]
    if [link.index for link in links] != list(range(len(links))):
        raise RuntimeError("CPU and GPU root assignments do not agree")
    phases = np.array([link.phase for link in links])
    result = {
        "minimum_root_overlap": min(link.overlap for link in links),
        "minimum_root_margin": min(link.margin for link in links),
        "ground_energy_error_hartree": abs(a.ground_energy_hartree - b.ground_energy_hartree),
        "excitation_error_hartree": max_error(a.excitation_energies_hartree, b.excitation_energies_hartree),
        "gpu_excitation_vs_retained_cpu_hartree": max_error(b.excitation_energies_hartree,
                                                         reference["excitation_energies_hartree"]),
        "phase_aligned_dipole_error_au": max_error(a.transition_dipoles_au, phases[:, None] * b.transition_dipoles_au),
        "ground_gradient_error_hartree_per_bohr": max_error(cpu.ground_gradient_hartree_per_bohr,
                                                          gpu.ground_gradient_hartree_per_bohr),
        "excited_gradient_error_hartree_per_bohr": max_error(cpu.excited_gradient_hartree_per_bohr,
                                                           gpu.excited_gradient_hartree_per_bohr),
        "ground_hessian_error_hartree_per_bohr2": max_error(cpu.ground_hessian_hartree_per_bohr2,
                                                           gpu.ground_hessian_hartree_per_bohr2),
        "mode_frequency_error_cm-1": HARTREE2WAVENUMBER * max_error(cpu.model.frequencies_hartree,
                                                                 gpu.model.frequencies_hartree),
        "origin_error_cm-1": HARTREE2WAVENUMBER * abs(cpu.model.gap_00_hartree - gpu.model.gap_00_hartree),
    }
    for process in ("absorption", "emission"):
        options = dict(temperature_kelvin=TEMPERATURE_K, initial_max_quanta=INITIAL_QUANTA,
                       final_max_quanta=FINAL_QUANTA, process=process)
        spectra = [x.spectrum(**options) for x in (cpu, gpu)]
        keyed = [{(tuple(i), tuple(f)): (energy, strength)
                  for i, f, energy, strength in zip(s.initial_quanta, s.final_quanta,
                                                    s.photon_energy_hartree, s.thermal_strength_au2)}
                 for s in spectra]
        if keyed[0].keys() != keyed[1].keys():
            raise RuntimeError("CPU and GPU generated distinct positive-energy assignments")
        left = np.array(list(keyed[0].values()))
        right = np.array([keyed[1][assignment] for assignment in keyed[0]])
        result[f"{process}_line_energy_error_cm-1"] = HARTREE2WAVENUMBER * max_error(left[:, 0], right[:, 0])
        result[f"{process}_max_strength_error_au2"] = max_error(left[:, 1], right[:, 1])
        result[f"{process}_relative_total_strength_error"] = abs(right[:, 1].sum() / left[:, 1].sum() - 1)
        result[f"{process}_line_count"] = len(keyed[0])
    return result


def max_error(first, second):
    import numpy as np

    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def run_series(run, coordinates, settings, synchronize, repeats, started):
    # Every trial creates fresh electronic objects from MINAO, even after
    # library/device warmup. Include both sets of thermal line assignments.
    wall_times = []
    analytic_times = []
    result = None
    for _ in range(repeats + 1):
        start = perf_counter()
        result = run(("O", "H", "H"), coordinates, settings, target_index=0)
        synchronize()
        analytic_times.append(perf_counter() - start)
        for process in ("absorption", "emission"):
            result.spectrum(temperature_kelvin=TEMPERATURE_K, initial_max_quanta=INITIAL_QUANTA,
                            final_max_quanta=FINAL_QUANTA, process=process)
        synchronize()
        wall_times.append(perf_counter() - start)
        print(f"{run.__name__} trial {len(wall_times)}  {wall_times[-1]:.3f} s", flush=True)
        if len(wall_times) == 1:
            first_use = perf_counter() - started
        if result.displacements or result.normal_dipole_derivatives_au is not None:
            raise RuntimeError("the analytic VG benchmark launched displaced work")
    return result, {
        "first_workflow_seconds": wall_times[0],
        "first_use_with_local_imports_seconds": first_use,
        "warm_seconds": wall_times[1:],
        "warm_median_seconds": median(wall_times[1:]),
        "warm_analytic_vg_seconds": analytic_times[1:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="retained Water_CPU reference.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("positive thread and repeat counts are required")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = {"symbols": ["O", "H", "H"], "basis": "6-31+g*", "xc": "pbe0", "grid_level": 5, "roots": 5}
    if reference.get("settings") != expected:
        raise ValueError("the retained water reference uses a different method or grid")

    cpu_start = perf_counter()
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf_esd.cpu import Settings
    from pyscf_esd.cpu_workflow import run_cpu_vibronic

    lib.num_threads(args.threads)
    coordinates = np.array(reference["geometry_bohr"])
    settings = Settings("6-31+g*", "pbe0", nstates=5, grid_level=5, scf_conv_tol=1e-12,
                        scf_residual_tol=1e-8, td_residual_tol=1e-9, direct_scf_tol=1e-14)
    cpu, cpu_time = run_series(run_cpu_vibronic, coordinates, settings, lambda: None, args.repeats, cpu_start)

    gpu_start = perf_counter()
    import cupy
    import gpu4pyscf
    from pyscf_esd.gpu import run_gpu_vibronic

    device = cupy.cuda.Device()
    gpu, gpu_time = run_series(run_gpu_vibronic, coordinates, settings, device.synchronize, args.repeats, gpu_start)
    device_name = cupy.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
    values = comparison(cpu, gpu, reference)
    thresholds = {
        "ground_energy_error_hartree": 1e-7,
        "excitation_error_hartree": 1e-6,
        "gpu_excitation_vs_retained_cpu_hartree": 1e-6,
        "phase_aligned_dipole_error_au": 1e-5,
        "ground_gradient_error_hartree_per_bohr": 1e-5,
        "excited_gradient_error_hartree_per_bohr": 1e-5,
        "ground_hessian_error_hartree_per_bohr2": 2e-4,
        "mode_frequency_error_cm-1": 1.0,
        "origin_error_cm-1": 1.0,
        "absorption_line_energy_error_cm-1": 2.0,
        "emission_line_energy_error_cm-1": 2.0,
        "absorption_max_strength_error_au2": 1e-4,
        "emission_max_strength_error_au2": 1e-4,
        "absorption_relative_total_strength_error": 0.01,
        "emission_relative_total_strength_error": 0.01,
    }
    report = {
        "geometry_bohr": reference["geometry_bohr"],
        "recipe": expected,
        "threads": args.threads,
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "temperature_kelvin": TEMPERATURE_K,
        "initial_max_quanta": INITIAL_QUANTA,
        "final_max_quanta": FINAL_QUANTA,
        "target_index": 0,
        "displaced_geometries": 0,
        "pyscf_version": pyscf.__version__,
        "gpu4pyscf_version": gpu4pyscf.__version__,
        "cupy_version": cupy.__version__,
        "numpy_version": np.__version__,
        "cutensor_version": importlib.metadata.version("cutensor-cu12"),
        "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "device": device_name,
        "cpu_timing": cpu_time,
        "gpu_timing": gpu_time,
        "timing_protocol": "Fresh CPU then GPU object series in one process; GPU first-use import time adds GPU imports after the common Python stack is loaded. Per-trial times include fresh electronic objects and both spectra, with CUDA synchronization.",
        "median_speedup": cpu_time["warm_median_seconds"] / gpu_time["warm_median_seconds"],
        "ground_hessian_quadrature_grid_response": {"cpu": False, "gpu": False},
        "gpu_direct_jk": gpu.center.direct_jk_evidence(),
        "cpu": {
            "ground_energy_hartree": cpu.center.ground_energy_hartree,
            "vertical_s1_hartree": float(cpu.center.excitation_energies_hartree[0]),
            "vg_origin_hartree": float(cpu.model.gap_00_hartree),
            "scf_residual_hartree": cpu.center.scf_residual,
            "max_tda_residual_hartree": float(max(cpu.center.tda_residuals)),
            "tda_route": cpu.center.tda_route,
        },
        "gpu": {
            "ground_energy_hartree": gpu.center.ground_energy_hartree,
            "vertical_s1_hartree": float(gpu.center.excitation_energies_hartree[0]),
            "vg_origin_hartree": float(gpu.model.gap_00_hartree),
            "scf_residual_hartree": gpu.center.scf_residual,
            "max_tda_residual_hartree": float(max(gpu.center.tda_residuals)),
            "tda_route": gpu.center.tda_route,
        },
        "accuracy": values,
        "acceptance_maxima": thresholds,
    }
    report["checks"] = {name: bool(0 <= values[name] <= limit) for name, limit in thresholds.items()}
    report["checks"]["same_root_character"] = bool(values["minimum_root_overlap"] >= 0.9999)
    report["passed"] = all(report["checks"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("matched water benchmark exceeded one or more accuracy thresholds")


if __name__ == "__main__":
    main()
