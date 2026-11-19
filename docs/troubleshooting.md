# Troubleshooting and convergence

Use the run status, the failing stage and the observable you need to choose the next action. A converged SCF, a stationary geometry and a converged radiative rate each require different checks. The [workflow](guide.md) records these checks in `diagnostics.json` and [physical quantities](numerical-results.md) in `properties.json`.

## Inspect a completed or incomplete run

The CLI diagnostics contain an outer run record and an inner workflow record. This script reads a calculation directory and prints the available failure and convergence information without starting an electronic calculation.

```python
import json
from pathlib import Path

run = Path("molecule-auto.esd")
record = json.loads((run / "diagnostics.json").read_text())
workflow = record.get("diagnostics", {})
print("Run status", record["status"])
print("Reason", record.get("reason"))
print("Model", record.get("model_used", workflow.get("model_used")))
print("Backend", workflow.get("electronic_backend"))
print("Stage", workflow.get("stage"))
print("Per-spectrum status", workflow.get("spectral_status", {}))
print("Available files", record.get("spectra", {}))
for attempt in workflow.get("reference_attempts", []):
    print("Reference attempt", attempt)
for process, failure in workflow.get("spectrum_failures", {}).items():
    print(process, failure["reason"])
```

Save the complete diagnostics when reporting a failure. A missing `properties.json` can mean preparation stopped before a spectrum was available. A missing `diagnostics.json` can indicate invalid arguments, an unavailable dependency, an unwritable destination or an interrupted file publication. Check the terminal output and process exit status as well.

## Installation and import failures

**`pyscf-esd` is unavailable.** Activate the environment where the package was installed. `python -m pyscf_esd --help` tests whether that interpreter can locate the module. Use `python -m pip`, so installation and execution use the same interpreter.

**`from pyscf import esd` fails.** Check both package locations with `python -c "import pyscf, pyscf_esd; print(pyscf.__file__, pyscf_esd.__file__)"`. An editable checkout needs to remain available at its installation path. The extension lives in the same distribution as the standalone package.

**PySCF installation fails on Windows.** Create the virtual environment inside Linux/WSL and install there. Native Windows has a different binary environment. Follow the [installation guide](installation.md).

**CUDA or cuTENSOR fails to load.** Check the GPU4PySCF, CuPy and cuTENSOR versions, the CUDA driver and the interpreter used to run the job. Use the [GPU environment check](gpu.md) before a molecular run.

## Input and destination failures

The XYZ reader accepts one complete frame with a comment line and at least two atoms. A trajectory with multiple frames must be reduced to the intended geometry. Coordinates must be finite and in ångström. Python `run_vibronic` takes bohr, while a PySCF `Mole` can be built with an explicitly declared coordinate unit.

Use a new output directory for each CLI run. An existing destination is rejected. Keep the previous calculation to compare its inputs and diagnostics with the new attempt.

The automatic singlet interface requires a closed-shell all-electron molecule. Charge and electron count must be consistent. Use the [open-shell interfaces](phosphorescence.md) for unrestricted or spin–orbit calculations.

## Electronic convergence and reference stability

SCF acceptance checks the residual as well as the solver's convergence flag. TDA roots are checked against the response operator. Failures can originate in a poor geometry, nearly degenerate orbital spaces, an insufficient root window, or the chosen electronic approximation.

For advanced CPU settings, construct `Settings` and pass it to `run_vibronic`. Increasing an iteration budget can allow the existing criteria to be reached.

```python
from pyscf_esd import run_vibronic
from pyscf_esd.cpu import Settings

def run_with_more_iterations(symbols, coordinates_bohr):
    settings = Settings(
        basis="def2-tzvpd", xc="pbe0", nstates=8,
        max_cycle=200, cphf_max_cycle=200,
    )
    return run_vibronic(symbols, coordinates_bohr, settings, progress=print)
```

This keeps the default SCF, response and residual tolerances. More iterations address an iterative convergence limit. They do not establish orbital stability.

An **internal instability** means a downhill restricted orbital rotation exists. The workflow can follow that direction, restart the same restricted model and require a lower-energy stable solution. A **restricted-to-unrestricted instability** means the restricted state has a spin-breaking direction. A verified instability can activate the explicit `fallback_xc` list. An **unresolved stability calculation** records that qualification failed and needs its own diagnosis.

When a fallback changes the functional, compare the effective `settings` with `settings_requested`, and reassess the target character. [Electronic reference checks](methods.md#electronic-reference-and-state-tracking)

## State identity and phase

The target root is labeled at relaxed S₀. Its overlap and separation from competing states are checked as geometries change. Large steps, near-degeneracy, a root crossing or loss of the target from the requested window can prevent an isolated-state assignment.

Inspect the recorded root labels, gaps, overlaps and optimizer rejection information. A larger `Settings.nstates` can help when the target window was too narrow. If the state cannot remain isolated, the present single-surface model cannot establish the requested path.

For prepared scans, both electronic states need consistent signed overlaps. Close the last-to-first link for a rotor, then phase the dipole and select each nuclear boundary condition. A smooth curve of squared dipoles alone cannot establish the sign required for coherent HT and rotor amplitudes. [Prepared state tracking](prepared-data.md)

## Minimum and derivative checks

Inspect gradient convergence, alignment, Hessian symmetry and vibrational eigenvalues at each optimized endpoint. A stationary planar geometry may be a saddle on the excited surface. Multiple minima, shallow torsions and changes of electronic character can require a larger nuclear model.

The analytic and fitted derivative routes have different checks. The CPU DFT AH route fits local excited-state curvature and signed dipole slopes from displaced electronic data. The supported GPU AH route differentiates the response analytically and performs independent consistency probes. Identify the actual route in the electronic diagnostics. [Derivative methods](methods.md#geometry-and-derivative-routes)

For prepared derivatives, retain the geometry, atom order, masses, electronic method, Cartesian frame and dipole phase. A Hessian supplied at S₀ does not establish the relaxed S₁ curvature required for AH.

## Nuclear basis and rate convergence

At finite temperature, the initial basis must represent its Boltzmann population. The final basis must cover dipole strength. Positive-photon moments qualify the observables with energy weighting. Absorption uses M₁, radiative emission rate uses M₃ and emitted power uses M₄. High-energy tails can matter more for these totals than for the visual envelope.

Inspect `basis_attempts`, `target_coverage`, `spectral_status` and each spectrum's `positive_photon_moments` report. Increase the vibrational budgets only when these fields identify a truncated basis.

```bash
pyscf-esd molecule.xyz --model ah --temperature 298 \
  --max-quanta 40 --max-overlaps 500000 --output ah-expanded.esd
```

More overlaps increase runtime and memory use. Compare successive totals, centroids and relevant band ratios in addition to the recorded bounds. Anharmonic solvers have their own retained-level, partition, domain and potential checks. The generic harmonic omitted-moment bound does not apply to every prepared anharmonic profile.

## Automatic anharmonic screening

`auto` reports whether its sampled model was accepted as harmonic, corrected with a supported local anharmonic model, or unsupported. Inspect `diagnostics["anharmonic"]` and its recorded reasons. A constrained one-well local expansion may fail on a periodic torsion, a double well or strong multimode coupling.

If a harmonic comparison is scientifically useful, request a separate `--model ah` run and retain its model label. For a large-amplitude coordinate, assemble the [prepared rotor or coupled potential](prepared-data.md) and converge its nuclear grid and thermal coverage.

## Band shape, broadening and energy offsets

Check energy units, whether the ordinate is a density, the absorption/emission photon factor, the width convention and the initial temperature before interpreting a spectral mismatch. Compare the native-energy prediction and any aligned display with their actual widths recorded. A rigid shift preserves band separations. Broadening changes peak heights and overlap between neighboring bands. [Spectral comparisons](spectral-comparison.md)

Relative-intensity discrepancies can reflect equilibrium geometry, mode displacements, Duschinsky rotation, HT interference, anharmonic coupling or missing electronic bands. The [validation guide](validation.md) gives controls for these effects in the retained benchmarks.

## Report a reproducible issue

Include the source revision, Python/PySCF/GPU4PySCF versions, platform, exact command or Python entry point, input geometry and charge, selected functional/basis/state, terminal failure, diagnostics and whether the issue uses an external archive. For GPU reports include the GPU and driver versions. Share only data you are authorized to publish. [Issue tracker](https://github.com/HenriqueCSJ/PySCF_ESD/issues)
