# Installation and workflow guide

[Project home](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/README.md) · [Python API](api.md) · [Methods](methods.md) · [Validation](validation.md)

This guide follows the automatic singlet calculation from input to output. For a complete environment setup use [installation](installation.md). The [first calculation](quickstart.md) provides a runnable molecular example and the [command-line reference](command-line.md) lists every flag and default.

## Install

Use Python 3.10 or newer. For electronic structure, use Linux or WSL and create the environment in that system.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[workflow]'
python -c "import pyscf, scipy; print(pyscf.__version__, scipy.__version__)"
pyscf-esd --version
```

Choose the extra that matches the calculation.

- `.` installs NumPy and the base spectral solvers, including harmonic sticks and the one-coordinate constant-mass DVR.
- `.[pyscf]` installs PySCF 2.15 or newer for the adapters and explicitly selected electronic routes. The minimum version includes the corrected CPHF Krylov solver. [PySCF v2.15.0](https://github.com/pyscf/pyscf/releases/tag/v2.15.0)
- `.[anharmonic]` adds SciPy for the additional anharmonic solvers.
- `.[workflow]` installs both extras for the CLI and full CPU workflow. Use `pip install -e '.[workflow]'` when editing the repository.

The command `python -m pyscf_esd` is equivalent to `pyscf-esd` in the active environment.

## Input and model

Use one XYZ frame with an atom count, a comment line, and one element symbol and three finite Cartesian coordinates per atom, in ångström. The CLI preserves atom order. Pass the molecular charge explicitly when needed. The automatic singlet workflow requires an even electron count and a closed-shell all-electron reference. See the [README water input](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/README.md#quick-start) and the [separate open-shell and phosphorescence APIs](phosphorescence.md).

```sh
# Default automatic calculation, both directions at 0 K
pyscf-esd molecule.xyz --output auto.esd

# Harmonic AH at 298 K with an 80 cm^-1 Gaussian FWHM
pyscf-esd molecule.xyz --model ah --temperature 298 --sigma 33.9728720115 --output ah-298.esd

# VG absorption sticks, selecting the second singlet root at relaxed S0
pyscf-esd molecule.xyz --model vg --process absorption --state 2 --sigma 0 --output vg-s2.esd
```

- `auto` optimizes both surfaces and screens direct electronic cuts. It returns the tested harmonic model when the screening permits it. The direct DVR and local VCI routes support positive temperatures when the partition and spectral coverage meet the requested tolerances. [Finite-temperature options](using-methods.md#finite-temperature-anharmonicity)
- `ah` builds harmonic wells at the two independently optimized geometries. It includes Duschinsky rotation and first-order Herzberg–Teller amplitudes. Use it for a specified harmonic model at either 0 K or finite temperature.
- `vg` uses the ground-state Hessian and the excited-state gradient at the relaxed ground geometry to define two harmonic wells of equal curvature. The CLI uses a constant transition dipole. Its emission comes from the relaxed minimum of the VG model.

The periodic torsion workflow uses [a separate Python API](api.md#periodic-rotor-and-thermal-profile). In that interface the Gaussian width is a **FWHM**, while the CLI takes the **standard deviation**. The conversion is `FWHM = 2 sqrt(2 ln 2) × sigma`.

## Command-line options

The [full CLI reference](command-line.md) also covers backend selection, solvents, exit status and output publication. The most frequently used controls are listed below.

- `--output PATH` selects a new directory. Default is the input basename with extension `.esd`.
- `--model {auto,ah,vg}` and `--process {both,absorption,emission}` select the model and direction.
- `--state N` selects the one-based singlet root at the relaxed ground minimum. Default is 1. Confirm the character and continuity of the tracked state in the diagnostics.
- `--basis BASIS`, `--xc XC`, and `--charge Q` define the electronic input. Defaults are `def2-tzvpd`, `pbe0`, and 0.
- `--fallback-xc XC` may be repeated to allow ordered changes in the functional following a restricted-to-unrestricted instability. Each candidate starts the whole calculation with its own reference and state ordering. The same root number can denote a different transition under a different functional.
- `--reference-checks {minima,all}` sets the internal and restricted-to-unrestricted orbital checks at the relevant minima (default) or all electronic points.
- `--temperature K` sets the initial population temperature separately for each direction. Default is 0 K. Emission assumes a relaxed emitting ensemble.
- `--sigma CM^-1` controls Gaussian broadening. Default is 50 cm⁻¹. Set 0 to write sticks only.
- `--max-quanta N` and `--max-overlaps N` bound the adaptive vibrational calculation. Defaults are 32 and 250,000. The CLI targets 99.5% population and, where defined, final dipole-strength coverage. Python can set an explicit `coverage`.

Set `OMP_NUM_THREADS` before running for reproducible thread counts. Otherwise the CLI chooses up to four threads for at most 12 atoms and up to eight for larger inputs. Record the settings, thread count and package versions when comparing performance or numerical tolerances.

## Read the output

- **`diagnostics.json`** records the run status, model, file names, effective and requested electronic settings, reference attempts, and available checks on geometry, tracking, derivatives, anharmonicity and vibrational coverage. It is written last. A recorded incomplete result exits with code 2. Invalid CLI input, unavailable dependencies or filesystem failures may produce only an error message.
- **`absorption_sticks.csv` and `emission_sticks.csv`** give photon energies in Hartree and cm⁻¹, initial populations, squared transition dipoles, assignment labels, signed Cartesian dipole components where available, and process-specific physical strengths. Empty fields represent quantities outside the selected model. Check the assignment scheme before comparing harmonic quanta with dominant VCI labels or DVR levels.
- **`*_profile.csv`** gives the Gaussian-broadened photon-weighted density on a wavenumber grid. The CSV uses relative prefactors, with weights proportional to photon energy for absorption and its cube for emission. Use the absolute strength columns in the sticks or the rates in `properties.json` for physical units.
- **`properties.json`** summarizes retained oscillator strengths, vacuum Einstein A coefficients and aggregate rates, radiative timescales, powers, moments, assignments, and electronic or harmonic diagnostics supported by the model. Harmonic HT results resolve Condon, linear HT and coherent interference contributions. Output with both completed profiles includes their peak Stokes shift. A difference below one common grid-spacing scale is flagged in the report.
- **`*_partial_sticks.csv` and optional `*_partial_profile.csv`** retain available lines when the numerical coverage target was missed. The diagnostics show the population and closure reached. The tabulated quantities then describe only the retained lines.

Peak positions, centroids, strongest sticks and the 0–0 energy measure different features. A radiative timescale estimated from the ensemble-averaged vacuum rate describes spontaneous emission within this model. A measured lifetime or quantum yield also depends on nonradiative processes and excited-state kinetics. Duschinsky-mixed mode displacements and Huang–Rhys factors describe local coordinates. Individual spectral intensities depend on the complete vibronic amplitudes.

## Troubleshoot an incomplete result

- **Reference unstable.** Check the channel and geometry under `reference_attempts`. The default can restart in the same model and follow an internal orbital instability. Spin-breaking instability requires an explicit alternative functional or a different electronic method outside the automatic workflow. See [the reference policy](methods.md#electronic-reference-and-state-tracking).
- **State ambiguous.** Inspect the selected root, root-energy gaps and overlaps. The workflow treats an isolated adiabatic surface. Changes of electronic rank during optimization or local derivative sampling can invalidate this model.
- **Minimum, curvature or HT check fails.** Verify the input, target and electronic recipe. Check for a nearby competing minimum, a shallow mode or near-degenerate states before interpreting local harmonic spectra. Numerical derivative fits carry their own held-out checks.
- **Anharmonic model unsupported.** Multiple wells, strong coupling or a finite-temperature population tail beyond the automatic model can block `auto`. Select `ah` explicitly to study the harmonic approximation, or provide a validated larger-amplitude model through the [Python routes](api.md).
- **Population, closure or a positive-photon moment fails its coverage check.** Review the initial and final basis cutoffs and the omitted-moment bounds in `diagnostics.json`. Increase the applicable vibrational budget if the expected memory and cost are manageable. Converge line positions or profile metrics of interest independently.

## NVIDIA GPU workflow

See the dedicated [GPU guide](gpu.md) for AH and automatic screening, analytic derivative support, C-PCM, environment checks and prepared response objects.

Install compatible GPU4PySCF, CuPy and cuTENSOR packages in a CUDA-enabled Linux or WSL environment. The [high-level GPU workflow](using-methods.md#gpu-vg-ah-and-automatic-screening) supports `vg`, analytic `ah`, and `auto`. The prepared call below runs VG from a stationary S₀ geometry. The electronic operators use native FP64 GPU arrays and the harmonic assembly runs on the host. The default fixed-grid ground Hessian matches the effective CPU route. The prepared VG call can request analytic grid response with `hessian_grid_response=True`. [GPU4PySCF installation](https://github.com/pyscf/gpu4pyscf/tree/v1.9.2#installation)

```python
from pyscf_esd.cpu import Settings
from pyscf_esd.gpu import run_gpu_vibronic

water_bohr = [[0., 0., 0.], [-1.4508663771, 0., 1.1054578981],
              [1.4508663771, 0., 1.1054578981]]
result = run_gpu_vibronic(
    ("O", "H", "H"), water_bohr, Settings("6-31+g*", "pbe0", grid_level=5),
    target_index=0,
)
emission = result.spectrum(process="emission", final_max_quanta=8)
```

The [water benchmark](validation.md#water-cpu-and-gpu) illustrates how to compare CPU and GPU at matched settings and why a small molecule can be slower on a GPU.
