# Command-line reference

`pyscf-esd` and `python -m pyscf_esd` call the same driver. Run `pyscf-esd --help` in your active environment for the installed version. The flags below are defined by [cli.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cli.py).

```text
pyscf-esd XYZ [--output PATH] [--model auto|ah|vg] [--backend cpu|gpu]
          [--solvent NAME] [--process both|absorption|emission] [--state N]
          [--charge Q] [--basis NAME] [--xc NAME] [--fallback-xc NAME]
          [--reference-checks minima|all] [--temperature K] [--sigma CM^-1]
          [--max-quanta N] [--max-overlaps N]
```

## Geometry and output

### `XYZ`

Required path to one XYZ frame. The first line gives the atom count, the second is a comment, and subsequent lines contain element symbols and three Cartesian coordinates in ångström. At least two atoms are required. Preserve atom order throughout a calculation and its prepared data.

### `--output PATH`

Fresh result directory. The default replaces the input file suffix with `.esd`. The destination must not already exist. The driver publishes the available files and writes `diagnostics.json` last.

## Model and direction

### `--model {auto,ah,vg}`

Default `auto`. AH optimizes the two surfaces and uses their distinct Hessians, Duschinsky rotation and available first-order HT amplitudes. Automatic screening starts from that preparation and tests direct electronic cuts for supported anharmonic corrections. VG uses the relaxed S₀ curvature and the vertical excited-state gradient, with a constant dipole in this CLI. [Model selection](guide.md#input-and-model)

### `--process {both,absorption,emission}`

Default `both`. Each direction uses its own initial vibrational populations. Emission assumes equilibration on the emitting surface before radiative decay. A calculation requesting both directions can preserve one complete spectrum and an available partial spectrum for the other direction.

### `--backend {cpu,gpu}`

Default `cpu`. Selects the electronic preparation and derivative implementation. Both backends expose VG, AH and automatic screening with the supported electronic models. [CPU derivative scope](methods.md#geometry-and-derivative-routes) · [GPU guide](gpu.md)

## Electronic structure

### `--xc NAME`

Default `pbe0`. Functional name or expression accepted by the selected backend. The CPU gas-phase workflow supports compatible LDA/GGA functionals and hybrids, including range-separated hybrids. The analytic GPU workflow requires a supported global hybrid. Inspect the [method guide](using-methods.md#choosing-the-level-of-theory) before choosing an XC expression.

### `--basis NAME`

Default `def2-tzvpd`. PySCF basis name for real atoms with spherical all-electron orbitals.

### `--charge Q`

Default `0`. Integer total molecular charge. The automatic CLI requires an even electron count and a closed-shell reference. Open-shell and SOC workflows use the [Python interfaces](phosphorescence.md).

### `--state N`

Default `1`. One-based singlet root at relaxed S₀. The CLI requests a root window of at least five roots and at least two entries above the specified root, subject to the supported response space. Tracking qualifies the selected state at subsequent geometries. Python `target_index` is zero-based.

### `--solvent NAME`

Optional named equilibrium C-PCM solvent with `--backend gpu`. The CPU CLI rejects this option. CPU and GPU prepared response objects have separate solvent support contracts. [Selecting a solvent](using-methods.md#choosing-a-solvent)

### `--reference-checks {minima,all}`

Default `minima`. Checks internal and restricted-to-unrestricted orbital stability at the relevant optimized stationary geometries. `all` extends the checks to each electronic point. The workflow may repeat a restricted calculation to follow an internal instability while retaining its convergence criteria.

### `--fallback-xc NAME`

Optional and repeatable. Authorizes ordered functional alternatives following a verified restricted-to-unrestricted instability. Each candidate starts a fresh preparation and has its own state assignment. An unresolved stability solve does not qualify a functional switch. The report records the requested and effective settings and every reference attempt.

```bash
pyscf-esd molecule.xyz --xc pbe0 --fallback-xc cam-b3lyp \
  --reference-checks all --output reference-check.esd
```

## Temperature and broadening

### `--temperature K`

Default `0.0`. Finite nonnegative temperature in kelvin for the initial nuclear populations in each requested direction. Converged hot-band populations can require larger vibrational bases. Supported anharmonic routes also evaluate their finite-temperature coverage.

### `--sigma CM^-1`

Default `50.0`. Finite nonnegative Gaussian standard deviation in cm⁻¹. Set `0` to write sticks without a profile. For 80 cm⁻¹ FWHM use `33.9728720115`. The width changes the displayed profile and its peak position. Sticks and their native-energy physical totals remain the numerical starting point. [Comparison guide](spectral-comparison.md)

## Nuclear convergence budgets

### `--max-quanta N`

Default `32`. Nonnegative ceiling on the adaptive vibrational cutoff. For the harmonic basis, the cutoff limits the total number of quanta over all modes. The solver chooses separate initial and final cutoffs within this ceiling.

### `--max-overlaps N`

Default `250000`. Positive limit on the number of initial–final overlaps considered by an adaptive attempt. The initial and final basis sizes multiply. The solver searches affordable cutoff pairs and can retain the best available partial result when the requested coverage cannot be reached.

The CLI targets coverage `0.995`. Python `run_vibronic` and `run_prepared_ah` expose the `coverage` keyword. Harmonic qualification considers the initial population, dipole closure and positive-photon moments for absorption, radiative rate and power. [Convergence diagnostics](troubleshooting.md#nuclear-basis-and-rate-convergence)

## Informational flags

`--help` or `-h` prints usage and exits. `--version` prints the installed package version. An uninstalled source tree may report `source` through the package metadata fallback.

## Exit status and recoverable output

Exit code 0 means the requested workflow completed. Exit code 2 represents rejected arguments, missing dependencies, a failed preparation, or incomplete numerical coverage. A recorded incomplete workflow writes `status="incomplete"` and a reason to `diagnostics.json`. Filesystem or early input failures may occur before that file can be written.

Output carrying `_partial` in its name describes the retained lines with their recorded coverage. Always check the status and individual observables before using a rate or comparing intensities. A full run does not resume from a prior output directory. Selected lower-level APIs provide their own checkpoint facilities. [Recovery and troubleshooting](troubleshooting.md)
