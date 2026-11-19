# GPU calculations

PySCF ESD uses GPU4PySCF for electronic structure on NVIDIA CUDA devices. The main workflow supports the vertical-gradient model, adiabatic Hessians with Duschinsky rotation and linear Herzberg–Teller coupling, and automatic anharmonic screening. The same nuclear solvers and physical-property reports serve the CPU and GPU workflows.

## Install and check the environment

Use a CUDA-enabled Linux installation or WSL with a working NVIDIA driver. Install PySCF ESD with the `workflow` extra, then install the GPU4PySCF, CuPy and cuTENSOR distributions appropriate for that CUDA environment. GPU packages are installed separately from the project's Python extras. Follow the upstream instructions for their compatible package names. [GPU4PySCF installation](https://pyscf.org/user/gpu.html#installation)

Run these checks inside the environment that will execute the calculation.

```bash
python -c "import pyscf, gpu4pyscf, cupy; print(pyscf.__version__, gpu4pyscf.__version__, cupy.__version__); print(cupy.cuda.runtime.getDeviceCount())"
python -c "from pyscf_esd.tda_xc_curvature import require_semilocal_xc_fourth; require_semilocal_xc_fourth('PBE0'); print('PBE0 fourth-order XC available')"
pyscf-esd --version
```

The device count must be positive. Record the CUDA runtime, driver, GPU model, available device memory, package versions and CPU thread count with any timing result. The matched water validation used PySCF 2.15.0 and GPU4PySCF 1.9.2 on an RTX 4070 Laptop GPU. This is a tested configuration. Validate an updated environment with the [water controls](validation.md#water-cpu-and-gpu).

## Start from an XYZ or PySCF molecule

The CLI reads XYZ coordinates in ångström. This command requests independently relaxed harmonic surfaces, Duschinsky rotation and the available analytic HT derivative.

```bash
pyscf-esd molecule.xyz --backend gpu --model ah --xc pbe0 --basis def2-svp --temperature 298 --sigma 33.9728720115 --output gpu-ah.esd
```

Here `--sigma` is the Gaussian standard deviation in cm⁻¹, giving FWHM 80 cm⁻¹. See the [output guide](numerical-results.md) for absolute strengths and rates.

The PySCF module accepts a built, closed-shell molecule with spherical all-electron basis functions and `symmetry=False`.

```python
from pyscf import esd, gto

mol = gto.M(
    atom="O 0 0 0; H 0.758 0 0.504; H -0.758 0 0.504",
    basis="def2-svp",
    unit="Angstrom",
    spin=0,
    symmetry=False,
)
result = esd.run(
    mol,
    backend="gpu",
    model="ah",
    xc="pbe0",
    temperature_kelvin=298.0,
    progress=print,
)
report = result.physical_properties
```

`esd.run` uses the molecule's geometry, named basis, charge and atom-ordered masses, including isotope overrides. It prepares its own mean-field objects. When individual electronic thresholds are needed, call `run_vibronic` with a `Settings` object as described in the [API guide](api.md).

Select `model="vg"` for the high-level VG workflow or `model="auto"` for the automatic workflow. Both optimize the ground-state geometry. AH and automatic calculations also follow the selected singlet to its emitting minimum. `target_index` is the zero-based state label at the relaxed ground-state geometry. The tracker determines its identity at subsequent points from orbital and response overlaps.

## Electronic models and solvents

The high-level GPU workflow uses Coulomb-metric density fitting for J and K and a global hybrid made from LDA or GGA components. PBE0 is the simplest tested starting point. The derivative validation rejects range-separated functionals, meta-GGAs, nonlocal correlation, empirical dispersion, ECPs and unsupported core Hamiltonians. A functional accepted by a generic GPU4PySCF SCF calculation must also satisfy the ESD derivative requirements.

Named solvents select equilibrium linear-response C-PCM.

```bash
pyscf-esd molecule.xyz --backend gpu --model ah --xc pbe0 --basis def2-svp --solvent cyclohexane --temperature 298 --output solution-ah.esd
```

```python
result = esd.run(
    mol, backend="gpu", model="ah", xc="pbe0",
    solvent="water", temperature_kelvin=298.0,
)
```

The RKS and TDA responses share a SWIG cavity, an unpruned primary XC grid and the same CPHF grid. The reference sets `small_rho_cutoff = 0`. Equilibrium solvation is part of the potential and response model used for both surfaces. Comparisons with solution experiments should report this choice alongside the solvent and temperature. Prepared references expose finer native cavity and dielectric settings. [Selecting solvent parameters](using-methods.md#choosing-a-solvent)

AH and automatic calculations require fourth derivatives of the semilocal XC energy density. PBE and PBE0 have an analytic implementation of this contribution. Other supported functionals use a suitable native LibXC implementation or a same-version LibXC shared library compiled with fourth-order support, selected through `PYSCF_ESD_LIBXC_LXC`. Check the required functional before starting a long run.

```bash
export PYSCF_ESD_LIBXC_LXC=/absolute/path/to/libxc.so
python -c "from pyscf_esd.tda_xc_curvature import require_semilocal_xc_fourth; require_semilocal_xc_fourth('B3LYP')"
```

Use an absolute path to the compatible library installed in your environment. The check reports missing derivative support before the Hessian is built.

## Analytic derivatives and state tracking

The GPU AH path computes the ground and excited Hessians at their respective minima. The excited Hessian includes orbital response, TDA eigenvector response, density-fitting and auxiliary-basis derivatives, moving XC quadrature, fourth-order semilocal XC terms and C-PCM response when applicable. The transition-dipole derivative includes the occupied-orbital and excitation response and uses the phase of the tracked state.

The workflow checks the excited-state gradient against the Hessian's first derivative, root residuals and separation, Hessian symmetry and the phase-matched dipole. A failure reports the electronic or response stage that could not be qualified. The automatic model can also evaluate electronic energies and dipoles along nuclear coordinates to test anharmonicity. Those direct surface samples are inputs to the nuclear potential model.

For a prepared, solved GPU4PySCF root, evaluate derivatives at that geometry directly.

```python
from pyscf_esd import (
    analytic_molecular_tda_hessian,
    analytic_tda_transition_dipole,
)

def record_progress(stage, done, total):
    print(stage, done, total, flush=True)

curvature = analytic_molecular_tda_hessian(
    td_at_s1,
    root_index=0,
    cphf_tolerance=1e-10,
    response_tolerance=1e-8,
    root_gap_tolerance=1e-5,
    max_cycle=150,
    progress=record_progress,
)
dipole = analytic_tda_transition_dipole(
    td_at_s0, root_index=0, phase=tracked_phase_at_s0,
    cphf_conv_tol=1e-10, response_tolerance=1e-8,
)
```

`td_at_s0` and `td_at_s1` are separately solved, isolated real singlet roots at the two geometries. Use the identity and index established at each endpoint. `tracked_phase_at_s0` is `+1` or `-1` from the state tracker. Each TDA window must include a root above the selected state or the complete occupied–virtual spectrum. Preserve the same atom order, basis, grid prescription and physical model throughout.

`curvature.hessian` has shape `(atom, atom, Cartesian, Cartesian)` and units Hartree/bohr². Its `excitation_gradient` has shape `(atom, 3)` in Hartree/bohr and differentiates the excitation contribution. The result also reports `orbital_response_residual`, `orbital_adjoint_residual`, `root_residual`, `root_separation_hartree` and `tda_response_residual`.

`dipole.dipole` has shape `(3,)` in e·bohr. `dipole.derivative` has shape `(atom, displacement Cartesian, dipole Cartesian)` in e. Multiply it by Cartesian mode derivatives in the same frame to obtain HT slopes. Root, adjoint and rigid-motion diagnostics accompany the derivative.

The Hessian's callback uses `(stage, done, total)`. It includes `ground_hessian`, `fixed_density_solvent` when applicable, `fixed_density_df`, XC derivative stages, `tda_root_response` and `mixed_operator_derivatives`. The high-level `run_vibronic` callback accepts a single status string. Use the callback matching the API you call.

## Prepared direct-J/K VG

`run_gpu_vibronic` evaluates the original direct-J/K PBE0 VG model at a supplied stationary ground-state geometry. It checks the ground gradient and curvature and returns a prepared VG result. It takes Cartesian coordinates in bohr and an explicit `Settings` object.

```python
from pyscf_esd.cpu import Settings
from pyscf_esd.gpu import run_gpu_vibronic

vg = run_gpu_vibronic(
    symbols,
    stationary_s0_coordinates_bohr,
    Settings(basis="6-31+g*", xc="pbe0", grid_level=5, nstates=5),
    target_index=0,
    hessian_grid_response=False,
)
emission = vg.spectrum(process="emission", final_max_quanta=8)
```

This call requires a visible CUDA device, PBE0, an all-electron basis supported by the native direct integral engine, and a target root below at least one higher stored root. The default fixed-grid Hessian matches the corresponding CPU validation route. `hessian_grid_response=True` requests the native moving-grid Hessian. Choose the same convention for a CPU–GPU comparison. The high-level `esd.run(..., backend="gpu")` route uses the density-fitted electronic session described above.

## Memory, timing and recovery

Electronic matrices and GPU4PySCF response operations use FP64 CUDA arrays. State tracking, selected derivative contractions, XC fourth derivatives and nuclear spectral assembly also use host code. Device-to-host transfer and response setup contribute to elapsed time. Report the complete workflow time alongside any electronic kernel timings.

The electronic session retains a small cache of recent points and their device objects. Device memory use therefore depends on the basis, occupied–virtual space, fitting basis, XC grid, root count and retained response objects. Check available memory before a large job. The high-level CLI exposes the electronic model and vibrational budget. Prepared native PySCF/GPU4PySCF objects expose their own resource settings.

When diagnosing a failure, keep the selected physical model and numerical thresholds recorded in `diagnostics.json`.

- An import or CUDA initialization error points to the GPU environment. Confirm that the same Python can import CuPy and GPU4PySCF and see the device.
- A fourth-order XC error identifies the functional or LibXC build required by the Hessian.
- A reference-residual error requires SCF convergence before the derivative can be used. Review the actual orbital residual and the attempted guesses.
- A state-gap or overlap error means that the isolated-state model could not be established at that geometry. Review state character, root window and the geometry path.
- A response-residual or symmetry failure requires a converged analytic response for the same operator. Record the failing diagnostic before changing a tolerance.
- A device-memory failure can be investigated with a smaller model as a distinct calculation. Document any basis, grid or state-space changes and recheck the observables used in the comparison.

The benchmark guide explains the matched [water CPU/GPU checks](validation.md#water-cpu-and-gpu). The [spin-flip section](using-methods.md#spin-flip-tda-on-gpu) covers prepared GPU UKS spin-flip response. The [open-shell guide](phosphorescence.md) describes UKS endpoints, spin-conserving spectra and the CPU SOC machinery.

## Implementation reference

The [GPU electronic adapter](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/gpu.py), [adiabatic session](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/gpu_adiabatic.py), [analytic molecular Hessian](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/molecular_tda_hessian.py), [transition-dipole derivative](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/tda_dipole.py) and [XC curvature implementation](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/tda_xc_curvature.py) define these contracts.
