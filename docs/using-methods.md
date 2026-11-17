# Choosing and running a method

Install the package with the `.[workflow]` extra as shown in the [quick start](../README.md#quick-start). The same installation provides the CLI and the Python entry points below.

In a PySCF script, use `from pyscf import esd`. The [`esd.run(mol, ...)` example](../README.md#use-as-a-pyscf-module) starts from an existing `pyscf.gto.Mole`. Core APIs such as `esd.run_prepared_ah` and `esd.solve_dvr` are also available through the module.

| What you want to run | Entry point |
| --- | --- |
| Automatic CPU workflow with anharmonic screening | [`--model auto`](#from-a-single-xyz) |
| CPU adiabatic Hessian (AH) with Duschinsky rotation and Herzberg–Teller (HT) | [`--model ah`](#from-a-single-xyz) |
| CPU vertical gradient (VG) with a constant dipole | [`--model vg`](#from-a-single-xyz) |
| Explicit finite differences for HT or excited-state curvature | [`run_cpu_vibronic`](#explicit-central-differences-on-cpu) |
| Analytic VG, AH and automatic screening on NVIDIA GPU | [`--backend gpu`](#gpu-vg-ah-and-automatic-screening) |
| SF-TDA roots, optical transitions and analytic gradients | [`solve_spin_flip_tda`](#spin-flip-tda-on-gpu) |
| Analytic Hessian and HT derivative for isolated LR-TDA/C-PCM roots | [Prepared C-PCM derivatives](#analytic-derivatives-for-prepared-c-pcm-roots) |
| AH with custom electronic data or selected FC, HT and Duschinsky terms | [`run_prepared_ah` and `harmonic_spectrum`](#spectra-from-prepared-electronic-data) |
| DVR, thermal rotor plus harmonic spectators, coupled modes | [Prepared anharmonic models](#prepared-anharmonic-models) |

Use the [numerical output guide](numerical-results.md) for radiative rates, lifetimes, oscillator strengths, mode analysis, and reading the CLI or Python results.

## From a single XYZ

```bash
pyscf-esd molecule.xyz --model auto --output automatic.esd
pyscf-esd molecule.xyz --model ah --temperature 298 --sigma 33.972872 --output ah.esd
pyscf-esd molecule.xyz --model vg --process absorption --state 2 --sigma 0 --output vg.esd
```

- **`auto`** starts from the two relaxed minima and screens direct electronic cuts. It retains AH when the cuts qualify as harmonic and can apply local anharmonic corrections at 0 K. If a correction is required at positive temperature or the sampled potential exceeds the supported model, the run reports why it could not complete. Request `ah` for a harmonic finite-temperature calculation.
- **`ah`** uses separate curvatures at the two minima and maps their normal coordinates through the Duschinsky matrix. The linear HT dipole term interferes with the constant-dipole Franck–Condon (FC) term. The CPU DFT route uses analytic gradients and ground-state curvature. It obtains excited-state curvature and HT by fitting displaced gradients and signed dipoles and checking the fits at independent points. The pure-HF control can use conventional RHF–CIS analytic derivatives when the dense response fits memory.
- **`vg`** optimizes the ground state and uses its curvature and the vertical excited-state gradient to define two displaced harmonic surfaces of equal curvature. The CLI uses a constant dipole.

Supply a closed-shell XYZ in ångström and a new output directory. Defaults are PBE0/def2-TZVPD, the first singlet root at the relaxed ground geometry, both spectra and 0 K. `--state` is one-based. Python `target_index` is zero-based. `--sigma` is a Gaussian standard deviation in cm⁻¹. The second command uses 80 cm⁻¹ FWHM via `FWHM = 2 sqrt(2 ln 2) σ`. Zero gives sticks only.

Use `--xc`, `--basis` and `--charge` for the electronic model, `--reference-checks all` to test the reference at every electronic point, and repeat `--fallback-xc XC` for ordered alternatives in the supported recovery path. `--max-quanta` and `--max-overlaps` bound the vibrational calculation. Run `pyscf-esd --help` for the full list. Check `diagnostics.json` for state assignments, the derivative route and per-spectrum completion. See the [output files](../README.md#output) for spectra and observables.

## Choosing the level of theory

Select the functional, orbital basis and vibrational model independently. These commands use the same harmonic model in the gas phase.

```bash
pyscf-esd molecule.xyz --xc pbe0 --basis def2-TZVPD --model ah --output pbe0-ah.esd
pyscf-esd molecule.xyz --xc b3lyp --basis def2-SVP --model ah --output b3lyp-ah.esd
```

Use PySCF [functional aliases](https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases) such as `pbe`, `pbe0`, `b3lyp` and `cam-b3lyp` with [PySCF basis names](https://pyscf.org/user/gto.html#basis-set). The automatic DFT workflow requires a closed-shell all-electron reference and LDA or GGA XC with the required LibXC derivatives. This includes compatible global and range-separated hybrids. Meta-GGA, nonlocal correlation, empirical dispersion and ECPs are outside this workflow. The prepared analytic C-PCM derivatives require a global hybrid.

Use `--charge 2` for a closed-shell dication or `--state 2` for the second singlet at the relaxed ground geometry. In Python, `Settings` from [`pyscf_esd.cpu`](../src/pyscf_esd/cpu.py) exposes `nstates`, `grid_level`, and the SCF and response tolerances. Pass it to [`run_vibronic`](../src/pyscf_esd/workflow.py) with coordinates in bohr.

```python
from pyscf_esd import run_vibronic
from pyscf_esd.cpu import Settings

def gas_phase_spectra(symbols, coordinates_bohr):
    settings = Settings(basis="def2-TZVPD", xc="pbe0", nstates=6, grid_level=4)
    return run_vibronic(symbols, coordinates_bohr, settings, model="ah", target_index=0)
```

## Choosing a solvent

Use `--backend gpu --solvent cyclohexane` from XYZ or `backend="gpu", solvent="cyclohexane"` in `esd.run`. Examples include `water`, `n-hexane` and `acetonitrile`. Named solvents use the [PySCF solvent parameters](https://pyscf.org/user/solvent.html#solvent-parameters). With prepared inputs, set `mf.with_solvent.eps` to override the static dielectric constant. Solvent selection changes the electronic calculation. The ESD temperature controls the vibrational populations.

```python
from gpu4pyscf import dft

def prepare_equilibrium_tda(mol, solvent="cyclohexane", *, xc="pbe0", nstates=5, eps=None):
    # mol is a PySCF Mole with the desired geometry, charge and all-electron basis.
    mf = dft.RKS(mol, xc=xc).density_fit().PCM(solvent)
    mf.with_solvent.method = "C-PCM"
    if eps is not None:
        mf.with_solvent.eps = eps
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.cphf_grids = mf.grids
    mf.kernel()
    td = mf.TDA(equilibrium_solvation=True)
    td.nstates = nstates
    td.kernel()
    return td
```

This prepares the response at the supplied geometry. For AH, use the relaxed S₀ and tracked excited-state minima under the same electronic and solvent model. Follow the [analytic derivative example](#analytic-derivatives-for-prepared-c-pcm-roots) and the [prepared AH example](#spectra-from-prepared-electronic-data) to build the spectrum. Other models available in PySCF, including SMD and nonequilibrium PCM, are outside this analytic derivative route.

## Explicit central differences on CPU

Supply symbols and coordinates in bohr at a stationary S₀ geometry. `include_ht=True` differentiates phase-aligned dipoles. `compute_excited_hessian=True` independently differentiates analytic excited-state gradients. Both use central differences with state tracking. The ground Hessian is analytic.

```python
from pyscf_esd.cpu import Settings
from pyscf_esd.cpu_workflow import run_cpu_vibronic

def numerical_vg(symbols, coordinates_bohr):
    return run_cpu_vibronic(
        symbols, coordinates_bohr, Settings(basis="def2-SVP", xc="pbe0", nstates=5),
        target_index=0, include_ht=True, compute_excited_hessian=True,
        step_bohr=0.005, checkpoint_dir="fd-checkpoints",
    )
```

Inspect `result.normal_dipole_derivatives_au` and `result.excited_hessian_hartree_per_bohr2`. `result.spectrum(process="absorption")` includes available HT by default. Set `include_ht=False` on the spectrum call for a constant dipole. The spectrum retains the VG curvatures. The optional excited-state Hessian describes S₁ at the **supplied S₀ geometry**. AH uses S₁ curvature at the relaxed S₁ geometry. See the [CPU workflow API](../src/pyscf_esd/cpu_workflow.py).

## GPU VG, AH and automatic screening

Follow the [GPU4PySCF installation instructions](https://pyscf.org/user/gpu.html#installation). The main command and PySCF module support all three models on this backend.

```bash
pyscf-esd molecule.xyz --backend gpu --model ah --temperature 298 --output gpu-ah.esd
```

```python
from pyscf import esd

def gpu_spectra(mol):
    return esd.run(mol, backend="gpu", model="ah", xc="pbe0")
```

AH follows the same closed-shell singlet through both geometry optimizations, computes analytic Hessians at the minima, and projects the signed analytic transition-dipole derivative at S₀ onto its normal modes. Absorption and fluorescence use the shared Duschinsky and linear HT model. The differentiations include the moving XC quadrature and DF response. Failure of a derivative or state-continuity check ends the GPU run with its reason. The automatic model can sample electronic energies and dipoles for nuclear anharmonicity on the GPU. These surface samples are distinct from numerical estimates of electronic derivatives.

The electronic workflow uses CUDA for GPU4PySCF response calculations and host routines for parts of derivative assembly, state tracking and the nuclear calculation. For the original direct-J/K VG route at a **supplied stationary ground geometry**, use the single-geometry API below.


```python
from pyscf_esd.cpu import Settings
from pyscf_esd.gpu import run_gpu_vibronic

def gpu_vg(symbols, coordinates_bohr):
    model = run_gpu_vibronic(
        symbols, coordinates_bohr, Settings(basis="def2-SVP", xc="pbe0", nstates=5),
        target_index=0,
    )
    return model.spectrum(process="emission")
```

See the [GPU backend](../src/pyscf_esd/gpu.py) and the [water benchmark driver](../examples/validate_water_gpu.py).

## Spin-flip TDA on GPU

Use GPU4PySCF with `gpu4pyscf.tdscf.uks.SpinFlipTDA` and a converged, double-precision GPU UKS reference with `mol.spin = 2`. For example, start with `gpu4pyscf.dft.UKS(mol, xc="pbe0").density_fit()` and converge it with `uks.kernel()`. Check the auxiliary triplet reference and identify the physical roots by spin and electronic character. The wrapper checks orbital and response residuals and uses orbital-relaxed optical moments. The optical response requires third-order XC derivatives.

```python
from pyscf_esd.gpu_spin_flip import solve_spin_flip_tda, spin_flip_tda_gradient

def sf_roots(uks):
    return solve_spin_flip_tda(uks, nroots=8)

def sf_transition(sf, lower, upper):
    # lower and upper are assigned physical states, indexed from zero.
    gap_hartree = float(sf.e[upper] - sf.e[lower])
    dipole_au = sf.transition_dipole(ref=lower + 1, state=upper + 1)[0]
    lower_gradient = spin_flip_tda_gradient(sf, lower)
    upper_gradient = spin_flip_tda_gradient(sf, upper)
    return gap_hartree, dipole_au, lower_gradient, upper_gradient
```

The SF eigenvalues are offsets from the auxiliary triplet. Physical root `i` has total energy `float(uks.e_tot) + float(sf.e[i])`. The gradients already differentiate those totals and have units Hartree/bohr. Optical labels are **one-based**, and gradient indices are **zero-based**. Choose the root count within the response dimension.

The [SF API](../src/pyscf_esd/gpu_spin_flip.py) operates at a supplied geometry. For a path, the [prepared tracking API](../src/pyscf_esd/prepared_state_tracking.py) propagates state assignments and phases from caller-supplied signed state overlaps. A complete SF run from XYZ through relaxed minima, curvatures and spectra is a user-assembled workflow.

## Analytic derivatives for prepared C-PCM roots

Supply solved, isolated singlet roots of GPU4PySCF restricted LR-TDA in vacuum or equilibrium linear-response C-PCM, a real global hybrid LDA/GGA functional and a shared RKS and CPHF grid with `small_rho_cutoff = 0`. The analytic Hessian requires a spherical all-electron basis, Coulomb-metric DF for J and K, fourth XC derivatives and a matching SWIG cavity when solvent is present. See the [installation note](../README.md#default-and-options) for `PYSCF_ESD_LIBXC_LXC` and the [integration test setup](../tests/test_molecular_tda_hessian.py) for the grid and cavity. Solve through at least one root above the target or provide the complete occupied–virtual spectrum.

```python
from pyscf_esd import analytic_molecular_tda_hessian, analytic_tda_transition_dipole

def endpoint_response(td_at_s0, root_at_s0, td_at_s1, root_at_s1, phase_at_s0):
    curvature = analytic_molecular_tda_hessian(td_at_s1, root_at_s1)
    dipole = analytic_tda_transition_dipole(td_at_s0, root_at_s0, phase=phase_at_s0)
    return curvature, dipole
```

Use zero-based indices for the same tracked excited state at the two geometries. `phase_at_s0` is `+1` or `-1` and acts on both the transition dipole and its derivative. `curvature.hessian` is the **total** excited-state Hessian in Hartree/bohr². Add `curvature.excitation_hartree` and `curvature.excitation_gradient` to the matching RKS reference energy and gradient at the excited geometry. The Cartesian `dipole.derivative` has axes (atom, displacement Cartesian, dipole Cartesian) and units e, the derivative of e·bohr with respect to bohr. See the [Hessian](../src/pyscf_esd/molecular_tda_hessian.py) and [dipole](../src/pyscf_esd/tda_dipole.py) APIs.

## Spectra from prepared electronic data

Give [`adiabatic_hessian`](../src/pyscf_esd/pyscf_adapter.py) two stationary PySCF `Mole` objects with matching atom order, their Hessians in Hartree/bohr², total energies in Hartree on a common reference and total gradients in Hartree/bohr. It constructs the mass-weighted normal modes, the Duschinsky matrix and the harmonic 0–0 energy. Here is how to connect the analytic dipole output above to AH.

```python
import numpy as np
from pyscf_esd import run_prepared_ah
from pyscf_esd.pyscf_adapter import adiabatic_hessian

def prepared_spectra(mol0, mol1, h0, curv1, e0, e_ref1, g0, g_ref1, dipole):
    parameters = adiabatic_hessian(
        mol0, mol1, h0, curv1.hessian,
        ground_energy_hartree=e0, excited_energy_hartree=e_ref1 + curv1.excitation_hartree,
        ground_gradient=g0, excited_gradient=g_ref1 + curv1.excitation_gradient,
    )
    dmu_dq = np.einsum(
        "qax,axd->qd", parameters.ground_cartesian_mode_derivatives_au, dipole.derivative,
    )
    return run_prepared_ah(
        parameters, dipole.dipole, electronic_method="DF-PBE0/TDA, equilibrium LR C-PCM",
        dipole_derivatives_au=dmu_dq, temperature_kelvin=298,
    )
```

The label should match the electronic method actually used. Use the signed length-gauge dipole and its derivative in the S₀ Cartesian frame and gauge. `result.spectra["absorption"]` and `result.spectra["emission"]` contain sticks. Omit `dipole_derivatives_au` for FC only. To control Duschinsky mixing, [`harmonic_spectrum`](../src/pyscf_esd/harmonic.py) accepts `rotation=None` for an identity transform while keeping the other inputs. Pair the modes consistently for such a control. [`thermal_harmonic_spectrum`](../src/pyscf_esd/thermal_separable.py) computes a broadened all-quanta FC/HT profile. For these low-level harmonic calls use the signed final-minus-initial 0–0 gap and the dipole data in the initial-state normal coordinates. The signed gap for emission is negative. Use [the line-property API](../src/pyscf_esd/spectral_properties.py) for oscillator strengths and photon emission rates from sticks.

## Prepared anharmonic models

[`solve_dvr`](../src/pyscf_esd/anharmonic_dvr.py) takes a uniform grid and sampled potential in Hartree. Choose a Dirichlet boundary for a mass-weighted inversion cut or a periodic or antiperiodic boundary for an angle in radians. For an angle use the inverse moment of inertia as `kinetic_coefficient`, with inertia in electron-mass·bohr². Use a common energy reference for both potentials. [`anharmonic_spectrum`](../src/pyscf_esd/anharmonic.py) combines the levels with a constant or coordinate-dependent signed dipole and optional harmonic spectators.

### Thermal torsion

[`phase_periodic_scan`](../src/pyscf_esd/anharmonic_sampling.py) takes uniform angle samples without a duplicate endpoint. The signed overlaps must link every pair of adjacent states including the link back to the original first point. The two closure signs set the nuclear boundary conditions and the product sign of the dipole over one period. This synthetic example runs with zero electronic calculations.

```python
import numpy as np
from pyscf_esd import phase_periodic_scan, solve_dvr, thermal_separable_spectrum

theta = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
v0 = 0.001 * (1 - np.cos(3 * theta))
v1 = 0.08 + 0.002 * (1 - np.cos(3 * theta - 0.5))
zeros = np.zeros_like(theta)
mu = np.column_stack((0.1 + 0.02 * np.cos(theta), zeros, zeros))
scan = phase_periodic_scan(
    theta, v0, v1, mu, ground_overlaps=np.ones(theta.size), excited_overlaps=np.ones(theta.size),
)
kinetic = 1 / 20_000.0
s0 = solve_dvr(scan.grid, scan.ground_potential_hartree, boundary=scan.ground_boundary,
               kinetic_coefficient=kinetic, n_levels=theta.size)
s1 = solve_dvr(scan.grid, scan.excited_potential_hartree, boundary=scan.excited_boundary,
               kinetic_coefficient=kinetic, n_levels=theta.size)
options = dict(transition_dipole_au=scan.transition_dipole, temperature_kelvin=298,
               fwhm_cm_inverse=80)
absorption = thermal_separable_spectrum(s0, s1, process="absorption", **options)
emission = thermal_separable_spectrum(s1, s0, process="emission", **options)
```

The result provides `wavenumber_cm_inverse` and `photon_weighted_density_per_cm_inverse`. The [thermal solver API](../src/pyscf_esd/thermal_separable.py) describes physical-unit conversion and accepts `spectators`. Put the emitting surface first and reverse the spectator mapping for emission.

To assemble a molecular problem, use the [separable rotor assembler](../examples/assemble_separable_rotor.py) with a JSON document containing the three tracked cuts (`ground`, `excited` and `midpoint`), masses in amu, and zero-based `axis_atoms` and `moving_atoms`. For the assembler, each cut **includes** the repeated endpoint. The surface cuts contain absolute energies, sequential signed overlaps, a duplicate-to-first closure overlap and one stationary geometry with its analytic Hessian. The midpoint cut supplies Cartesian dipoles and signed links for each of the two states separately. The assembler documents the exact fields and units. With Matplotlib installed:

```bash
python examples/assemble_separable_rotor.py cuts.json --output rotor-298 \
    --temperature 298 --fwhm 80 --dvr-grids 256 512
```

### Coupled modes and tunneling

[`SineGalerkin2D`](../src/pyscf_esd/anharmonic_plane.py) and [`SineDVR3D`](../src/pyscf_esd/anharmonic_volume.py) accept prepared multi-coordinate potentials. The [formaldehyde driver](../examples/validate_h2co_3d.py) shows the double-well and dipole inputs. [`mix_vibrational_states`](../src/pyscf_esd/vibrational_mixing.py) accepts a symmetric vibrational Hamiltonian in Hartree and signed Cartesian transition dipoles in the same basis. Couplings mix the amplitudes before squaring, retaining intensity borrowing and interference. The [benzene coupling driver](../examples/validate_benzene_spectator_mixing.py) shows how to use prepared cuts.

## Replaying a benchmark

The [CF₃NO replay](../examples/replay_cf3no_thermal.py) assembles the retained PBE0/TDA and SF-TDA cases at 298 K and 80 cm⁻¹ FWHM. PBE0/TDA is the replay default and `--method sf-tda` selects spin flip. Run the script with `--help` for its archive and output arguments. The public repository includes the article and selected plots. This replay and some molecule-specific drivers require separate electronic calculation archives. The generic APIs accept your own data, and the [synthetic rotor check](../examples/validate_anharmonic_rotor.py) runs without the archives.
