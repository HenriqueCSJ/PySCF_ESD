# PySCF_ESD

PySCF_ESD predicts vibronic absorption and emission from one molecular geometry. It also accepts separately prepared electronic and vibrational inputs through its Python solvers.

## Run from one XYZ

On Linux or WSL, install the workflow and supply a single XYZ geometry in Ångström.

```bash
python -m pip install '.[workflow]'
pyscf-esd molecule.xyz
```

The default predicts S₁ absorption and relaxed emission at 0 K on the closed-shell
singlet CPU route. It starts with PBE0/def2-TZVPD and optimizes both geometries
with analytic gradients. It tracks the root and electronic phase, builds the
Duschinsky transformation and linear Herzberg–Teller amplitudes, and screens
anharmonicity on direct electronic cuts at both minima. It uses analytic
derivatives where the electronic backend supports them and independently checks
the local fits used elsewhere. The spectral cutoff grows until 99.5% population
and transition-strength coverage or the declared numerical budget.

`molecule.esd/` contains line lists, Gaussian profiles (σ = 50 cm⁻¹ by default),
and `diagnostics.json`. The report records the methods, checked state and
fit diagnostics, the anharmonic decision, and the coverage at each attempt.
The local variational model covers a bounded state space and samples selected
intermode couplings. Multiple wells, ambiguous electronic states, unconverged
fits, or insufficient vibrational coverage can produce an incomplete run with
diagnostics. When a checked spectrum exists at lower coverage, the incomplete
run retains it with a partial filename and reports its measured coverage.
The XYZ and calculations determine the prediction. Experimental
origins, line positions and intensity ratios enter benchmark comparisons only.
For matched DFT benchmark observables, report the absolute and relative error.
Up to 5% is good agreement and up to 10% is acceptable with reservation.
For near-zero reference values, use the absolute error. These benchmarks do not
set the electronic or vibrational convergence tolerances.

`--model ah` requests harmonic adiabatic Hessians with HT explicitly.
`--model vg` requests the vertical-gradient approximation explicitly.
Use `pyscf-esd --help` for temperature, broadening, electronic state, and
resource settings. Python callers can use `pyscf_esd.run_vibronic` with
Cartesian coordinates in bohr.

For a single vibrational coordinate, the automatic path solves a direct common
potential and transition-dipole surface with an adaptive open-boundary DVR.
Multimode local VCI retains the linear Herzberg–Teller approximation. Automatic
anharmonic qualification is available at 0 K. Explicit harmonic spectra and
screened harmonic cases use the harmonic thermal partition function.

## Install

```sh
python -m pip install .
```

Install with `python -m pip install '.[pyscf]'` in Linux or WSL to use the PySCF adapters. The electronic workflows use PySCF 2.15 or later for the corrected coupled-perturbed SCF (CPHF) Hessian solver. The harmonic solver needs only NumPy. [PySCF 2.15 release](https://github.com/pyscf/pyscf/releases/tag/v2.15.0)

## Calculate assigned transitions

This synthetic example gives a one-mode Franck–Condon progression. Frequencies and the 0–0 gap are in Hartree. The displacement is in `sqrt(electron mass) * bohr` and the dipole is in `e * bohr`.

```python
from pyscf_esd import harmonic_spectrum

spectrum = harmonic_spectrum(
    [0.005], [0.005],
    gap_00_hartree=0.12,
    transition_dipole_au=[1.0, 0.0, 0.0],
    shift_au=[15.0],
    temperature_kelvin=298.15,
    initial_max_quanta=4,
    final_max_quanta=8,
)

print(spectrum.photon_energy_hartree)
print(spectrum.thermal_strength_au2)
print(spectrum.initial_population_coverage, spectrum.final_dipole_strength_coverage)
```

The signed 0–0 gap is the difference of the final and initial electronic minima plus the difference of their vibrational zero-point energies. For emission, supply the excited surface first, use a negative 0–0 gap for a downhill electronic transition, and select `process="emission"`.

The Duschinsky convention is `Q_final = rotation @ Q_initial + shift_au`. The two surfaces may have different positive frequencies. The supplied rotation must be square and orthogonal. For linear Herzberg–Teller coupling, `dipole_derivatives_au[k, c]` gives the derivative of Cartesian dipole component `c` with respect to initial normal coordinate `k`. The transition dipole is evaluated at `Q_initial = 0`. Use the same electronic phase for the dipole and its derivatives and the same mode signs for the transformation and derivatives. The solver combines Franck–Condon and Herzberg–Teller amplitudes before squaring.

Both cutoffs bound the **total** number of vibrational quanta. `initial_population_coverage` is the fraction of the exact harmonic partition function represented in the initial basis. `final_dipole_strength_coverage` is the retained strength divided by the dipole closure sum for those initial states, before filtering for positive photon energies. Check both when increasing the cutoffs. The dense reference recurrence is meant for small bases and stops when the requested overlap count exceeds `max_overlaps`. The returned strengths are squared dipole matrix elements. Observable-specific photon weighting, absolute rates, and lineshape convolution remain outside this interface.

## Prepare a vertical-gradient model from PySCF derivatives

The adapter reads completed PySCF data at a ground-state minimum. Supply a PySCF `Mole`, the ground-state Hessian in PySCF `(natm, natm, 3, 3)` layout, the total gradient of the chosen excited root, the vertical excitation energy, and its length-gauge transition dipole. The Hessian and gradient are in Hartree/bohr² and Hartree/bohr respectively. All inputs share the atom order and Cartesian frame. [PySCF TDDFT](https://pyscf.org/user/tddft.html) · [PySCF harmonic analysis](https://pyscf.org/_modules/pyscf/hessian/thermo.html)

```python
from pyscf_esd import harmonic_spectrum
from pyscf_esd.pyscf_adapter import vertical_gradient


def absorption_from_pyscf(mol, ground_hessian, excited_gradient, vertical_energy, dipole):
    model = vertical_gradient(
        mol, ground_hessian, excited_gradient,
        vertical_excitation_hartree=vertical_energy,
        transition_dipole_au=dipole,
    )
    return harmonic_spectrum(
        model.frequencies_hartree, model.frequencies_hartree,
        gap_00_hartree=model.gap_00_hartree,
        shift_au=model.shift_au,
        transition_dipole_au=model.transition_dipole_au,
        initial_max_quanta=0,
        final_max_quanta=4,
    )
```

The vertical-gradient model uses the ground-state curvature for both surfaces and subtracts the excited-state relaxation energy from the vertical gap. The adapter projects translations and rotations using PySCF's harmonic analysis and rejects imaginary or zero vibrational frequencies. It also returns signed Cartesian mode derivatives for projecting a phase-consistent Cartesian dipole derivative. The caller supplies the derivatives and identifies the electronic state. Calling the adapter launches no electronic calculations.

## Prepare an adiabatic-Hessian model from two PySCF minima

Supply the two independently optimized `Mole` objects, total electronic energies on a common energy reference, total gradients and Cartesian Hessians. The atom order and isotope mapping must correspond between the two inputs. The caller follows the target electronic state through the optimization and derivative calculations. When using PySCF TDA, the total excited-state energy at the excited geometry is the reference SCF total energy plus the excitation energy of the tracked root at that geometry.

```python
from pyscf_esd.pyscf_adapter import adiabatic_hessian

model = adiabatic_hessian(
    ground_mol,
    excited_mol,
    ground_hessian,
    excited_hessian,
    ground_energy_hartree=ground_energy,
    excited_energy_hartree=excited_energy,
    ground_gradient=ground_gradient,
    excited_gradient=excited_gradient,
)
absorption = model.spectrum(
    transition_dipole_au=dipole_at_ground,
    initial_max_quanta=0,
    final_max_quanta=4,
)
emission = model.spectrum(
    transition_dipole_au=dipole_at_ground,
    initial_max_quanta=0,
    final_max_quanta=4,
    process="emission",
)
```

The adapter aligns the geometries by a proper mass-weighted rigid fit, projects translations and rotations for each surface, and derives the 0–0 energy, displacement and Duschinsky transformation. It checks both minima for positive vibrational curvature and bounds the largest per-atom gradient norm by `stationary_tolerance` (default `1e-5` Hartree/bohr). The two vibrational subspaces can differ when the geometries distort. The adapter reports the singular values of their overlap, the orthogonalization correction, and the displacement projection residual. A full-rank overlap within `subspace_tolerance` of the unit singular values (default `1e-3`) is required. This criterion limits how far the orthogonal model may depart from the original Cartesian projection.

For linear Herzberg–Teller, pass `dipole_derivatives_au` to `model.spectrum`. Use the signed ground-state normal modes of the model and Cartesian dipoles in the same ground-state frame and electronic phase. For emission the adapter reverses the affine coordinate map and evaluates the same linear dipole at the excited minimum.

Exactly linear geometries leave the axial rotation undetermined by the fit. Mode labels within degenerate bending spaces therefore depend on the chosen basis. A physically anisotropic model at a linear geometry needs an explicit relative axial frame.

### Analytic S₁ Hessian for RHF–TDA/CIS

`analytic_cis_hessian` computes the total excited-state gradient and Hessian at one geometry from a converged molecular RHF calculation. Pass the amplitude of the tracked, isolated singlet root in the same MO basis. `tracked_root` here is the zero-based index in a PySCF singlet TDA calculation at the excited geometry.

```python
from pyscf_esd.cis_hessian import analytic_cis_hessian

s1 = analytic_cis_hessian(excited_rhf, excited_cis.xy[tracked_root][0])
# Pass s1.hessian, s1.gradient and s1.total_energy_hartree to adiabatic_hessian.
```

The method computes analytic AO integral derivatives and first and second RHF orbital response. It also includes the response of the targeted CIS eigenvalue. The route is dense and CPU-only, with conventional real, all-electron RHF, spherical AOs and the full singlet CIS excitation space. The caller checks spin and follows state identity between geometries. Near-degenerate targets and unsupported Hamiltonians are rejected. Analytic TDDFT curvature remains separate work.

The LiH/6-31G benchmark used signed CIS overlaps to track the state between geometries and used the analytic S₁ Hessian for the two-minimum Franck–Condon spectrum. The largest Cartesian discrepancy against central differences of native PySCF gradients at 0.002 bohr was 3.10×10⁻⁸ Hartree/bohr². The independent ORCA 6.1.1 calculation checked the Li–H bond projection. `examples/validate_lih_analytic_ah.py` reproduces the report and plots from the saved PySCF reference checkpoint. [PySCF CIS gradient](https://pyscf.org/_modules/pyscf/grad/tdrhf.html)

### Analytic Herzberg–Teller input for RHF–TDA/CIS

`analytic_cis_transition_dipole` follows the same reference and amplitude contract. It returns the length-gauge dipole and its Cartesian nuclear derivative in the phase of the input amplitude. The derivative includes the AO position integrals, RHF orbital response and CIS eigenvector response at the expansion geometry.

```python
import numpy as np

from pyscf_esd.cis_hessian import analytic_cis_transition_dipole

ht = analytic_cis_transition_dipole(ground_rhf, ground_cis.xy[tracked_root_at_ground][0])
derivatives = np.einsum("qax,axd->qd", model.ground_cartesian_mode_derivatives_au, ht.derivative)
absorption = model.spectrum(transition_dipole_au=ht.dipole, dipole_derivatives_au=derivatives)
```

The formaldehyde/6-31G(d) benchmark follows the lowest singlet from the planar S₀ geometry to a pyramidal S₁ minimum with six positive vibrational frequencies. The analytic S₁ Hessian agrees with two Cartesian PySCF gradient stencils within 1.84×10⁻⁶ Hartree/bohr². An independent ORCA calculation checks two projected curvatures. `examples/validate_h2co_analytic_ah.py` takes the archived PySCF electronic directory and an `--output` directory. It compares local, single-well AH+HT spectra with the full Duschinsky matrix and with its best signed mode correspondence, holding the modeled displacement in S₀ coordinates fixed. Both spectra use the same analytic HT expansion at S₀. The single-well comparison omits inversion tunneling and double-well parity. [Tatchen and Pollak](https://doi.org/10.1063/1.3074100)

## Run the CPU workflow

For an optimized ground-state geometry, the CPU workflow can compute the inputs and track an isolated singlet TDA state across optional Cartesian displacements. This first route uses all-electron, closed-shell RKS with screened direct integrals, unpruned grids, and HF/LDA/GGA-type functionals including hybrid GGAs. It has a projected-orbital warm start, cold DIIS and second-order SCF retries, and independent SCF and TDA residual checks. A small TDA problem can retry with a dense eigensolve of the same response operator if Davidson fails. The central reference receives RKS stability and stationarity checks.

```python
from pyscf_esd.cpu import Settings
from pyscf_esd.cpu_workflow import run_cpu_vibronic

# H2O geometry optimized at direct RKS/PBE0/6-31+G*, in bohr.
geometry = [[0, 0, 0], [-1.4508663771, 0, 1.1054578981], [1.4508663771, 0, 1.1054578981]]
result = run_cpu_vibronic(
    ("O", "H", "H"), geometry, Settings("6-31+g*", "pbe0", grid_level=5),
    target_index=0,
)
absorption = result.spectrum(final_max_quanta=8)
emission = result.spectrum(process="emission", final_max_quanta=8)
```

`target_index` selects a zero-based root at the center. The workflow matches displaced roots using signed, AO-overlap-weighted transition-density similarity, which measures particle–hole character. It rejects weak, ambiguous, or near-degenerate assignments and checks that it sampled at least one root above the match. Every displaced gradient is evaluated after matching. Raw amplitudes and dipoles are saved with each completed gradient, so a rerun can verify the match and reapply the phase even when the fresh center changes sign. The reference is rebuilt on every invocation. Use a separate checkpoint directory for each calculation and one writer per directory.

The default for this entry point uses analytic ground-state curvature and a single analytic excited-state gradient for VG Franck–Condon sticks. It performs no displaced electronic calculations. Set `include_ht=True` to request finite-difference transition dipoles, and `compute_excited_hessian=True` to request a separate local S₁ Hessian at the supplied ground-state geometry. Both can share displaced electronic solutions. Supply `checkpoint_dir` and `step_bohr` for either option. When using HT, emission extrapolates the dipole to the modeled VG minimum. For AH, use the separate two-minimum adapter above with the S₁ Hessian at its own minimum.

The CPU analytic TD-RKS gradient checks the absolute, orbital-gap-preconditioned Z-vector residual against `cphf_conv_tol` (default `1e-10`). If the native Krylov restart discards the small residual direction, the workflow normalizes that residual, corrects it with the same response operator, and verifies the result at the original tolerance. The earlier PySCF 2.14 water stencil checked numerical grid and step consistency within one TDA model. Experimental accuracy, full LR-TDDFT, near-degenerate state subspaces, ECP, dispersion and solvent routes require separate validation. [PySCF CPHF solver](https://github.com/pyscf/pyscf/blob/v2.15.0/pyscf/scf/cphf.py) · [PySCF Krylov solver](https://github.com/pyscf/pyscf/blob/v2.15.0/pyscf/lib/linalg_helper.py)

Run the molecular benchmark with `python examples/validate_water_cpu.py --stencil /path/to/dedicated/workdir` after installing the PySCF extra. It compares to direct PySCF reference energies and dipoles, checks derivatives by finite differences and displacement refinement, compares a pair of independent cold displacement points, and verifies reciprocal VG absorption and emission dipole strengths. The script writes numerical results and acceptance limits to `validation.json`.

## Run analytic VG on an NVIDIA GPU

In a Linux or WSL environment with compatible CuPy, cuTENSOR and GPU4PySCF packages, use `pyscf_esd.gpu.run_gpu_vibronic` with the same geometry, `Settings`, and `target_index` as the CPU example. The GPU entry point is currently a single-geometry direct FP64 route for all-electron, closed-shell PBE0 with singlet TDA. It constructs the same Franck–Condon model from the analytic ground-state Hessian and analytic excited-state gradient. It does not accept HT or numerical Hessian requests. [GPU4PySCF installation](https://github.com/pyscf/gpu4pyscf/tree/v1.9.2#installation)

The matched CPU/GPU comparison includes quadrature-grid response for the ground-state gradient and uses a fixed grid for the ground-state Hessian. PySCF's CPU semilocal Hessian omits this response, so the GPU default matches it. The GPU response equations reuse the primary unpruned grid. Explicit `hessian_grid_response=True` selects GPU4PySCF's distinct analytic Hessian option. [CPU RKS Hessian](https://raw.githubusercontent.com/pyscf/pyscf/v2.15.0/pyscf/hessian/rks.py) · [GPU RKS Hessian](https://raw.githubusercontent.com/pyscf/gpu4pyscf/v1.9.2/gpu4pyscf/hessian/rks.py)

For water, PySCF 2.14 and GPU4PySCF 1.9.2 both returned a nuclear CPHF solution with a residual above the requested tolerance. At the same determinant, the fixed-orbital Hessian and induced potential already agreed to within 6×10⁻¹² and 3×10⁻¹⁴ in atomic units, respectively. The CPU workflow uses the corrected upstream PySCF solver. The GPU workflow checks each nuclear response equation and, when needed, corrects the residual at the original tolerance using PySCF's host Krylov algebra with the same native GPU response operator. The entire Hessian is analytic. [PySCF Hessian CPHF issue](https://github.com/pyscf/pyscf/issues/2702) · [Upstream Krylov correction](https://github.com/pyscf/pyscf/pull/3384)

Run `python examples/validate_water_gpu.py /path/to/reference.json /path/to/report.json` with the retained water benchmark reference. Both workflows start from fresh electronic objects and generate absorption and emission sticks. The report includes CUDA-synchronized first and repeated workflow timings, direct GPU FP64 evidence, state-overlap and dipole-phase checks, derivative differences, frequencies and assigned-line comparisons.

The water benchmark on the RTX 4070 Laptop GPU passes all 16 original acceptance checks with zero displaced geometries. After the CPU gradient correction, the independently checked S₁ response residual is 8.51×10⁻¹¹ at the original 1×10⁻¹⁰ tolerance. The maximum CPU–GPU S₁ gradient and S₀ Hessian differences are 3.64×10⁻¹¹ Hartree/bohr and 7.04×10⁻¹¹ Hartree/bohr². The largest paired-line differences are 1.13×10⁻⁵ cm⁻¹ for absorption and 1.04×10⁻⁵ cm⁻¹ for emission, within the original 2.0 cm⁻¹ limit. Warm end-to-end medians over three repeats are 36.80 s on GPU and 21.37 s on the two-thread CPU path. The GPU route remains explicit and experimental.

## Anharmonic absorption and emission

Treat one large-amplitude coordinate explicitly with a discrete variable representation (DVR). Use periodic Fourier DVR for a torsion or sine DVR for a bound coordinate with vanishing wavefunctions at its two walls. Sample both electronic surfaces at the same geometries, in the same atom order and Cartesian frame. Supply potential energies on **one common electronic-energy zero**, and supply either a constant Cartesian transition dipole or its signed value at every grid point. For example, use $V_0=E_{\rm SCF}-E_{\rm SCF}(q_0)$ and $V_1=V_0+\Delta E_{\rm TDA}$. The DVR levels already include the active coordinate's zero-point energy. For the lowest sticks, the photon energy is the difference of the resulting vibronic levels. Apply the electronic gap once through the potential arrays.

```python
import numpy as np

from pyscf_esd import anharmonic_spectrum, solve_dvr

angle = 2 * np.pi * np.arange(96) / 96  # radians; omit the endpoint
inertia = 20_000.0                     # electron mass * bohr**2
settings = dict(boundary="periodic", n_levels=8, kinetic_coefficient=1 / inertia)
ground = solve_dvr(angle, 0.001 * (1 - np.cos(angle)), **settings)
excited = solve_dvr(angle, 0.12 + 0.0015 * (1 - np.cos(angle - 0.5)), **settings)
dipoles = np.column_stack((0.5 + 0.1 * np.cos(angle), np.zeros((angle.size, 2))))

absorption = anharmonic_spectrum(ground, excited, transition_dipole_au=dipoles)
emission = anharmonic_spectrum(excited, ground, transition_dipole_au=dipoles, process="emission")
```

With a nonperiodic mass-weighted coordinate in `sqrt(electron mass) * bohr`, set `boundary="dirichlet"` and use the default kinetic coefficient of one. The grid contains interior points; the walls are one grid spacing outside the first and last points. Check levels and line strengths by widening the domain and refining the grid. A torsion may use different constant inertias for the two states on the same angular grid. Ordinary periodic boundary conditions are used.

Optionally pass `HarmonicSpectators` to combine the anharmonic coordinate with independent harmonic modes. Spectators may have different frequencies, displacements, and an orthogonal Duschinsky rotation within their own subspace. They contribute their own zero-point energies and harmonic Franck–Condon overlaps. Exclude the explicitly treated coordinate from the spectator frequencies. For emission, order the spectator data from emitting to receiving surface as well. To reverse a supplied map `(J, K)`, use `(J.T, -J.T @ K)`. Spectator–active coupling, coordinate-dependent inertia, and nuclear-spin statistical weights require a richer nuclear model. [Hybrid DVR–harmonic spectroscopy](https://doi.org/10.1021/acs.jctc.7b00236)

At finite temperature, the active-coordinate Boltzmann weights come from the complete **finite DVR grid** partition. The harmonic spectator weights use their harmonic partition. The returned `initial_population_coverage` reflects the retained states, and `final_dipole_strength_coverage` reflects the closure sum for those initial states before the positive-photon-energy filter. Keep complete degenerate multiplets at each cutoff and converge the grid and state counts separately. Sticks report dipole strengths with optional Boltzmann weights, as in the harmonic solver.

For completed CPU electronic points, `pyscf_esd.anharmonic_pyscf.track_anharmonic_scan` builds the two common-zero potentials and uses cross-geometry AO overlaps to follow the chosen TDA state and sign. It checks the last-to-first link on periodic scans. `resample_periodic` can evaluate already sampled, phase-consistent potentials and dipoles on a denser nuclear grid. The electronic curves retain the resolution of the original scan. This workflow launches no electronic calculations or nuclear finite differences. Run `python examples/validate_anharmonic_rotor.py` for a finite-barrier model with an exact transition energy and dipole strength. A computational benchmark also used archived CF₃NO full-TDDFT energies and externally phase-tracked dipoles on a single shared cut. The DVR levels and absorption agree with an independent plane-wave implementation for that constant-inertia model. Experimental validation is pending.

For a coupled vibrational subspace such as a Fermi resonance, `mix_vibrational_states(hamiltonian_hartree, basis_transition_dipole_au)` diagonalizes a real symmetric Hamiltonian in an orthonormal basis. Supply signed Cartesian transition dipoles in the same basis order and phase convention. A harmonic calculation exposes these amplitudes as `HarmonicSpectrum.transition_dipole_au` in stick order. The mixing step rotates amplitudes coherently before squaring and returns the term energies, eigenvectors and dipole strengths. For fluorescence from a fixed initial term, subtract the mixed final-state terms from the initial term on the same energy reference. The benzene benchmark uses a Raman-parametrized block for ν₈ and ν₁+ν₆ in S₀.

### Coupled inversion and relaxation

`pyscf_esd.anharmonic_plane.SineGalerkin2D` solves coupled constant-mass surfaces on a rectangular domain symmetric in the first coordinate. It uses an analytic product-sine kinetic operator, Gauss–Legendre quadrature and separate exact inversion-parity blocks. Its transition matrices retain the sign of the sampled dipole. Install `.[anharmonic]` for the solver and quadrature utilities, or `.[pyscf,anharmonic]` for the electronic benchmark.

The formaldehyde example uses a common mass-weighted Cartesian surface for S₀ and S₁, with pyramidalization and one coupled planar relaxation coordinate. Both S₁ minima are included. Matched electronic points on opposite sides of the plane verify the phase before reflecting the energy and dipole grids. The four remaining modes supply a constant local zero-point correction. The spectrum uses the signed transition dipole on the plane and is compared with a harmonic model on the same plane and with the archived six-mode AH calculation. Both references use linear Herzberg–Teller. The plane limits relaxation in the remaining coordinates, and the electronic planar barrier is sensitive to that choice. [Formaldehyde anharmonic spectrum](https://doi.org/10.1063/1.3074100)

Run `python examples/validate_h2co_inversion.py /path/to/previous_AH /path/to/electronic_scan --output /path/to/output` with the archived PySCF data. Run again with `--plot` to render the saved spectra. The example reports the two lowest inversion parities and energies, dark 0–0 intensity, first bright transition, and the effects of basis, electronic interpolation and wall positions.

### Three-coordinate formaldehyde inversion

`InversionSpace` extends the two-coordinate cut with the mass-weighted residual needed to include the relaxed planar S₁ geometry. The volume interpolator preserves the finer archived `t=0` surface and interpolates corrections from direct PySCF points at nonzero `t`. `SineDVR3D` acts on a tensor sine grid with an analytic kinetic spectrum, fast sine transforms and exact reflection-parity sectors. It supports converged low eigenpairs and a Lanczos spectral measure for Gaussian broadening. The quadrature nodes describe a broadened observable. Use the low eigenpairs to assign the inversion interval and first allowed transition.

`examples/validate_h2co_3d.py` compares the three-coordinate surface and spectrum with the retained 2D inversion and 6D AH benchmarks. The model uses constant Cartesian masses, an absolute common electronic zero and three spectator modes through a constant local zero-point correction. The effective transverse curve is the lowest two-coordinate slice energy at each inversion coordinate. Its barrier diagnoses the relation of the full 3D levels to the inversion region. The example tests the grid, third-coordinate walls, separately computed electronic holdouts and the Lanczos resolution at the same σ = 50 cm⁻¹.

### Benzene Fermi resonance from electronic points

`examples/validate_benzene_3d.py` uses direct PySCF energies and phase-tracked transition dipoles along Wilson ν₁ and matching components of ν₆ and ν₈. The analytic S₀ Hessian defines the coordinates and supplies a separate curvature reference. The independently tested energy fit determines the surfaces. The calculation diagonalizes the anharmonic potentials in a product oscillator basis and projects the states onto the Fermi pair. The report includes a local quadratic control, a control with the direct cubic Fermi term removed, and sensitivity to the fit and vibrational basis. The 3D model retains one component of each degenerate mode.

`examples/compare_benzene_experiment.py` uses the standalone theoretical report to compare gas-phase emission maxima and absorption band centers. It keeps the computed origin in the report and uses the measured origin on the comparison axes. A separate legacy curve uses Raman input and the phase chosen from the fluorescence data. The plots use the benzene benchmark width of σ = 5 cm⁻¹. Experimental height readings describe the observed rotational envelopes. [Gas-phase fluorescence](https://doi.org/10.1021/ja00841a001) · [Gas-phase absorption centers](https://doi.org/10.1063/1.1767515)

`examples/validate_benzene_absorption.py` adds corrections from pairs of directly calculated energies taken at the same molecular geometry. Radial energy holdouts select the fit, and transverse holdouts test its transfer to ν₆ and ν₈. The original phase-tracked TDA dipole supplies intensities for this composite model. `examples/compare_benzene_absorption.py` extracts the ν₆ to ν₁+ν₆ progression by state overlap and compares it with gas-phase band centers at σ = 5 cm⁻¹.

`examples/benzene_spectator_coordinates.py` resolves the 27 spectator coordinates into 19 symmetry-adapted cuts from the analytic S₀ Hessian. `examples/validate_benzene_spectators.py` uses direct S₁ energies to couple those cuts to ν₁, compares conditional spectator ground states with explicit vibrational channels, and can jointly contract the two B₂g modes from mixed cuts. It evaluates nuclear derivative couplings analytically from the fitted potentials and local eigenstates. `examples/validate_benzene_spectator_mixing.py` measures sensitivity to the other sampled same-symmetry pairs and to the E₂g angular cubic terms.

## Verification and development

Run `python -m unittest discover -s tests -v` after installation. The tests use analytic oscillators, independent two-mode quadrature, synthetic inputs to PySCF's normal-mode analysis, and synthetic root-swap and phase-flip cases.

1. **End-to-end harmonic spectroscopy.** Generalize the state-tracked LiH and H₂CO workflows. Extend analytic excited-state curvature and transition-dipole response to TDDFT, sharing intermediates where possible. Validate derivatives on small systems before comparing cost and spectral accuracy against equivalent models. Develop a scalable time-domain solver checked against assigned sticks with the same inputs. [Analytic TDA Hessian](https://doi.org/10.1063/1.3605504) · [Analytic HT response](https://hdl.handle.net/11368/2304126) · [Baiardi, Bloino, and Barone](https://pmc.ncbi.nlm.nih.gov/articles/PMC6485600/)
2. **Coherent resonance Raman.** Sum complex tensors before taking intensities, over multiple electronic states at every laser energy. Test destructive interference, then benchmark hot bands and higher orders. ORCA 6.1.1 documents adding state spectra after intensity evaluation and disallows multiple states together with multiple laser energies. [ORCA resonance Raman](https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/esd.html#specific-keywords-and-details) · [Bianchi and Bloino](https://pmc.ncbi.nlm.nih.gov/articles/PMC13602930/)
3. **Nonradiative and spin-dependent processes.** Benchmark phosphorescence, intersystem crossing and internal conversion channels with explicit coupling, phase, units and energy-gap conventions. Extend IC to specified excited-state pairs. ORCA 6.1.1 documents an IC workflow with the final state fixed to S₀. [ORCA internal conversion](https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/esd.html#internal-conversion-rates-unpublished)
4. **Anharmonic and coupled nuclear motion.** Extend the one-coordinate DVR model to coupled coordinates and curvilinear kinetic operators using the earlier CF₃NO prototype. Treat coupled wavepacket propagation as its own observable and validate norm and population transfer independently.
