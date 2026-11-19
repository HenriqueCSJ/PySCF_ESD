# API reference

The public namespace is `pyscf_esd`. With PySCF installed, `from pyscf import esd` exposes the same objects and adds convenience functions for molecular electronic calculations. Use the [Python API guide](api.md) for complete examples and [prepared-data contracts](prepared-data.md) when supplying your own surfaces or overlaps.

Signatures below show the user-facing parameters. `ArrayLike` accepts NumPy arrays and compatible numeric sequences. Shapes are given in atom order, Cartesian order `x, y, z`, and signed normal-mode order. Values must be finite. Function docstrings remain available through `help(function)` in the installed version.

## Molecular workflow

### `pyscf.esd.run`

```text
esd.run(mol, *, xc="pbe0", target_index=0, **options)
```

`mol` is a built `pyscf.gto.Mole` with `spin=0`, a single named basis string, spherical all-electron orbitals, point nuclei, real atoms and `symmetry=False`. The function transfers its geometry, charge, basis and ordered isotope masses into the workflow. `target_index` is zero-based and identifies the singlet at the relaxed ground-state geometry. Other options are passed to `run_vibronic`. The root request includes at least one state above the target where the excitation space allows it.

Use `run_vibronic` directly when you need to supply an explicit `Settings` object. [Namespace source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf/esd/__init__.py)

### Run vibronic

```text
run_vibronic(
    symbols, coordinates_bohr, settings=None, *,
    target_index=0, masses_amu=None, model="auto", backend="cpu",
    solvent=None, process="both", temperature_kelvin=0.0,
    coverage=0.995, max_quanta=32, max_overlaps=250_000,
    fallback_xc=(), progress=None,
)
```

- `symbols` contains one element symbol per atom. `coordinates_bohr` has shape `(n_atoms, 3)` and requires at least two atoms.
- `settings=None` selects PBE0/def2-TZVPD. An explicit `Settings` object fixes the electronic recipe and tolerances.
- `masses_amu` optionally supplies a positive `(n_atoms,)` mass array for the same atom order. Otherwise the workflow uses the PySCF isotope-averaged masses.
- `model` selects `auto`, `ah` or `vg`. `backend` selects `cpu` or `gpu`. A named `solvent` requests equilibrium C-PCM and requires `backend="gpu"`.
- `process` selects `both`, `absorption` or `emission`. Each direction uses its own initial-surface populations at `temperature_kelvin`.
- `coverage` is the target for the numerical coverage checks. `max_quanta` bounds the adaptive total vibrational quantum cutoff. `max_overlaps` bounds the product of initial and final harmonic basis sizes.
- `fallback_xc` is an ordered sequence of allowed alternative functional names. Each attempted functional starts a complete electronic preparation with its own root ordering.
- `progress` accepts a callback receiving one stage message as a string. `progress=print` is sufficient for console feedback.

Returns `VibronicResult`. A checked calculation that cannot qualify the requested result raises `WorkflowIncomplete`. Invalid input can raise `ValueError`. [Workflow source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/workflow.py)

### `Settings`

Import with `from pyscf_esd.cpu import Settings` or `from pyscf import esd` followed by `esd.Settings`.

```text
Settings(
    basis, xc, nstates=5, charge=0, grid_level=4,
    scf_conv_tol=1e-11, scf_residual_tol=1e-7,
    td_residual_tol=1e-8, max_cycle=100,
    cphf_conv_tol=1e-10, cphf_max_cycle=100,
    direct_scf_tol=1e-13, stability_checks="minima",
)
```

`basis` is a PySCF basis name and `xc` uses PySCF/LibXC syntax. `nstates` requests response roots. `grid_level` is an integer from 0 through 9. `scf_conv_tol` controls the SCF energy convergence criterion. `scf_residual_tol`, `td_residual_tol` and `cphf_conv_tol` control the checked orbital, TDA and coupled-perturbed equations. `max_cycle` and `cphf_max_cycle` set their iteration budgets. `direct_scf_tol` controls direct integral screening. `stability_checks` accepts `minima` or `all`.

The supported electronic models depend on the selected derivative route. Consult [method selection](using-methods.md) before choosing an XC expression. A functional accepted by PySCF still needs the derivative orders and solvent response used by the requested ESD route. [Settings source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cpu.py)

### `VibronicResult` and `WorkflowIncomplete`

`VibronicResult` provides `model_requested`, `model_used`, `spectra`, `diagnostics` and the computed `physical_properties` property. `spectra` maps process names to `HarmonicSpectrum` or `AnharmonicSpectrum`. Read `model_used` to determine which model the automatic calculation qualified.

`WorkflowIncomplete` provides `diagnostics` and `partial_spectra`. The partial dictionary can be empty when failure occurred during electronic preparation. Coverage and status in the diagnostics determine how to interpret any retained lines. The numerical report aggregates those retained lines using their supplied populations. [Output interpretation](numerical-results.md)

## Prepared harmonic models

### `adiabatic_hessian`

Import from `pyscf_esd.pyscf_adapter` or `pyscf.esd`.

```text
adiabatic_hessian(
    ground_mol, excited_mol, ground_hessian, excited_hessian, *,
    ground_energy_hartree, excited_energy_hartree,
    ground_gradient, excited_gradient, masses_amu=None,
    stationary_tolerance=1e-5, subspace_tolerance=1e-3,
)
```

The two endpoint geometries have matching ordered atoms and charge. Each Hessian has PySCF shape `(n_atoms, n_atoms, 3, 3)` and units Hartree/bohr². Each gradient has shape `(n_atoms, 3)` and units Hartree/bohr. Each electronic total energy is evaluated on its own surface at its own geometry.

The stationarity limit applies to the largest per-atom gradient norm. `subspace_tolerance` bounds the departure of each vibrational overlap singular value from one before the overlap is fitted to an orthogonal Duschinsky matrix. The default is `1e-3`. Both surfaces use the same ordered masses.

Returns `AdiabaticHessianParameters` with the two frequency arrays, `gap_00_hartree`, `rotation`, `shift_au`, signed Cartesian normal-mode derivatives, alignment transformations and subspace diagnostics. Its `.spectrum(...)` method accepts a real dipole expanded in the ground-state frame and handles the inverse coordinate map for emission. [Adapter source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/pyscf_adapter.py)

### `run_prepared_ah`

```text
run_prepared_ah(
    parameters, transition_dipole_au, *, electronic_method,
    dipole_derivatives_au=None, process="both",
    temperature_kelvin=0.0, coverage=0.995,
    max_quanta=32, max_overlaps=250_000, progress=None,
)
```

`parameters` is an `AdiabaticHessianParameters` object. `electronic_method` is a nonempty description of the method that produced both surfaces. The dipole has shape `(3,)` in e·bohr and refers to the ground minimum. Optional derivatives have shape `(n_modes, 3)` with respect to its signed ground normal coordinates. The resulting `VibronicResult` identifies electronic validation as `caller_supplied` and runs the ordinary adaptive nuclear coverage checks. [Prepared AH example](prepared-data.md#reuse-two-completed-minima)

### `vertical_gradient`

```text
vertical_gradient(
    mol, ground_hessian, excited_gradient, *,
    vertical_excitation_hartree, transition_dipole_au, masses_amu=None,
)
```

Import from `pyscf_esd.pyscf_adapter`. Supply the ground-state Hessian and total excited-state gradient at the same stationary ground geometry. The adapter returns `VerticalGradientParameters` with equal surface frequencies, signed displacement, modeled 0–0 gap, reorganization energy and signed Cartesian normal-mode derivatives. This adapter consumes completed electronic data.

## Harmonic spectral solvers

### `harmonic_spectrum`

```text
harmonic_spectrum(
    initial_frequencies_hartree, final_frequencies_hartree, *,
    gap_00_hartree, transition_dipole_au,
    initial_max_quanta, final_max_quanta,
    rotation=None, shift_au=None, dipole_derivatives_au=None,
    temperature_kelvin=0.0, process="absorption", max_overlaps=250_000,
)
```

Each frequency vector contains positive vibrational energy quanta ℏω in Hartree. `gap_00_hartree` is the signed final-minus-initial energy including zero-point energies. The coordinate relation is `Q_final = rotation @ Q_initial + shift_au`. `rotation=None` gives the identity and `shift_au=None` gives zero displacement. The dipole and optional `(n_modes, 3)` derivatives refer to `Q_initial=0`. This low-level solver also supports complex dipole amplitudes.

The two integer cutoffs bound total quanta across all modes on each surface. The returned `HarmonicSpectrum` gives positive-photon-energy sticks. At finite temperature, omitted initial populations retain their missing weight in the coverage accounting. [Harmonic source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/harmonic.py)

### `HarmonicSpectrum`

- `photon_energy_hartree` is a one-dimensional array with one positive photon energy per line.
- `initial_quanta` and `final_quanta` have shape `(n_lines, n_modes)` and describe the corresponding harmonic basis states.
- `population` gives the initial-state thermal population for each line. Multiple final transitions from the same initial state share that population.
- `dipole_strength_au2` gives the squared Cartesian transition dipole norm. `thermal_strength_au2` multiplies it by `population`.
- `transition_dipole_au`, when supplied by the solver, has shape `(n_lines, 3)` and retains the Cartesian amplitudes. `condon_transition_dipole_au` gives the constant-dipole contribution in the chosen expansion.
- `initial_population_coverage` reports retained initial weight. `final_dipole_strength_coverage` reports dipole closure and can be `None` for an exactly dark model.
- `positive_photon_moments` holds retained moments and bounds used to qualify photon-weighted quantities, including absorption strength and emission rate.

### `thermal_harmonic_spectrum`

```text
thermal_harmonic_spectrum(
    initial_frequencies_hartree, final_frequencies_hartree, *,
    gap_00_hartree, transition_dipole_au,
    dipole_derivatives_au=None, rotation=None, shift_au=None,
    process, temperature_kelvin, fwhm_cm_inverse,
    step_cm_inverse=None, tail_tolerance=1e-12,
    max_grid_points=2**20, include_dipole_strength_density=False,
)
```

Uses the same harmonic coordinate and phase conventions. It returns a `ThermalSeparableSpectrum` evaluated through the all-quanta thermal characteristic function. `fwhm_cm_inverse` is the full Gaussian width. `step_cm_inverse` controls the spectral mesh independently. `include_dipole_strength_density=True` retains the reduced dipole-strength profile as an additional array. Photon-weighted profiles retain the E or E³ factor for absorption or emission. [Thermal source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/thermal_separable.py)

## Sampled anharmonic surfaces

### `solve_dvr`

```text
solve_dvr(
    grid, potential_hartree, *, boundary, n_levels,
    kinetic_coefficient=1.0, period=None,
)
```

Solves `H = -kinetic_coefficient / 2 * d²/dq² + V(q)` on a strictly increasing uniform grid. `boundary` selects `dirichlet`, `periodic` or `antiperiodic`. A mass-weighted coordinate uses `kinetic_coefficient=1`. A rotor angle in radians uses `1 / inertia`, with inertia in electron-mass·bohr². Rotor samples omit the duplicate endpoint and default to a period of `2*pi`. The current grid-size limit is 1,024 points.

`DVRResult` contains `grid`, `spacing`, `domain`, `boundary`, `kinetic_coefficient`, all finite-grid eigenvalues and retained eigenvectors. `coefficients[:, level]` is Euclidean-normalized. `wavefunctions` divides those coefficients by `sqrt(spacing)`. Grid-operator matrix elements use the coefficients directly. [DVR source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/anharmonic_dvr.py)

### `HarmonicSpectators`

```text
HarmonicSpectators(
    initial_frequencies_hartree, final_frequencies_hartree,
    initial_max_quanta, final_max_quanta,
    rotation=None, shift_au=None,
)
```

Defines modes separable from an active DVR coordinate, with the same harmonic affine convention as above. Exclude the active coordinate. The cutoffs apply to explicit sticks. The thermal characteristic-function solver sums all harmonic spectator states and uses the frequency and coordinate data independently of those cutoffs.

### `anharmonic_spectrum`

```text
anharmonic_spectrum(
    initial, final, *, transition_dipole_au,
    temperature_kelvin=0.0, process="absorption",
    spectators=None, max_transitions=250_000,
)
```

`initial` and `final` are `DVRResult` objects on the same grid and common electronic energy reference. The dipole is a real `(3,)` vector or `(n_grid, 3)` samples. If either nuclear boundary is antiperiodic, pass a `PhasedRotorDipole` with the product of the two nuclear closure signs. `max_transitions` bounds active-times-spectator pairs before photon filtering.

The returned `AnharmonicSpectrum` has the common energy, population, strength and coverage arrays. Its assignments use `initial_level`, `final_level`, `initial_spectator_quanta` and `final_spectator_quanta`. DVR level labels are zero-based. Retained level cutoffs must preserve complete degenerate manifolds. [Anharmonic source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/anharmonic.py)

### `thermal_separable_spectrum`

```text
thermal_separable_spectrum(
    initial, final, *, transition_dipole_au,
    process, temperature_kelvin, fwhm_cm_inverse,
    spectators=None, step_cm_inverse=None,
    tail_tolerance=1e-12, max_grid_points=2**20,
)
```

Uses the same active DVR contract and all-quanta harmonic spectators. The active partition function includes all finite-grid eigenvalues, so retained coefficients can expose a population tail. Increase retained levels and refine the electronic scan independently of the output spectral mesh.

### `ThermalSeparableSpectrum`

The two main output arrays are `wavenumber_cm_inverse` and `photon_weighted_density_per_cm_inverse`. Their entries use the same spectral grid. The ordinate contains dipole strength multiplied by photon energy in Hartree for absorption or its cube for emission, per cm⁻¹.

Metadata include `process`, `temperature_kelvin`, `fwhm_cm_inverse`, `step_cm_inverse`, `origin_00_cm_inverse`, `centroid_cm_inverse`, both coverage measures, `signed_dipole_strength_au2`, `positive_photon_weighted_strength`, `tail_probability_bound`, `auxiliary_sigma_cm_inverse` and `negative_weight_grid_fraction`. The optional `dipole_strength_density_per_cm_inverse` is present when requested from `thermal_harmonic_spectrum`.

## Tracking and periodic phase

`phase_periodic_scan(grid, ground_potential_hartree, excited_potential_hartree, transition_dipole_au, *, ground_overlaps, excited_overlaps, period=2*pi, minimum_overlap=0.65, n_points=None)` returns `PhasedPeriodicScan`. Its two overlap vectors include the last-to-first link. It exposes the two nuclear boundary strings, individual closure signs, source phase arrays and a `PhasedRotorDipole` for direct use by the nuclear solver.

`resample_periodic(samples, n_points, *, max_harmonic=None)` evaluates a trigonometric interpolant on a denser uniform period. `evaluate_rotor_samples(source_grid, samples, target_grid, *, period=2*pi, closure_sign=1)` evaluates periodic or antiperiodic samples at arbitrary angles. The latter accepts `closure_sign=-1` for a signed antiperiodic field.

`track_prepared_states` is imported from `pyscf_esd.prepared_state_tracking`. It selects two isolated physical states from candidate-root energies and signed intergeometry overlap matrices. [Root-index, overlap and phase contracts](prepared-data.md#track-states-before-building-a-surface)

The rotor-coordinate helpers `bond_rotation`, `project_torsion`, `project_spectator_modes`, `transport_bond_spectators` and `map_transported_spectators` construct analytic angular tangents, remove rigid motion and align the spectator basis. Their mass arguments are `masses_me`, in electron masses. [Rotor coordinate source](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/rotor_coordinates.py)

## Electronic derivative results

`analytic_molecular_tda_hessian(td, root_index, *, cphf_tolerance=1e-10, response_tolerance=1e-8, root_gap_tolerance=1e-5, max_cycle=150, progress=None)` accepts the supported GPU4PySCF equilibrium LR-TDA reference. `progress` receives `(stage, completed, total)`.

`MolecularTDAHessian.hessian` is the total excited-state Cartesian Hessian in Hartree/bohr². `excitation_gradient` is the gradient of the TDA excitation gap, so add the reference gradient for a total excited-state gradient. The other fields provide `excitation_hartree`, `orbital_response_residual`, `orbital_adjoint_residual`, `root_residual`, `root_separation_hartree` and `tda_response_residual`.

`analytic_tda_transition_dipole(td, root_index, *, phase=1.0, root_gap_tolerance=1e-5, response_tolerance=1e-8, cphf_conv_tol=1e-10, cphf_max_cycle=100)` returns `TDADipoleResult`. Its `dipole` has shape `(3,)` in e·bohr. `derivative[a, x, d]` gives the derivative of dipole component `d` with respect to nuclear coordinate `(a, x)`, in e. Additional fields report the root index, separation, root and adjoint residuals, translation error and rotation error. [Supported reference and setup](api.md#analytic-hessian-and-ht-from-a-prepared-solvated-tda-root)

The lower-level conventional RHF/CIS derivative helpers are imported from `pyscf_esd.cis_hessian`. CPU and GPU fixed-geometry VG helpers are imported from `pyscf_esd.cpu_workflow` and `pyscf_esd.gpu`, respectively. Their keyword options and derivative choices are explained in [method selection](using-methods.md).

## Observables and vibrational mixing

`spectral_line_properties(spectrum, process)` returns input-order arrays of photon-weighted strengths and direction-specific observables. Absorption provides `oscillator_strength` and `population_weighted_oscillator_strength`. Emission provides `einstein_a_s_inverse`, `population_weighted_einstein_a_s_inverse` and `population_weighted_radiant_power_w`. Populations enter once and keep the supplied normalization.

`summarize_spectral_properties(spectra, *, spectral_status=None, temperature_kelvin=None, max_dominant_lines=5, assignment_scheme=None, target_coverage=None)` returns a JSON-ready dictionary. Supply a mapping such as `{"absorption": absorption, "emission": emission}`. `assignment_scheme` selects `harmonic_quanta`, `vci_dominant_harmonic_labels` or `dvr_levels` when the corresponding labels are meaningful. [Numerical fields and interpretation](numerical-results.md)

`electronic_relaxation_properties` accepts the four cross-geometry total energies `ground_at_ground_hartree`, `ground_at_excited_hartree`, `excited_at_ground_hartree` and `excited_at_excited_hartree`, plus optional endpoint dipoles. `harmonic_model_properties` accepts an AH or VG model with optional normal-coordinate dipole derivatives and endpoint dipoles. Both report physical descriptors of the supplied model.

`mix_vibrational_states(hamiltonian_hartree, basis_transition_dipole_au)` accepts an exactly symmetric real `(n_basis, n_basis)` Hamiltonian and signed dipoles with shape `(..., n_basis, 3)`. `VibrationalMixingResult` contains ordered `energies_hartree`, eigenvectors by column, mixed Cartesian dipoles and squared strengths. The transformation acts on amplitudes, preserving interference and intensity borrowing.

## Spin-resolved public objects

The same public namespace exports `analytic_uks_triplet_endpoint`, `unrestricted_tda_vertical_spectrum`, `run_pyscf_tda_soc`, `assemble_pyscf_tda_soc`, `tda_soc_response`, `spin_orbit_phosphorescence`, `spin_orbit_dipole_derivatives` and `run_prepared_phosphorescence`.

Their result types are `UKSTripletEndpoint`, `UnrestrictedTDAVerticalSpectrum`, `PySCFTDASOC`, `TDASOCResponse`, `PhosphorescenceResult` and `VibronicPhosphorescenceResult`. The [phosphorescence and open-shell guide](phosphorescence.md) defines their state, spin, units and response conventions and supplies the associated execution examples.

## Inspect the installed interface

```python
import inspect
import pyscf_esd

print(pyscf_esd.__all__)
print(inspect.signature(pyscf_esd.run_vibronic))
help(pyscf_esd.thermal_harmonic_spectrum)
```

Use the installed package version alongside the documentation version selector when reproducing a calculation. Source links on this site point to the current `main` branch. Record a release tag or commit when the exact implementation matters. [Public exports](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/__init__.py)
