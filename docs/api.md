# Python API

[Project home](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/README.md) · [Workflow guide](guide.md) · [Methods](methods.md) · [Validation](validation.md)

This guide connects the principal Python entry points to complete calculations. The [API reference](api-reference.md) gives signatures, result attributes and defaults. The [prepared-data guide](prepared-data.md) explains how to reuse completed electronic calculations, import signed state overlaps and assemble periodic rotor models.

## Use as a PySCF module

Install the package into the same Python environment as PySCF. Both namespaces expose the spectral solvers. The `pyscf.esd` namespace also supplies the `run` convenience function and the electronic `Settings` object.

```python
from pyscf import esd, gto

mol = gto.M(
    atom="O 0 0 0; H 0 -0.76 0.58; H 0 0.76 0.58",
    basis="6-31+g*", unit="Angstrom", charge=0, spin=0,
    symmetry=False,
)
result = esd.run(
    mol, xc="pbe0", target_index=0, model="ah",
    backend="gpu", solvent="water", temperature_kelvin=298.15,
    process="both", progress=print,
)
print(result.model_used)
print(result.physical_properties["spectra"]["emission"])
```

This example launches a complete electronic and nuclear calculation and requires the GPU dependencies for its selected backend. For a CPU gas-phase calculation, set `backend="cpu"` and omit `solvent`. Named C-PCM solvents in this high-level interface require the GPU backend. The starting `Mole` supplies geometry, charge, a single named basis and atomic masses, including isotope overrides. Use `masses_amu` to override that ordered mass list explicitly.

`esd.run` requires a built closed-shell `Mole`, spherical all-electron orbitals, point nuclei, `symmetry=False` and real atoms. It accepts one named basis string. For a custom `Settings` instance, call [`run_vibronic`](api-reference.md#run-vibronic) directly. A caller-prepared `Mole` can start in ångström because `esd.run` obtains its internal coordinates in bohr. The coordinate-array interface expects bohr.

## Choose an API entry point

- Use `esd.run(mol, ...)` or `run_vibronic(...)` to start from one molecular geometry and request optimization, electronic derivatives and a checked spectrum.
- Use `run_prepared_ah(...)` after constructing an AH model from two completed minima. It performs the adaptive nuclear calculation using the supplied surfaces and dipole expansion.
- Use `harmonic_spectrum(...)` for explicit line assignments at chosen quantum cutoffs, or `thermal_harmonic_spectrum(...)` for a broadened all-quanta harmonic profile.
- Use `solve_dvr(...)` with `anharmonic_spectrum(...)` or `thermal_separable_spectrum(...)` for sampled potentials and a coordinate-dependent dipole.
- Use the [phosphorescence API](phosphorescence.md) for SOC-mediated emission, analytic spin-vibronic HT and open-shell transitions.

The fixed-geometry `run_cpu_vibronic` and `run_gpu_vibronic` functions construct VG models at a stationary S₀ input. The complete optimized AH and automatic routes are selected through `run_vibronic(model="ah" or "auto", backend=...)`.

## Units and input conventions

- Electronic, vibrational and photon energies in solver inputs and results are in Hartree. A harmonic frequency input is the energy quantum ℏω. To convert from cm⁻¹, divide by `pyscf_esd.spectral_properties.HARTREE_TO_WAVENUMBER`.
- Electronic Cartesian positions are in bohr in Python. PySCF gradients are in Hartree/bohr with shape `(n_atoms, 3)`. The PySCF adapters expect Hessians in Hartree/bohr² with shape `(n_atoms, n_atoms, 3, 3)`.
- Mass-weighted normal coordinates `Q` use √(electron mass)·bohr. Atomic masses for the rotor tools are given in electron masses. Rotor angles are in radians and angular inertias are in electron mass·bohr².
- Real, length-gauge transition dipoles are signed Cartesian vectors in e·bohr. Use one body frame, consistent Cartesian mode signs, and one relative electronic phase wherever amplitudes enter a common calculation.
- Keep one atom order and isotope mapping when combining geometries, derivatives, or modes. Python electronic indices are zero-based.

For unrestricted DFT, open-shell optical transitions and weak-SOC phosphorescence, see the [spin-resolved API guide](phosphorescence.md). It includes the full-space TDA moment and analytic gas-phase Herzberg–Teller profile example.

## End-to-end workflow

```python
from pyscf_esd import WorkflowIncomplete, run_vibronic
from pyscf_esd.cpu import Settings

water_bohr = [[0., 0., 0.], [-1.4508663771, 0., 1.1054578981],
              [1.4508663771, 0., 1.1054578981]]
try:
    result = run_vibronic(
        ("O", "H", "H"), water_bohr,
        Settings(basis="6-31+g*", xc="pbe0"),
        target_index=0, model="vg", temperature_kelvin=0.0,
    )
except WorkflowIncomplete as error:
    print(error.diagnostics)
    raise

print(result.model_used, result.physical_properties["spectra"]["emission"])
```

`run_vibronic` returns `VibronicResult`, with `spectra` keyed by direction, `diagnostics`, and derived `physical_properties`. Use `process` to request one direction or both. A failed checked workflow raises `WorkflowIncomplete`; inspect its `diagnostics` and `partial_spectra` before consuming partial results. Invalid API arguments may raise ordinary `ValueError`.

The solver accepts `coverage`, `max_quanta`, `max_overlaps`, ordered `fallback_xc` and a `progress` callback. Default numerical budgets match the [CLI](guide.md#command-line-options). `Settings` also exposes the grid level, root count, SCF and TDA tolerances, coupled-perturbed response tolerance, and `stability_checks`. Request a higher root than your target where required to diagnose isolation.

For a geometry already optimized on S₀, `pyscf_esd.cpu_workflow.run_cpu_vibronic` skips ground-state optimization and builds VG directly. It can track roots on caller-supplied displacements, optionally evaluate finite-difference HT or a local excited-state Hessian at S₀, and checkpoint the electronic points. Provide `checkpoint_dir` and `step_bohr` for the optional derivatives. Use one writer and calculation per checkpoint directory. Curvature at S₀ belongs to the local derivative test, while AH requires S₁ curvature at its own minimum. The [GPU counterpart](guide.md#nvidia-gpu-workflow) implements the analytic VG path.

## Analytic Hessian and HT from a prepared solvated TDA root

`pyscf_esd.analytic_tda_transition_dipole(td, root_index, phase=...)` accepts a converged singlet GPU4PySCF RKS/TDA object with equilibrium LR C-PCM. Use a real, all-electron closed-shell global hybrid LDA or GGA reference without nonlocal correlation. Set `mf.small_rho_cutoff = 0` and `mf.cphf_grids = mf.grids` before SCF. Solve at least one root above the target to check isolation. Use the full singlet excitation space and set `phase` from the tracked electronic overlap.

The result gives the signed length-gauge dipole in e·bohr and `derivative[a, x, d]` in e, with axes for atom, nuclear displacement and dipole component. It also gives the TDA root residual, adjoint residuals and rigid-motion checks. Project `derivative` onto the signed Cartesian normal-mode derivatives at the expansion geometry before using it in the nuclear solver. The occupied orbital response is verified against the native CPHF equation. `cphf_conv_tol` also sets the tolerance requested from GPU4PySCF's TD gradient Z-vector solver.

`pyscf_esd.analytic_molecular_tda_hessian(td, root_index)` accepts the same electronic reference with density fitting for both J and K, spherical AOs, and the same equilibrium SWIG cavity for the RKS reference and the TDA root. `hessian` is the **total** excited-state Hessian in PySCF atom order. `excitation_gradient` is the gradient of the TDA gap. Add it to the ground-state gradient at the supplied geometry to obtain the total excited-state gradient. The result records residuals for the reference orbital response, its adjoint, the TDA eigenpair, and the projected TDA response. The optional `progress(stage, completed, total)` callback receives milestones.

At a tracked S₁ minimum, pass this Hessian with the S₀-minimum Hessian and the endpoint energies and gradients to `pyscf_adapter.adiabatic_hessian`. This yields the Duschinsky map and both sets of vibrational frequencies. If the TDA eigenvector contribution misses the Hessian symmetry tolerance, the solver retries the projected response once at tighter residual tolerances using the existing derivative actions. The fourth semilocal XC derivative uses an analytic expression for PBE0. Other supported LDA/GGA functionals require a fourth-order native LibXC or a matching `PYSCF_ESD_LIBXC_LXC` shared library built with `MAXORDER=4`.

## Harmonic transitions from prepared parameters

```python
from pyscf_esd import harmonic_spectrum, spectral_line_properties

sticks = harmonic_spectrum(
    [0.005], [0.005], gap_00_hartree=0.12,
    transition_dipole_au=[1.0, 0.0, 0.0], shift_au=[15.0],
    temperature_kelvin=298, initial_max_quanta=4, final_max_quanta=8,
)
lines = spectral_line_properties(sticks, "absorption")
print(sticks.initial_population_coverage, sticks.final_dipole_strength_coverage)
strongest = lines.population_weighted_oscillator_strength.argmax()
print(sticks.photon_energy_hartree[strongest], lines.population_weighted_oscillator_strength[strongest])
```

`harmonic_spectrum` takes initial and final positive vibrational quanta, a signed 0–0 energy, a transition dipole at the initial minimum, and total-quanta cutoffs. The relation is `Q_final = rotation @ Q_initial + shift_au`, with an orthogonal square `rotation`. For linear HT, supply `dipole_derivatives_au` with shape `(n_modes, 3)`, with respect to the **initial** signed normal coordinates. The phase of the expansion point and derivatives must agree. The solver adds amplitudes before squaring.

For emission, put the excited surface first, reverse the affine map as `(J.T, -J.T @ K)`, use the signed final-minus-initial 0–0 gap, and set `process="emission"`. The transition dipole and derivatives must refer to the new initial coordinates. The [AH adapter](#pyscf-adapters) performs these transformations automatically. Returned sticks retain positive photon energies and store raw and population-weighted dipole strengths. Coverage is evaluated before the positive-energy filter.

`pyscf_esd.thermal_harmonic_spectrum` takes the same frequency, rotation, shift, dipole and signed 0–0 conventions and returns a Gaussian-broadened profile at the specified temperature and `fwhm_cm_inverse`. The thermal characteristic function sums the coherent Condon and linear HT strengths over all harmonic quanta. Spectral ordinates are weighted by positive photon energy for absorption and its cube for emission. Multiply by the corresponding oscillator-strength or Einstein-A prefactor to obtain physical units. Individual line assignments are available from the stick solver.

## PySCF adapters

`pyscf_esd.pyscf_adapter.vertical_gradient` takes a PySCF `Mole`, a ground-state Hessian, the **total** gradient of the excited root at the same geometry, and the vertical excitation energy and dipole. It returns `VerticalGradientParameters` with equal surface frequencies, a displacement, a signed 0–0 gap and Cartesian mode derivatives. Pass them to `harmonic_spectrum`, or use `CpuVibronicResult.spectrum` if you ran the CPU workflow.

`pyscf_esd.pyscf_adapter.adiabatic_hessian` takes two `Mole` objects, two Hessians, two total electronic energies and two gradients. An excited-state TDA total energy is the reference SCF total energy **at the excited geometry** plus the energy of the tracked root there. The adapter mass-aligns the geometries and assesses stationarity, vibrational curvature and subspace compatibility. It launches no electronic calculations.

```python
from pyscf_esd.pyscf_adapter import adiabatic_hessian

def ah_spectra(mol0, mol1, h0, h1, e0, e1, g0, g1, dipole0, dmu_dq0):
    model = adiabatic_hessian(
        mol0, mol1, h0, h1, ground_energy_hartree=e0, excited_energy_hartree=e1,
        ground_gradient=g0, excited_gradient=g1,
    )
    return {
        process: model.spectrum(
            transition_dipole_au=dipole0, dipole_derivatives_au=dmu_dq0,
            process=process, initial_max_quanta=0, final_max_quanta=8,
        )
        for process in ("absorption", "emission")
    }
```

In this example, `dipole0` and `dmu_dq0` are supplied by the caller in the frame and signed modes of the resulting model. With a Cartesian derivative `dmu_dR` in the common frame, project it after building the model through `np.einsum("qax,axd->qd", model.ground_cartesian_mode_derivatives_au, dmu_dR)`.

For conventional all-electron RHF, use `pyscf_esd.cis_hessian.analytic_cis_hessian(reference, amplitude)` and `analytic_cis_transition_dipole(reference, amplitude)`. The amplitude refers to an isolated singlet CIS root in the same MO basis, for example `tda.xy[root_index][0]`. The Hessian result provides `total_energy_hartree`, `gradient`, and `hessian`. The dipole result provides `dipole` and `derivative`; use the expansion-point root and project its Cartesian derivative as above. Explicitly match roots across geometries before combining their data.

For a converged PySCF state-averaged CASSCF object, call `pyscf_esd.casscf_response.checked_sacasscf_gradient(method, root)` with the raw zero-based root index at that geometry. It returns the analytic Cartesian gradient and the measured linear-response residual and limit for every attempt. The response retries preserve the tolerances and electronic method. Use a caller-tracked root and retain its identity and phase separately. This API supplies first derivatives. A CASSCF calculation that needs nuclear curvatures requires an independently qualified source of curvature.

For independently prepared electronic data, `run_prepared_ah(parameters, transition_dipole_au, electronic_method=...)` accepts the same AH parameters and runs both directions with the ordinary workflow's adaptive thermal and dipole-closure checks. Pass optional `dipole_derivatives_au` for linear HT. The result carries the method name and labels electronic validation as caller supplied. The electronic source must provide a consistent state pair, energy reference, curvature, and dipole phase. The workflow uses the supplied 0–0 energy directly.

For a converged CPU RKS object, `pyscf_esd.cpu.assess_rks_stability(mean_field)` independently checks SCF stationarity and tests the internal, real-to-complex and restricted-to-unrestricted orbital Hessians, including symmetry-breaking rotations. Supply `scf_residual_tolerance` and `minimum_curvature` to use a declared benchmark policy. The default positive margin is 10⁻⁶ in each operator's native PySCF scaling. Each channel reports its status, curvature and residual, or the reason the check remained unresolved. A failed channel can report a Rayleigh quotient even before an eigenpair is converged. Functionals with nonlocal correlation raise an unresolved-reference error because their full internal response is unavailable. This API evaluates the supplied determinant and leaves orbital recovery to the ordinary workflow.

For a converged molecular ROKS object, `pyscf_esd.roks_stability.assess_roks_internal_stability` checks stationarity from fresh spin Fock matrices and evaluates the analytic curvature for real rotations between different occupation classes. `internal_qualified` applies to the fixed spin and orbital-sharing constraints. External relaxation, complex orbital rotations, and continuity of the high-spin reference across geometries require separate evidence. The supported routine uses the PySCF 2.15 shared-orbital operator for gas-phase CPU LDA and GGA references, including hybrid GGAs. [Native implementation](https://github.com/pyscf/pyscf/blob/v2.15.0/pyscf/soscf/newton_ah.py#L103-L178)

The experimental `pyscf_esd.openqp_mrsf_prepared` module reads full, untransported OpenQP MRSF singlet checkpoint JSON from the 1.3.1 snapshot `1db79089b285406338bd28a255b967e4011f8df7`. It keeps total response-root energies on their original zero and reconstructs signed state-interaction dipoles in the checkpoint gauge. `exact_openqp_mo_overlap` evaluates the raw AO contraction and can check the native MO overlap for additional rescaling or sign changes. The caller verifies both electronic settings and the AO convention across points. This interface currently has synthetic fixture coverage. Qualifying the signed overlap of the physical MRSF states requires an additional native checkpoint test. [OpenQP transition properties](https://open-quantum-platform.github.io/openqp-docs/workflows/mrsf-analysis/)

Pass the verified doubly occupied core count as `n_core` to `read_openqp_mrsf_point` to also read the checkpoint's response vectors. This mode reconstructs the stored difference and signed transition densities from the vectors in the same root order. Evaluate `previous.signed_state_overlap(current, ao_cross_overlap)` with raw AO overlap rows in the previous checkpoint's basis and columns in the current checkpoint's basis. The method contracts in the native MO gauges and returns the raw signed overlap of the restricted two-SOMO response model. Use the [prepared-overlap checks](validation.md#archived-mixed-reference-tddft-investigation) before transporting root phases. Validation with independent untransported energy checkpoints is pending.

When both checkpoints include the pinned portable shell data, `previous.cross_ao_overlap(current)` evaluates the raw cross-AO integrals with PySCF in a common Cartesian frame. It restores native shell order and the original contraction norms, and independently reconstructs both self metrics before returning the cross matrix. It uses the raw AO convention of the OpenQP coefficient tags. The portable viewer's Molden orbital order is a separate convention. Omit `ao_cross_overlap` from the signed-state or auxiliary-reference method to evaluate it automatically. Pass the returned matrix explicitly if both methods will use the same link. The default convention covers Cartesian s/p and spherical s through f. [PySCF cross-geometry integrals](https://pyscf.org/pyscf_api_docs/pyscf.gto.html#pyscf.gto.mole.intor_cross)

`pyscf_esd.mrsf_overlap.expand_restricted_mrsf_singlets(vectors, n_core=..., n_orbitals=...)` also accepts decoded amplitudes directly. Pass one root per column and particle-major excitation rows with the hole varying fastest. The pinned OpenQP JSON serializer emits these logical axes. Its in-memory tag buffers use a different interpretation. The decoder checks the unused SOMO slot and root orthonormality. `bra.overlap(ket, raw_mo_overlap)` factors the raw bra-by-ket MO metric into four oriented determinant-minor blocks. The explicit `coefficients`, `alpha`, and `beta` arrays are available for independent checks.

For the auxiliary open-shell reference, `previous.auxiliary_reference_link(current, ao_cross_overlap, minimum_singular_value=...)` reports the core, two-SOMO and full alpha-occupied singular spectra. It also reports how well each occupied class is represented in the complete opposite MO space in both directions. Choose the link cutoff together with the geometry spacing. The same screen accepts an already contracted MO metric through `pyscf_esd.roks_stability.assess_roks_reference_link`.

`pyscf_esd.openqp_mrsf_prepared.paired_openqp_triplet_gap_hartree(singlet, triplet_checkpoint, engine_commit=...)` accepts a raw companion checkpoint for a separately converged triplet response at that same geometry. Verify the input recipes and that the response window covers the two lowest triplets. The function checks the AO metric, shared auxiliary energy and core/SOMO projectors. It returns the difference of the first two response offsets. Small response gaps flag possible reference changes. Local stationarity, orbital-Hessian stability and reference continuity between samples require their own evidence. [MRSF reference diagnostics](https://pubs.acs.org/doi/10.1021/acs.jpclett.6c01200)

`pyscf_esd.openqp_roks_prepared.read_openqp_saved_roks_reference` reads the auxiliary open-shell determinant and checks both saved spin densities against the occupied orbitals. It returns the orbital residual formed from the *recorded* alpha and beta Focks in the original MO gauge. The native records may reflect the solver's last stored Focks. A fresh final-density evaluation is a separate check.

`assess_openqp_roks_internal_stability` in the same module imports those MOs into PySCF with their column phases and occupations intact. Pass the independently verified gas-phase, all-electron XC and exact Coulomb/exchange recipe, charge, and the actual final XC grid coordinates in bohr and partitioned weights. A fresh PySCF evaluation must match the recorded energy and spin Focks and meet the orbital residual tolerance before the analytic real shared-orbital Hessian is evaluated. The workflow performs zero SCF iterations. The pinned OpenQP full JSON lacks the final quadrature arrays. The generic `pyscf_esd.roks_stability.assess_fixed_roks_internal_stability` accepts an already configured PySCF ROKS object for other prepared sources. Its reference-energy check is required and the native spin-Fock comparison is optional. [PySCF DFT model](https://pyscf.org/user/dft.html) · [PySCF analytic shared-orbital Hessian](https://github.com/pyscf/pyscf/blob/v2.15.0/pyscf/soscf/newton_ah.py)

### Spin-flip optical properties

`pyscf_esd.gpu_spin_flip.solve_spin_flip_tda(uks)` accepts a converged GPU4PySCF UKS triplet. Identify the physical roots using the calculated spin and state character. Pass their one-based labels to `response.transition_dipole(ref=initial, state=final)` and `response.oscillator_strength(ref=initial, state=final)`. The optical energy is the difference between the two spin-flip root energies. The returned dipole uses the PySCF length-gauge position convention and follows the phases of the two roots.

The spin-flip TDA optical calculation includes the analytic orbital response at the energy difference between the roots. It uses SF-TDA amplitudes and the spin-conserving TDDFT response operator. This route is benchmarked at two fixed CF₃NO geometries. Frequency-dependent adiabatic response can become unreliable near a pole of the spin-conserving operator. See [Parker, Rappoport and Furche](https://pubs.acs.org/doi/10.1021/acs.jctc.7b01008).

## Periodic rotor and thermal profile

```python
import numpy as np
from pyscf_esd import anharmonic_spectrum, solve_dvr, thermal_separable_spectrum

angle = np.linspace(0, 2 * np.pi, 96, endpoint=False)
inertia = 20_000.0
opts = dict(boundary="periodic", kinetic_coefficient=1 / inertia, n_levels=angle.size)
ground = solve_dvr(angle, 0.001 * (1 - np.cos(angle)), **opts)
excited = solve_dvr(angle, 0.12 + 0.0015 * (1 - np.cos(angle - 0.5)), **opts)
dipoles = np.column_stack((0.5 + 0.1 * np.cos(angle), np.zeros((angle.size, 2))))

sticks = anharmonic_spectrum(ground, excited, transition_dipole_au=dipoles)
emission = thermal_separable_spectrum(
    excited, ground, transition_dipole_au=dipoles, process="emission",
    temperature_kelvin=298, fwhm_cm_inverse=80,
)
print(emission.origin_00_cm_inverse, emission.initial_population_coverage)
```

- `solve_dvr` accepts a uniform grid with the duplicate periodic endpoint excluded. For a nonperiodic mass-weighted coordinate, choose `boundary="dirichlet"` and a unit kinetic coefficient. Its walls are one spacing outside the supplied interior grid. Both potentials must share one electronic energy zero. Different constant inertias may be supplied on the same angular grid.
- `anharmonic_spectrum` returns assigned DVR lines and optional spectator quanta. It accepts a constant three-vector or signed samples with shape `(n_grid, 3)`. Order the surfaces by initial and final state. Along a closed scan verify phase closure, including the last-to-first link. `pyscf_esd.anharmonic_pyscf.track_anharmonic_scan` can follow the state from completed PySCF TDA points, and `resample_periodic` can evaluate periodic fits on a denser nuclear grid.
- For two explicitly tracked real electronic states, `phase_periodic_scan` takes both signed overlap arrays in the original wavefunction gauges, including the closing link. Use the returned `ground_boundary` and `excited_boundary` in the corresponding `solve_dvr` calls. Pass the returned `transition_dipole` directly to the stick or thermal spectrum solver. If one surface requires an antiperiodic nuclear function, the matching dipole changes sign across the seam. If both surfaces are antiperiodic, the dipole is periodic and both nuclear boundary conditions still apply. The caller verifies the labeled geometries close in a common frame and the two states remain isolated.

- If the data source provides signed overlaps between all relevant real electronic roots at adjacent geometries, call `pyscf_esd.prepared_state_tracking.track_prepared_states` first. It checks forward and reverse assignments, separation from the supplied competitors, and closure of both root identities. Select a raw transition dipole for each returned root pair. Pass its two columns of `signed_link_overlaps` and the raw dipoles to `phase_periodic_scan`. For an open path, use `align_transition_dipoles` to apply the transported relative phase. A state-interaction transition density supplies dipoles and character diagnostics. The individual electronic phases require the signed physical-state overlaps.

- To calibrate a new signed overlap evaluator, `pyscf_esd.prepared_overlap_qualification.qualify_prepared_overlap` checks an informative pair of different geometries. Supply the two self-overlaps, forward and reversed overlaps, separate input gauge flips on each axis, and an independent forward result in the same physical-state representation. A successful calibration returns the raw forward matrix for the existing state tracker. The caller establishes how the matrices were evaluated. Later production links can use the calibrated evaluator.
- To put separately prepared loops on a common physical angle, `evaluate_rotor_samples` evaluates one uniform source grid at arbitrary angles. Subtract each source's physical angular origin from the target coordinate. Use closure sign +1 for energies and the product of the two electronic phase signs for a dipole. A measured cross-template overlap is required to compare signed dipoles that were originally transported in independent gauges.
- `HarmonicSpectators` takes the two spectator frequency arrays and optional orthogonal map and displacement. Exclude the active coordinate from those arrays, use a shared angular frame, and reverse the map for emission. The stick route truncates the thermal spectator basis at the supplied cutoffs. The `thermal_separable_spectrum` route sums the harmonic spectator basis with a characteristic function and returns a profile without assignments.
- `bond_rotation` supplies a geometry and exact angular tangent. `project_torsion` removes rigid motion, `project_spectator_modes` obtains the local spectator modes, `transport_bond_spectators` moves their bases to a common angle, and `map_transported_spectators` constructs the affine spectator map. Use overlap singular values and step-size refinement to assess the mapping.
- Check grid size, retained DVR levels, walls where applicable, periodic fit quality, thermal coverage and mesh convergence separately. Retain complete degenerate manifolds when cutting the level list. The thermal profile uses **FWHM in cm⁻¹**, and its ordinate contains the dipole strength weighted by photon energy in Hartree (absorption) or its cube (emission) per cm⁻¹. See [spectral observables](methods.md#physical-observables) for the absolute prefactors.

### Separate torsional cuts and harmonic spectators

Run `python examples/assemble_separable_rotor.py input.json --output spectra` from a workspace with `pyscf_esd` and Matplotlib installed. Supply closed signed angular samples for each state and a separate midpoint dipole cut. The input declares the rotor axis, moving atoms, masses, and two Cartesian analytic Hessians with their electronic-method provenance. The adapter projects the harmonic spectator blocks, transports their modes to a common angle, and calls the thermal separable solver on the absolute photon-energy axis. Set `hessian_is_same_surface` only when each Hessian belongs to its input electronic surface. The output identifies auxiliary curvatures as an approximation. Experimental cross-section points can be overlaid with `--experiment`.

## Reduced coupled vibrational models

- `mix_vibrational_states` diagonalizes a real symmetric Hamiltonian, in Hartree, in an orthonormal vibrational basis. Supply signed Cartesian transition amplitudes in the same order and phase convention. It returns mixed eigenvectors, term energies, amplitudes and strengths. For emission from a fixed initial level, subtract the mixed final-state terms from the initial term on the common energy reference.
- `pyscf_esd.anharmonic_plane.SineGalerkin2D` and `pyscf_esd.anharmonic_volume.SineDVR3D` are lower-level tools for sampled constant-mass surfaces. They treat a reflection-symmetric inversion coordinate with parity sectors. Use sampled dipoles in a common phase, and use the formaldehyde replay examples in the [validation guide](validation.md#formaldehyde-inversion) for the associated input contracts.
- `spectral_line_properties` converts either stick type into photon-weighted strengths and absolute oscillator strengths or vacuum Einstein A coefficients and powers. `summarize_spectral_properties` aggregates one or both directions and can retain coverage and assignment metadata. `electronic_relaxation_properties` and `harmonic_model_properties` report the corresponding input-model diagnostics.
