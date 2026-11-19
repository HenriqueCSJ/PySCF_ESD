# Reuse prepared electronic data

Prepared APIs let you recompute nuclear spectra from completed electronic calculations. They also provide a common interface for retained geometries, analytic derivatives, signed overlaps and dipole functions. This guide defines the input contracts so that a reusable calculation archive carries the physical information needed by the nuclear model.

Start with [the molecular workflow](api.md#end-to-end-workflow) when electronic calculations should be launched automatically. Use [the API reference](api-reference.md) for signatures and defaults.

## Choose the representation

- Two stationary minima with Hessians, total energies and a dipole expansion define an AH model. Use `adiabatic_hessian` and `run_prepared_ah`.
- Positive harmonic frequencies, an affine coordinate map, a 0–0 gap and a dipole expansion define a direct harmonic model. Use `harmonic_spectrum` or `thermal_harmonic_spectrum`.
- A common coordinate grid with two potentials, kinetic coefficients and a signed dipole function defines a one-coordinate DVR model. Add `HarmonicSpectators` for the separable modes.
- A real vibrational Hamiltonian and signed basis transition dipoles define a coupled vibrational block. Use `mix_vibrational_states`.

The input determines the model being solved. Increasing a spectral grid or nuclear basis refines that model. Changing the electronic surface, coordinate metric, spectator coupling or dipole model changes the physical approximation and belongs in the calculation record.

## Preserve the energy reference

Both electronic surfaces must use a common total-energy zero. Store total electronic energies before removing offsets for plots or fitting a potential representation. A constant removed from both surfaces is harmless. Removing separate constants from each surface destroys the electronic gap unless that gap is restored explicitly.

For a TDA excited state at geometry `R1`, use `E_reference(R1) + omega_target(R1)` as its total energy. The reference energy and response root must belong to that same geometry and recipe. The AH adapter adds the difference of harmonic zero-point energies once when constructing `gap_00_hartree`.

In a direct harmonic solver, `gap_00_hartree` already includes zero-point energy and uses the signed final-minus-initial convention. It is positive for the usual S₀→S₁ absorption input. Reverse it for S₁→S₀ emission. The final returned photon energies remain positive.

For DVR surfaces, the eigenvalues already contain active-coordinate zero-point and tunneling contributions. The spectral solvers obtain their active energy difference directly. A harmonic spectator bath contributes its own zero-point difference. Include each coordinate in one of these contributions.

## Geometry, masses and frames

Keep one element order, charge and isotope mapping across the two surfaces. Python Cartesian geometries use bohr. Endpoint Hessians and gradients use each endpoint's own native Cartesian frame. The AH adapter aligns both endpoints and rotates the derivatives internally.

`adiabatic_hessian` accepts `masses_amu` in atomic mass units. Rotor coordinate tools accept `masses_me` in electron masses. Convert explicitly when crossing those interfaces. The mass-weighted normal coordinate convention is √(electron mass)·bohr.

After AH alignment, the transition dipole and Cartesian dipole derivative supplied to the nuclear model must use the ground molecule's frame. Derivative rows projected onto normal modes must follow the signs and order of the returned ground normal modes. Degenerate subspaces can rotate within the eigensolver, so move the amplitudes with the mode basis whenever that basis changes.

## Reuse two completed minima

The following function assembles a checked AH calculation from caller-supplied endpoint objects. It performs nuclear assembly after receiving the completed electronic data.

```python
import numpy as np
from pyscf_esd import run_prepared_ah
from pyscf_esd.pyscf_adapter import adiabatic_hessian

def spectra_from_endpoints(
    mol0, mol1, hessian0, hessian1, energy0, energy1,
    gradient0, gradient1, dipole0, cartesian_dipole_derivative0,
):
    parameters = adiabatic_hessian(
        mol0, mol1, hessian0, hessian1,
        ground_energy_hartree=energy0,
        excited_energy_hartree=energy1,
        ground_gradient=gradient0,
        excited_gradient=gradient1,
    )
    derivative_q = np.einsum(
        "qax,axd->qd",
        parameters.ground_cartesian_mode_derivatives_au,
        cartesian_dipole_derivative0,
    )
    return run_prepared_ah(
        parameters, dipole0,
        electronic_method="PBE0/def2-TZVPD LR-TDA, analytic derivatives",
        dipole_derivatives_au=derivative_q,
        process="both", temperature_kelvin=298.15,
        coverage=0.995, max_quanta=32, max_overlaps=250_000,
    )
```

`dipole0` has shape `(3,)` in e·bohr. `cartesian_dipole_derivative0[a, x, d]` has shape `(n_atoms, 3, 3)` in e and is evaluated at the ground minimum in the same electronic phase. Each Hessian has PySCF shape `(n_atoms, n_atoms, 3, 3)` in Hartree/bohr². Each gradient has shape `(n_atoms, 3)` in Hartree/bohr. Replace the method description with the actual source recipe.

Pass `dipole_derivatives_au=None` for a Condon model. A calculated dipole at the other endpoint can be retained for an independent comparison with the linear expansion evaluated there. `run_prepared_ah` records that the electronic validation is supplied by the caller. The caller establishes electronic convergence, state identity and common phase. The function checks the adaptive nuclear coverage and returns `WorkflowIncomplete` if it misses the requested target.

## A complete synthetic harmonic calculation

This example requires the base package and NumPy. Its frequencies and displacement define a synthetic two-mode model. It illustrates the signed absorption/emission convention, linear HT and extraction of absolute radiative quantities.

```python
import numpy as np
from pyscf_esd import harmonic_spectrum, spectral_line_properties

frequency0 = np.array([0.004, 0.006])
frequency1 = np.array([0.0042, 0.0058])
rotation = np.eye(2)
shift = np.array([6.0, -3.0])
mu0 = np.array([0.2, 0.0, 0.0])
derivative0 = np.array([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0]])
gap = 0.12

absorption = harmonic_spectrum(
    frequency0, frequency1, gap_00_hartree=gap,
    transition_dipole_au=mu0, dipole_derivatives_au=derivative0,
    rotation=rotation, shift_au=shift,
    initial_max_quanta=4, final_max_quanta=12,
    temperature_kelvin=298.15, process="absorption",
)

inverse_shift = -rotation.T @ shift
mu1 = mu0 + inverse_shift @ derivative0
derivative1 = rotation @ derivative0
emission = harmonic_spectrum(
    frequency1, frequency0, gap_00_hartree=-gap,
    transition_dipole_au=mu1, dipole_derivatives_au=derivative1,
    rotation=rotation.T, shift_au=inverse_shift,
    initial_max_quanta=4, final_max_quanta=12,
    temperature_kelvin=298.15, process="emission",
)
lines = spectral_line_properties(emission, "emission")
rate = float(lines.population_weighted_einstein_a_s_inverse.sum())
print("Retained radiative rate in s^-1", rate)
print("Retained radiative timescale in s", 1.0 / rate)
print("Initial population coverage", emission.initial_population_coverage)
print("Final dipole coverage", emission.final_dipole_strength_coverage)
```

The explicit quantum cutoffs are choices for this demonstration. For a production prepared AH model, `run_prepared_ah` adapts those cutoffs and also qualifies the positive-photon moments needed by rates. `spectral_line_properties` converts the returned lines using their actual retained thermal populations.

## Track states before building a surface

`pyscf_esd.prepared_state_tracking.track_prepared_states` accepts a sequence of candidate-root energy vectors and a sequence of signed overlap matrices. At point `i`, its energy vector has shape `(n_roots_i,)`. Each link matrix has shape `(n_roots_i, n_roots_next)` with source states along rows and destination states along columns.

Supply `initial_indices=(ground_index, excited_index)` as two distinct zero-based candidate indices at the first point. These labels specify the physical states to track even if the energetic root ranking changes later. Include the relevant competitors so that overlap margins and energy gaps can be assessed.

For an open path with `N` points, provide `N-1` link matrices. For a closed path with `periodic=True`, omit the repeated endpoint and provide `N` matrices, including the link from the last point to the original first-point checkpoint in its original gauge.

The default thresholds are `minimum_overlap=0.65`, `minimum_margin=0.10` and `minimum_gap_hartree=1e-4`. Every chosen state must pass forward and reverse assignment, remain distinct from its tracked partner and return to its initial identity on a closed path. The signed phase may reverse around the loop and is recorded independently for each state.

```python
import numpy as np
from pyscf_esd.prepared_state_tracking import track_prepared_states

# Synthetic root reordering at the middle point.
energies = [
    np.array([0.0, 0.1, 0.2]),
    np.array([0.001, 0.201, 0.101]),
    np.array([0.002, 0.102, 0.202]),
]
permutation = np.eye(3)[:, [0, 2, 1]]
path = track_prepared_states(
    energies, [permutation, permutation.T],
    initial_indices=(0, 1), periodic=False,
)
print(path.root_indices)
print(path.energies_hartree)
```

`root_indices` and `phases` both have shape `(N, 2)` in the initial label order. `signed_link_overlaps` contains the selected raw links. `path.align_transition_dipoles(raw_dipoles)` applies both transported electronic phases to an `(N, 3)` dipole array for an open path.

The overlap evaluator must provide normalized physical-state overlaps with their signs. For a new prepared evaluator, `qualify_prepared_overlap` checks self-overlaps, reverse orientation, separate state gauge flips and an independent forward result on an informative pair of different geometries. Preserve raw checkpoints and evaluator identity for that calibration. [Calibration contract](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/prepared_overlap_qualification.py)

## Close the dipole phase around a rotor

For two already identified real electronic states, `phase_periodic_scan` takes two common-reference potential vectors and raw signed Cartesian dipoles on a uniform angular grid. Its `ground_overlaps` and `excited_overlaps` each have length `N`, including the closing link. These overlaps belong to the raw electronic gauges used for the dipole samples.

The result provides `ground_boundary` and `excited_boundary`. A closure sign of −1 gives an antiperiodic nuclear boundary. The dipole closure sign is the product of the two electronic signs, making the complete vibronic integrand single-valued. Pass the returned `PhasedRotorDipole` to the nuclear solver without applying the phases a second time.

The following complete synthetic model includes a deliberately flipped excited electronic gauge at several points. Its phase links and raw dipoles refer to that same gauge. All active finite-grid levels are retained.

```python
import numpy as np
from pyscf_esd import (
    phase_periodic_scan, solve_dvr, thermal_separable_spectrum,
)

count = 48
angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
potential0 = 0.001 * (1.0 - np.cos(3.0 * angle))
potential1 = 0.12 + 0.0015 * (1.0 - np.cos(3.0 * (angle - 0.2)))
smooth_dipole = np.column_stack((
    0.2 + 0.03 * np.cos(angle),
    0.02 * np.sin(angle),
    np.zeros(count),
))
raw_phase1 = np.ones(count)
raw_phase1[12:24] = -1.0
raw_dipole = raw_phase1[:, None] * smooth_dipole
links1 = raw_phase1 * np.roll(raw_phase1, -1)

scan = phase_periodic_scan(
    angle, potential0, potential1, raw_dipole,
    ground_overlaps=np.ones(count), excited_overlaps=links1,
)
np.testing.assert_allclose(scan.transition_dipole.values_au, smooth_dipole)
ground = solve_dvr(
    scan.grid, scan.ground_potential_hartree,
    boundary=scan.ground_boundary, n_levels=count,
    kinetic_coefficient=1.0 / 20_000.0,
)
excited = solve_dvr(
    scan.grid, scan.excited_potential_hartree,
    boundary=scan.excited_boundary, n_levels=count,
    kinetic_coefficient=1.0 / 22_000.0,
)
profile = thermal_separable_spectrum(
    excited, ground, transition_dipole_au=scan.transition_dipole,
    process="emission", temperature_kelvin=298.15,
    fwhm_cm_inverse=80.0,
)
print(profile.origin_00_cm_inverse)
print(profile.initial_population_coverage)
```

The unit signed overlaps above describe a synthetic gauge test. A molecular calculation supplies measured signed overlaps from its electronic state representation. Increasing `n_points` in `phase_periodic_scan` interpolates the supplied information. Establish electronic sampling convergence by adding or refining electronic points when they are needed.

## Add separable spectator modes

Construct `HarmonicSpectators` with the frequencies and affine spectator map for the initial and final surfaces. Supply the object as `spectators=...` in either anharmonic solver. The active rotor is removed from the spectator Hessian subspace so that its kinetic and potential contributions occur once.

For emission, swap the spectator frequency arrays and invert their map. If absorption uses `(J, K)`, emission uses `(J.T, -J.T @ K)`. The active solver likewise receives the emitting DVR first. Keep the spectator bases and Cartesian dipoles in one transported angular frame.

`project_torsion` projects an analytic angular tangent against overall translation and rotation. `project_spectator_modes` removes the rigid and active directions from a stationary Hessian. `transport_bond_spectators` and `map_transported_spectators` form a compatible spectator map across the chosen angular path. Inspect the resulting inertias, subspace singular values and displacement residuals before using the separable approximation.

The solver uses a constant kinetic coefficient on each active surface. Where projected inertia changes over a molecular path, the chosen effective inertia is part of the model and should be reported together with the sampled variation. The example assembler records this information.

## Import a complete rotor archive

The [separable rotor assembler](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/examples/assemble_separable_rotor.py) reads a JSON archive containing separate `ground`, `excited` and `midpoint` cuts. It also needs `masses_amu`, `axis_atoms` and `moving_atoms`, with zero-based atom indices, plus descriptive `method` and `geometry_path` strings for the output record. Its input uses degrees for `angle_deg` and bohr for Cartesian geometries and analytic angular tangents.

Each cut contains a uniformly spaced full loop **including the repeated endpoint**. Every point records `tracking_passed=true`, `coordinates_bohr`, `tangent_bohr_per_rad` and its energy or dipole. Potential points use `energy_eh` and `overlap`. Midpoint points use `dipole_au`, `overlap_s0` and `overlap_s1`. Each cut also declares its duplicate-to-original-first closing overlap through `closure_overlap` or `closure_overlap_s0` and `closure_overlap_s1`.

The surface cuts include a `stationary` record containing `angle_deg`, `coordinates_bohr`, `hessian_eh_bohr2`, `hessian_method` and `hessian_is_same_surface`. Describe the actual source of each analytic curvature. The assembler reports auxiliary curvature as an approximation when it belongs to a different electronic surface.

The duplicate endpoint is a property of this archive reader. Its internal solver drops the duplicate before using the periodic DVR grid. The low-level APIs described above receive the grid without that endpoint. [Assembler input description and options](using-methods.md#prepared-anharmonic-models)

Run the assembler from the repository with the workflow dependencies and Matplotlib installed.

```sh
python examples/assemble_separable_rotor.py rotor-input.json --output rotor-spectra
```

Inspect `python examples/assemble_separable_rotor.py --help` for spectral mesh, temperature, width, data overlay and output arguments. The JSON and CSV outputs retain the model inputs and absolute spectral quantities. [Thermal output units](numerical-results.md#thermal-profiles-and-prepared-rotors)

## Archive what is needed to replay a calculation

Keep endpoint geometries and atom labels, isotope masses, charge and spin, electronic method and basis, grid and solver tolerances, energies and their common zero, gradients, Hessians, signed dipoles, dipole derivatives and state-tracking evidence. For a rotor retain each raw sample, the angular origin, analytic tangent, individual signed state links and the closure link. For a coupled block retain the basis order, Hamiltonian and signed amplitudes together.

Record the nuclear grid or basis sizes, requested temperature, broadening definition and numerical coverage. Retain the unshifted physical arrays alongside any comparison plot. Replaying a molecular benchmark may require its separate electronic archive even when the selected published plots and generic drivers are distributed in the repository. [Validation and reproducibility](validation.md)
