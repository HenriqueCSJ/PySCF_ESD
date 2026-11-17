# Unrestricted DFT and phosphorescence

Use a converged UKS reference for spin-free triplet derivatives and spin-conserving open-shell spectra. For phosphorescence, provide a common set of spin-free states, a spin-dependent Hamiltonian, and their electric-dipole matrices. These inputs support vertical electronic rates and adiabatic Hessian (AH) emission with vibrational structure.

## Analytic derivatives of a UKS triplet

Supply a converged unrestricted Kohn–Sham object from PySCF or GPU4PySCF with `mol.spin = 2`, the difference between the alpha and beta electron counts. Use a gas-phase, all-electron LDA or GGA functional, including hybrid GGAs. The adapter checks the occupied–virtual orbital residual, nuclear derivative arrays, and unrestricted determinant spin diagnostics. Inspect `spin_squared` and `spin_contamination`, the excess over the pure-triplet value $\langle S^2\rangle=2$.

```python
from pyscf import esd

endpoint = esd.analytic_uks_triplet_endpoint(uks, include_hessian=True)
energy_eh = endpoint.energy_hartree
gradient_eh_per_bohr = endpoint.gradient_hartree_per_bohr
hessian_eh_per_bohr2 = endpoint.hessian_hartree_per_bohr2
print(endpoint.spin_squared, endpoint.scf_residual_hartree)
```

The Hessian uses axes (atom A, atom B, Cartesian A, Cartesian B). The first Hessian route holds the XC quadrature fixed. The gradient supports moving XC quadrature and reports which convention was used. Match the gradient and Hessian conventions when validating curvature against an optimization. A converged determinant can still represent a different triplet or have an orbital instability. Qualify the triplet identity and stability along the geometry path and check its vibrational minimum before using the derivatives in AH.

Use `gradient_grid_response=False` to match the fixed-quadrature convention of the Hessian. For matched GPU response calculations use the same primary and CPHF grids on the prepared reference. [PySCF UKS Hessian](https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/uks.py) · [GPU4PySCF UKS Hessian](https://github.com/pyscf/gpu4pyscf/blob/master/gpu4pyscf/hessian/uks.py)

## Electronic phosphorescence at a supplied geometry

Supply all inputs at the same geometry, typically the relaxed emitting triplet geometry for a vertical rate:

- A vector of spin-free energies in Hartree on one common zero, including the ground singlet, all three components of the target triplet, and the states that supply spin–orbit mixing. The target triplet components share one spin-free energy.
- A complex Hermitian spin–orbit matrix in Hartree. Include any additional spin-dependent term such as spin–spin coupling in this matrix once. Keep the supplied spin-free energies free of those corrections.
- Three Hermitian electric-dipole matrices as one array with shape `(n_states, n_states, 3)` in e·bohr. Include the permanent and excited-state dipoles supplied by the electronic model. The ground-to-target-triplet block vanishes before spin mixing and the target's permanent dipole is the same for all three components. All matrices use the same orthonormal electronic basis, ordering, phases, and Cartesian frame.

```python
from pyscf import esd

rates = esd.spin_orbit_phosphorescence(
    energies_eh, soc_eh, dipole_au,
    ground_index=0, triplet_indices=(2, 3, 4), temperature_kelvin=77,
)
print(rates.sublevel_vacuum_rates_s_inverse)
print(rates.sublevel_radiative_lifetimes_s)
print(rates.vacuum_radiative_rate_s_inverse, rates.vacuum_radiative_lifetime_s)
```

Indices are zero-based. The labeled ground singlet must remain the isolated lowest state after mixing. The triplet label is assigned as an isolated three-dimensional subspace with majority overlap in every principal direction. The solver reports its weakest squared principal overlap and individual triplet characters. The splitting reported by `sublevel_splittings_hartree` describes the supplied Hamiltonian. Total zero-field splitting requires all relevant terms in that Hamiltonian.

With spin-free energies $E$ and a spin-dependent correction $V$, the solver diagonalizes $H=\operatorname{diag}(E)+V$ and transforms each dipole matrix as $U^\dagger\mu_\xi U$. Ground- and excited-state mixing contribute coherently to the same transition amplitude. The electric-dipole rate for each sublevel is $A_i=4\alpha^3\Delta E_i^3\sum_\xi|\tilde{\mu}_{0i,\xi}|^2/(3t_\mathrm{au})$, using Hartree and e·bohr for the energy and dipole. [TDDFT phosphorescence and vibronic theory](https://pubmed.ncbi.nlm.nih.gov/30721046/)

With a temperature, the report also gives the conditional Boltzmann average $k_p=\sum_i p_i A_i$ and radiative lifetime $1/k_p$. Omitting the temperature leaves the population model to the caller. Within an exactly degenerate subspace, a rotation can redistribute individual line strengths. Its summed intensity or equal-mixture rate is invariant. `sublevel_rates_resolved` flags sublevels with energy gaps larger than the input degeneracy tolerance. At 0 K the solver distributes the population across the lowest sublevels within that tolerance. Positive temperatures use the computed splittings. Experimental decay and yield also depend on sublevel relaxation, triplet formation, and nonradiative channels.

## Analytic spin–orbit dipole derivatives

For an isolated mixed state $k$ with eigenvector $u_k$, the analytic derivative in parallel-transport gauge is

$$
\partial_q u_k = \sum_{\ell\ne k} u_\ell
\frac{u_\ell^\dagger (\partial_q H) u_k}{E_k-E_\ell}.
$$

`spin_orbit_dipole_derivatives` applies the product rule to $\tilde\mu_{0k,\xi}=u_0^\dagger\mu_\xi u_k$. Both state responses contribute, along with the derivative of the spin-free dipole matrix. Supply the full $\partial_q H$ and $\partial_q\mu_\xi$ as arrays with shapes `(state, state, coordinate)` and `(state, state, coordinate, 3)`. These are derivatives of matrix entries in a smoothly transported spin-free basis. They include the energy, orbital, state, and operator response of the electronic input model. Use Hartree and e·bohr per chosen coordinate. The adapter evaluates this response without finite differences. If a selected state is degenerate within the input tolerance, an individual derivative requires a separate treatment of the coupled subspace.

## Prepared vibronic phosphorescence

Compute the electronic matrices at the emitting triplet minimum and express the moments in the same Cartesian frame as the AH model. Prepare `AdiabaticHessianParameters` for the relaxed $S_0$ and triplet surfaces with independent Hessians using `pyscf_esd.pyscf_adapter.adiabatic_hessian`. Supply the adiabatic 0–0 energy of the lowest spin sublevel in Hartree, including the vibrational zero-point difference and any adiabatic spin correction. The reported line energies use this origin plus the supplied electronic spin splittings.

```python
from pyscf import esd

slopes = esd.spin_orbit_dipole_derivatives(
    rates, dipole_au, hamiltonian_derivatives_eh, dipole_derivatives_au,
)
result = esd.run_prepared_phosphorescence(
    ah_parameters, rates, lowest_sublevel_00_hartree=adiabatic_origin_eh,
    dipole_derivatives_au=slopes, temperature_kelvin=77,
)
report = result.physical_properties
print(report["vacuum_radiative_rate_s_inverse"])
print(report["vacuum_radiative_lifetime_s"])
```

In this example the derivative coordinates are the mass-weighted triplet normal coordinates. Slopes have shape `(sublevel, T normal mode, 3)`. Use the phases and ordering in `rates.transition_dipole_au` and the mode signs and order in `ah_parameters.excited_cartesian_mode_derivatives_au`. Omit `dipole_derivatives_au` for the Condon model. If nuclear motion couples degenerate electronic components, a coupled spin-vibrational Hamiltonian is needed to describe that motion.

The AH solver uses one nuclear surface for all triplet components and treats the spin splittings as constant. Duschinsky rotation and displacement enter the harmonic overlaps for emission. For each vibronic transition the complex Franck–Condon and Herzberg–Teller amplitudes are added before taking the squared norm. If $C$ is the constant-dipole amplitude and $L$ the linear-dipole amplitude, the interference term is $2\operatorname{Re}(C^\dagger L)$. The rate weights every line by its positive photon energy cubed and the thermal population of its initial vibrational state. The thermal sublevel weights enter once when summing the component spectra.

`result.sublevel_spectra` contains the sticks, `result.sublevel_populations` gives the spin weights, and `result.basis_attempts` reports the adaptive harmonic coverage. The JSON-ready report gives per-sublevel 0–0 energies, vacuum rates, lifetimes, dominant lines, and Condon/Herzberg–Teller contributions when slopes are supplied. Multiply each sublevel's photon-rate sticks by its spin weight before summing a total spectrum. The aggregate radiative lifetime is $1/\sum_i p_i A_i$ under thermal equilibration of the emitting sublevels. [TDDFT phosphorescence and Herzberg–Teller theory](https://pubs.acs.org/doi/10.1021/acs.jctc.8b00841)

## Spin-conserving open-shell absorption and emission

`unrestricted_tda_vertical_spectrum` takes a prepared, converged molecular UKS/TDA object from PySCF or GPU4PySCF. For a high-spin reference, set `mol.spin` to $N_\alpha-N_\beta=2S$. Examples are `1` for a doublet, `2` for a triplet, `3` for a quartet, and `4` for a quintet. Spin-conserving unrestricted response keeps $M_S$ fixed. Verify the character and total spin of the relevant states before assigning a band to, for example, $T_1\to T_n$ or $D_0\to D_n$. The result exposes the reference determinant's `reference_ms` and `reference_spin_squared`. Native UKS/TDA leaves `root_spin_squared` unavailable. Open-shell singlets and other low-spin roots require a suitable spin-adapted or spin-flip method with explicit spin diagnostics. [PySCF unrestricted TDA](https://pyscf.org/_modules/pyscf/tdscf/uhf.html) · [Spin-adapted open-shell TDDFT](https://pubmed.ncbi.nlm.nih.gov/21476737/)

```python
from pyscf import esd

lines = esd.unrestricted_tda_vertical_spectrum(td, roots=(0, 2))
print(lines.reference_ms, lines.reference_spin_squared)
print(lines.photon_energy_hartree)
print(lines.oscillator_strength)
print(lines.einstein_a_s_inverse)
```

The selected indices are zero-based. Oscillator strengths describe excitation out of the UKS reference and the Einstein coefficients describe the reverse transition back to it at the same geometry. The signed length-gauge dipoles retain the native response phases. The adapter reads the solved roots and evaluates the dipoles without launching new SCF or TDA runs.

## Electronic preparation still needed

- For an automatic relaxed open-shell spectrum, optimize and verify both states, track the response root and phase, and provide independent curvatures and transition-moment response. The existing prepared AH interface can use qualified inputs. The analytic UKS endpoint gives the derivatives of a determinant surface. A response-excited root needs derivatives of its own energy and transition moment.
- For automatic multiplicity-changing emission, construct the spin-adapted electronic state space and its magnetic components, a documented relativistic SOC operator, and intra- and inter-manifold dipole matrices in a common gauge. Converge the contributing state space and, for HT, supply its analytic derivative response. The current prepared solver targets triplet-to-singlet emission. Experimental lifetimes and quantum yields also require a population model and the relevant nonradiative rates.
