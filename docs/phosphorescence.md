# Unrestricted DFT and phosphorescence

Use a converged UKS reference for spin-free triplet derivatives and spin-conserving open-shell spectra. For phosphorescence from a closed-shell reference, PySCF can build spin–orbit coupling (SOC) and electric-dipole matrices from restricted DFT/TDA. These inputs support vertical electronic rates and adiabatic Hessian (AH) emission with vibrational structure.

## Automatic SOC and interstate dipoles

Start with a converged nonrelativistic, all-electron molecular RKS calculation at the geometry of interest. For phosphorescence use the independently checked triplet minimum. The CPU driver solves the singlet and triplet TDA manifolds on the same reference, then builds the matrices from PySCF integrals. It supports gas-phase RKS, including density-fitted references, with real orbitals and integer occupations.

```python
from pyscf import esd

matrices = esd.run_pyscf_tda_soc(rks_at_triplet_minimum, n_singlets=12, n_triplets=12)
rates = matrices.phosphorescence(triplet_root=0, temperature_kelvin=77)
print(rates.transition_dipole_au)
print(rates.vacuum_radiative_lifetime_s)
```

Increase the number of singlets and triplets until the rates stabilize. `response_tolerance` defaults to a TDA residual norm of $10^{-7}$ Hartree. The Davidson linear-dependence cutoff scales with the requested residual to retain small correction directions. When both native PySCF TDA objects are already solved, use `esd.assemble_pyscf_tda_soc(singlet_tda, triplet_tda)` with optional `singlet_roots` and `triplet_roots`. The selection preserves zero-based native root identities and order. The state basis has the ground state, the selected singlets, and three Cartesian spin components for each selected triplet.

The operator is the molecular Breit–Pauli spin–orbit mean field (SOMF). For the spin-summed reference density $D$ and PySCF momentum integrals $G_{\xi,ijkl}$, the spatial coefficient multiplying the electron spin $S_\xi=\sigma_\xi/2$ is

$$
h_\xi = \frac{i}{2c^2}\left[V^{\mathrm{nuc}}_\xi+J_\xi-\frac{3}{2}(K_\xi-K_\xi^\mathsf{T})\right],
\qquad J_{\xi,ij}=\sum_{kl}G_{\xi,ijkl}D_{lk},\qquad
K_{\xi,il}=\sum_{jk}G_{\xi,ijkl}D_{jk}.
$$

The AO-shell contraction evaluates $J_\xi$ and $K_\xi$ without storing a four-index AO tensor. The normalized spatial TDA root is $\sqrt{2}X$, with spin excitations $I/\sqrt{2}$ and $\sigma_\xi/\sqrt{2}$. The projected effective one-body matrix includes the ground-to-excited block and both the virtual-particle and occupied-hole contractions between excited states. The electric dipole uses the physical length-gauge sign and includes the common nuclear contribution on the diagonal. Interstate and permanent dipoles are unrelaxed TDA singles matrix elements. Spin–spin zero-field splitting and nuclear derivatives of these electronic matrices require separate calculations. [Breit–Pauli SOMF in TDDFT](https://doi.org/10.1063/5.0130868) · [TDA state-interaction dipoles](https://doi.org/10.1039/D5CP01656H)

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

## Full-space first-order SOC and analytic Herzberg–Teller coupling

`esd.tda_soc_response` uses the same molecular Breit–Pauli SOMF operator to compute weak-SOC intensity borrowing across every occupied–virtual configuration in the finite RKS/TDA basis. It solves linear systems at the target triplet energy for singlet mixing and at the ground-state energy for triplet mixing. The two contributions add as complex amplitudes. It needs a converged target triplet and positive shifted TDA operators. This first-order model uses a Cartesian basis for the spin components of the spin-free triplet and neglects fine-structure energies.

For real orbitals, let $h_m$ be the imaginary part of the spatial SOC coefficient that multiplies $\sigma_m/2$, and let $\ell_O Q=QO_{vv}^\mathsf{T}-O_{oo}^\mathsf{T}Q$ denote the particle and hole action of an orbital operator on the unit spatial triplet amplitude $Q$. With $d_\xi=\sqrt{2}(\mu_\xi)_{ov}$, $g_m=(h_m)_{ov}/\sqrt{2}$, $b_m=\ell_{h_m}Q/2$, and $t_\xi=\ell_{\mu_\xi}Q$, the mixed moment is

$$
M_{m\xi}=-i\left[
b_m^\mathsf{T}(A_S-E_T I)^{-1}d_\xi+
g_m^\mathsf{T} A_T^{-1} t_\xi
\right].
$$

The common permanent dipole cancels between the ground-state contribution to the emitting state and the target-triplet contribution to the ground state. The first-order model recovers the sum over higher states without enumerating them.

The selected triplet must satisfy a native TDA eigenpair residual of at most $10^{-7}$ Hartree. The lowest eigenvalue of $A_S-E_T I$ must exceed $10^{-7}$ Hartree plus that residual. The Condon path also requires the lowest eigenvalue of $A_T$ to exceed $10^{-7}$ Hartree. These checks use dense diagonalization for small spaces and residual-verified, matrix-free Davidson with independent full-support starts for larger spaces. Optical selection rules can exclude singlets from the dipole response vectors even when their energy denominators are singular.

Supply nuclear directions to differentiate the moment. The analytic path differentiates the electric dipole and SOMF integrals, the spin-summed density, the occupied and virtual orbitals, and the singlet and triplet TDA operators. Adjoint systems account for the change of the triplet root and both resolvents. They are solved at one geometry. The derivative requires the lowest spatial triplet, separated by more than $10^{-5}$ Hartree from the remaining triplets. With $q=\mathrm{vec}(Q)$ and $P=I-qq^\mathsf{T}$, the lowest eigenvalue of $P(A_T-E_T I)P$ on the orthogonal complement gives that gap for an exact target eigenpair. For a numerical target, the check uses the Rayleigh quotient and the coupling $\lVert PA_Tq\rVert$ to allow for the target residual. Removing this coupling changes each ordered eigenvalue by at most its norm, so the projected check can establish triplet positivity and isolation with a single stored target root.

The analytic path uses CPU PySCF with spherical all-electron basis functions in the gas phase and global LDA or GGA, including global hybrid GGAs. Set `mf.small_rho_cutoff = 0` before SCF so the primary atom-centered grid is shared by the reference and response without density pruning. [Analytic SOC derivatives](https://doi.org/10.1021/acs.jctc.5c01442)

```python
from pyscf import esd

directions = ah_parameters.excited_cartesian_mode_derivatives_au @ ah_parameters.excited_alignment_rotation.T
response = esd.tda_soc_response(triplet_tda, directions=directions)
moments = response.transition_dipole_au       # spin component, dipole axis
slopes = response.dipole_derivatives_au      # mode, spin component, dipole axis
```

The Cartesian axes are those of the emitting geometry in the supplied TDA object. For equal spin populations, an all-quanta harmonic profile for one polarization uses `moments[:, axis] / sqrt(3)` and `slopes[:, :, axis] / sqrt(3)` as the three independent amplitudes. Use the inverse Duschinsky transform for emission and sum the three photon-weighted polarization profiles to obtain the isotropic rate.

```python
from math import sqrt

profiles = [
    esd.thermal_harmonic_spectrum(
        ah_parameters.excited_frequencies_hartree,
        ah_parameters.ground_frequencies_hartree,
        gap_00_hartree=-ah_parameters.gap_00_hartree,
        rotation=ah_parameters.rotation.T,
        shift_au=-(ah_parameters.rotation.T @ ah_parameters.shift_au),
        transition_dipole_au=moments[:, axis] / sqrt(3),
        dipole_derivatives_au=slopes[:, :, axis] / sqrt(3),
        temperature_kelvin=77.0,
        fwhm_cm_inverse=150.0,
        process="emission",
        include_dipole_strength_density=True,
    )
    for axis in range(3)
]
```

Use each profile's `positive_photon_weighted_strength` for the rate and `dipole_strength_density_per_cm_inverse` for comparison with a reduced experimental trace. [Rate and lifetime conversion](numerical-results.md#thermal-profiles-and-prepared-rotors)

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

For a Condon profile with many modes, use `thermal_harmonic_spectrum` on each sublevel to include all harmonic quanta without enumerating sticks. Give it `rotation=ah_parameters.rotation.T`, `shift_au=-(ah_parameters.rotation.T @ ah_parameters.shift_au)`, and `gap_00_hartree=-(adiabatic_origin_eh + splitting_eh)` for emission. A real vector with norm $|\tilde{\boldsymbol\mu}_{0i}|$ gives the same isotropic Condon strength as the original complex vector. Multiply each sublevel's rate or profile by its population before summing. See [numerical results](numerical-results.md#thermal-profiles-and-prepared-rotors) for the rate prefactor and reduced intensity convention.

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

## Further electronic response

- For an automatic relaxed open-shell spectrum, optimize and verify both states, track the response root and phase, and provide independent curvatures and transition-moment response. The existing prepared AH interface can use qualified inputs. The analytic UKS endpoint gives the derivatives of a determinant surface. A response-excited root needs derivatives of its own energy and transition moment.
- For Condon emission the restricted TDA driver supplies equilibrium SOC and dipole matrices. Automated spin-vibronic HT needs analytic derivatives of the full SOMF and dipole matrices, including orbital and TDA response. The present spin-adapted driver starts from a closed-shell reference. Experimental decay lifetimes and quantum yields also require a population model and the relevant nonradiative rates.

## Naphthalene benchmark

Gas-phase DF-B3LYP/def2-SVP with def2-universal-jkfit was used for the RKS ground-state and UKS triplet minima, with analytic fixed-quadrature gradients and Hessians. The harmonic model retains 48 real modes on each surface. The spin-free adiabatic gap including zero-point energy is 2.5545 eV (20,603 cm⁻¹), 0.0797 eV below the 77 K light-petroleum-glass origin. The SOC calculation adds relative fine-structure splittings to the spin-free origin. The spectrum uses the 77 K Condon model and Duschinsky rotation, with a Gaussian FWHM of 0.019 eV. For relative intensity, the harmonic dipole-strength profile is compared with the reduced 77 K hydrocarbon-glass trace. The integrated radiative rate uses cubed photon-energy weights and the vacuum Einstein-A prefactor. [Absolute origin measurement](https://doi.org/10.1039/JR9540003160) · [Reduced spectrum, Figure S2 and Table S2](https://pstorage-acs-6854636.s3.amazonaws.com/6569049/jp6b07709_si_001.pdf)

The same RKS reference at the UKS triplet minimum supplied 12 singlet and 12 triplet TDA roots with residuals below $10^{-7}$ Hartree. The RKS TDA $T_1$ and UKS determinant have squared overlap 0.945 in their common $M_S=1$ representation. The vertical rate changes by 0.014% from 8 to 12 roots of each spin. Four roots of each spin miss most of the intensity, including the opposite-reflection singles that start at the fifth singlet root. Zeroing SOC makes the transition exactly dark. The first-order and diagonalized total electronic dipole strengths agree to 0.031% in the 12+12 window.

The 77 K Condon rate is $2.603\times10^{-3}\ \mathrm{s}^{-1}$, equivalent to a radiative lifetime of 384 s. Published solution estimates are 75 s in ethanol at 77 K and 19.3 s inferred from the polar-medium data compiled by de Souza and coworkers. The Condon model captures several spacings but overemphasizes the band near 1,630 cm⁻¹ below origin. Only 0.016% of the computed electronic dipole strength is perpendicular to the molecular plane. A later polarization measurement in EPA glass found perpendicular emission. The comparisons span different media. This 12+12 Condon calculation omits spin-vibronic HT and higher-state borrowing. [Ethanol lifetime](https://doi.org/10.1063/1.1674576) · [Compiled radiative rate](https://doi.org/10.1021/acs.jctc.8b00841.s001) · [Polarization reappraisal](https://www.sciencedirect.com/science/article/abs/pii/S0301010405001928)

### Full-space response and analytic HT at 77 K

Reusing the B3LYP/def2-SVP endpoints, the first-order response over 4,964 spatial occupied–virtual configurations per spin recovers the perpendicular intensity. The Condon rate is $0.0567612\ \mathrm{s}^{-1}$, equivalent to 17.62 s, with 94.61% of the rate polarized perpendicular to the plane. Higher opposite-reflection configurations supply this intensity through both singlet and triplet mixing. Such channels have also been identified in independent analyses of aromatic phosphorescence. [Higher-state intensity borrowing](https://doi.org/10.1016/S0166-1280(02)00035-0) · [Polarization reappraisal](https://doi.org/10.1016/j.chemphys.2005.04.047)

Analytic HT along all 48 triplet modes increases the rate by 13.4% to $0.0643706\ \mathrm{s}^{-1}$, corresponding to 15.54 s, with 88.49% perpendicular emission. An independent harmonic third-moment calculation agrees with the integrated positive-photon rate within $8\times10^{-9}$ relative. The comparison uses the same 0.019 eV Gaussian width and native energies. The calculated double feature near 0.2 eV below the origin remains resolved where the hydrocarbon-glass trace has a broad band. The gas-phase 0–0 energy is 2.5545 eV, and the two radiative reference estimates above describe different condensed-phase conditions.

### Dielectric and linewidth controls

The resolved gas-phase maxima come primarily from $T_1(v=0)$ transitions to two totally symmetric $S_0$ ring fundamentals at 1,421.48 and 1,628.67 cm⁻¹ (0.176 and 0.202 eV). The split is present in the 0 K Condon sticks. The symmetry-matched Raman fundamentals occur at 1,381 and 1,579 cm⁻¹. [Mode assignments](benchmarks/naphthalene/gas-assignments.json) · [Raman measurements](https://doi.org/10.1016/0301-0104(77)85115-X)

The DF-B3LYP/def2-SVP control uses analytic gradients and Hessians at relaxed $S_0$ and $T_1$ minima in C-PCM. PySCF's methylcyclohexane settings ($\epsilon=2.024$, $\epsilon_\mathrm{opt}=2.0252$) represent the mean dielectric of the experimental 3:1 methylcyclohexane/isopentane glass. The optical solvent response enters the singlet TDA operator. The triplet response has no first-order spin-summed transition density. The SOC and dipole operators are evaluated on the solute. The change in 0–0 energy is +0.00231 eV. Both matched fundamentals move by less than 3 cm⁻¹, and the Condon maxima remain separated by 0.0248 eV at 0.0186 eV Gaussian FWHM.

On the same vacuum photon prefactor, gas Condon gives 17.62 s, continuum surfaces with the gas dipole give 17.54 s, and the continuum surfaces with the continuum Condon moment give 15.84 s ($0.0631166\ \mathrm{s}^{-1}$). The continuum control omits analytic HT in solvent and the distribution of local glass sites. The 15.54 s result includes gas-phase HT. Each reference lifetime comes from a different medium than the spectral comparison.

The experimental 0–0 has an apparent FWHM of about 239 cm⁻¹. Using that independently measured width as a resolution check merges the calculated gas HT peaks into a maximum at 1,595 cm⁻¹ below the origin. The experimental maximum is at 1,491 cm⁻¹ below the origin. The main comparison retains the original calculated broadening and plots loss from each curve's own 0–0 energy. [Comparison figure](benchmarks/naphthalene-phosphorescence.png) · [Experimental trace](https://pstorage-acs-6854636.s3.amazonaws.com/6569049/jp6b07709_si_001.pdf)

The benchmark data package includes the [matched rates, mode overlaps and linewidth analysis](benchmarks/naphthalene/glass-diagnostic.json) and the [gas HT](benchmarks/naphthalene/gas-ht-77k.csv.gz), [gas Condon](benchmarks/naphthalene/gas-condon-77k.csv.gz), and [continuum Condon](benchmarks/naphthalene/pcm-condon-77k.csv.gz) profiles. The profiles are tabulated on native photon energy in eV with photon-rate and reduced dipole-strength densities for each Cartesian polarization.
