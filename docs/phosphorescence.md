# Unrestricted DFT and phosphorescence

Two prepared-input APIs start the unrestricted DFT workflow. One evaluates the spin-free triplet surface from an $M_S=+1$ UKS calculation. The other evaluates electronic phosphorescence from a common set of spin-free states, a spin-dependent Hamiltonian, and their electric-dipole matrices.

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

## Machinery for the molecular workflow

1. Qualify and track the spin-free triplet with an unrestricted DFT reference. Relax both surfaces and verify gradient, Hessian and stability diagnostics.
2. Construct phase-consistent spin-adapted singlet and triplet state matrices from PySCF DFT response, with a documented relativistic SOC operator and basis. The $M_S=+1$ UKS determinant supplies one spin-free surface. It alone supplies neither the three interacting components nor their singlet–triplet couplings. Check convergence with the electronic state space.
3. Build the $T_1$ and $S_0$ vibrational models with independent curvatures and a Duschinsky map. Propagate the complex SOC-mixed dipole through Franck–Condon amplitudes and analytically differentiate it for spin-vibronic Herzberg–Teller terms. Those derivatives need the change of the spin-free dipoles, spin–orbit operator, and electronic state mixing in a consistent gauge. Retain analytic derivatives where a qualified operator response exists.
4. Compare sublevel splittings, electronic moments, the vibronic envelope and integrated radiative rates at matched conditions. Calculate intersystem crossing and other nonradiative rates through their own dynamical models when estimating experimental lifetimes or yields.

The current APIs deliver the triplet endpoint and fixed-geometry electronic rates from prepared inputs. Automatic SOC matrix construction, triplet optimization and vibronic phosphorescence are separate implementation milestones.
