# Methods and implementation

[Project home](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/README.md) · [Workflow guide](guide.md) · [Python API](api.md) · [Validation](validation.md)

The high-level workflow combines a tracked electronic surface, a nuclear Hamiltonian, transition amplitudes and a spectral observable. These layers have separate acceptance checks. Select the electronic backend and derivative route explicitly when they affect the approximation. The [calculation guide](using-methods.md) gives runnable choices and the [numerical-results guide](numerical-results.md) defines the exported observables.

## Electronic reference and state tracking

The CPU electronic backend builds a closed-shell, all-electron molecular reference with direct screened integrals, unpruned atom-centered grids for DFT, and isolated singlet TDA roots. Its accepted exchange–correlation types are HF, LDA and GGA, including hybrid GGAs. The CPU automatic workflow describes gas-phase electric-dipole absorption and fluorescence. Its input checks exclude ECPs, empirical dispersion, nonlocal correlation, symmetry constraints and named solvents. [PySCF TDDFT and TDA](https://pyscf.org/user/tddft.html)

The high-level GPU workflow uses density-fitted global hybrid LDA/GGA references and accepts gas phase or a named equilibrium C-PCM solvent. The AH and automatic GPU routes use analytic excited-state Hessians and transition-dipole derivatives within that supported domain. Open-shell absorption and phosphorescence have [dedicated Python interfaces](phosphorescence.md). Prepared nuclear models can use electronic inputs supplied by the caller, with their method, units and derivative provenance stated explicitly.

For an isolated TDA root $n$, the response problem is $A X_n=\omega_n X_n$, where $A$ is the TDA response matrix, $X_n$ is the excitation vector and $\omega_n$ is the excitation energy. The tracked excited surface has total energy $E_n(R)=E_0(R)+\omega_n(R)$ at geometry $R$. Optical properties and geometry derivatives belong to that tracked root. A converged ground reference alone does not establish the convergence or isolation of its excited response.

SCF attempts can use projected occupied orbitals from a neighboring point, cold DIIS and a second-order solver. Acceptance requires the independent SCF residual alongside native convergence. TDA eigenpairs are checked against the same response operator. Small problems can use a dense solution of that operator if the iterative solver fails. The CPU TD-RKS analytic gradient checks the absolute orbital-gap-preconditioned Z-vector residual against the stated response tolerance (default 10⁻¹⁰). A residual correction uses the original operator and tolerance if the Krylov restart has discarded a small residual direction. [PySCF response correction](https://github.com/pyscf/pyscf/pull/3384)

At each stationary geometry used to locate a minimum, the internal restricted and restricted-to-unrestricted checks run before the nuclear curvature calculation. An internal saddle can trigger a fresh calculation with all-point checks and a restricted orbital rotation followed by a new SCF solve. The retry must lower the energy and pass the original convergence checks. An instability toward unrestricted orbitals blocks spectroscopy with that reference. Ordered alternate functionals are optional and each starts the complete workflow afresh, including later electronic sampling. The selected root must be assigned separately for each functional. [PySCF orbital stability](https://pyscf.org/user/scf.html#stability-analysis)

The CPU DFT workflow separately records an unresolved orbital-stability calculation under `reference_attempts` as `stability_unresolved`, with the channel and geometry stage. A change of functional requires a verified restricted-to-unrestricted instability and an explicitly supplied alternate functional. SCF convergence qualifies the stationarity of the electronic determinant. The orbital-stability calculation also tests its curvature along occupied–virtual rotations.

Across two geometries the matcher contracts particle–hole amplitudes through cross-geometry AO overlaps projected into the occupied and virtual spaces. The gap, overlap and runner-up checks reject an under-resolved, near-degenerate or ambiguous match. The sign of the accepted overlap fixes the real-state phase on each link and is applied to the transition dipole. The automatic isolated-state workflow requires the target's energy rank to remain the same during optimization, local derivative fits and direct spectral sampling. A periodic real-state scan also checks the closing link.

For a separate restricted CASSCF calculation, `casscf_tracking.casscf_state_overlap` computes a signed many-electron overlap for the same active-space specification at two geometries. The determinant of the common doubly occupied core enters twice. The active-space Schur complement includes core–active mixing. An ill-conditioned core raises a tracking error so the caller can retry on a shorter link. `state_tracking.match_state_overlaps` accepts these signed scores and applies the same assignment checks with thresholds appropriate to the metric. Track both states to phase a transition dipole. A periodic nuclear model also needs individual electronic phase closure for each surface. This is a tracking utility for prepared CASSCF states. The XYZ workflow uses its current DFT or RHF–CIS routes.

## Geometry and derivative routes

The CPU AH route optimizes S₀ and the tracked excited surface using analytic gradients. The excited-state step uses the ground Hessian as a starting preconditioner. Optimization removes rigid directions and checks gradient norms, line-search behavior, state identity, and stationary curvature. Steepest-descent recovery and local retries preserve the electronic acceptance criteria. The Hessian must be symmetric to the stated tolerance and positive in the vibrational subspace.

- **Analytic VG.** A ground-state Hessian and one analytic excited-state gradient define a second harmonic well with the same curvature. The direct CPU and GPU VG calculations require zero displaced geometries for their standard Condon sticks. The optional displaced-point CPU reference API has explicit step and checkpoint settings.
- **Analytic RHF–TDA/CIS AH and HT.** The dense CPU engine differentiates the AO integrals and includes first and second RHF orbital response and CIS root response. The length-gauge transition-dipole derivative includes AO position-integral, orbital and CIS eigenvector derivatives. Its domain is conventional real, all-electron RHF, spherical AOs and the complete singlet CIS space. Dense integral and response costs confine it to small examples.
- **RKS/TDA AH and HT.** For the default hybrid DFT route, local S₁ curvature is estimated from analytic gradients on checked displacements, and first-order HT from fitted signed transition dipoles. Symmetric Hessian constraints, held-out points and radius checks diagnose the fits. These entries are numerical derivative models. The analytics and fitted quantities are identified separately in the electronic diagnostics.
- **GPU VG.** The explicit single-geometry PBE0 route computes the S₀ Hessian and singlet TDA S₁ gradient with GPU4PySCF. The same native GPU operator is used in any CPHF residual correction, with host Krylov algebra when needed. Fixed-grid S₀ Hessians are used by default for CPU/GPU parity. An optional GPU grid-response Hessian describes a distinct derivative convention. [GPU4PySCF](https://doi.org/10.1021/acs.jpca.4c05876)
- **Prepared analytic HT in LR C-PCM.** Three projected TDA adjoints give the response of the Cartesian transition dipole to the TDA eigenvector. The derivative combines same-geometry polarized TDA gradient contractions, the moving XC quadrature correction, analytic AO position-integral derivatives and occupied orbital response. The API supports isolated real singlets and all-electron global hybrid LDA/GGA references in equilibrium C-PCM. It returns the gradient of the dipole in the tracked phase.
- **Analytic Hessian in equilibrium LR C-PCM.** The prepared GPU4PySCF route differentiates the TDA action in an orthonormal moving MO frame. It uses occupied RKS response and an adjoint to eliminate second-order occupied–virtual rotations. A projected TDA resolvent gives the excitation eigenvector contribution. The explicit mixed terms include orbital and auxiliary basis motion in the DF Coulomb and exchange operators, the solvated SCF Fock potential, the SWIG response to a transition density, and moving XC nodes and weights through the fourth derivative of the semilocal XC energy density. The molecular route adds the native ground Hessian at the same geometry with XC quadrature and auxiliary basis response enabled. [Analytic TDA Hessians](https://doi.org/10.1063/1.3605504) · [C-PCM extension](https://doi.org/10.1063/1.4773397)

The automatic AH route also checks linear HT at the two signed zero-point displacements of the mode with the largest predicted dipole change when its vibrational quantum is at least the nearest computed electronic root gap. The direct dipoles use the tracked phase and serve as validation points. If the linear extrapolation fails, the workflow reports the mode and residual and blocks the isolated-state spectrum and rate.

The fixed-grid convention in the single-geometry VG benchmark and the moving-quadrature convention in the high-level analytic GPU AH route describe different derivative settings. The method report identifies each route. A local regression of analytic gradients uses several geometries to estimate curvature and is a numerical derivative model. Analytic differentiation and direct electronic sampling for an anharmonic potential have distinct roles. The direct potential samples define a nuclear model and require their own state and interpolation checks.

## Harmonic vibronic model

AH aligns the two minima in a common mass-weighted Cartesian frame, removes rigid motion and compares the two vibrational subspaces. The polar factor of their overlap gives an orthogonal Duschinsky transformation. Overlap singular values, the rotation correction and the displacement projection residual measure the quality of that approximation. The low-level adapter defaults to a subspace singular-value loss tolerance of 10⁻³. The higher-level workflow also reports its separate explicit fit criteria. For linear molecules, relative axial orientation remains undetermined by the rigid fit, so coordinate-dependent amplitudes may need an externally specified frame.

The affine map is `Q_f = J Q_i + K`. The signed 0–0 gap is the final-minus-initial electronic minima difference plus the final-minus-initial zero-point difference. VG uses shared frequencies, so that zero-point difference is zero. For harmonic HT, each Cartesian dipole component is expanded to first order in the initial coordinates. The solver computes harmonic Franck–Condon overlaps with a Gaussian generating-function recurrence and obtains the linear-coordinate matrix elements with ladder operators. The constant and linear dipole terms combine as amplitudes, including signed interference. Reversing an AH transition swaps surfaces, applies `(Jᵀ, −JᵀK)` and re-expands the same dipole in the emitting coordinates.

In atomic units the local harmonic Hamiltonian on surface $s$ is

$$
\hat H_s=E_s(R_s)+\frac12\sum_k\left(-\frac{\partial^2}{\partial Q_{s,k}^2}
+\omega_{s,k}^2Q_{s,k}^2\right),
$$

where $Q_{s,k}$ is a mass-weighted normal coordinate in $\sqrt{m_e}\,a_0$, and $\omega_{s,k}$ is its energy quantum in Hartree. The harmonic absorption origin is

$$
E_{00}=E_1(R_1)-E_0(R_0)+\frac12\sum_k(\omega_{1,k}-\omega_{0,k}).
$$

The Duschinsky rotation $J$ maps ground-mode coordinates into excited-mode coordinates, and $K$ is the corresponding displacement. Reversal uses $J^{-1}=J^{\mathsf T}$ and $-J^{\mathsf T}K$. The linear transition-dipole expansion in the initial coordinate basis is

$$
\boldsymbol\mu(Q_i)=\boldsymbol\mu_0+\sum_k\boldsymbol d_kQ_{i,k},\qquad
\boldsymbol M_{if}=\boldsymbol\mu_0\langle\chi_f|\chi_i\rangle+
\sum_k\boldsymbol d_k\langle\chi_f|Q_{i,k}|\chi_i\rangle.
$$

Here $\boldsymbol d_k$ is a signed Cartesian dipole derivative and $\chi_i$, $\chi_f$ are initial and final nuclear eigenfunctions. Squaring the full amplitude preserves FC–HT interference. A state phase or mode-sign change must be propagated into every quantity sharing that gauge.

At temperature $T>0$, initial eigenstates have populations

$$
p_i=\frac{\exp[-(\epsilon_i-\epsilon_0)/(k_{\mathrm B}T)]}{Z},\qquad
Z=\sum_i\exp[-(\epsilon_i-\epsilon_0)/(k_{\mathrm B}T)].
$$

$k_{\mathrm B}$ is the Boltzmann constant and $\epsilon_i$ are initial nuclear levels. The harmonic partition function is evaluated analytically. Retained hot-state populations keep the denominator of the complete model, allowing the reported coverage to reveal truncation. The thermal absorption and relaxed emission ensembles belong to their respective initial surfaces.

The harmonic basis is cut on total excitation quanta. The exact partition function gives initial population coverage. Dipole closure provides an independent final-state strength denominator when it is nonzero. Both coverage checks apply before the positive photon-energy filter. The adaptive harmonic solver separately bounds omitted initial and final contributions to the positive-photon moments `M₁` for absorption and `M₃` and `M₄` for emission. For emission from one initial level, the maximum photon energy is its excitation energy above the final vibrational ground level. Together with dipole closure this bounds the omitted strength weighted by the photon energy. For absorption, harmonic expectations give the signed first moment analytically. Bounding the magnitude of its negative-energy contribution then bounds the positive first moment. Thermal harmonic averages bound the hot-state contribution. These checks qualify integrated observables within the harmonic model. Individual bands and broadened profiles can have their own convergence requirements.

The thermal harmonic profile solver sums all quanta with an affine dipole multiplying the Bargmann overlap. The Gaussian contraction preserves coherent FC–HT interference and checks the thermal dipole closure including the linear-dipole variance. It supports unequal frequencies and Duschinsky rotation and uses a cached covariance for matching curvatures. For equal, undisplaced Hamiltonians it applies the positive photon-energy cutoff directly to the Condon origin and linear HT sidebands and centers the Gaussians at their transition energies. Other profiles use the auxiliary Gaussian cutoff described below.

## Anharmonic models

**Automatic local correction.** Direct electronic samples along normal coordinates screen the harmonic model on both surfaces. Selected mode pairs and triples are probed under finite budgets. A local fitted potential and a product-basis vibrational configuration interaction (VCI) solve correct supported multimode cases. The constraints include direct holdouts and sensitivity to the basis and local domain. Multimode dipoles use the linear HT expansion. The one-mode branch uses the direct potential and signed dipole with an open-boundary DVR. At positive temperature the DVR tests its finite-grid partition and confinement, and VCI checks retained Boltzmann populations and bounds the omitted weight in the tapered model. [Thermal qualification](using-methods.md#finite-temperature-anharmonicity)

**Supplied periodic coordinate and harmonic bath.** The periodic coordinate uses a constant-inertia spectral DVR at a common energy zero. Its transition amplitudes integrate the signed dipole along the coordinate. For a closed loop of isolated real states, each electronic phase closure determines its own nuclear boundary. A sign reversal uses a half-integer Fourier kinetic basis. The transition dipole closure is the product of the two electronic signs. The local rotor tools can remove rigid motion and transport harmonic spectator modes between angles. The reduced model allows different constant inertias between electronic surfaces and Duschinsky mixing within the harmonic spectator subspace. It neglects variable inertia, nonseparable spectator–rotor interactions, and nuclear-spin statistical weights. [Hybrid DVR–harmonic model](https://doi.org/10.1021/acs.jctc.7b00236)

For angular coordinate $\theta$ and constant angular inertia $I_s$ in atomic units, the active Hamiltonian and transition amplitude are

$$
\hat H_s=-\frac{1}{2I_s}\frac{d^2}{d\theta^2}+V_s(\theta),\qquad
\boldsymbol M_{if}=\int\psi_f^*(\theta)\boldsymbol\mu(\theta)\psi_i(\theta)\,d\theta.
$$

Both $V_s$ potentials share one electronic energy zero, and $\boldsymbol\mu(\theta)$ is a signed vector in one Cartesian frame. The electronic closure sign $\eta_s=\pm1$ determines $\psi_s(\theta+2\pi)=\eta_s\psi_s(\theta)$ in the transported real gauge. The dipole has closure sign $\eta_0\eta_1$. Removing separate potential minima without retaining their offset would destroy the optical energy reference. The [prepared-data contracts](api.md#periodic-rotor-and-thermal-profile) describe endpoint and overlap conventions.

For a thermal broadened profile the separable solver combines a finite active-coordinate dipole measure with the all-state harmonic spectator characteristic function. The active populations are normalized against the full finite-grid DVR spectrum. Retained active levels determine population and dipole-closure coverage. The inverse Fourier calculation uses an energy mesh, a signed-measure tail bound and an auxiliary smoothing width for the cutoff at positive photon energy. The auxiliary width is removed in quadrature from the physical Gaussian. The solver raises an error when estimated leakage across zero or the magnitude of removed negative photon-weighted bins exceeds 10⁻⁸ of the corresponding positive measure. The tail bound concerns the underlying dipole measure. Check the final photon-weighted profile against a finer mesh, especially near zero energy. Checks on an interpolated nuclear grid are also separate from checks on the sampled electronic potential and dipole.

**Coupled reduced spaces.** Sine Galerkin (2D) and sine DVR (3D) solvers treat prescribed constant-mass surfaces, including parity blocks for a reflection-symmetric inversion. The 3D operator applies the kinetic term by discrete sine transforms and can use a Lanczos spectral measure for a broadened response. The lower eigenpairs identify physical low levels. Lanczos quadrature nodes represent a spectral measure. Explicit Fermi mixing in an orthonormal vibrational subspace can use a real symmetric Hamiltonian and signed dipoles. These models require the coordinates and electronic information appropriate to the molecule.

## Physical observables

Spin–orbit phosphorescence uses either prepared state-interaction matrices or first-order full-space TDA resolvents. The analytic spin-vibronic route differentiates the mixed moment at the emitting geometry and carries the complex derivatives into the thermal harmonic solver. [Spin–orbit method and limitations](phosphorescence.md#full-space-first-order-soc-and-analytic-herzbergteller-coupling)

Let `E > 0` be the photon energy in Hartree, `p_i` the normalized initial-level population, and `S = Σ_c |μ_c|²` the squared transition dipole in atomic units. PySCF ESD uses the following length-gauge conventions.

- Absorption oscillator strength `f = (2/3) E S` for a line, and `p_i f` for its thermal contribution.
- Vacuum spontaneous-emission coefficient `A = [4 α³ / (3 t_a)] E³ S`, with fine-structure constant `α` and atomic unit of time `t_a` in seconds. The population-weighted photon rate is `p_i A`. The corresponding radiant power multiplies this rate by the photon energy in joules.
- The ensemble-averaged radiative rate sums `p_i A` over retained lines, with inverse giving a model radiative timescale. An emitting-level rate sums the unweighted `A` values for that level.
- The Gaussian peak, photon-weighted centroid and strongest line are separate observables. Positive peak Stokes shift means the absorption maximum lies above the emission maximum in wavenumber. A common energy-zero convention preserves their connection to the 0–0 gap.

Use the physical rate prefactors for comparisons to absolute oscillator strengths or vacuum rates. Refractive-index corrections, nonradiative rates, nonequilibrium populations and quantum yield require additional modeling. [NIST electric-dipole relations](https://physics.nist.gov/Pubs/AtSpec/node17.html)

## Interpreting a completed calculation

A completed calculation meets the implemented electronic residual, geometry, state-identity and nuclear-coverage checks for its selected model. Those checks qualify the numerical result within the specified electronic and nuclear approximations. DFT functional error, basis incompleteness, an omitted conformer, solvent dynamics and higher-order vibronic effects can still affect agreement with experiment.

Transfer to a new molecule uses the same physical settings and acceptance criteria. Experimental peaks may define a declared comparison shift or linewidth in postprocessing. They do not enter the default electronic energies, transition moments or nuclear Hamiltonian. Keep the native-energy numerical report when comparing aligned envelopes. The [comparison guide](spectral-comparison.md) gives the density conversions and the [validation guide](validation.md) states the conditions of the retained benchmarks.

## Source map

- [workflow.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/workflow.py) selects the model, retries a reference and grows the vibrational basis. [electronic_workflow.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/electronic_workflow.py), [cpu.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cpu.py), and [state_tracking.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/state_tracking.py) own electronic preparation and phase checks.
- [cis_hessian.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cis_hessian.py) and [cis_integrals.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cis_integrals.py) implement analytic CIS derivatives. [pyscf_adapter.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/pyscf_adapter.py) converts derivatives and aligned geometries to vibrational models. [gpu.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/gpu.py) holds the single-point GPU route.
- [harmonic.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/harmonic.py), [automatic_anharmonic.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/automatic_anharmonic.py), [automatic_1d.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/automatic_1d.py), [anharmonic.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/anharmonic.py) and [thermal_separable.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/thermal_separable.py) evaluate the corresponding spectral models.
- [spectral_properties.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/spectral_properties.py) and [vibronic_properties.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/vibronic_properties.py) interpret strengths and surface parameters. [cli.py](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/src/pyscf_esd/cli.py) writes the CSV and JSON outputs.
