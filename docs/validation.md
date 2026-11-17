# Validation and reproducibility

[Project home](../README.md) · [Workflow guide](guide.md) · [Methods](methods.md) · [Companion mini-paper](mini-paper.md)

The runnable tests verify numerical behavior and software contracts. Molecular replay scripts also use retained electronic inputs from particular benchmark runs. The complete external archive and experimental digitizations are separate from this repository. Each replay below identifies the inputs it needs.

## Reproduce from the repository

```sh
python -m pip install -e '.[workflow]'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python examples/validate_anharmonic_rotor.py
python examples/validate_water_cpu.py water-validation --threads 2 --stencil
```

- The unit suite checks harmonic oscillator limits, independent quadrature, coherent phase and amplitude handling, numerical stability, derivative response, workflow failure modes, and convergence diagnostics. Optional electronic tests require the relevant packages.
- The synthetic periodic rotor example checks a finite-barrier model against an exact 0–0 energy and dipole strength. The thermal test compares a DVR plus harmonic bath against direct broadening of assigned vibronic lines at 298 K for absorption and emission. These are independent checks within a model. [Rotor example](../examples/validate_anharmonic_rotor.py) · [Thermal tests](../tests/test_thermal_separable.py)
- The native H₂/STO-3G stability control distinguishes an internal orbital saddle from restricted-to-unrestricted instability. At 1.4 bohr, starting from the stationary antibonding solution, the recovery reaches the independent lower-energy RHF solution to within 5×10⁻¹⁶ Hartree. At 4.5 bohr the restricted solution has the external instability and the workflow rejects the reference. The test suite also checks restart and failure propagation with controlled inputs. [CPU tests](../tests/test_cpu.py) · [Workflow tests](../tests/test_workflow.py)
- A prepared LiH RHF–CIS control reproduces the archived 31,354.50396 cm⁻¹ origin using the caller's analytic Hessians and the same adaptive nuclear coverage checks as the automatic workflow. A matched PySCF 2.15.0 water VG replay gives a 0–0 gap difference of 4.71×10⁻¹³ Hartree between the current and archived post-correction sources. [Prepared AH tests](../tests/test_prepared_ah.py)
- Closed-rotor tests check all four combinations of the two electronic closure signs, including the half-integer free-rotor spectrum and dipole selection rules for mixed nuclear boundary conditions. Both the stick and finite-temperature paths require matching dipole parity. [Phase tests](../tests/test_phased_rotor.py)

## Water CPU and GPU

The CPU example computes a fresh water reference and compares analytic gradients with validation stencils and cold independent displacements. Its `reference.json` is an input to the GPU comparison. Use a fresh output directory for the CPU benchmark.

```sh
python examples/validate_water_gpu.py water-validation/reference.json water-gpu-report.json --threads 2 --repeats 3
```

The matched analytic VG run on the RTX 4070 Laptop GPU passed 16 acceptance checks with zero displaced production geometries. It gave maximum CPU–GPU differences of 3.64×10⁻¹¹ Hartree/bohr in the S₁ gradient and 7.04×10⁻¹¹ Hartree/bohr² in the S₀ Hessian. The largest assigned-line differences were 1.13×10⁻⁵ cm⁻¹ in absorption and 1.04×10⁻⁵ cm⁻¹ in emission. The CPU S₁ response residual was 8.51×10⁻¹¹ at the requested 10⁻¹⁰ tolerance. The warmed three-repeat medians were 21.37 s on the two-thread CPU route and 36.80 s on GPU. This is one small-molecule, matched-model result. In a separate instrumented run, SCF/TDA setup and validation and orbital stability checks accounted for about 89% of the additional GPU time. [CPU driver](../examples/validate_water_cpu.py) · [GPU driver](../examples/validate_water_gpu.py)

## CPU default from XYZ

The CPU `auto` workflow completed LiH from the XYZ below with the default PBE0/def2-TZVPD, first singlet and 0 K. It optimized both minima, tracked the electronic state and dipole phase, and selected the direct diatomic DVR. The scan used 51 direct energy and dipole evaluations. Absorption and emission each produced six positive-energy sticks, with final dipole-strength coverage of 99.52% and 99.83%, respectively. The harmonic moment bounds do not cover this direct anharmonic model. Its numerical report labels M1, M3 and M4 `unassessed` and provides the sums over the retained sticks.

```sh
cat > lih.xyz <<'EOF'
2
LiH test input, angstrom
Li 0.0 0.0 0.0
H  0.0 0.0 1.60
EOF
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 pyscf-esd lih.xyz --output lih-default.esd
```

The [harmonic convergence tests](../tests/test_prepared_ah.py) separately exercise hot-band emission and absorption in cases where unweighted dipole closure passes before the corresponding photon moment converges.

## Analytic excited-state derivatives

- **LiH/6-31G, conventional RHF–TDA/CIS.** The analytic total S₁ Hessian at the relaxed S₁ minimum agreed with a 0.002 bohr central stencil of independent analytic gradients to 3.10×10⁻⁸ Hartree/bohr² in the largest Cartesian component. The stencil is a validation control. The analytic S₁ Hessian and native RHF S₀ Hessian supply the two-surface AH spectrum. The replay takes the retained directory containing `lih_rhf_cis_631g_ah_reference.npz` and its adjoining JSON. [LiH replay](../examples/validate_lih_analytic_ah.py)
- **Formaldehyde/6-31G(d), conventional RHF–TDA/CIS.** The corresponding polyatomic Cartesian Hessian discrepancy was within 1.84×10⁻⁶ Hartree/bohr² against two stencil steps. The benchmark includes an analytic HT dipole derivative, local AH spectra, and a Duschinsky control using signed mode correspondence with the ground-frame displacement held fixed. The AH model describes one S₁ well. The script takes the retained electronic reference directory. [Formaldehyde AH replay](../examples/validate_h2co_analytic_ah.py)

```sh
python examples/validate_lih_analytic_ah.py PATH_TO_LIH_REFERENCE --output lih-replay
python examples/validate_h2co_analytic_ah.py PATH_TO_H2CO_ELECTRONIC --output h2co-ah-replay
```

The original independent ORCA calculations checked projected curvatures in these demonstrations. The PySCF spectra use their own electronic inputs.

## Formaldehyde inversion

The reduced models use direct electronic potential and signed dipole samples over the double well, in a shared mass-weighted Cartesian coordinate system with parity resolution. The 3D model includes the direction toward the relaxed planar S₁ geometry. Its direct electronic barrier is 133.90 cm⁻¹ and its computed lowest even–odd interval is 211.84 cm⁻¹. The barrier and interval refer to different quantities. The even level lies close to the top of the effective transverse vibrational profile, which limits a tunneling interpretation. Grid, domain and interpolation tests are part of the retained benchmark reports. [Formaldehyde spectroscopy context](https://doi.org/10.1063/1.3074100)

```sh
python examples/validate_h2co_inversion.py PATH_TO_AH PATH_TO_2D_SCAN --output h2co-2d
python examples/validate_h2co_3d.py PATH_TO_AH PATH_TO_2D_REPLAY PATH_TO_3D_SCAN --output h2co-3d
```

Use the retained AH, planar and volume electronic arrays. The latter include direct tests of the electronic fit. The `--plot` options require Matplotlib and render already calculated arrays.

## CF₃NO at 298 K

The benchmark replay defaults to the earlier PBE0/TDA model with a separate periodic rotor and 11 harmonic spectators. Use `--method sf-tda` for the alternative with SF-TDA torsional cuts and auxiliary UKS spectator force constants. Both replays use retained electronic inputs, 298 K and Gaussian FWHM 80 cm⁻¹ on the calculated energy axis. The first argument is the calculation archive containing both benchmark runs.

```sh
python examples/replay_cf3no_thermal.py /path/to/calculation/archive --output cf3no-pbe0
python examples/replay_cf3no_thermal.py /path/to/calculation/archive --output cf3no-sf --method sf-tda
```

After aligning to the experimental 0–0, PBE0/TDA gives the closer absorption envelope in this comparison. Its restricted reference has a known restricted-to-unrestricted instability, which the replay reports. The general XYZ workflow continues to check orbital stability.

The active CF₃NO stability investigation uses PySCF and DFT. Declare the candidates and stopping criteria before comparing spectra. A restricted candidate needs independently verified SCF and positive orbital curvature in the internal, real-to-complex and restricted-to-unrestricted channels, with the benchmark's existing +10⁻⁶ margin in the native convention. It also needs a tracked isolated singlet, stationary minima and qualified nuclear curvature. Check the electronic reference and phase along the entire labeled torsional path, including its closing point, before assembling the thermal spectrum. The target remains 298 K, Gaussian FWHM 80 cm⁻¹ and the calculated photon-energy axis without shifts or experimental fitting.

The fresh vacuum PBE/def2-TZVPD electronic check gave a positive restricted-to-unrestricted stability eigenvalue at the reoptimized S₀ minimum (+0.00393723 in the native convention). At the near-relaxed tracked n→π* geometry, the same channel remained unstable (−0.01741351), and the raw gradient maximum (1.20×10⁻⁵ Hartree/bohr) was above the 10⁻⁵ threshold. The run stopped before computing the S₁ Hessian and torsional surfaces. Earlier PBE0 inputs also failed the spin-breaking stability check. Fixed-geometry outcomes apply to the tested reference and geometry.

An independent CPU PBE check at the retained near-S₁ geometry found a restricted-to-unrestricted Rayleigh curvature of −0.01233414 in the native convention. A 24-dimensional low-gap trial space supplied the direction, and a full 5,040-dimensional native analytic Hessian application verified its sign. The eigenvector residual was 0.11249, so this number measures one downhill direction. The exact semilocal real-to-complex factorization gave +0.0298603900, with a residual of 3.81×10⁻¹³. The retained GPU body-frame calculation reports a different minimum for that positive channel. The CPU result independently establishes negative spin curvature and blocks spectral production from this RKS determinant.

At the retained S₀ geometry the optimized independent CPU assessor qualified the real-to-complex channel at +0.04933863569, with residual 2.03×10⁻¹³. Its internal and restricted-to-unrestricted eigensolves remained unresolved. The archived positive GPU stability result and this CPU numerical disposition concern their respective solver checks.

A CPU PBE0/STO-3G H₂ control checked the stability verdict against direct matrices of the two native operators. The lowest restricted-to-unrestricted curvatures were +0.48749038 at 1.4 bohr and −0.21561424 at 4.5 bohr. The checked CPU route accepted the first reference and reported the spin-breaking channel for the second. On the retained PBE0/6-31+G* water geometry, the checked CPU VG origin was 0.290601164769128 Hartree, 3.24×10⁻¹³ Hartree from the matched archived benchmark. For the same converged water determinant, one four-thread stability timing was 3.15 s for the native wrapper and 11.29 s for the independent check.

In a separate four-thread PBE0 water control, paired two-vector Hessian applications took median times of 0.133 s and 0.103 s for scalar and batched internal products, and 0.131 s and 0.099 s for scalar and batched spin-breaking products. Their largest component difference was 1.43×10⁻¹⁴. The timings cover the operator applications. The eigensolver retains the native scalar operator for independent residual and Rayleigh checks.

For a separate numerical diagnostic, the retained PBE0 potentials, phase-tracked dipole and 11 harmonic spectators were reprocessed at 298 K with the paper's Gaussian FWHM of 80 cm⁻¹. The photon-energy axis is unshifted, with no experimental parameter fit. The calculated absorption maximum is 13,668.86 cm⁻¹ and the digitized measured maximum is 14,429.60 cm⁻¹ (error −760.74 cm⁻¹, −5.27%). Relative to their corresponding origins, the peak offsets are 498.50 and 499.90 cm⁻¹. The observed reference origin comes from a separate jet-cooled measurement. The calculated emission maximum is 12,768.13 cm⁻¹, and the predicted peak Stokes shift is 900.73 cm⁻¹. Figure 3 supplies an absorption comparison. The diagnostic tests one spectral spacing under a reduced model with an unstable electronic reference. [298 K comparison, Fig. 3](https://doi.org/10.1021/acsomega.6c11480) · [Mason and Banus absorption experiment](https://doi.org/10.1039/JR9570003904)

The retained numerical comparison requires the common-zero torsional potentials, the signed dipole and spectator inputs, and digitized experimental points. These inputs are outside this checkout. The periodic API example and synthetic thermal test reproduce the nuclear calculation pattern. [Periodic rotor API](api.md#periodic-rotor-and-thermal-profile)

### Archived mixed-reference TDDFT investigation

MRSF-TDDFT gives two singlet response roots from a common auxiliary open-shell Kohn–Sham reference. A fixed-geometry pilot input set uses BHHLYP and the same spherical def2-TZVPD basis as the restricted benchmark. It stages six singlets and three triplets at each retained PBE geometry. The required checks include the high-spin reference stability and continuity, the gap between the two lowest triplets, the identity of the n→π* singlet, and convergence and phase conventions for the two physical-singlet transition properties. Published examples show response discontinuities when the triplet reference changes character. BHHLYP is a candidate for the pilot; production requires qualified properties at the relaxed minima and along the complete torsion. [MRSF theory](https://doi.org/10.1063/1.5044202) · [Analytic gradients](https://doi.org/10.1063/1.5086895) · [Reference-branch limitations](https://doi.org/10.1021/acs.jpclett.6c01200)

A triplet H₄/PBE0/STO-3G control tested the analytic internal ROKS assessment. All five directions involve rotations between occupation classes. An independent analytic spin-density variation matched each Hessian element within 10⁻⁸ Hartree, and the smallest native curvature was +0.22642789. This control uses a PySCF ROKS determinant. Applying the assessment to a different engine's orbitals requires verifying the AO map, functional, integration quadrature, and fresh orbital stationarity at those exact orbitals.

The pinned OpenQP engine starts and accepts the four staged CF₃NO inputs in a parser-only check. In a native stored-vector formaldehyde control, the identity, sign and order tests passed at one geometry. With a rigidly translated basis, the raw MO metric confirmed by PySCF differs from the native normalized metric by up to 0.164. Native state-overlap reversal fails by 5.33×10⁻³. Providing the raw metric improves the transpose discrepancy to 1.68×10⁻⁴, above the 10⁻⁸ tolerance. The discrepancy persisted with the same checkpoint context while alternating the raw metric and its transpose. The bundled Python contraction replicated both native matrices within 4.45×10⁻¹⁶ and isolated the residual to mixed determinant terms that reuse one directional minor for both factors. [Pinned OpenQP release](https://github.com/Open-Quantum-Platform/oqp-studio-releases/releases/tag/v0.2.7)

The independent determinant expansion contracts both directions from the raw bra-by-ket metric. On the same control the four-block implementation reproduced identity within 5.56×10⁻¹⁶, transpose reciprocity within 2.78×10⁻¹⁶, and both imposed root signs exactly. All four spin-summed one-body density matrices matched the bundled OpenQP transition-density formula within 6.67×10⁻¹⁶. The calculation used the stored vectors and a translated basis with the same electronic coefficients. The decoder implements the restricted MRSF response representation, including the original theory's incomplete spin adaptation of core-to-virtual configurations. [MRSF theory, Eqs. 3.9–3.10](https://doi.org/10.1063/1.5044202)

On a synthetic six-root control with 23 core orbitals and 234 total orbitals, the explicit determinant overlap and the four-block overlap differed by at most 2.78×10⁻¹⁵. The warmed one-thread CPU medians were 1.167 s and 3.927 ms for the respective overlap calls. The comparison times the signed overlap alone. To enforce a shared phase for tracking and dipoles, the prepared reader can reconstruct the density from the response vectors and rejects inconsistent signs or root orders that change the density. Its density reconstruction agrees with the OpenQP helper on the stored formaldehyde control within 2.23×10⁻¹⁶. Independent electronic checkpoints, production CF₃NO timing, the full torsion loop, and the absolute 298 K spectra remain unvalidated.

An artificial core–SOMO exchange in the stored formaldehyde orbitals leaves every alpha-occupied singular value at one within rounding, while the smallest core and SOMO singular values vanish. A rigidly translated basis reduces the smallest core overlap to 0.808 and its full-space retention to 0.816, exposing a loss of basis support in a control with the same electronic coefficients. The companion-triplet reader independently rejects a different auxiliary determinant even with the same SCF energy. The archived BHHLYP/6-31G water response offsets give 1.6683 eV for the gap between the lowest two triplets. The archive provides the energies only. Full native paired-checkpoint validation on independently solved points remains open. [MRSF reference diagnostics](https://pubs.acs.org/doi/10.1021/acs.jpclett.6c01200)

A basis-integral replay used the exact 234 spherical def2-TZVPD functions from the staged CF₃NO geometries in the pinned OpenQP engine. The direct PySCF adapter restores the native shell order and evaluates contraction scales from the supplied exponents and coefficients. Native self integrals agree within 1.93×10⁻¹⁵. Five signed cross-matrix comparisons including both endpoint orientations and an off-axis rigid translation agree within 4.62×10⁻¹⁵, including the spherical d and f blocks. The native AO self diagonals range from 0.05915 to 1, so replacing raw normalization with unit contractions changes matrix elements substantially. The tests cover AO integrals and the prepared interface. Electronic convergence, orbital-Hessian stability, and phase continuity of independently computed CF₃NO states still need electronic results. [Pinned OpenQP engine](https://github.com/Open-Quantum-Platform/oqp-studio-releases/releases/tag/v0.2.7) · [PySCF cross integrals](https://pyscf.org/pyscf_api_docs/pyscf.gto.html#pyscf.gto.mole.intor_cross)

The H₄/PBE0 control now serializes its converged determinant in a scaled AO convention and reimports it with SCF disabled. The fixed check recovers the native PySCF control's energy, both spin Focks and lowest real shared-orbital curvature within 10⁻¹⁰ in their respective Hartree units. On the OpenQP BHHLYP/STO-3G formaldehyde archive, imported orbital metric error is 1.34×10⁻¹⁵ and both recorded spin densities agree within 2.23×10⁻¹⁵. The recorded Focks give a residual of 1.375×10⁻⁶ Hartree in PySCF's half-gradient convention. This archive lacks a native grid and therefore supports orbital import and the recorded-Fock residual only. [PySCF analytic shared-orbital Hessian](https://github.com/pyscf/pyscf/blob/v2.15.0/pyscf/soscf/newton_ah.py)

The recoverable exact OpenQP commit was used for a standalone GNU spatial-grid replay at the two CF₃NO geometries. The radial, Lebedev, SSF and final weight-filter kernels give 373,592 and 373,178 retained points. An independent Python implementation matches every point within its parent atom, including an off-axis translation of R₀. Maximum coordinate and absolute weight differences were 9.0×10⁻¹⁴ bohr and 6.82×10⁻¹³ bohr³. The pinned Windows parser confirms the staged grid recipe, including separate AO screening. These comparisons concern the spatial grid before density-dependent and AO-value screening during Fock construction. A comparison of fresh native and PySCF Focks and orbital response still requires independent electronic checkpoints or a documented fixed-density native response interface. [Pinned source revision](https://github.com/VladimirMakhnev/oqp/commit/1db79089b285406338bd28a255b967e4011f8df7) · [Grid assembly](https://raw.githubusercontent.com/VladimirMakhnev/oqp/1db79089b285406338bd28a255b967e4011f8df7/source/dftlib/dft.F90)

### Archived wavefunction investigation

An earlier SA-CASSCF(10,7)/def2-SVP pilot converged spin-pure states and analytic S₁ gradient response at two retained geometries. A nearly doubly occupied orbital then exchanged between the active and inactive spaces during both CAS(10,7) geometry optimizations. Two orbital solvers reproduced the exchange with clear singlet root assignments. Including the exchanged occupied direction and its electron pair in CAS(12,8) restored local continuity through both rejected geometries. The forward–reverse checks recovered the two state energies within 2×10⁻⁹ Hartree.

An archived CAS(12,8) S₁ trial also exercises analytic gradient-response recovery. The native 45-cycle response failed with a residual of 7.14×10⁻⁸ against a limit of 1.23×10⁻⁸. Increasing the iteration budget to 135 for the same operator and preconditioner reduced the verified residual to 8.99×10⁻⁹, at the same tolerances. A separate native H₂ control checks the state-specific gradient interface. [Response tests](../tests/test_casscf_response.py)

Both full-period wavefunction scans were stopped with partial checkpoints. The archived wavefunction pilot is outside the current DFT benchmark.

## Benzene and de Souza comparisons

- The gas-phase benzene examples compare a three-coordinate model using ν₁ and selected components of ν₆ and ν₈ with measured band centers and the emission Fermi pair. Further examples probe mode coupling and the ν₁ progression in S₁. They require the retained electronic scans and mode metadata. The gas-phase plotting comparison uses a measured origin on the comparison axis and retains the computed origin in the report. The Raman-parametrized Fermi control uses external information and has a separate interpretation. [Three-coordinate solve](../examples/validate_benzene_3d.py) · [Experimental comparison](../examples/compare_benzene_experiment.py) · [Spectator model](../examples/validate_benzene_spectators.py) · [Gas-phase emission](https://doi.org/10.1021/ja00841a001)
- The 298 K benzene-in-hexane and naphthalene comparisons to de Souza and co-workers are exploratory. For benzene the default AH+HT model misses the relative emission peak heights. A separate composite sensitivity control replaces the computed totally symmetric bond expansion with a published theoretical geometry difference while retaining the benchmark harmonic and dipole inputs. These runs answer a model sensitivity question, and the geometry replacement is specific to that benchmark. The naphthalene one-mode linear HT extrapolation is amplitude-sensitive. Neither result supports a general claim of matching or outperforming ORCA. [Fluorescence benchmarks](https://doi.org/10.1063/1.5010895) · [Benzene theoretical geometries](https://doi.org/10.1063/1.480854)

For comparable DFT observables report absolute and relative discrepancies. The project's interpretation treats up to 5% as good agreement and up to 10% as acceptable with reservation. Use absolute discrepancies for quantities close to zero. State the observable, environment, reference and any presentation shift explicitly.

## Naphthalene phosphorescence at 77 K

The full-space gas-phase SOC calculation resolves intensity missed by the 12-singlet and 12-triplet state window. Analytic spin-vibronic HT raises the gas-phase rate from 0.05676 to 0.06437 s⁻¹. The matched solvent Condon calculation isolates the effect of mean dielectric polarization and gives 0.06312 s⁻¹ with the same vacuum photon prefactor. Two totally symmetric vibrational fundamentals account for the resolved structure in all three calculations. The measured 0–0 linewidth can merge those features in a resolution check. [Methods and interpretation](phosphorescence.md#naphthalene-benchmark) · [Benchmark data](benchmarks/naphthalene/)
