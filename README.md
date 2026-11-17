# PySCF ESD

**Vibronic absorption and fluorescence from PySCF.** Start with an XYZ geometry for an automatic calculation, or supply electronic surfaces and transition dipoles to the Python solvers.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-326E9A)](pyproject.toml)
[![PySCF 2.15+](https://img.shields.io/badge/PySCF-2.15%2B-087E82)](https://pyscf.org/)
[![License GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-596773)](LICENSE)
[![Methods paper PDF](https://img.shields.io/badge/Methods%20paper-PDF-913F62)](docs/paper/pyscf-esd.pdf)

**[Quick start](#quick-start) · [PySCF module](#use-as-a-pyscf-module) · [Default and options](#default-and-options) · [Method guide](docs/using-methods.md) · [Spectra gallery](#spectra-gallery) · [Output](#output) · [Methods paper](#methods-paper)**

## Quick start

Use Python 3.10 or later on Linux or Windows Subsystem for Linux.

```bash
git clone https://github.com/HenriqueCSJ/PySCF_ESD.git
cd PySCF_ESD
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[workflow]'
pyscf-esd molecule.xyz --output molecule.esd
```

Replace `molecule.xyz` with a single XYZ geometry in ångström. The output directory must be new. The CLI supports closed-shell singlets and accepts a molecular charge with `--charge`.

<details>
<summary><strong>Try a small water input</strong></summary>

```bash
cat > water.xyz <<'EOF'
3
Water, angstrom
O  0.0000000  0.0000000  0.0000000
H -0.7677654  0.0000000  0.5849831
H  0.7677654  0.0000000  0.5849831
EOF
pyscf-esd water.xyz --model vg --basis '6-31+g*' --output water-vg.esd
```

This example explicitly selects vertical gradient (VG) for a small installation check.

</details>

## Use as a PySCF module

After installing `.[workflow]` in your PySCF environment, import the [namespace extension](https://pyscf.org/user/extensions.html) from a Python script.

```python
from pyscf import esd, gto

mol = gto.M(
    atom="O 0 0 0; H -0.7677654 0 0.5849831; H 0.7677654 0 0.5849831",
    basis="6-31+g*", unit="Angstrom",
)
result = esd.run(mol, xc="pbe0", model="vg")
absorption = result.spectra["absorption"]
fluorescence = result.spectra["emission"]
```

`esd.run` takes a built, closed-shell molecule with a named all-electron spherical basis and uses its geometry, basis and charge. Choose `model="auto"`, `"ah"` or `"vg"` and use the zero-based `target_index` to select a singlet. The result includes `diagnostics` and `physical_properties`. Use `backend="gpu"` for analytic GPU derivatives and add `solvent="cyclohexane"` for equilibrium C-PCM. The [advanced API](docs/using-methods.md#choosing-the-level-of-theory) is also available as `esd.Settings` and `esd.run_vibronic`.

## Default and options

See the [method guide](docs/using-methods.md) for CLI and Python examples, including explicit finite differences, SF-TDA, analytic derivatives, and prepared anharmonic models.

The default CPU workflow uses PBE0/def2-TZVPD and the first singlet LR-TDA root at the relaxed ground-state geometry. It targets both directions at 0 K, optimizes the two geometries with analytic gradients, follows the electronic root and transition-dipole phase, and checks restricted orbital stability. The automatic nuclear model builds an adiabatic harmonic reference and screens direct cuts for anharmonicity. A case outside the model or numerical coverage is reported as incomplete. For gas-phase RKS/TDA, the automatic route can use checked local fits to analytic gradients and signed dipoles for the excited-state curvature and dipole derivative.

<details open>
<summary><strong>Harmonic and faster models</strong></summary>

```bash
# Adiabatic Hessian with Duschinsky and linear Herzberg–Teller contributions
pyscf-esd molecule.xyz --model ah --temperature 298 --sigma 33.972872 --output ah-298.esd

# Vertical-gradient absorption sticks from the second singlet root at S₀
pyscf-esd molecule.xyz --model vg --process absorption --state 2 --sigma 0 --output vg-s2.esd
```

AH uses the two endpoint curvatures and includes Duschinsky rotation and coherent Franck–Condon/Herzberg–Teller amplitudes in the supported route. VG uses the ground-state curvature and the vertical excited-state gradient with a constant transition dipole. The CLI Gaussian setting is a standard deviation in cm⁻¹. The first command above uses **80 cm⁻¹ FWHM**. The default `--sigma 50` corresponds to about 118 cm⁻¹ FWHM.

</details>

<details>
<summary><strong>Reference checks and numerical controls</strong></summary>

- Select the electronic model with `--xc`, `--basis`, `--charge`, and one-based `--state`.
- Choose `--reference-checks all` to check the restricted reference at every electronic point. The default checks the relevant minima.
- Repeat `--fallback-xc XC` to specify ordered functional alternatives for the supported stability recovery path. Each functional starts a separate calculation. Inspect the resulting state assignment.
- Set `--process`, `--temperature`, `--max-quanta`, and `--max-overlaps` to control the desired spectrum and its vibrational budget. Run `pyscf-esd --help` for the accepted flags.

</details>

<details>
<summary><strong>Analytic AH, automatic screening and VG on NVIDIA GPU</strong></summary>

```bash
pyscf-esd molecule.xyz --backend gpu --model ah --output gpu-ah.esd
pyscf-esd molecule.xyz --backend gpu --model ah --solvent cyclohexane --output gpu-solvent.esd
```

The GPU workflow shares the optimizer and phase tracking with the CPU workflow. `ah` uses analytic S₀ and excited-state Hessians and a phase-matched analytic Herzberg–Teller derivative. `auto` uses the same GPU backend for electronic sampling of anharmonic cuts. `vg` relaxes S₀ and uses the analytic vertical gradient. Install a compatible [GPU4PySCF environment](https://pyscf.org/user/gpu.html#installation).

For GPU AH use a compatible global hybrid LDA/GGA such as PBE0. The analytic route includes density fitting and XC quadrature response. PBE0 has built-in analytic fourth XC derivatives. Other functionals can require `PYSCF_ESD_LIBXC_LXC` pointing to a same-version LibXC library built with `MAXORDER=4`. Unsupported or failed analytic responses are reported as errors. The CPU DFT route retains checked local fits when analytic excited-state derivatives are unavailable on that backend. See the [GPU method guide](docs/using-methods.md#gpu-vg-ah-and-automatic-screening).

</details>

<details>
<summary><strong>Prepared inputs and anharmonicity</strong></summary>

- The [Python API](src/pyscf_esd/__init__.py) accepts prepared harmonic surfaces, supports periodic DVR with harmonic spectators at finite temperature, and exposes oscillator strengths and radiative properties.
- Automatic local anharmonic spectra support positive temperatures when the thermal partition and transition coverage qualify. See [thermal anharmonic spectra](docs/using-methods.md#finite-temperature-anharmonicity). For periodic rotations use supplied phase-consistent torsional inputs and the [separable rotor example](examples/assemble_separable_rotor.py).
- The [CF₃NO replay](examples/replay_cf3no_thermal.py) defaults to the retained PBE0/TDA model with one rotor and 11 spectators. SF-TDA is an explicit `--method sf-tda` alternative with Matplotlib for plotting. Both routes require the retained calculation archive, which is separate from this repository. The PBE0 comparison is a nuclear-model diagnostic with a restricted-reference instability.
- Prepared isolated DF-RKS/TDA roots use the same [analytic excited-state curvature](src/pyscf_esd/molecular_tda_hessian.py) and [analytic transition-dipole response](src/pyscf_esd/tda_dipole.py) in vacuum or equilibrium C-PCM.

</details>

## Spectra gallery

These selected plots retain their original axes and comparison parameters. The captions give the method, linewidth, and alignment for each case. Pyridine and quinine retain the absolute absorption scale. Comparison widths and shifts are applied after calculation and leave the software defaults and transition strengths unchanged.

<details open>
<summary><strong>Benzene · n-hexane · geometry sensitivity and anharmonicity</strong></summary>

<p>
  <img src="docs/benchmarks/benzene.svg" alt="Six-membered aromatic ring of benzene" width="155">
</p>

![Benzene at 298 K with the original and published CASPT2 equilibrium displacements](docs/benchmarks/benzene-geometry-control.png)

Replacing the symmetric breathing displacement in the B3LYP5/TDA model with a published geometry-derived value changes the fluorescence peak ratio from **1.013 to 0.754**, versus **0.688** measured. The two calculations share the same Hessians, dipole model, origin, Lorentzian width and display shift. [Published CASPT2 geometries](https://doi.org/10.1063/1.480854)

![Benzene PBE harmonic and anharmonic absorption and fluorescence compared with published curves](docs/benchmarks/benzene.png)

The PBE anharmonic model reproduces the three main fluorescence bands and reduces the profile RMSE by **38%** relative to the harmonic control at the same adjusted width. The PBE anharmonic RMSE is **0.063** in absorption and **0.081** in fluorescence, versus **0.085** and **0.071** for the published comparison. [de Souza et al., Fig. 3](https://doi.org/10.1063/1.5010895)

</details>

<details>
<summary><strong>CF₃NO · periodic rotor and 11 harmonic spectators</strong></summary>

![CF3NO absorption at 298 K against the measured trace and calculated fluorescence for PBE0 and spin-flip TDA](docs/benchmarks/cf3no-298k.png)

The periodic CF₃ rotor recovers the low-frequency progression and broad absorption envelope after alignment to the measured jet origin. The peak-to-origin interval is **498.50 cm⁻¹**, compared with **499.90 cm⁻¹** measured. The PySCF curves at 298 K use Gaussian FWHM **80 cm⁻¹**, and fluorescence retains the computed photon-rate density. The article reports the reference-stability test and the native energy offset alongside this nuclear benchmark. [Hage et al.](https://doi.org/10.1021/acsomega.6c11480)

![CF3NO gas-phase absorption oscillator strength and emission photon rate on the native axis at 0 K](docs/benchmarks/cf3no-0k.png)

The 0 K panel shows the native energy axis and absolute spectral densities with Gaussian **σ = 50 cm⁻¹**. The common 0–0 origin separates the main absorption and fluorescence bands.

</details>

<details>
<summary><strong>Quinine dication · water · AS–Franck–Condon</strong></summary>

<p>
  <img src="docs/benchmarks/quinine.svg" alt="Stereochemical structure of quinine protonated at both nitrogen atoms" width="220">
</p>

![Calculated and measured absorption and fluorescence of the quinine dication](docs/benchmarks/quinine.png)

The broadened absorption maximum is **3.606 eV** before alignment, compared with **3.568 eV** in the experiment. A display shift of **−0.496 eV** brings the calculated fluorescence envelope close to the measured one. The unshifted peak separation in the energy-domain plot is **0.41 eV**, compared with **0.85 eV** measured. Retesting the widths retained **0.496 eV** for absorption and **0.397 eV** for fluorescence. The 50-atom pilot used one conformer in implicit water. Its relaxed S₁ geometry met the force criterion and gave stable successive spectra. The strict geometry convergence cycle was incomplete. [PhotochemCAD measurement in 0.5 M aqueous H₂SO₄](https://omlc.org/spectra/PhotochemCAD/html/081.html)

</details>

<details>
<summary><strong>9-Cyanoanthracene · cyclohexane · AS–Franck–Condon</strong></summary>

<p>
  <img src="docs/benchmarks/cyanoanthracene.svg" alt="Chemical structure of anthracene-9-carbonitrile" width="240">
</p>

![Calculated and measured spectra of 9-cyanoanthracene at three Gaussian linewidths on native and aligned axes](docs/benchmarks/cyanoanthracene-widths.png)

The wavelength-domain comparison shows Gaussian FWHM values of **80**, **147**, and **400 cm⁻¹**, both on the native axis and with a common **+203 cm⁻¹** shift. The red absorption band carries too much relative intensity at all three linewidths. [Boens et al., supporting Fig. S1-D, CC BY-NC 4.0](https://doi.org/10.1021/ac062160k.s001)

</details>

<details>
<summary><strong>Pyridine · vapor · AH + Duschinsky + Herzberg–Teller</strong></summary>

<p>
  <img src="docs/benchmarks/pyridine.svg" alt="Chemical structure of neutral pyridine" width="160">
</p>

![Calculated and measured spectra of pyridine on the native energy axis and after alignment](docs/benchmarks/pyridine.png)

The local harmonic origin is **4.217 eV** versus **4.311 eV** measured. The comparison retains Gaussian FWHM values of **0.010** and **0.020 eV**, with display shifts of **−0.127** and **+0.356 eV** for absorption and fluorescence. A shallow out-of-plane well and evolving electronic character limit the local model. [Experimental vibronic origin](https://doi.org/10.1016/j.cplett.2008.07.080) · [Absorption](https://doi.org/10.1016/0022-2852(84)90051-1) · [Fluorescence trace](https://doi.org/10.1063/1.433862) · [Traced through Wang et al.](https://doi.org/10.1021/acs.jpclett.2c01475)

</details>

<details>
<summary><strong>Lithium hydride · one-mode analytic AH control</strong></summary>

![LiH 0 K absorption and fluorescence from the archived analytic RHF-CIS AH test](docs/benchmarks/lih.png)

This archived RHF–CIS/6-31G test exercises the analytic derivatives and AH assembly with independent curvatures. At 0 K the strongest absorption line lies one excited-state harmonic quantum above the common 0–0 origin. Fluorescence peaks at the origin.

</details>

## Output

Read `run.esd/properties.json` for numerical results. With emission enabled, `spectra.emission.vacuum_radiative_rate_s_inverse` gives $k_f$ in s⁻¹ and `spectra.emission.vacuum_radiative_lifetime_s` gives $1/k_f$ in seconds. Python exposes the same values in `result.physical_properties`.

The [numerical output guide](docs/numerical-results.md) shows how to extract these values, oscillator strengths, 0–0 and Stokes energies, mode contributions, and line assignments. Use `diagnostics.json` to check convergence, state tracking, derivative provenance, and coverage. The CSV files provide all retained lines and Gaussian profiles.

## Unrestricted DFT and phosphorescence

The [unrestricted DFT and phosphorescence APIs](docs/phosphorescence.md) cover analytic derivatives of a high-spin UKS triplet, vertical absorption and emission from prepared UKS/TDA roots, and AH phosphorescence from prepared spin–orbit matrices. The phosphorescence workflow resolves triplet sublevels and supports complex Franck–Condon and Herzberg–Teller amplitudes, Duschinsky rotation, and temperature-dependent vacuum radiative rates.

## Methods paper

The methods paper documents the earlier 0 K qualification policy for the automatic anharmonic route. The [finite-temperature method guide](docs/using-methods.md#finite-temperature-anharmonicity) describes the current workflow.

The [complete article (PDF)](docs/paper/pyscf-esd.pdf) includes the abstract, introduction, methods, results and discussion, and conclusions. It presents nine molecular structures, a quantitative benchmark table, and the eight selected spectra within the results that discuss them. The [main LaTeX source](docs/paper/main.tex) includes the [electronic response](docs/paper/electronic.tex), [nuclear methods](docs/paper/nuclear.tex), and [benchmark table](docs/paper/benchmark-table.tex).

[Download the PDF directly](https://raw.githubusercontent.com/HenriqueCSJ/PySCF_ESD/main/docs/paper/pyscf-esd.pdf).

Build the article from its source with `latexmk -cd -pdf docs/paper/main.tex`.

## Develop

```bash
python -m pip install -e '.[workflow]'
python -m unittest discover -s tests
```

[GNU GPL v3.0 only](LICENSE).
