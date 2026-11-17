# PySCF ESD

**Vibronic absorption, fluorescence, and phosphorescence with PySCF.** Start from an XYZ file for closed-shell absorption and fluorescence. Use `pyscf.esd` for prepared states, open-shell transitions, and spin–orbit phosphorescence. Get spectra, oscillator strengths, radiative lifetimes, and mode assignments.

[Quick start](#quick-start) · [Python](#use-as-a-pyscf-module) · [Choose a method](#default-and-options) · [Results](#output) · [Spectra](#spectra-gallery) · [Methods paper](docs/paper/pyscf-esd.pdf)

## Quick start

Use Python 3.10 or later on Linux or WSL.

```bash
git clone https://github.com/HenriqueCSJ/PySCF_ESD.git
cd PySCF_ESD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[workflow]'

pyscf-esd examples/water.xyz --model vg --basis '6-31+g*' --output water.esd
```

The [bundled water example](examples/water.xyz) uses vertical gradient and writes absorption and emission sticks, Gaussian profiles, and `water.esd/properties.json`.

For your own closed-shell molecule, supply one XYZ geometry in ångström and a new output directory.

```bash
pyscf-esd molecule.xyz --output run.esd
```

## Use as a PySCF module

```python
from pyscf import esd, gto

mol = gto.M(atom="examples/water.xyz", basis="6-31+g*", unit="Angstrom")
result = esd.run(mol, xc="pbe0", model="vg")

absorption = result.spectra["absorption"]
emission = result.physical_properties["spectra"]["emission"]
if emission["coverage_status"] == "complete":
    print("Radiative rate (s^-1)", emission["vacuum_radiative_rate_s_inverse"])
    print("Radiative lifetime (s)", emission["vacuum_radiative_lifetime_s"])
```

`esd.run` uses the geometry, basis, and charge of the built PySCF molecule. Select a singlet root with zero-based `target_index` in Python or one-based `--state` on the CLI. [Python API](docs/api.md)

## Default and options

The default is **CPU · PBE0/def2-TZVPD · first singlet at S₀ · 0 K · absorption and emission**. PySCF ESD follows the electronic state and transition-dipole phase, checks orbital stability, and reports the derivative route and vibrational coverage.

- **`auto`** relaxes both states, screens for anharmonicity, and uses a corrected nuclear model when it qualifies at the requested temperature.
- **`ah`** uses separate curvatures at the two minima, Duschinsky rotation, and coherent Franck–Condon and Herzberg–Teller amplitudes.
- **`vg`** uses the ground-state curvature, the vertical excited-state gradient, and a constant transition dipole.

```bash
# B3LYP/def2-SVP harmonic AH at 298 K with 80 cm^-1 Gaussian FWHM
pyscf-esd molecule.xyz --model ah --xc b3lyp --basis def2-SVP \
  --temperature 298 --sigma 33.972872 --output ah.esd

# Analytic GPU AH with equilibrium C-PCM
pyscf-esd molecule.xyz --backend gpu --model ah --solvent cyclohexane --output solution.esd
```

Use `--xc`, `--basis`, and `--charge` to select the electronic model. `--sigma` is the Gaussian standard deviation in cm⁻¹. Use `--sigma 0` for sticks. [Command-line options](docs/guide.md#command-line-options) · [Solvents](docs/using-methods.md#choosing-a-solvent)

<details>
<summary><strong>Analytic derivatives, GPU setup, and prepared inputs</strong></summary>

- **Derivative coverage.** CPU DFT AH uses analytic gradients and checked local fits for excited-state curvature and HT. For supported functionals and references, GPU AH evaluates both Hessians and HT analytically. Some functionals require a matching fourth-derivative LibXC build via `PYSCF_ESD_LIBXC_LXC`. [GPU setup and derivative requirements](docs/using-methods.md#gpu-vg-ah-and-automatic-screening)
- **Prepared electronic and vibrational models.** Use isolated TDA/C-PCM derivatives, custom AH surfaces, coupled-mode DVR, or a periodic rotor with harmonic spectators. [Prepared-input examples](docs/using-methods.md#spectra-from-prepared-electronic-data) · [Anharmonic models](docs/using-methods.md#prepared-anharmonic-models)
- **Spin-resolved spectra.** Use prepared UKS/TDA for spin-conserving transitions, SF-TDA for spin-flip states, or full-space SOC with gas-phase analytic spin-vibronic HT for an isolated triplet. [SF-TDA](docs/using-methods.md#spin-flip-tda-on-gpu) · [Open-shell spectra and phosphorescence](docs/phosphorescence.md)

</details>

## Output

- **`properties.json`** gives oscillator strengths, radiative rates and lifetimes, 0–0 and Stokes energies, and mode contributions when the model supplies them.
- **`*_sticks.csv`** lists individual transitions, assignments, and line strengths. With broadening enabled, **`*_profile.csv`** gives relative photon-weighted spectral density.
- **`diagnostics.json`** records state tracking, convergence, the derivative source, and coverage. Use it to check whether a spectrum completed.

Use `coverage_status` to check whether a reported rate is complete. The vacuum radiative lifetime is the reciprocal of a positive emission rate. Observed decay also depends on nonradiative channels. [Read the numerical results](docs/numerical-results.md)

## Spectra gallery

**Benzene in n-hexane.** The anharmonic PBE model resolves the three main fluorescence bands and reduces the fluorescence profile RMSE by **38%** relative to harmonic PBE at the same linewidth.

![Benzene absorption and fluorescence in eV with harmonic and anharmonic PySCF curves and experimental data](docs/benchmarks/benzene.png)

<details>
<summary><strong>Naphthalene phosphorescence at 77 K</strong></summary>

![Naphthalene phosphorescence in eV with gas-phase, continuum, and experimental glass spectra](docs/benchmarks/naphthalene-phosphorescence.png)

Full-space SOC and analytic spin-vibronic HT give **88.5% perpendicular emission** and a **15.54 s** gas-phase radiative lifetime. The matched Condon solvent comparison leaves the two ring-mode peaks resolved at the shared linewidth. [Methods and numerical data](docs/phosphorescence.md#naphthalene-benchmark)

</details>

See the [full spectra gallery](docs/spectra-gallery.md) for molecular structures, plot settings, and all selected comparisons, including the CF₃NO rotor, benzene geometry control, pyridine, 9-cyanoanthracene, quinine, and LiH.

## Documentation

[Method guide](docs/using-methods.md) · [Python API](docs/api.md) · [Numerical results](docs/numerical-results.md) · [Validation](docs/validation.md) · [Methods paper PDF](docs/paper/pyscf-esd.pdf) · [LaTeX source](docs/paper/main.tex)

Develop with `python -m pip install -e '.[workflow]'` and `python -m unittest discover -s tests`. [GPL-3.0-only](LICENSE)
