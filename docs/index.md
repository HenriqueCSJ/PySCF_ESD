# PySCF ESD

**Vibronic absorption, fluorescence and phosphorescence with PySCF**

PySCF ESD connects electronic structure, state tracking, nuclear motion and radiative observables. Start from a molecular geometry, use the package as a PySCF module, or supply prepared surfaces and signed transition moments. The documentation explains each route, its physical assumptions and the numerical checks used to qualify its results.

## Start a calculation

Follow [installation](installation.md) to set up a CPU or GPU environment, then run the [water example](quickstart.md). From an installed checkout, a small VG calculation starts with the following command.

```bash
pyscf-esd examples/water.xyz --model vg --basis '6-31+g*' --output water-vg.esd
```

The full automatic default uses CPU, PBE0/def2-TZVPD, the first singlet at relaxed S₀, 0 K and both absorption and emission. It prepares the two surfaces, includes Duschinsky rotation and available HT amplitudes, then screens for supported anharmonic corrections. [Workflow and models](guide.md)

The same package can be used in a PySCF script.

```python
from pyscf import esd, gto

mol = gto.M(atom="examples/water.xyz", basis="6-31+g*", unit="Angstrom")
result = esd.run(mol, xc="pbe0", model="vg")
print(result.model_used)
print(result.physical_properties["spectra"]["emission"]["vacuum_radiative_rate_s_inverse"])
```

## Choose the physical model

**Absorption and fluorescence.** Use VG for a shared-curvature model, AH for the two relaxed harmonic surfaces, or `auto` for local anharmonic screening. Prepared APIs expose FC, HT, Duschinsky rotation, finite-temperature spectra and caller-supplied curvatures. [Select methods, states and solvents](using-methods.md)

**GPU and solution calculations.** The high-level GPU workflow supports VG, analytic AH and automatic screening with its supported global-hybrid LR-TDA model. Named solvents select equilibrium C-PCM. The GPU guide states the derivative requirements and explains where host-side work enters the calculation. [GPU calculations](gpu.md)

**Phosphorescence and open shell.** Separate Python routes provide UKS endpoints, spin-conserving unrestricted TDA, SF-TDA optical properties, SOC state interaction and analytic spin-vibronic HT in their supported domains. [Phosphorescence and open-shell guide](phosphorescence.md)

**Large-amplitude motion.** Prepared rotors, double wells and coupled vibrational Hamiltonians accept coordinate-dependent signed dipoles. Periodic scans retain the electronic closure phase and the corresponding nuclear boundary conditions. [Prepared data and nuclear models](prepared-data.md)

## Obtain the physics from the output

Read assigned lines, oscillator strengths, radiative rates and lifetimes, emitted powers, spectral moments and relaxation quantities from JSON and CSV. Check the completion status and the coverage for the requested observable. [Physical quantities and files](numerical-results.md)

For experimental comparisons, choose an energy or wavelength axis, a width convention and any displayed energy alignment explicitly. The comparison guide explains density conversion, normalization and how to retain the native prediction alongside an aligned spectrum. [Broadening and experimental comparison](spectral-comparison.md)

If a calculation stops, inspect its electronic, state, derivative and nuclear-basis diagnostics. The troubleshooting guide gives a recovery path for each stage. [Troubleshooting and convergence](troubleshooting.md)

## Reference and examples

- [Command-line reference](command-line.md) lists every flag and its default.
- [Python examples](api.md) show end-to-end and prepared calculations.
- [Python API reference](api-reference.md) documents settings, signatures, shapes and result objects.
- [Theory and conventions](methods.md) defines the electronic and nuclear models and their observables.
- [Spectra gallery](spectra-gallery.md) presents the retained molecular comparisons.
- [Validation and reproducibility](validation.md) describes numerical controls and the data needed for each replay.
- [Methods manuscript (PDF)](paper/pyscf-esd.pdf) develops the accompanying scientific account.
- [Contributing](contributing.md) covers development, tests and documentation publication.

The code is distributed under GPL-3.0-only. Report reproducible issues through the repository and include the model, package versions and diagnostic output. [Source code](https://github.com/HenriqueCSJ/PySCF_ESD) · [Issue tracker](https://github.com/HenriqueCSJ/PySCF_ESD/issues) · [License](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/LICENSE)
