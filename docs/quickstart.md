# First calculation

Start with the bundled water input to check that the electronic and spectral workflow runs in your environment. This example requests VG with a small basis. The [automatic default](guide.md#input-and-model) uses a larger basis and a more extensive nuclear model.

## Run from an XYZ file

From the installed repository, run the following in a Linux or WSL shell.

```bash
pyscf-esd examples/water.xyz --model vg --basis '6-31+g*' \
  --output water-vg.esd
```

The XYZ contains Cartesian coordinates in ångström. The driver optimizes the ground-state geometry, resolves the selected singlet, forms the VG model and converges absorption and relaxed-state emission. Electronic and basis-growth progress appears on standard error.

The output directory contains `diagnostics.json`, `properties.json` and the available absorption and emission sticks and profiles. A successful run exits with code 0. Use the diagnostics to establish completion before interpreting the spectra.

```python
import json
from pathlib import Path

run = Path("water-vg.esd")
diagnostics = json.loads((run / "diagnostics.json").read_text())
print("Run status", diagnostics["status"])
print("Model", diagnostics.get("model_used"))
print("Output files", diagnostics["spectra"])
```

The [numerical results guide](numerical-results.md) shows how to extract oscillator strengths, rates, radiative lifetimes, individual lines and their convergence status.

## Use the full automatic workflow

For your own closed-shell molecule, supply a single XYZ frame and a fresh output directory.

```bash
pyscf-esd molecule.xyz --output molecule-auto.esd
```

The default selects PBE0/def2-TZVPD, the first singlet at relaxed S₀, CPU, 0 K and both spectral directions. It optimizes the two surfaces, constructs AH with Duschinsky rotation and HT, then screens direct electronic cuts for supported anharmonic corrections. The [workflow guide](guide.md) explains the model selection and derivative routes.

For a specified harmonic model at room temperature with 80 cm⁻¹ Gaussian FWHM, use the following.

```bash
pyscf-esd molecule.xyz --model ah --temperature 298 \
  --sigma 33.9728720115 --output molecule-ah-298.esd
```

The CLI takes Gaussian standard deviation in cm⁻¹. The display width in this example is about 0.00992 eV. [Broadening and energy axes](spectral-comparison.md)

## Use the PySCF module

The module accepts a built `Mole` and returns an in-memory result. Geometry construction below uses ångström, and the wrapper converts the coordinates for the ESD workflow.

```python
from pyscf import esd, gto

mol = gto.M(
    atom="O 0 0 0; H -0.7677654 0 0.5849831; H 0.7677654 0 0.5849831",
    unit="Angstrom",
    basis="6-31+g*",
    spin=0,
    verbose=0,
)
result = esd.run(mol, xc="pbe0", model="vg", progress=print)

print(result.model_used)
for process, spectrum in result.spectra.items():
    print(process, len(spectrum.photon_energy_hartree))
print(result.physical_properties["spectra"]["emission"]["vacuum_radiative_rate_s_inverse"])
```

The Python workflow returns arrays and dictionaries. The CLI writes its JSON and CSV files. Prepared solver functions accept your own surfaces, Hessians and signed dipoles. [Python API](api.md) · [Prepared data](prepared-data.md)

## Select the target state

Use `--state 2` for the second singlet on the CLI and `target_index=1` in Python. The label is assigned at the optimized S₀ geometry. State tracking then follows its character and electronic phase as the geometry changes.

Inspect the root identity and reference checks in `diagnostics.json` before assigning experimental bands. The [troubleshooting guide](troubleshooting.md#state-identity-and-phase) explains ambiguous matches and root-window failures.

## Continue with a specific task

- [Choose a functional, basis or solvent](using-methods.md).
- [Use the GPU for AH or automatic screening](gpu.md).
- [Compute open-shell spectra or phosphorescence](phosphorescence.md).
- [Assemble thermal torsions and coupled nuclear models](prepared-data.md).
- [Read complete and partial numerical output](numerical-results.md).
- [Compare broadened spectra with experiment](spectral-comparison.md).
