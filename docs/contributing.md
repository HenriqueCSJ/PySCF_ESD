# Development and reproducibility

The repository contains the electronic adapters, nuclear solvers, tests, benchmark drivers, documentation sources and the methods manuscript. Changes should preserve the supported public input conventions and the physical meaning of reported outputs. Use a molecule as a regression or transfer test when changing a general algorithm.

## Set up a development environment

Clone the repository and install an editable environment as described in [installation](installation.md).

```bash
python -m pip install -e '.[workflow]'
python -m pip check
```

Optional GPU tests need a compatible GPU4PySCF stack. Some archived molecular replays also need electronic data stored outside the public repository. Check the individual script's `--help` and the [validation guide](validation.md) before running it.

## Run relevant tests

The suite uses Python's `unittest`. From the repository root, run a focused module after editing the associated code.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_harmonic.py' -v
```

The full suite includes electronic integration checks when their dependencies are available. Review dependency-based skips in the output.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

The synthetic rotor control is a small independent solver check.

```bash
python examples/validate_anharmonic_rotor.py
```

Describe whether a validation used new electronic calculations, retained electronic data, synthetic inputs or controlled test doubles. The [benchmark records](validation.md) identify the source of each reported comparison.

## Preserve physical input contracts

Keep coordinate units, state indices, atom order, isotope masses, Hessian layout and normal-mode signs explicit. Track state identity independently of root number and retain signed amplitudes through coherent operations. Verify the same electronic model at both endpoints of a prepared AH calculation.

Check observable convergence at the energy powers used by the requested property. A basis change must preserve partial-result diagnostics and maintain the distinction between coverage reached and a retained numerical estimate. Rate calculations use native photon energies. Experimental alignment and broadening are recorded in the comparison layer.

General changes should be tested on inputs with different dimensions, temperatures or phase patterns when those differences exercise the affected behavior. Regression tests should reproduce a specific failure and verify the physical or API contract it violated.

## Documentation changes

Documentation is written in Markdown under `docs/`. Navigation lives in `mkdocs.yml` and the hosted environment in `.readthedocs.yaml`. API examples should name the entry point, required prepared inputs and units. A snippet containing an electronic run should make that cost apparent to the reader.

Use a docs-only environment and run the strict build.

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

Keep existing page and section anchors working when reorganizing a guide. Link to repository files with an absolute GitHub URL when the target is outside `docs/`. Figures, downloadable benchmark data and the manuscript PDF are static assets copied into the documentation build.

The public site is hosted on [Read the Docs](https://pyscf-esd.readthedocs.io/en/latest/). The project was imported from its public GitHub URL and currently uses manually triggered builds. Maintainers can select **Builds** in the [project dashboard](https://app.readthedocs.org/projects/pyscf-esd/) and trigger `latest` after pushing an update. Confirm the built Git commit and the live pages before announcing a documentation release.

## Submit a change

Open an issue or pull request describing the user-visible problem, the implementation, supported scope and validation. Include a small reproducer when possible. Source code and examples use the GPL-3.0-only license distributed with the repository. Check the permissions attached to external spectra and digitized experimental data before redistributing them. [Repository](https://github.com/HenriqueCSJ/PySCF_ESD) · [License](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/LICENSE)
