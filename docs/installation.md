# Installation

Install the electronic workflow on Linux or in Windows Subsystem for Linux (WSL). PySCF also provides macOS wheels, while the molecular CPU/GPU benchmarks here were run on Linux/WSL. Native Windows users should create and run the Python environment inside WSL. Python 3.10 or newer is required. [PySCF installation](https://pyscf.org/user/install.html)

## Install from the repository

Run these commands in a Linux, macOS or WSL shell. In Windows, open the WSL terminal first.

```bash
git clone https://github.com/HenriqueCSJ/PySCF_ESD.git
cd PySCF_ESD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[workflow]'
```

This installs the package, PySCF and SciPy. The checkout also contains the examples, molecular inputs, test suite and documentation. Install an editable checkout when developing the package.

```bash
python -m pip install -e '.[workflow]'
```

The optional dependency groups are defined in [pyproject.toml](https://github.com/HenriqueCSJ/PySCF_ESD/blob/main/pyproject.toml).

- The base package requires NumPy 1.24 or newer and provides the prepared harmonic and one-coordinate DVR solvers.
- `.[pyscf]` adds PySCF 2.15.0 or newer for electronic structure and PySCF adapters.
- `.[anharmonic]` adds SciPy 1.10 or newer for solvers that use SciPy.
- `.[workflow]` combines the PySCF and anharmonic dependencies for the CPU command-line workflow.

For the same installation from outside the checkout, use a Git URL. Keep a local checkout as well if you need the bundled examples.

```bash
python -m pip install 'pyscf-esd[workflow] @ git+https://github.com/HenriqueCSJ/PySCF_ESD.git'
```

## Verify the environment

Run the checks in the same activated environment that will run the calculation.

```bash
python -m pip check
python -c "import pyscf, scipy, numpy; print(pyscf.__version__, scipy.__version__, numpy.__version__)"
python -c "from pyscf import esd; import pyscf_esd; print(pyscf_esd.__file__); print(esd.run.__name__)"
pyscf-esd --help
```

Both imports refer to the installed package. `from pyscf import esd` exposes the PySCF extension interface and `import pyscf_esd` exposes the standalone solvers. `python -m pyscf_esd` runs the same command as `pyscf-esd`.

The following check runs only the nuclear solver and compares a prepared periodic rotor with its analytic solution.

```bash
python examples/validate_anharmonic_rotor.py
```

Its expected 0–0 energy is `0.120000000000` Hartree. It also compares the computed dipole strength with an independent analytic expression. Follow the [first calculation](quickstart.md) for an electronic run.

## Add GPU support

Install GPU4PySCF, CuPy and the compatible cuTENSOR package in this same environment using the CUDA-specific upstream instructions. GPU dependencies are installed separately from `.[workflow]`. The benchmark environment used PySCF 2.15.0 and GPU4PySCF 1.9.2. Check the upstream compatibility guidance when choosing newer versions. [GPU4PySCF installation](https://pyscf.org/user/gpu.html#installation)

The [GPU guide](gpu.md) covers analytic AH, VG, automatic anharmonic screening, C-PCM and the distinction between the high-level workflow and a prepared single-geometry calculation.

## Threads and reproducible environments

Choose the thread count before starting Python. This example assigns four OpenMP threads and one OpenBLAS thread.

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
pyscf-esd examples/water.xyz --model vg --basis '6-31+g*' --output water-vg.esd
```

When `OMP_NUM_THREADS` is unset, the CLI asks PySCF for at most four threads for up to 12 atoms and at most eight for larger molecules, bounded by available CPU count. Direct Python callers can use `pyscf.lib.num_threads`.

Record the environment together with the source revision and the run output.

```bash
python -m pip freeze > environment.txt
git rev-parse HEAD > source-revision.txt
```

A Python environment alone does not capture the CUDA driver, GPU model, thread settings, input geometry or external electronic archives. Retain those details for comparisons and benchmark replays.

## Upgrade an existing checkout

Use a clean checkout or resolve your local edits before updating.

```bash
git pull --ff-only
python -m pip install -e '.[workflow]'
python -m pip check
```

Keep completed calculation directories under their original version and use a new output directory for each run. The CLI requires a destination that does not already exist.

## Build the documentation locally

The documentation has its own small dependency set. A separate environment lets you build it without the electronic stack.

```bash
python3 -m venv .docs-venv
source .docs-venv/bin/activate
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The terminal prints the local preview address. Stop the server with Ctrl+C. See [contributing](contributing.md#documentation-changes) for the review and publication steps.
