"""Reassemble the retained CF3NO rotor benchmark at 298 K and 80 cm^-1 FWHM."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="PySCF_ESD calculation archive containing both CF3NO runs")
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("--method", choices=("pbe0-tda", "sf-tda"), default="pbe0-tda",
                        help="retained electronic model (default: pbe0-tda)")
    args = parser.parse_args(argv)

    if args.method == "pbe0-tda":
        script = args.archive / "CF3NO_Stable_298K_20261115/thermal_rotor_spectra.py"
        source = args.archive / "CF3NO_From_XYZ_20261115"
        command = ["--input", str(source), "--label", "PBE0/TDA unstable-reference diagnostic"]
        qualification = "PBE0/TDA uses the retained restricted reference with known external instability."
        if not source.is_dir():
            parser.error(f"missing retained PBE0/TDA input directory: {source}")
    else:
        script = Path(__file__).with_name("assemble_separable_rotor.py")
        source = args.archive / "CF3NO_SFTDA_Separ_298K/input_separable_298k.json"
        command = [str(source)]
        qualification = "SF-TDA uses auxiliary UKS spectator curvatures."
        if not source.is_file():
            parser.error(f"missing retained SF-TDA input: {source}")

    if not script.is_file():
        parser.error(f"missing spectral assembler: {script}")
    try:
        args.output.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"result destination already exists: {args.output}")
    print(qualification, file=sys.stderr, flush=True)
    return subprocess.run(
        [sys.executable, str(script), *command, "--output", str(args.output),
         "--temperature", "298", "--fwhm", "80"],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
