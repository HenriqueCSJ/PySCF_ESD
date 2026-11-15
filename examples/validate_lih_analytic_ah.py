"""Validate analytic CIS curvature and assemble AH from a native LiH reference.

The reference archive supplies both PySCF minima and the saved RHF orbitals.
Displaced gradients in the archive are used only to check the analytic result.
With --plot, render the saved sticks without importing PySCF.
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np


CM_PER_HARTREE = 219474.6313632
SIGMA_CM = 50.0
REFERENCE = "lih_rhf_cis_631g_ah_reference"


def validate(reference, output):
    import pyscf
    from pyscf import gto, lib, scf
    from pyscf_esd.cis_hessian import analytic_cis_hessian
    from pyscf_esd.pyscf_adapter import adiabatic_hessian

    lib.num_threads(2)
    reported = json.loads((reference / f"{REFERENCE}.json").read_text(encoding="utf-8"))
    with np.load(reference / f"{REFERENCE}.npz") as archive:
        data = dict(archive)
    s0 = gto.M(atom=list(zip(("Li", "H"), data["s0_coords_bohr"])), unit="Bohr", basis="6-31g")
    s1 = gto.M(atom=list(zip(("Li", "H"), data["s1_coords_bohr"])), unit="Bohr", basis="6-31g")
    ground = reported["states"]["at_s0_minimum"]
    excited = reported["states"]["at_s1_minimum"]
    mf = scf.RHF(s1)
    mf.mo_coeff = data["s1_mo_coeff"]
    mf.mo_energy = data["s1_mo_energy"]
    mf.mo_occ = data["s1_mo_occ"]
    mf.e_tot = excited["hf_energy_hartree"]
    mf.converged = True
    started = perf_counter()
    result = analytic_cis_hessian(mf, data["s1_tda_amplitude"])
    elapsed = perf_counter() - started

    hessian = result.hessian.transpose(0, 2, 1, 3).reshape(6, 6)
    coarse = data["s1_hessian_fd_coarse"].reshape(6, 6)
    fine = data["s1_hessian_fd_fine"].reshape(6, 6)
    extrapolated = (4 * fine - coarse) / 3
    translation = np.tile(np.eye(3), (2, 1))
    rotation = np.cross(np.eye(3)[:, None, :], data["s1_coords_bohr"]).reshape(3, 6)
    rotation_rhs = np.cross(np.eye(3)[:, None, :], result.gradient).reshape(3, 6)
    bond_direction = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.5])
    errors = {
        "energy_hartree": abs(result.total_energy_hartree - excited["s1_total_energy_hartree"]),
        "gradient_hartree_per_bohr": float(np.max(np.abs(result.gradient - data["s1_gradient"]))),
        "hessian_coarse_hartree_per_bohr2": float(np.max(np.abs(hessian - coarse))),
        "hessian_fine_hartree_per_bohr2": float(np.max(np.abs(hessian - fine))),
        "hessian_extrapolated_hartree_per_bohr2": float(np.max(np.abs(hessian - extrapolated))),
        "translation_hartree_per_bohr2": float(np.max(np.abs(hessian @ translation))),
        "rotation_covariance_hartree_per_bohr": float(np.max(np.abs(rotation @ hessian - rotation_rhs))),
    }
    limits = {
        "energy_hartree": 1e-10,
        "gradient_hartree_per_bohr": 1e-9,
        "hessian_fine_hartree_per_bohr2": 1e-7,
        "hessian_extrapolated_hartree_per_bohr2": 1e-8,
        "translation_hartree_per_bohr2": 1e-9,
        "rotation_covariance_hartree_per_bohr": 1e-9,
    }
    for key, tolerance in limits.items():
        if errors[key] > tolerance:
            raise RuntimeError(f"{key} = {errors[key]:.3g} exceeds {tolerance:.3g}")
    if result.root_index != 0 or errors["hessian_coarse_hartree_per_bohr2"] <= errors["hessian_fine_hartree_per_bohr2"]:
        raise RuntimeError("expected the tracked S1 root and a tighter Hessian check upon step refinement")

    masses = s0.atom_mass_list(isotope_avg=True)
    ah = adiabatic_hessian(
        s0, s1, data["s0_hessian"].transpose(0, 2, 1, 3), result.hessian,
        ground_energy_hartree=ground["hf_energy_hartree"],
        excited_energy_hartree=result.total_energy_hartree,
        ground_gradient=data["s0_gradient"], excited_gradient=result.gradient, masses_amu=masses,
    )
    dipole = ground["transition_dipole_au"]
    rows, maxima, origins, basis_coverage, positive_photon_coverage = [], {}, {}, {}, {}
    for process, power in (("absorption", 1), ("emission", 3)):
        spectrum = ah.spectrum(transition_dipole_au=dipole, final_max_quanta=40, process=process)
        photon = spectrum.photon_energy_hartree * CM_PER_HARTREE
        weight = spectrum.thermal_strength_au2 * spectrum.photon_energy_hartree**power
        maximum = int(np.argmax(weight))
        normalized = weight / weight[maximum]
        maxima[process] = {"cm_inverse": float(photon[maximum]), "final_quanta": int(spectrum.final_quanta[maximum, 0])}
        origins[process] = float(spectrum.dipole_strength_au2[spectrum.final_quanta[:, 0] == 0][0])
        basis_coverage[process] = spectrum.final_dipole_strength_coverage
        positive_photon_coverage[process] = float(spectrum.dipole_strength_au2.sum() / np.dot(dipole, dipole))
        if basis_coverage[process] is None or basis_coverage[process] < 0.999999:
            raise RuntimeError(f"incomplete {process} dipole-strength sum")
        for i, cm in enumerate(photon):
            rows.append((process, int(spectrum.initial_quanta[i, 0]), int(spectrum.final_quanta[i, 0]),
                         float(cm), float(spectrum.dipole_strength_au2[i]), float(normalized[i])))
    np.testing.assert_allclose(origins["absorption"], origins["emission"], rtol=1e-10, atol=0)

    report = {
        "method": "LiH RHF/singlet TDA (CIS)/6-31G; harmonic AH Franck-Condon at 0 K",
        "pyscf_version": pyscf.__version__,
        "reference_pyscf_version": reported["settings"]["pyscf_version"],
        "root_index_zero_based": result.root_index,
        "s1_overlap_at_analytic_hessian": result.overlap,
        "s1_gap_to_nearest_root_hartree": result.separation_hartree,
        "max_orbital_response_residual": result.max_response_residual,
        "root_tracking_across_geometries": reported["root_tracking_counts"],
        "analytic_hessian_elapsed_seconds_two_threads": elapsed,
        "excitation_at_s1_minimum_hartree": result.excitation_hartree,
        "s1_total_energy_hartree": result.total_energy_hartree,
        "s1_bond_curvature_hartree_per_bohr2": float(bond_direction @ hessian @ bond_direction),
        "native_refs": reported["curvature_comparison"],
        "errors": errors, "acceptance_limits": limits,
        "s0_bond_angstrom": ground["bond_length_bohr"] * 0.529177210903,
        "s1_bond_angstrom": excited["bond_length_bohr"] * 0.529177210903,
        "isotope_averaged_masses_amu": masses.tolist(),
        "s0_frequency_cm_inverse": float(ah.ground_frequencies_hartree[0] * CM_PER_HARTREE),
        "s1_frequency_cm_inverse": float(ah.excited_frequencies_hartree[0] * CM_PER_HARTREE),
        "origin_cm_inverse": float(ah.gap_00_hartree * CM_PER_HARTREE),
        "condon_dipole_at_s0_au": dipole,
        "duschinsky_rotation": ah.rotation.tolist(),
        "displacement_au": ah.shift_au.tolist(),
        "mode_subspace_singular_values": ah.subspace_singular_values.tolist(),
        "basis_dipole_strength_coverage": basis_coverage,
        "positive_photon_dipole_strength_coverage": positive_photon_coverage,
        "intensity_weighting": {
            "absorption": "photon energy times dipole strength",
            "emission": "photon energy cubed times dipole strength",
            "normalization": "each channel divided by its own strongest line",
        },
        "origin_dipole_strength_au2": origins,
        "maxima": maxima, "stokes_separation_cm_inverse": maxima["absorption"]["cm_inverse"] - maxima["emission"]["cm_inverse"],
        "plot_gaussian_sigma_cm_inverse": SIGMA_CM,
    }
    np.savez(output / "lih_analytic_ah.npz", hessian_s1=result.hessian, gradient_s1=result.gradient)
    (output / "lih_analytic_ah.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "lih_ah_sticks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("process", "initial_v", "final_v", "wavenumber_cm_inverse", "dipole_strength_au2", "relative_intensity"))
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


def plot(output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    report = json.loads((output / "lih_analytic_ah.json").read_text(encoding="utf-8"))
    with (output / "lih_ah_sticks.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.1, 4.6), layout="constrained")
    xmin, xmax = 16800, 37500
    axis = np.linspace(xmin, xmax, 7500)
    origin = report["origin_cm_inverse"]
    sigma = report["plot_gaussian_sigma_cm_inverse"]
    series = (("absorption", "Absorção", "#17698c"), ("emission", "Emissão", "#a04427"))
    for ax, (process, label, color) in zip(axes, series):
        selected = [row for row in rows if row["process"] == process]
        frequency = np.array([float(row["wavenumber_cm_inverse"]) for row in selected])
        strength = np.array([float(row["relative_intensity"]) for row in selected])
        visible = (frequency >= xmin) & (frequency <= xmax)
        curve = np.exp(-0.5 * ((axis[:, None] - frequency) / sigma)**2) @ strength
        ax.plot(axis, curve, color=color, lw=1.2)
        ax.vlines(frequency[visible], 0, strength[visible], color=color, lw=0.85, alpha=0.55)
        ax.axvline(origin, color="#555d63", lw=0.85, ls=(0, (3, 3)), zorder=0)
        ax.text(0.018, 0.87, label, transform=ax.transAxes, color=color, fontsize=10, weight="medium")
        ax.set(xlim=(xmin, xmax), ylim=(0, 1.14), yticks=(0, 0.5, 1.0))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ",")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e6e8eb", linewidth=0.55, zorder=-1)
    axes[0].text(origin, 1.04, "0–0", color="#40474d", ha="center", va="bottom", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    axes[0].set_title("LiH  |  Espectro vibrônico AH", loc="left", fontsize=15, pad=13)
    axes[0].set_title(f"RHF–CIS/6-31G · 0 K · σ = {sigma:g} cm⁻¹", loc="right", fontsize=9, pad=13)
    axes[-1].set_xlabel("Número de onda (cm⁻¹)")
    axes[-1].xaxis.set_major_locator(MultipleLocator(2500))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
    fig.supylabel("Intensidade relativa")
    fig.savefig(output / "lih_ah_spectrum.png", dpi=230)
    fig.savefig(output / "lih_ah_spectrum.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true", help="plot saved sticks instead of recalculating")
    args = parser.parse_args()
    output = args.output or args.reference
    output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot(output)
    else:
        validate(args.reference, output)
