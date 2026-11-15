"""Build and plot the local harmonic H2CO AH benchmark from PySCF data."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np


CM_PER_HARTREE = 219474.6313632
SIGMA_CM = 50.0
SUBSPACE_TOLERANCE = 0.05
PROCESSES = ("absorption", "emission")
MODELS = ("duschinsky", "matched_modes")


def matched_mode_control(model):
    """Remove mixing while retaining signed mode correspondence and displacement."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-np.abs(model.rotation))
    permutation = np.zeros_like(model.rotation)
    permutation[rows, cols] = np.copysign(1.0, model.rotation[rows, cols])
    # Both models send the same S0-coordinate point to the S1 minimum.
    shift = permutation @ (model.rotation.T @ model.shift_au)
    return replace(model, rotation=permutation, shift_au=shift)


def broaden(frequencies, weights, axis, sigma):
    density = np.zeros_like(axis)
    normal = 1 / (np.sqrt(2 * np.pi) * sigma)
    lower = np.searchsorted(axis, frequencies - 8 * sigma)
    upper = np.searchsorted(axis, frequencies + 8 * sigma)
    for energy, weight, lo, hi in zip(frequencies, weights, lower, upper):
        if hi > lo and weight:
            density[lo:hi] += (weight * normal) * np.exp(-0.5 * ((axis[lo:hi] - energy) / sigma)**2)
    return density


def spectra(base, control, dipole, derivatives):
    for cutoff in (12, 16, 20):
        print(f"Building four AH spectra with up to {cutoff} total final quanta", flush=True)
        result = {
            (process, label): model.spectrum(
                transition_dipole_au=dipole, dipole_derivatives_au=derivatives, final_max_quanta=cutoff,
                process=process, max_overlaps=2_000_000,
            )
            for process in PROCESSES
            for label, model in zip(MODELS, (base, control))
        }
        if min(value.final_dipole_strength_coverage for value in result.values()) >= 0.999:
            return cutoff, result
    raise RuntimeError("the four spectra did not reach 99.9% vibrational basis coverage")


def assemble(reference, output):
    import pyscf
    from pyscf import gto, lib, scf
    from pyscf_esd.cis_hessian import analytic_cis_transition_dipole
    from pyscf_esd.pyscf_adapter import adiabatic_hessian

    lib.num_threads(2)
    with np.load(reference / "electronic_results.npz") as archive:
        data = dict(archive)
    endpoints = json.loads((reference / "endpoint_summary.json").read_text(encoding="utf-8"))
    basis = endpoints["basis"]
    s0, s1 = [gto.M(atom=list(zip(data["symbols"], data[f"geometry_{state}_bohr"])),
                    basis=basis, unit="Bohr", verbose=0) for state in ("s0", "s1")]
    mf = scf.RHF(s0)
    mf.mo_coeff = data["mo_coeff_s0"]
    mf.mo_occ = data["mo_occ_s0"]
    mf.mo_energy = data["mo_energy_s0"]
    mf.e_tot = float(data["ground_energy_hartree"])
    mf.converged = True
    ht = analytic_cis_transition_dipole(mf, data["tda_amplitude_s0"])
    if ht.root_index != 0:
        raise RuntimeError("the planar dipole derivative did not follow S1")
    np.testing.assert_allclose(ht.dipole, data["dipole_s0"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ht.derivative.sum(axis=0), 0, rtol=0, atol=1e-9)
    rigid_rotations = np.cross(np.eye(3)[:, None, :], s0.atom_coords())
    np.testing.assert_allclose(
        np.einsum("rax,axd->rd", rigid_rotations, ht.derivative), np.cross(np.eye(3), ht.dipole),
        rtol=0, atol=1e-9,
    )
    if np.linalg.norm(ht.dipole) > 1e-6:
        raise RuntimeError("unexpected nonzero length dipole at the symmetry-forbidden S0 geometry")

    model = adiabatic_hessian(
        s0, s1, data["hessian_s0"], data["hessian_s1"],
        ground_energy_hartree=float(data["ground_energy_hartree"]), excited_energy_hartree=float(data["excited_energy_hartree"]),
        ground_gradient=data["gradient_s0"], excited_gradient=data["gradient_s1"],
        masses_amu=data["masses_amu"], subspace_tolerance=SUBSPACE_TOLERANCE,
    )
    control = matched_mode_control(model)
    np.testing.assert_allclose(control.rotation.T @ control.shift_au, model.rotation.T @ model.shift_au, atol=1e-12, rtol=0)
    dipole, derivatives = ht.dipole, np.einsum("qax,axd->qd", model.ground_cartesian_mode_derivatives_au, ht.derivative)
    native_excited = data["dipole_s1"] @ model.excited_alignment_rotation
    approximate_excited = dipole - model.rotation.T @ model.shift_au @ derivatives
    sum_rule = float(dipole @ dipole + np.sum(derivatives**2 / (2 * model.ground_frequencies_hartree[:, None])))
    if sum_rule < 1e-8:
        raise RuntimeError("no appreciable Herzberg–Teller intensity found")

    cutoff, calculated = spectra(model, control, dipole, derivatives)
    gap = model.gap_00_hartree
    report = {
        "method": f"H2CO RHF–TDA/CIS/{basis}; one S1 well; harmonic AH with linear HT at S0",
        "pyscf_version": pyscf.__version__, "basis": basis, "temperature_kelvin": 0, "sigma_cm_inverse": SIGMA_CM,
        "masses_amu": data["masses_amu"].tolist(), "origin_cm_inverse": gap * CM_PER_HARTREE,
        "frequencies_s0_cm_inverse": (model.ground_frequencies_hartree * CM_PER_HARTREE).tolist(),
        "frequencies_s1_cm_inverse": (model.excited_frequencies_hartree * CM_PER_HARTREE).tolist(),
        "subspace_singular_values": model.subspace_singular_values.tolist(),
        "subspace_tolerance": SUBSPACE_TOLERANCE,
        "rotation_correction_norm": model.rotation_correction_norm,
        "max_rotation_element_change_in_control": float(np.max(np.abs(model.rotation - control.rotation))),
        "duschinsky": model.rotation.tolist(), "matched_mode_control": control.rotation.tolist(),
        "shift_au": model.shift_au.tolist(), "control_shift_au": control.shift_au.tolist(),
        "same_s0_to_s1_minimum_displacement_in_s0_modes": (-model.rotation.T @ model.shift_au).tolist(),
        "displacement_norm_au": model.displacement_norm_au,
        "displacement_projection_residual_au": model.displacement_projection_residual_au,
        "dipole_s0_au": dipole.tolist(), "dipole_s1_native_au_in_s0_frame": native_excited.tolist(),
        "dipole_s1_linear_ht_au_in_s0_frame": approximate_excited.tolist(),
        "dipole_response_residual": ht.max_response_residual,
        "dipole_response_root_overlap": ht.overlap,
        "dipole_response_root_gap_hartree": ht.separation_hartree,
        "dipole_derivatives_in_s0_modes_au": derivatives.tolist(),
        "condon_total_strength_au2": float(dipole @ dipole), "absorption_ht_sum_rule_au2": sum_rule,
        "final_quanta_cutoff": cutoff, "intensity_weighting": "dipole strength times (E/E00) for absorption and (E/E00)^3 for emission",
        "series": {}, "process": {},
    }
    diagnostics = reference / "dipole_audit.json"
    if diagnostics.is_file():
        displaced = json.loads(diagnostics.read_text(encoding="utf-8"))
        exact = np.einsum("ax,axd->d", displaced["direction"], ht.derivative)
        reference_steps = [displaced[key] for key in ("coarse", "fine")]
        derivatives_fd = np.array([item["derivative_au_per_bohr"] for item in reference_steps])
        errors = np.max(np.abs(derivatives_fd - exact), axis=1)
        if errors[1] > 1e-6 or errors[1] >= errors[0]:
            raise RuntimeError("pyramidalization dipole derivative failed the tracked displacement check")
        report["dipole_derivative_reference"] = {
            "steps_bohr": [item["step_bohr"] for item in reference_steps],
            "analytic_e": exact.tolist(), "fd_e": derivatives_fd.tolist(), "max_abs_errors_e": errors.tolist(),
        }

    sticks, curves = {}, {}
    for process, power in zip(PROCESSES, (1, 3)):
        records = {}
        for label in MODELS:
            value = calculated[process, label]
            frequencies = value.photon_energy_hartree * CM_PER_HARTREE
            strengths = value.thermal_strength_au2
            weights = strengths * (value.photon_energy_hartree / gap)**power
            name = f"{process}_{label}"
            sticks[f"{name}_quanta"] = value.final_quanta
            sticks[f"{name}_values"] = np.column_stack((frequencies, strengths, weights))
            report["series"][name] = {
                "basis_coverage": value.final_dipole_strength_coverage, "line_count": frequencies.size,
                "positive_photon_dipole_strength_au2": float(strengths.sum()),
                "weighted_strength_au2": float(weights.sum()),
            }
            records[label] = frequencies, weights
        if process == "absorption":
            for label in MODELS:
                ratio = report["series"][f"{process}_{label}"]["positive_photon_dipole_strength_au2"] / sum_rule
                if not 0.999 <= ratio <= 1.00000001:
                    raise RuntimeError("absorption violated the linear-dipole closure sum")

        # An exact normalized Gaussian, sampled in a joint display window.
        band_edges = []
        for frequencies, weights in records.values():
            band = frequencies[weights >= weights.max() * 0.002]
            band_edges.extend((band.min(), band.max()))
        axis = np.arange(min(band_edges) - 300, max(band_edges) + 302, 2.0)
        curves[f"{process}_axis"] = axis
        for label, (frequencies, weights) in records.items():
            curves[f"{process}_{label}"] = broaden(frequencies, weights, axis, SIGMA_CM)
        on, off = (curves[f"{process}_{label}"] for label in MODELS)
        report["process"][process] = {
            "max_on_cm_inverse": float(axis[np.argmax(on)]), "max_off_cm_inverse": float(axis[np.argmax(off)]),
            "relative_l1_change_in_plot_window": float(np.sum(np.abs(on - off)) / np.sum(on)),
            "fraction_weight_shown_with_J": float(2 * np.sum(on) / records[MODELS[0]][1].sum()),
            "fraction_weight_shown_without_J": float(2 * np.sum(off) / records[MODELS[1]][1].sum()),
        }

    np.savez_compressed(output / "h2co_ah_sticks.npz", **sticks)
    np.savez_compressed(output / "h2co_ah_curves.npz", **curves)
    (output / "h2co_ah_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def plot(output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    report = json.loads((output / "h2co_ah_report.json").read_text(encoding="utf-8"))
    with np.load(output / "h2co_ah_curves.npz") as archive:
        data = dict(archive)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.1), layout="constrained")
    color_on, color_off = "#1c6586", "#9a472b"
    for ax, process, label in zip(axes, PROCESSES, ("Absorção", "Emissão")):
        axis = data[f"{process}_axis"]
        on, off = (data[f"{process}_{mode}"] for mode in MODELS)
        common = max(on.max(), off.max())
        ax.plot(axis, on / common, color=color_on, lw=1.45, label="Com Duschinsky")
        ax.plot(axis, off / common, color=color_off, lw=1.25, ls=(0, (4, 2.5)), label="Sem mistura")
        origin = report["origin_cm_inverse"]
        if axis[0] < origin < axis[-1]:
            ax.axvline(origin, color="#747f89", lw=0.8, ls=":", zorder=0)
            ax.text(origin, 1.03, "0–0", fontsize=8.5, ha="center", va="bottom", color="#4c5660",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.set(xlim=(axis[0], axis[-1]), ylim=(0, 1.13), yticks=(0, 0.5, 1.0))
        ax.set_title(label, loc="left", fontsize=12, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ",")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e8ebee", lw=0.7, zorder=0)
    fig.suptitle("Formaldeído  |  AH local de um poço de S₁", fontsize=15, x=0.04, ha="left")
    fig.supxlabel("Número de onda (cm⁻¹)")
    fig.supylabel("Intensidade relativa")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_title(f"RHF–CIS/6-31G(d) · HT linear em S₀ · 0 K · σ = {report['sigma_cm_inverse']:g} cm⁻¹",
                      loc="right", fontsize=9, pad=8)
    fig.savefig(output / "h2co_ah_duschinsky.png", dpi=230, facecolor="white")
    fig.savefig(output / "h2co_ah_duschinsky.pdf", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="plot the saved AH curves without electronic calculation")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.plot:
        plot(arguments.output)
    else:
        assemble(arguments.reference, arguments.output)
