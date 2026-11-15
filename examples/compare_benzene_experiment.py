"""Compare a predicted benzene Fermi pair with gas-phase fluorescence.

Comparisons on the measured origin use calculated vibrational term values.
The raw photon positions and absolute electronic-structure error are kept
separately. The experimental points represent band maxima and a reading of
the relative peak heights, not an intensity-calibrated trace.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_h2co_analytic_ah import broaden


ORIGIN_CM = 38086.1
SIGMA_CM = 5.0
MEASURED_EMISSION = np.array([36475.0, 36494.0])
MEASURED_ABSORPTION = np.array([38606.098, 39529.636])


def pair_comparison(terms, strengths, theoretical_origin):
    """Summarize the projected dyad in ascending photon energy order."""
    from scipy.signal import find_peaks

    terms, strengths = np.asarray(terms, dtype=float), np.asarray(strengths, dtype=float)
    if terms.shape != (2,) or strengths.shape != (2,) or not np.all(np.isfinite([terms, strengths])):
        raise ValueError("expected two finite term values and dipole strengths")
    if np.any(strengths < 0) or strengths.max() <= 0:
        raise ValueError("the pair needs nonnegative strengths, with at least one positive component")
    order = np.argsort(-terms)
    terms, strengths = terms[order], strengths[order]
    centers = ORIGIN_CM - terms
    weights = strengths * (centers / ORIGIN_CM)**3
    axis = np.arange(centers[0]-6*SIGMA_CM, centers[-1]+6*SIGMA_CM, 0.02)
    curve = broaden(centers, weights, axis, SIGMA_CM)
    peaks, _ = find_peaks(curve, prominence=curve.max()*1e-4)
    ratio = float(curve[peaks[0]] / curve[peaks[1]]) if len(peaks) == 2 else None
    return {
        "ground_term_values_cm_inverse_in_photon_order": terms.tolist(),
        "centers_cm_inverse_on_measured_origin": centers.tolist(),
        "centers_cm_inverse_on_computed_origin": (theoretical_origin - terms).tolist(),
        "stick_errors_vs_measured_maxima_cm_inverse": (centers - MEASURED_EMISSION).tolist(),
        "stick_separation_cm_inverse": float(np.ptp(centers)),
        "separation_error_vs_measured_maxima_cm_inverse": float(np.ptp(centers)-np.ptp(MEASURED_EMISSION)),
        "dipole_strengths_au2": strengths.tolist(),
        "photon_rate_weights_common_scale": weights.tolist(),
        "photon_rate_area_ratio_low_over_high": float(weights[0]/weights[1]) if weights[1] else None,
        "broadened_peaks_cm_inverse": axis[peaks].tolist(),
        "broadened_height_ratio_low_over_high": ratio,
    }


def absorption_progression(process):
    """Assign ν6 and ν1+ν6 by their distinct vibrational projections."""
    from scipy.optimize import linear_sum_assignment

    columns = {name: i for i, name in enumerate(process["spectrum_column_order"])}
    spectrum = np.asarray(process["spectrum_all_retained_levels_in_increasing_photon_order"])
    projections = spectrum[:, [columns["projection_weight_nu6"], columns["projection_weight_nu1+nu6"]]].T
    row, column = linear_sum_assignment(-projections)
    offsets = spectrum[column, columns["offset_from_calculated_origin_cm_inverse"]]
    positions = ORIGIN_CM + offsets
    return {
        "band_order": ["nu6", "nu1+nu6"],
        "centers_cm_inverse_on_measured_origin": positions.tolist(),
        "centers_cm_inverse_on_computed_origin": spectrum[column, columns["photon_cm_inverse"]].tolist(),
        "errors_cm_inverse_after_origin_alignment": (positions - MEASURED_ABSORPTION).tolist(),
        "progression_separation_cm_inverse": float(positions[1] - positions[0]),
        "assignment_weights": projections[row, column].tolist(),
        "dipole_strengths_au2": spectrum[column, columns["dipole_strength_au2"]].tolist(),
    }


def build_comparison(model, archived, reference):
    measured = reference["experimental_maxima_cm-1"]
    np.testing.assert_array_equal(measured, MEASURED_EMISSION)
    origin = model["models"]["anharmonic"]["calculated_active_space_origin_cm_inverse"]
    cases = {}
    absorption = {}
    for label in ("anharmonic", "harmonic", "without_cubic_fermi"):
        mode = model["models"][label]
        emission = mode["emission"]
        lines = emission["lines_in_increasing_photon_order"]
        cases[label] = pair_comparison([line["vibrational_energy_cm_inverse"] for line in lines],
                                      [line["dipole_strength_au2"] for line in lines],
                                      mode["calculated_active_space_origin_cm_inverse"])
        cases[label]["dyad_assignment_weights_nu8_and_nu1_plus_nu6"] = [line["projection_weights_in_dyad_order"] for line in lines]
        cases[label]["effective_vibrational_block"] = emission["effective_block"]
        effective = np.asarray(emission["effective_block"]["hamiltonian_relative_to_vibrational_ground_cm_inverse"])
        cases[label]["effective_detuning_nu8_minus_combination_cm_inverse"] = float(effective[0,0]-effective[1,1])
        cases[label]["effective_splitting_at_zero_detuning_cm_inverse"] = float(2*abs(effective[0,1]))

        process = mode["absorption"]
        if "spectrum_column_order" in process:
            absorption[label] = absorption_progression(process)
    fit = archived["cases"]["mu8_plus"]
    return {
        "experimental": {
            "source_doi": "10.1021/ja00841a001",
            "fluorescence_maxima_cm_inverse": MEASURED_EMISSION.tolist(),
            "fluorescence_band_maximum_separation_cm_inverse": float(np.ptp(MEASURED_EMISSION)),
            "well_defined_band_maximum_accuracy_cm_inverse": 1.5,
            "reported_instrument_bandpass_cm_inverse": 4,
            "estimated_peak_height_ratio_low_over_high": reference["raw_height_ratio_low_over_high"],
            "peak_height_ratio_pixel_reading_range": reference["height_ratio_range_pixel_sensitivity_only"],
            "gas_phase_absorption_doi": "10.1063/1.1767515",
            "gas_phase_absorption_centers_cm_inverse": MEASURED_ABSORPTION.tolist(),
            "gas_phase_absorption_progression_separation_cm_inverse": float(np.ptp(MEASURED_ABSORPTION)),
            "reference_origin_cm_inverse": ORIGIN_CM,
        },
        "computed_active_mode_origin_cm_inverse": origin,
        "computed_active_mode_origin_error_cm_inverse": origin - ORIGIN_CM,
        "sigma_cm_inverse": SIGMA_CM,
        "emission_cases": cases,
        "absorption_cases": absorption,
        "archived_raman_parameterized_control": {
            "centers_cm_inverse_on_measured_origin": fit["photon_centers_cm-1"],
            "photon_rate_weights_common_scale": fit["photon_count_areas_relative_to_harmonic"],
            "broadened_height_ratio_low_over_high": fit["broadened_height_ratio_low_over_high"],
            "splitting_input_cm_inverse": float(np.ptp(fit["photon_centers_cm-1"])),
            "phase_selected_using_fluorescence": True,
        },
    }


def plot_comparison(comparison, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    model = comparison["emission_cases"]
    experimental = comparison["experimental"]
    reference = comparison["archived_raman_parameterized_control"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.65), layout="constrained")
    labels = {"harmonic": ("Controle quadrático · PySCF", "#716a76", "--"),
              "anharmonic": ("Anarmônico 3D · PySCF", "#17637f", "-")}
    display = {label: model[label] for label in labels if label in model}
    edge = min(min(item["centers_cm_inverse_on_measured_origin"]) for item in display.values())
    upper = max(max(item["centers_cm_inverse_on_measured_origin"]) for item in display.values())
    overview = (min(edge-28, 36445), max(upper+28, 36520))
    step = 0.05
    x = np.arange(overview[0], overview[1]+step, step)
    curves = {}
    for label, numbers in {**display, "reference": reference}.items():
        curve = broaden(np.asarray(numbers["centers_cm_inverse_on_measured_origin"]),
                        np.asarray(numbers["photon_rate_weights_common_scale"]), x, SIGMA_CM)
        curves[label] = curve / curve.max()

    observed = experimental["estimated_peak_height_ratio_low_over_high"]
    lower, upper = experimental["peak_height_ratio_pixel_reading_range"]
    for ax, xlim in zip(axes, (overview, (36442, 36522))):
        for name, (label, color, line) in labels.items():
            if name in curves:
                ax.plot(x, curves[name], color=color, ls=line, lw=1.65, label=label)
        ax.plot(x, curves["reference"], color="#895f35", ls=":", lw=1.65, label="Raman · fase escolhida na fluorescência")
        ax.errorbar(MEASURED_EMISSION, [1., 1./observed], xerr=experimental["well_defined_band_maximum_accuracy_cm_inverse"],
                    yerr=[[0, 1./observed-1./upper], [0, 1./lower-1./observed]],
                    fmt="o", color="#292730", ms=4.5, capsize=3, zorder=4, label="Máximos experimentais")
        ax.set(xlim=xlim, ylim=(0,1.16), yticks=(0,.5,1.))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", ".")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e8ebee", lw=.7)
    axes[0].set_title("Dubleto de Fermi · posições previstas e medidas", loc="left", fontsize=11.5, pad=8)
    axes[0].set_title("B3LYP/TDA · 6-31G(d)", loc="right", fontsize=9, pad=8)
    axes[0].set_ylim(0, 1.40)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    axes[1].set_title("Janela do experimento", loc="left", fontsize=11.5, pad=8)
    axes[1].set_title("Gaussiana σ = 5 cm⁻¹ · máximo de cada modelo = 1", loc="right", fontsize=9, pad=8)
    fig.suptitle("Benzeno  |  Fluorescência do nível S₁(v = 0)", fontsize=15, x=.045, ha="left")
    fig.supxlabel("Número de onda (cm⁻¹) · origem experimental comum")
    fig.supylabel("Intensidade relativa")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"benzene_fermi_experiment.{suffix}", dpi=230, facecolor="white")
    plt.close(fig)


def plot_absorption(comparison, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    cases = comparison["absorption_cases"]
    if not cases:
        return
    extents = np.concatenate((MEASURED_ABSORPTION, *(cases[key]["centers_cm_inverse_on_measured_origin"]
                                                    for key in ("anharmonic", "harmonic"))))
    x = np.arange(extents.min()-45, extents.max()+45, 0.08)
    fig, ax = plt.subplots(figsize=(8.6, 3.55), layout="constrained")
    for key, label, color, line in (("anharmonic", "Anarmônico 3D · PySCF", "#17637f", "-"),
                                   ("harmonic", "Controle quadrático · PySCF", "#716a76", "--")):
        case = cases[key]
        positions = np.asarray(case["centers_cm_inverse_on_measured_origin"])
        profile = broaden(positions, np.asarray(case["dipole_strengths_au2"]) * positions/ORIGIN_CM, x, SIGMA_CM)
        if profile.max() <= 0:
            continue
        ax.plot(x, profile/profile.max(), color=color, ls=line, lw=1.55, label=label)
    ax.vlines(MEASURED_ABSORPTION, -.075, 0, color="#292730", lw=2, label="Centros de banda medidos")
    ax.set(xlim=(x[0],x[-1]), ylim=(-.09,1.15), yticks=(0,.5,1),
           xlabel="Número de onda (cm⁻¹) · origem experimental comum",
           ylabel="Intensidade relativa\n(máximo de cada modelo = 1)")
    ax.set_title("Benzeno  |  Absorção da progressão ν₆, ν₁+ν₆", loc="left", fontsize=14.5, pad=12)
    ax.set_title("B3LYP/TDA · 6-31G(d) · σ = 5 cm⁻¹", loc="right", fontsize=8.5, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", ".")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper center", frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e8ebee", lw=.7)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"benzene_absorption_experiment.{suffix}", dpi=230, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="theoretical report from validate_benzene_3d.py")
    parser.add_argument("reference", type=Path, help="archived benzene direct-HT reference directory")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    model = json.loads(opts.model.read_text(encoding="utf-8"))
    archived = json.loads((opts.reference / "benchmark.json").read_text(encoding="utf-8"))
    reference = json.loads((opts.reference / "reference_heights.json").read_text(encoding="utf-8"))
    comparison = build_comparison(model, archived, reference)
    (opts.output / "benzene_fermi_experiment.json").write_text(json.dumps(comparison, indent=2)+"\n", encoding="utf-8")
    plot_comparison(comparison, opts.output)
    plot_absorption(comparison, opts.output)
