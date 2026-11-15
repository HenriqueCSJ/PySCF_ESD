"""Compare a revised benzene absorption model with the gas-phase progression."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_benzene_experiment import (
    MEASURED_ABSORPTION,
    ORIGIN_CM,
    SIGMA_CM,
    absorption_progression,
)
from validate_h2co_analytic_ah import broaden


def plot_absorption(cases, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    measured = MEASURED_ABSORPTION
    positions = np.array([case["centers_cm_inverse_on_measured_origin"] for _, case in cases])
    error = positions - measured
    before = min(-20, float(error.min()) - 5 * SIGMA_CM)
    after = max(20, float(error.max()) + 5 * SIGMA_CM)
    axes_grids = [np.arange(center + before, center + after, .05) for center in measured]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65), sharey=True, facecolor="white")
    fig.subplots_adjust(left=.095, right=.98, bottom=.19, top=.68, wspace=.15)

    handles = []
    for (label, case), centers, color, linestyle in zip(
        cases, positions, ("#77717e", "#17637f", "#946840"), ("--", "-", "-."),
    ):
        satellites = case.get("borrowed_intensity_lines", [])
        centers = np.r_[centers, [row["center_cm_inverse"] for row in satellites]]
        strengths = np.r_[case["dipole_strengths_au2"], [row["dipole_strength_au2"] for row in satellites]]
        weights = strengths * centers / ORIGIN_CM
        profiles = [broaden(centers, weights, axis, SIGMA_CM) for axis in axes_grids]
        scale = max(profile.max() for profile in profiles)
        for ax, grid, profile in zip(axes, axes_grids, profiles):
            ax.plot(grid, profile / scale, color=color, ls=linestyle, lw=2)
            for line in satellites:
                center = line["center_cm_inverse"]
                if not grid[0] + 5 < center < grid[-1] - 5:
                    continue
                character = line.get("excited_spectator_channel_probabilities_by_group", {})
                dominant = max(character, key=character.get) if character else ""
                mode = dominant.removeprefix("nu").split("_")[0]
                label = "Satélite"
                if mode.isdecimal():
                    label += " ν" + mode.translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
                ax.annotate(label, (center, np.interp(center, grid, profile) / scale),
                            xytext=(0, 14), textcoords="offset points", ha="center", color=color,
                            fontsize=8.5, zorder=5, arrowprops={"arrowstyle": "-", "color": color, "lw": .75})
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=2, label=label))

    for ax, grid, center, label in zip(axes, axes_grids, measured, ("ν₆", "ν₁ + ν₆")):
        ax.axvline(center, color="#292730", ls=":", lw=1.4, zorder=2)
        ax.set_title(label, loc="left", fontsize=10.5, pad=8)
        ax.set(xlim=(grid[0], grid[-1]), ylim=(0, 1.12), yticks=(0, .5, 1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", ".")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e8ebee", lw=.7)
        ax.set_axisbelow(True)

    handles.append(Line2D([], [], color="#292730", ls=":", lw=1.4, label="Centro experimental"))
    fig.text(.095, .955, "Benzeno  |  Progressão de absorção", va="top", fontsize=15)
    fig.text(.095, .846, "Origem 0–0 experimental · Gaussiana σ = 5 cm⁻¹ · Máximo de cada modelo = 1",
             fontsize=9.5, color="#454552")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.095, .791),
               frameon=False, borderaxespad=0, ncols=len(handles), fontsize=9)
    fig.supylabel("Intensidade relativa", x=.020, y=.445, fontsize=10)
    fig.supxlabel("Número de onda (cm⁻¹)", y=.078, fontsize=10)
    old_spacing, new_spacing, *diagnostic = [f"{np.diff(pair).item():.2f}".replace(".", ",") for pair in positions]
    measured_spacing = f"{np.diff(measured).item():.2f}".replace(".", ",")
    spacing = f"{old_spacing} → {new_spacing}"
    if diagnostic:
        spacing = f"3D {old_spacing} · Canais {new_spacing} · Adiabático {diagnostic[0]}"
    fig.text(.095, .021, f"Separação (cm⁻¹) · {spacing} · Experimental {measured_spacing}",
             fontsize=9, color="#454552")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"benzene_absorption_corrected.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="original nuclear report")
    parser.add_argument("revised", type=Path, help="revised nuclear report")
    parser.add_argument("--baseline-model", default="anharmonic", help="baseline model key")
    parser.add_argument("--baseline-label", default="3D original", help="figure label for baseline")
    parser.add_argument("--model", default="anharmonic", help="revised model key")
    parser.add_argument("--adiabatic-model", help="optional adiabatic diagnostic in the revised report")
    parser.add_argument("--include-revised-satellites", action="store_true",
                        help="plot borrowed-intensity lines in the coupled model")
    parser.add_argument("--label", required=True, help="figure label for revised model")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    baseline = json.loads(options.baseline.read_text(encoding="utf-8"))
    revised = json.loads(options.revised.read_text(encoding="utf-8"))
    previous = absorption_progression(baseline["models"][options.baseline_model]["absorption"])
    revised_model = revised["models"][options.model]
    updated = absorption_progression(revised_model["absorption"])
    if options.include_revised_satellites:
        updated["borrowed_intensity_lines"] = [
            {**row, "center_cm_inverse": ORIGIN_CM + row["term_value_cm_inverse"]}
            for row in revised_model["upper_band_and_borrowed_projection_lines"]
            if not row["same_level_as_selected_upper_band"]
        ]
    comparison = {
        "experimental_doi": "10.1063/1.1767515",
        "experimental_centers_cm_inverse": MEASURED_ABSORPTION.tolist(),
        "experimental_spacing_cm_inverse": float(np.diff(MEASURED_ABSORPTION).item()),
        "measured_00_origin_cm_inverse": ORIGIN_CM,
        "gaussian_sigma_cm_inverse": SIGMA_CM,
        "baseline": previous,
        "revised": updated,
    }
    cases = [(options.baseline_label, previous), (options.label, updated)]
    if options.adiabatic_model:
        adiabatic = absorption_progression(revised["models"][options.adiabatic_model]["absorption"])
        comparison["adiabatic"] = adiabatic
        cases.append(("Contração adiabática", adiabatic))
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "benzene_absorption_corrected.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    plot_absorption(cases, options.output)


if __name__ == "__main__":
    main()
