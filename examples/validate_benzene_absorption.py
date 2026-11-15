"""Solve a benzene absorption progression with a direct radial energy correction.

Use paired electronic energies on the same geometries to correct the breathing dependence
of both three-mode TDA surfaces. Keep the TDA transition-dipole surface and its
phase convention. Accept CCSD/EOM-CCSD or a separately named electronic probe.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from validate_benzene_3d import (
    CM_PER_HARTREE,
    RadialCorrection,
    Surface,
    UnderdeterminedSurface,
    UnstableQuadratic,
    calculation,
    fit_errors,
    fit_residual,
    fit_surfaces,
    harmonic_parameters,
    monomials,
    read_scan,
    select_scan_rows,
)


def electronic_pairs(scan, target_rows, extra_tda, *, reference=None):
    direct = ({tuple(q): energy for q, energy in zip(scan["q"], scan["energies"])}
              if reference is None else reference.copy())
    for row in extra_tda:
        q = tuple(row["q"])
        energy = np.array([row["ground_energy"], row["tda"]["energy"]])
        if q in direct and np.max(np.abs(direct[q] - energy)) > 3e-7:
            raise ValueError(f"inconsistent direct TDA results at q={q}")
        direct.setdefault(q, energy)

    q = np.array([row["q"] for row in target_rows], dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or len(np.unique(q, axis=0)) != len(q):
        raise ValueError("require unique, finite three-mode coordinates for the target data")
    target = np.array([[row["ground_energy_hartree"], row["excited_energy_hartree"]] for row in target_rows])
    try:
        lower = np.array([direct[tuple(point)] for point in q])
    except KeyError as unmatched:
        raise ValueError(f"target point {unmatched} needs a direct electronic-energy partner") from unmatched
    holdout = np.array([row["holdout"] for row in target_rows])
    if holdout.dtype != np.bool_ or not np.isfinite(target).all() or not np.isfinite(lower).all():
        raise ValueError("require finite energies and a boolean holdout flag at each paired point")
    radial = np.all(q[:, 1:] == 0, axis=1)
    train = radial & ~holdout
    if np.any(~radial & ~holdout):
        raise ValueError("off-axis target points are validation points for a radial correction")
    center = np.flatnonzero(np.all(q == 0, axis=1) & train)
    if center.size != 1 or train.sum() < 3:
        raise ValueError("require the origin and at least two other direct radial training nodes")
    center = center.item()
    shifts = ((target - target[center]) - (lower - lower[center])) * CM_PER_HARTREE
    return q, target, shifts, train, radial, center


def fit_correction(q, shifts, train, radial, degree, extension):
    coordinate = q[:, :1]
    bounds = (float(coordinate[train].min()), float(coordinate[train].max()))
    coefficients, condition = fit_residual(coordinate[train], shifts[train],
                                           np.arange(1, degree + 1)[:, None], np.array([max(map(abs, bounds))]))
    corrections = [RadialCorrection(column, bounds, extension) for column in coefficients.T]
    estimate = np.column_stack([item.points(coordinate[:, 0]) for item in corrections])
    residual = estimate - shifts
    report = {
        "coefficients_cm_inverse_by_surface_and_q1_power_starting_at_one": coefficients.T.tolist(),
        "training_domain_q1": list(bounds),
        "q1_bounded_extension": extension,
        "scaled_condition_number": condition,
        "residuals_cm_inverse_at_paired_points": residual.tolist(),
    }
    for label, mask in (("training", train), ("radial_holdout", radial & ~train), ("off_axis_holdout", ~radial)):
        if mask.any():
            report[f"{label}_rms_cm_inverse_by_surface"] = np.sqrt(np.mean(residual[mask]**2, axis=0)).tolist()
    return corrections, report


def radial_minimum(surface, frequency, limits, correction=None):
    mask = np.all(surface.terms[:, 1:] == 0, axis=1)
    correction_degree = len(correction.coefficients) if correction is not None else 0
    coefficients = np.zeros(max(int(surface.terms[mask, 0].max()), correction_degree) + 1)
    coefficients[surface.terms[mask, 0]] = surface.values[mask]
    if correction is not None:
        stop = len(correction.coefficients) + 1
        coefficients[1:stop] += correction.coefficients
    potential = np.polynomial.Polynomial(coefficients)
    stationary = potential.deriv().roots()
    interior = stationary.real[(np.abs(stationary.imag) < 1e-10) &
                               (stationary.real > limits[0]) & (stationary.real < limits[1])]
    candidates = np.r_[limits, interior]
    minimum = float(candidates[np.argmin(potential(candidates))])
    curvature = float(potential.deriv(2)(minimum))
    return {
        "minimum_q1": minimum,
        "minimum_on_domain_boundary": minimum in limits,
        "analytic_fit_curvature_cm_inverse_per_q1_squared": curvature,
        "harmonic_cm_inverse_at_radial_minimum": float(np.sqrt(frequency * curvature)) if curvature > 0 else None,
    }


def fit_transverse(q, shifts, radial):
    radial_shift = {point[0]: shift for point, shift in zip(q[radial], shifts[radial])}
    points = q[~radial]
    if not set(points[:, 0]) <= radial_shift.keys():
        raise ValueError("each transverse slice requires a directly paired point at q6=q8=0")
    target = shifts[~radial] - np.array([radial_shift[point[0]] for point in points])
    q6_fit = (np.abs(points[:, 1]) == 1.5) & (points[:, 2] == 0)
    q8_fit = (np.abs(points[:, 2]) == 1.5) & (points[:, 1] == 0)
    q6_terms = np.array([(0, 2, 0), (1, 2, 0), (0, 3, 0), (1, 3, 0)])
    q6_coefficients, _ = fit_residual(points[q6_fit], target[q6_fit], q6_terms, np.array([3., 1.5, 1.5]))
    terms, coefficients = q6_terms, q6_coefficients
    if q8_fit.any():
        q8_terms = np.array([(0, 0, 2), (0, 0, 3)])
        q8_coefficients, _ = fit_residual(points[q8_fit], target[q8_fit], q8_terms, np.array([3., 1.5, 1.5]))
        terms = np.vstack((terms, q8_terms))
        coefficients = np.vstack((coefficients, q8_coefficients))

    residual = monomials(points, terms) @ coefficients - target
    test_rows = ~(q6_fit | q8_fit)
    validation = {}
    for q1 in np.unique(points[q6_fit, 0]):
        omitted = q6_fit & (points[:, 0] == q1)
        retained = q6_fit & ~omitted
        if np.unique(points[retained, 0]).size >= 2:
            variant, _ = fit_residual(points[retained], target[retained], q6_terms, np.array([3., 1.5, 1.5]))
            validation[str(q1)] = (monomials(points[omitted], q6_terms) @ variant - target[omitted]).tolist()
    return terms, coefficients, {
        "terms_exponents_in_q1_q6_q8_order": terms.tolist(),
        "coefficients_cm_inverse_by_term_and_surface": coefficients.tolist(),
        "points_q1_q6_q8": points.tolist(),
        "independent_transverse_holdout_mask": test_rows.tolist(),
        "raw_paired_transverse_correction_cm_inverse": target.tolist(),
        "transverse_residuals_cm_inverse_by_surface": residual.tolist(),
        "q6_leave_one_q1_slice_out_residuals_cm_inverse_by_surface": validation,
        "q6_training_rule": "Pure q6 slices at plus/minus 1.5, fitted in q6 squared/cubed with linear q1 dependence",
        "q8_training_rule": "Pure q8 slices at plus/minus 1.5, fitted in q8 squared/cubed with constant coefficients",
    }


def with_transverse_correction(surfaces, terms, coefficients):
    active = []
    for surface, column in zip(surfaces[:2], coefficients.T):
        values = {tuple(term): coefficient for term, coefficient in zip(surface.terms, surface.values)}
        for term, coefficient in zip(terms, column):
            key = tuple(term)
            values[key] = values.get(key, 0.0) + coefficient
        active.append(Surface(np.array(list(values)), np.array(list(values.values())),
                              surface.lengths, surface.linear_order))
    return (*active, surfaces[2])


def base_surface_fit(scan):
    mask = scan.get("holdout", np.zeros(len(scan["q"]), dtype=bool))
    if mask.dtype != bool or mask.shape != (len(scan["q"]),):
        raise ValueError("require a boolean holdout flag per TDA scan point")
    training, holdouts = select_scan_rows(scan, ~mask), select_scan_rows(scan, mask)
    degree = 4
    quartic = fit_surfaces(training, degree)
    chosen = quartic
    if mask.any():
        center = training["energies"][np.all(training["q"] == 0, axis=1)]
        quartic_error = np.linalg.norm(fit_errors(quartic[:3], holdouts, True, center)["rms_energy_cm_inverse"])
        try:
            quintic = fit_surfaces(training, 5)
            for surface in quintic[:2]:
                harmonic_parameters(surface, scan["frequencies_cm1"])
        except (UnderdeterminedSurface, UnstableQuadratic):
            quintic = None
        if quintic is not None:
            quintic_error = np.linalg.norm(fit_errors(quintic[:3], holdouts, True, center)["rms_energy_cm_inverse"])
            if quintic_error < quartic_error:
                degree, chosen = 5, quintic
    ground, excited, dipole, diagnostic = chosen
    return (ground, excited, dipole), degree, diagnostic["vertical_gap_cm_inverse"]


def absorption_progression(model):
    absorption = model["absorption"]
    columns = {name: i for i, name in enumerate(absorption["spectrum_column_order"])}
    spectrum = np.asarray(absorption["spectrum_all_retained_levels_in_increasing_photon_order"])
    weights = spectrum[:, [columns["projection_weight_nu6"], columns["projection_weight_nu1+nu6"]]].T
    row, state = linear_sum_assignment(-weights)
    selected = spectrum[state]
    terms = selected[:, columns["vibrational_energy_cm_inverse"]]
    summary = {
        "labels": ["nu6", "nu1+nu6"],
        "term_values_cm_inverse": terms.tolist(),
        "spacing_cm_inverse": float(terms[1] - terms[0]),
        "assignment_weights": weights[row, state].tolist(),
        "dipole_strengths_au2": selected[:, columns["dipole_strength_au2"]].tolist(),
    }
    outside = absorption.get("probability_outside_q1_correction_domain_by_spectrum_line")
    if outside is not None:
        summary["probability_outside_q1_correction_domain"] = np.asarray(outside)[state].tolist()
    return summary


def run(scan, target_rows, extra_tda, nmax, quadrature, *, reference=None, label="radial_eom_ccsd", transverse=False):
    if nmax < 8 or quadrature < nmax + 8:
        raise ValueError("require max quanta >= 8 and quadrature >= max quanta + 8")
    q, target, shifts, train, radial, center = electronic_pairs(scan, target_rows, extra_tda, reference=reference)
    fits = {degree: fit_correction(q, shifts, train, radial, degree, 0.5)
            for degree in range(2, min(4, int(train.sum()) - 1) + 1)}
    # Use the radial validation energies to select the degree. Preserve mixed
    # displacements for testing whether the radial-only approximation transfers.
    radial_holdout = radial & ~train
    if radial_holdout.any():
        degree = min(fits, key=lambda d: np.linalg.norm(fits[d][1]["radial_holdout_rms_cm_inverse_by_surface"]))
    else:
        degree = 2
    corrections = fits[degree][0]
    surfaces, base_degree, tda_gap = base_surface_fit(scan)
    target_gap = float(np.diff(target[center]).item() * CM_PER_HARTREE)
    reference_gap = (tda_gap if reference is None else
                     float(np.diff(reference[(0.0, 0.0, 0.0)]).item() * CM_PER_HARTREE))
    corrected_gap = tda_gap + target_gap - reference_gap
    frequencies = scan["frequencies_cm1"]
    shifted = surfaces
    transverse_fit = None
    if transverse:
        terms, coefficients, transverse_fit = fit_transverse(q, shifts, radial)
        shifted = with_transverse_correction(surfaces, terms, coefficients)
    if reference is None:
        target_relative = (target - target[center]) * CM_PER_HARTREE
        base_energies = np.column_stack([surface.points(q) for surface in shifted[:2]])
        for correction, diagnostic in fits.values():
            combined = base_energies + np.column_stack([item.points(q[:, 0]) for item in correction])
            diagnostic["combined_surface_residuals_vs_direct_target_cm_inverse"] = (combined - target_relative).tolist()

    def calculate(gap, size, order, correction=None, potential=surfaces):
        model = calculation(potential, frequencies, gap, size, order, "capped", "anharmonic",
                            radial_corrections=correction)
        model["absorption_progression"] = absorption_progression(model)
        return model

    models = {
        "anharmonic": calculate(tda_gap, nmax, quadrature),
        label: calculate(corrected_gap, nmax, quadrature, corrections, shifted),
        f"{label}_larger_nuclear_basis": calculate(corrected_gap, nmax + 2, quadrature + 4, corrections, shifted),
    }
    if transverse_fit is not None:
        models[f"{label}_radial_only"] = calculate(corrected_gap, nmax, quadrature, corrections)
    capped, _ = fit_correction(q, shifts, train, radial, degree, 0.0)
    models[f"{label}_capped_correction"] = calculate(corrected_gap, nmax, quadrature, capped, shifted)
    for alternative, (correction, _) in fits.items():
        if alternative != degree:
            models[f"{label}_degree_{alternative}"] = calculate(corrected_gap, nmax, quadrature, correction, shifted)
    return {
        "base_total_degree": base_degree,
        "primary_radial_correction_degree": degree,
        "radial_correction_degree_selection": "Independent radial energy holdouts" if radial_holdout.any() else "Quadratic pilot",
        "paired_coordinates_q1_q6_q8": q.tolist(),
        "paired_radial_training_mask": train.tolist(),
        "paired_radial_mask": radial.tolist(),
        "paired_target_minus_reference_relative_surfaces_cm_inverse": shifts.tolist(),
        "vertical_tda_gap_cm_inverse": tda_gap,
        "vertical_direct_target_gap_cm_inverse": target_gap,
        "vertical_direct_reference_gap_cm_inverse": reference_gap,
        "vertical_corrected_model_gap_cm_inverse": corrected_gap,
        "transition_dipole_surface": "Original phase-tracked TDA fit",
        "tail_convention": "Continue the correction through tanh saturation of q1 at 0.5 beyond the training interval. The TDA potential supplies the outer confining term.",
        "tda_radial_minima_by_surface": [radial_minimum(surface, frequencies[0], (-surface.lengths[0], surface.lengths[0]))
                                       for surface in surfaces[:2]],
        "corrected_radial_minima_by_surface": [radial_minimum(surface, frequencies[0], correction.limits, correction)
                                             for surface, correction in zip(surfaces[:2], corrections)],
        "radial_correction_fits": {f"degree_{key}": value[1] for key, value in fits.items()},
        "transverse_correction_fit": transverse_fit,
        "models": models,
    }


def read_probes(paths):
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data if isinstance(data, list) else [data])
    return rows


def probe_method(rows, response):
    settings = {(row["basis"], row["grid"], row.get("density_fit", "auxbasis" in row), row.get("auxbasis"))
                for row in rows}
    if len(settings) != 1:
        raise ValueError("a set of directly paired probes must share basis, grid and density fitting settings")
    basis, grid, density_fit, auxbasis = settings.pop()
    return {"response": response, "basis": basis, "grid": grid, "density_fit": density_fit,
            "auxiliary_basis": auxbasis}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument("--coupled-cluster", nargs="+", type=Path)
    method.add_argument("--target-probes", nargs="+", type=Path)
    parser.add_argument("--target-response", choices=("tda", "tddft"), default="tddft")
    parser.add_argument("--reference-probes", nargs="+", type=Path,
                        help="direct reference for an isolated method/basis shift instead of the baseline TDA scan")
    parser.add_argument("--reference-response", choices=("tda", "tddft"), default="tddft")
    parser.add_argument("--extra-tda", nargs="+", type=Path, default=(),
                        help="additional direct baseline TDA geometries for matched target points")
    parser.add_argument("--radial-holdout", type=float, nargs="+", default=(),
                        help="withhold these q1 coordinates when using --target-probes")
    parser.add_argument("--method-label", help="model key for the corrected spectrum")
    parser.add_argument("--fit-transverse", action="store_true",
                        help="fit pure q6 and available q8 cuts at their sampled amplitudes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-quanta", type=int, default=12)
    parser.add_argument("--quadrature", type=int, default=42)
    args = parser.parse_args()

    if args.reference_probes and args.extra_tda:
        parser.error("use one of --reference-probes and --extra-tda")
    label = args.method_label or ("radial_eom_ccsd" if args.coupled_cluster else "radial_electronic_correction")
    if not re.fullmatch("[a-z][a-z0-9_]+", label) or label == "anharmonic":
        parser.error("--method-label must be a lowercase underscored key other than 'anharmonic'")
    if args.coupled_cluster:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.coupled_cluster]
        settings = {tuple(report[key] for key in ("method", "basis", "ccsd_auxbasis")) for report in reports}
        if len(settings) != 1:
            raise ValueError("the coupled-cluster inputs must share a method, basis and auxiliary basis")
        method, basis, auxbasis = settings.pop()
        target_method = {"response": method, "basis": basis, "auxiliary_basis": auxbasis}
        target_rows = [row for report in reports for row in report["points"]]
    else:
        probes = read_probes(args.target_probes)
        target_method = probe_method(probes, args.target_response)
        available_radial = {row["q"][0] for row in probes if not any(row["q"][1:])}
        if not set(args.radial_holdout) <= available_radial:
            parser.error("all --radial-holdout coordinates must be present in the target radial probes")
        target_rows = [{"q": row["q"], "holdout": bool(any(row["q"][1:]) or row["q"][0] in args.radial_holdout),
                        "ground_energy_hartree": row["ground_energy"],
                        "excited_energy_hartree": row[args.target_response]["energy"]} for row in probes]
    reference = None
    reference_method = {"response": "Baseline phase-tracked TDA scan"}
    if args.reference_probes:
        probes = read_probes(args.reference_probes)
        reference_method = probe_method(probes, args.reference_response)
        reference = {tuple(row["q"]): np.array([row["ground_energy"], row[args.reference_response]["energy"]])
                     for row in probes}
        if len(reference) != len(probes):
            parser.error("--reference-probes must contain unique coordinates")
    result = run(read_scan(args.scan), target_rows, read_probes(args.extra_tda), args.max_quanta, args.quadrature,
                 reference=reference, label=label, transverse=args.fit_transverse)
    result["direct_target_electronic_method"] = target_method
    result["direct_reference_electronic_method"] = reference_method
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
