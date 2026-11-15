"""Compare coupled inversion/relaxation spectra with the archived H2CO AH run."""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_h2co_analytic_ah import CM_PER_HARTREE, SIGMA_CM, broaden


PROCESSES = ("absorption", "emission")
BASIS_SHAPES = ((42, 32), (56, 42))


def checked_parity(values, reflected, atol, label):
    error = float(np.max(np.abs(values - reflected)))
    if error > atol:
        raise ValueError(f"{label} reflection error {error:.3g} exceeds {atol:g}")
    return error


def read_cut(directory, path, energy_zero):
    from scipy.interpolate import CubicSpline
    from pyscf_esd.anharmonic_dvr import solve_dvr

    with np.load(directory / "electronic_scan.npz") as data:
        q = data["q"]
        signed_dipole = data["dipole_au"]
        e1 = data["cis_energy_hartree"] - energy_zero
        checked_parity(q, -q[::-1], 1e-12, "one-dimensional coordinate")
        checked_parity(e1, e1[::-1], 1e-8, "one-dimensional S1 energy")
        checked_parity(signed_dipole, -signed_dipole[::-1], 1e-7, "one-dimensional dipole")
    arc = np.array([path.arclength(value) for value in q])
    fit = CubicSpline(arc, (e1 + e1[::-1]) / 2)
    grid = np.linspace(arc[0], arc[-1], 385)[1:-1]
    states = solve_dvr(grid, fit(grid), n_levels=2, boundary="dirichlet")
    return {
        "barrier_cm_inverse": float((fit(0.0) - fit(path.arclength(1.0))) * CM_PER_HARTREE),
        "lowest_gap_cm_inverse": float(np.diff(states.energies_hartree)[0] * CM_PER_HARTREE),
    }


def load_plane(directory, path, energy_zero):
    from scipy.interpolate import RectBivariateSpline

    candidates = ("electronic_relaxation_plane_wide.npz", "electronic_relaxation_plane_extended.npz",
                  "electronic_relaxation_plane.npz")
    source = next((directory / filename for filename in candidates if (directory / filename).is_file()), None)
    if source is None:
        raise FileNotFoundError(f"no electronic relaxation plane in {directory}")
    with np.load(source) as data:
        q, r, dipoles = data["q"].round(12), data["r"].round(12), data["dipole_au"]
        v0 = data["rhf_energy_hartree"] - energy_zero
        v1 = data["cis_energy_hartree"] - energy_zero
        geometries = data["geometry_bohr"]
    origin = path.geometry(0.0)
    odd = (path.geometry(1.0) - path.geometry(-1.0)) / 2
    even = (path.geometry(1.0) + path.geometry(-1.0)) / 2 - origin
    np.testing.assert_allclose(geometries, origin + q[:, None, None] * odd + r[:, None, None] * even, atol=1e-10, rtol=0)
    lookup = {(qi, ri): i for i, (qi, ri) in enumerate(zip(q, r))}
    if len(lookup) != q.size:
        raise ValueError("duplicate electronic coordinates in the relaxation plane")
    qp, rp = np.unique(q[q >= 0]), np.unique(r[q >= 0])
    if qp[0] != 0 or not np.any(q < 0):
        raise ValueError("a planar row and independently calculated reflected points are required")
    positive = np.array([[lookup[qi, ri] for ri in rp] for qi in qp])
    negative = np.flatnonzero(q < 0)
    partners = np.array([lookup[-q[i], r[i]] for i in negative])
    sym = {
        "mirror_pairs": int(negative.size),
        "max_s0_error_hartree": checked_parity(v0[negative], v0[partners], 1e-8, "S0 energy"),
        "max_s1_error_hartree": checked_parity(v1[negative], v1[partners], 1e-8, "S1 energy"),
        "max_signed_dipole_error_au": checked_parity(dipoles[negative], -dipoles[partners], 1e-7, "dipole"),
        "max_planar_dipole_au": checked_parity(dipoles[positive[0]], 0, 1e-7, "planar dipole"),
        "max_transverse_dipole_au": checked_parity(dipoles[:, (0, 2)], 0, 1e-7, "perpendicular dipole components"),
    }
    mass_weighted = np.stack((odd, even)) * np.sqrt(path.masses_au)[None, :, None]
    lengths = np.linalg.norm(mass_weighted.reshape(2, -1), axis=1)
    directions = mass_weighted.reshape(2, -1).T / lengths
    np.testing.assert_allclose(directions.T @ directions, np.eye(2), atol=1e-10, rtol=0)
    x, y = qp * lengths[0], rp * lengths[1]
    values = v0[positive], v1[positive], dipoles[positive, 1].copy()
    values[2][0] = 0

    def fits(ix, iy):
        nodes = x[ix]
        signed_x = np.r_[-nodes[:0:-1], nodes]
        fields = []
        for symmetry, data in zip((1, 1, -1), values):
            positive_half = data[np.ix_(ix, iy)]
            mirrored = np.r_[symmetry * positive_half[:0:-1], positive_half]
            fields.append(RectBivariateSpline(signed_x, y[iy], mirrored, s=0))
        return fields

    coarse_y = np.unique(np.r_[0, np.arange(0, len(y), 2), np.flatnonzero(rp == 1), len(y)-1])
    fine_fits = fits(np.arange(len(x)), np.arange(len(y)))
    coarse_fits = fits(np.r_[np.arange(0, len(x)-1, 2), len(x)-1], coarse_y)
    return x, y, lengths, directions, fine_fits, coarse_fits, sym, source.name


def rotation_metric_correction(path, directions):
    """Magnitude of the rotation-removal term omitted by the Cartesian slice."""
    mass = path.masses_au
    derivatives = directions.T.reshape(directions.shape[1], -1, 3) / np.sqrt(mass)[None, :, None]
    corrections = {}
    for q in (0, 1):
        coords = path.geometry(float(q))
        inertia = np.einsum("a,ax,ax,ij->ij", mass, coords, coords, np.eye(3)) - np.einsum("a,ai,aj->ij", mass, coords, coords)
        angular = np.einsum("a,bax->xb", mass, np.cross(coords, derivatives))
        matrix = angular.T @ np.linalg.solve(inertia, angular)
        corrections[f"at_s{q}_minimum"] = float(np.linalg.eigvalsh(matrix)[-1])
    return corrections


def projected_modes(endpoints, path, directions):
    """Use archived analytic Hessians in the active plane and its complement."""
    from scipy.linalg import null_space

    mass_factor = np.repeat(np.sqrt(path.masses_au), 3)
    axis = path.excited_rotation
    cart_rotation = np.kron(np.eye(len(path.masses_au)), axis)
    active, spectator = [], []
    for index, name in enumerate(("s0", "s1")):
        hessian = endpoints[f"hessian_{name}"].transpose(0, 2, 1, 3).reshape(mass_factor.size, -1)
        if index:
            hessian = cart_rotation.T @ hessian @ cart_rotation
        hessian = hessian / np.outer(mass_factor, mass_factor)
        coords = path.geometry(float(index))
        translations = np.tile(np.eye(3), (len(coords), 1))
        rotations = np.cross(np.eye(3)[:, None, :], coords).reshape(3, -1).T
        rigid = np.column_stack((translations, rotations)) * mass_factor[:, None]
        vibrational = null_space(rigid.T)
        complement = vibrational @ null_space(directions.T @ vibrational)
        active_values, active_vectors = np.linalg.eigh(directions.T @ hessian @ directions)
        bath = np.linalg.eigvalsh(complement.T @ hessian @ complement)
        if active_values.min() <= 0 or bath.size != vibrational.shape[1] - directions.shape[1] or np.any(bath <= 0):
            raise RuntimeError("active-plane or spectator curvature is not stable")
        active.append((np.sqrt(active_values), active_vectors))
        spectator.append(np.sqrt(bath))
    return active, spectator


def linear_dipole(endpoints, directions, path):
    from pyscf import gto, scf
    from pyscf_esd.cis_hessian import analytic_cis_transition_dipole

    molecule = gto.M(atom=list(zip(endpoints["symbols"], endpoints["geometry_s0_bohr"])),
                     basis="6-31g*", unit="Bohr", verbose=0)
    reference = scf.RHF(molecule)
    reference.mo_coeff = endpoints["mo_coeff_s0"]
    reference.mo_occ = endpoints["mo_occ_s0"]
    reference.e_tot = float(endpoints["ground_energy_hartree"])
    reference.converged = True
    result = analytic_cis_transition_dipole(reference, endpoints["tda_amplitude_s0"])
    jacobian = result.derivative.reshape(-1, 3) / np.repeat(np.sqrt(path.masses_au), 3)[:, None]
    return directions.T @ jacobian


def harmonic_control(active, lengths, slope, electronic_gap, zpe):
    from pyscf_esd.harmonic import harmonic_spectrum

    (w0, u0), (w1, u1) = active
    gap = electronic_gap + zpe + 0.5 * (w1 - w0).sum()
    shift = -(u1.T @ lengths)
    result = {}
    for process, omega_i, omega_f, units_i, units_f, displacement, dipole, sign in (
        ("absorption", w0, w1, u0, u1, shift, np.zeros(3), 1),
        ("emission", w1, w0, u1, u0, u0.T @ lengths, lengths @ slope, -1),
    ):
        lines = harmonic_spectrum(
            omega_i, omega_f, gap_00_hartree=sign * gap, transition_dipole_au=dipole,
            initial_max_quanta=0, final_max_quanta=50, rotation=units_f.T @ units_i,
            shift_au=displacement, dipole_derivatives_au=units_i.T @ slope, process=process,
        )
        if lines.final_dipole_strength_coverage < 0.999:
            raise RuntimeError("insufficient harmonic control dipole coverage")
        result[process] = np.column_stack((lines.photon_energy_hartree * CM_PER_HARTREE, lines.dipole_strength_au2))
    return gap, result


def solve_plane(fields, x_limit, y_bounds, shape, zpe, n_levels):
    from pyscf_esd.anharmonic_plane import SineGalerkin2D

    grid = SineGalerkin2D(x_limit, y_bounds, shape)
    e0 = grid.solve(fields[0](grid.x, grid.y), n_levels=n_levels)
    e1 = grid.solve(fields[1](grid.x, grid.y) + zpe, n_levels=n_levels)
    return grid, e0, e1


def plane_spectra(grid, e0, e1, dipole, linear):
    if tuple(e1.parity[:2]) != (1, -1) or e0.parity[0] != 1:
        raise ValueError("unexpected parity ordering in the lowest inversion states")
    amplitude = grid.transition_matrix(e0, e1, dipole)
    linear_amplitude = grid.transition_matrix(e0, e1, linear)
    same_parity = e0.parity[:, None] == e1.parity[None, :]
    leakage = checked_parity(amplitude[same_parity], 0, 1e-9, "same-parity transition")
    squared = amplitude**2
    origin = float(e1.energies_hartree[0] - e0.energies_hartree[0])
    sticks = {}
    summary = {"max_same_inversion_parity_amplitude_au": leakage,
               "dark_0_0_strength_au2": float(squared[0, 0])}
    for process, i, f, strength, linear_strength, parity in (
        ("absorption", e0, e1, squared[0], linear_amplitude[0]**2, e1.parity),
        ("emission", e1, e0, squared[:, 0], linear_amplitude[:, 0]**2, e0.parity),
    ):
        sign = 1 if process == "absorption" else -1
        energy = sign * (f.energies_hartree - i.energies_hartree[0])
        positive = energy > 0
        closure = float(strength.sum() / grid.expectation(i, dipole**2)[0])
        if not 0.995 < closure <= 1.0000001:
            raise RuntimeError(f"{process} anharmonic dipole closure = {closure:.6g}")
        sticks[process] = np.column_stack((energy[positive] * CM_PER_HARTREE, strength[positive],
                                          np.flatnonzero(positive), parity[positive]))
        total, total_linear = float(strength[positive].sum()), float(linear_strength[positive].sum())
        summary[process] = {"dipole_coverage": closure,
                            "positive_photon_strength_au2": total,
                            "positive_photon_strength_linear_dipole_au2": total_linear,
                            "relative_strength_change_from_linear_dipole": total / total_linear - 1}
    summary["first_bright_absorption_cm_inverse"] = float((e1.energies_hartree[1] - e0.energies_hartree[0]) * CM_PER_HARTREE)
    summary["first_bright_absorption_strength_au2"] = float(squared[0, 1])
    summary["first_bright_emission_cm_inverse"] = float((e1.energies_hartree[0] - e0.energies_hartree[1]) * CM_PER_HARTREE)
    return origin, sticks, summary


def compare_lines(ref_sticks, grid, e0, e1, dipole, reference_cm):
    small = grid.transition_matrix(e0, e1, dipole)**2
    differences = {}
    for process, photons, strengths, power in (
        ("absorption", e1.energies_hartree - e0.energies_hartree[0], small[0], 1),
        ("emission", e1.energies_hartree[0] - e0.energies_hartree, small[:, 0], 3),
    ):
        ref = ref_sticks[process]
        positive = photons > 0
        energies = photons[positive] * CM_PER_HARTREE
        weights = strengths[positive] * (energies / reference_cm)**power
        ref_weights = ref[:, 1] * (ref[:, 0] / reference_cm)**power
        display = ref[ref_weights >= ref_weights.max()*1e-4, 0]
        axis = np.arange(display.min()-400, display.max()+402, 2.0)
        fit_curve = broaden(energies, weights, axis, SIGMA_CM)
        ref_curve = broaden(ref[:, 0], ref_weights, axis, SIGMA_CM)
        differences[process] = {
            "dominant_stick_cm_inverse": float(energies[np.argmax(weights)]),
            "ref_dominant_stick_cm_inverse": float(ref[np.argmax(ref_weights), 0]),
            "relative_l1_broadened_intensity_change": float(np.abs(fit_curve - ref_curve).sum() / ref_curve.sum()),
        }
        if process == "absorption":
            near_onset = axis <= display.min() + 1200
            differences[process]["relative_l1_change_within_1200_cm_inverse_of_onset"] = float(
                np.abs(fit_curve[near_onset] - ref_curve[near_onset]).sum() / ref_curve[near_onset].sum()
            )
    return differences


def inversion_profile(grid, e1, fields, lengths, y_bounds, zpe):
    from scipy.optimize import minimize_scalar
    from pyscf_esd.anharmonic_dvr import solve_dvr

    def slice_minimum(x):
        point = minimize_scalar(lambda y: float(fields[1].ev(x, y)), bounds=y_bounds, method="bounded")
        if not point.success:
            raise RuntimeError("unable to locate the minimum of an S1 slice")
        return point.x, point.fun

    y = np.linspace(*y_bounds, 145)[1:-1]
    q = np.linspace(-1.65, 1.65, 133)
    minima = [slice_minimum(x) for x in q * lengths[0]]
    effective = np.array([
        solve_dvr(y, fields[1].ev(x, y) + zpe, n_levels=1, boundary="dirichlet").energies_hartree[0]
        for x in q * lengths[0]
    ])
    effective_min = float(effective.min())
    center = q.size // 2
    side = grid.x > 0
    weights = grid.x_weights[:, None] * grid.y_weights[None, :]
    # Relative sign of two normalized exact-parity standing waves is arbitrary.
    localized = (e1.wavefunctions[:, :, 0] + e1.wavefunctions[:, :, 1]) / np.sqrt(2)
    right_well = float(np.sum(weights[side] * localized[side]**2))
    well_bottom = float(fields[1].ev(*lengths))
    saddle_y, saddle_energy = minima[center]
    summary = {
        "electronic_plane_barrier_cm_inverse": float((saddle_energy - well_bottom) * CM_PER_HARTREE),
        "planar_saddle_r": float(saddle_y / lengths[1]),
        "effective_transverse_barrier_cm_inverse": float((effective[center] - effective_min) * CM_PER_HARTREE),
        "levels_relative_to_effective_barrier_cm_inverse": ((e1.energies_hartree[:2] - effective[center]) * CM_PER_HARTREE).tolist(),
        "ground_pair_max_one_side_probability": max(right_well, 1-right_well),
        "effective_profile_zero_hartree": effective_min,
    }
    sampled_dipole = fields[2].ev(q * lengths[0], [item[0] for item in minima])
    return (q, (effective - effective_min) * CM_PER_HARTREE, sampled_dipole), summary


def assemble(reference, electronic, output):
    from pyscf import lib
    from pyscf_esd.inversion_path import InversionPath

    lib.num_threads(2)
    with np.load(reference / "electronic" / "electronic_results.npz") as saved:
        endpoints = dict(saved)
    energy_zero = float(endpoints["ground_energy_hartree"])
    gap_el = float(endpoints["excited_energy_hartree"]) - energy_zero
    path = InversionPath(endpoints["geometry_s0_bohr"], endpoints["geometry_s1_bohr"], endpoints["masses_amu"])
    cut = read_cut(electronic, path, energy_zero)
    x, y, lengths, directions, fit, coarse_fit, mirror, plane_file = load_plane(electronic, path, energy_zero)
    active, spectator = projected_modes(endpoints, path, directions)
    zpe = float(0.5 * (spectator[1] - spectator[0]).sum())
    slope = linear_dipole(endpoints, directions, path)
    np.testing.assert_allclose(fit[0].ev(0, 0), 0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(fit[1].ev(*lengths), gap_el, atol=1e-9, rtol=0)
    np.testing.assert_allclose(fit[2].ev(*lengths), (endpoints["dipole_s1"] @ path.excited_rotation)[1], atol=1e-7, rtol=0)

    y_box = (float(y.min()), float(y.max()))
    grid, e0, e1 = solve_plane(fit, x[-1], y_box, BASIS_SHAPES[-1], zpe, 144)
    slope_field = grid.x[:, None] * slope[0, 1] + grid.y[None, :] * slope[1, 1]
    origin, anharmonic, selection = plane_spectra(grid, e0, e1, fit[2](grid.x, grid.y), slope_field)
    convergence = {}
    for label, fields, x_limit, bounds, order in (
        ("smaller_basis", fit, x[-1], y_box, BASIS_SHAPES[0]),
        ("coarser_electronic_mesh", coarse_fit, x[-1], y_box, BASIS_SHAPES[-1]),
        ("penultimate_inversion_range", fit, x[-2], y_box, BASIS_SHAPES[-1]),
        ("penultimate_relaxation_range", fit, x[-1], (y_box[0], y[-2]), BASIS_SHAPES[-1]),
        ("smaller_inversion_range", fit, 0.90*x[-1], y_box, BASIS_SHAPES[-1]),
        ("smaller_relaxation_range", fit, x[-1], (y_box[0], y_box[1]*0.90), BASIS_SHAPES[-1]),
    ):
        test, e0_test, e1_test = solve_plane(fields, x_limit, bounds, order, zpe, 80)
        convergence[label] = {
            "max_s0_first_ten_difference_cm_inverse": float(np.max(np.abs(e0_test.energies_hartree[:10] - e0.energies_hartree[:10])) * CM_PER_HARTREE),
            "max_s1_first_ten_difference_cm_inverse": float(np.max(np.abs(e1_test.energies_hartree[:10] - e1.energies_hartree[:10])) * CM_PER_HARTREE),
            "lowest_s1_gap_cm_inverse": float(np.diff(e1_test.energies_hartree[:2])[0] * CM_PER_HARTREE),
            "spectral_differences": compare_lines(anharmonic, test, e0_test, e1_test, fields[2](test.x, test.y), origin * CM_PER_HARTREE),
        }

    control_gap, control = harmonic_control(active, lengths, slope, gap_el, zpe)
    profile, profile_report = inversion_profile(grid, e1, fit, lengths, y_box, zpe)
    with (reference / "spectra" / "h2co_ah_report.json").open(encoding="utf-8") as stream:
        previous = json.load(stream)
    if previous["sigma_cm_inverse"] != SIGMA_CM or previous["temperature_kelvin"] != 0:
        raise ValueError("the archived AH settings are incompatible")
    origin_ah = previous["origin_cm_inverse"]
    with np.load(reference / "spectra" / "h2co_ah_sticks.npz") as saved:
        full_ah = {process: saved[f"{process}_duschinsky_values"][:, :2] for process in PROCESSES}

    data = {"profile_q": profile[0], "profile_effective_cm_inverse": profile[1], "profile_dipole_au": profile[2],
            "scan_q": x / lengths[0], "scan_r": y / lengths[1],
            "s1_wavefunctions": e1.wavefunctions[:, :, :2], "x_quadrature": grid.x, "y_quadrature": grid.y,
            "x_weights": grid.x_weights, "y_weights": grid.y_weights,
            "s1_levels_relative_to_profile_min_cm_inverse": (e1.energies_hartree[:2] - profile_report["effective_profile_zero_hartree"]) * CM_PER_HARTREE}
    spectra_summary = {}
    for process, power in zip(PROCESSES, (1, 3)):
        models = {"coupled": anharmonic[process], "control": control[process], "archived_ah": full_ah[process]}
        band = []
        for name, model in models.items():
            weights = model[:, 1] * (model[:, 0] / origin_ah)**power
            data[f"{process}_{name}_sticks"] = model
            visible = model[weights >= weights.max()*0.002, 0]
            band.extend((visible.min(), visible.max()))
        axis = np.arange(min(band)-250, max(band)+252, 2.0)
        data[f"{process}_axis"] = axis
        spectra_summary[process] = {}
        for name, model in models.items():
            weights = model[:, 1] * (model[:, 0] / origin_ah)**power
            curve = broaden(model[:, 0], weights, axis, SIGMA_CM)
            data[f"{process}_{name}_curve"] = curve
            spectra_summary[process][name] = {"band_max_cm_inverse": float(axis[np.argmax(curve)]),
                                               "weighted_strength_in_plot_fraction": float(2*curve.sum()/weights.sum())}

    planar = json.loads((electronic / "planar_relaxation.json").read_text(encoding="utf-8"))
    relaxed_planar = {key: planar[key] for key in (
        "relaxed_planar_barrier_cm1", "gradient_max", "optimizer_message", "min_overlap",
    )}
    report = {
        "method": "two-dimensional mass-weighted Cartesian RHF–TDA/CIS/6-31G(d) cut, sine Galerkin",
        "electronic_zero_s0_hartree": energy_zero, "adiabatic_electronic_gap_cm_inverse": gap_el * CM_PER_HARTREE,
        "four_spectator_zpe_difference_cm_inverse": zpe * CM_PER_HARTREE,
        "four_spectator_frequencies_cm_inverse": [(omega*CM_PER_HARTREE).tolist() for omega in spectator],
        "two_harmonic_control_frequencies_cm_inverse": [(omega*CM_PER_HARTREE).tolist() for omega, _ in active],
        "anharmonic_00_cm_inverse": origin * CM_PER_HARTREE, "harmonic_2d_00_cm_inverse": control_gap * CM_PER_HARTREE,
        "archived_ah_6d_00_cm_inverse": origin_ah,
        "s1_lowest_gap_cm_inverse": float(np.diff(e1.energies_hartree[:2])[0] * CM_PER_HARTREE),
        "s1_levels_relative_to_ground_cm_inverse": ((e1.energies_hartree[:8] - e1.energies_hartree[0]) * CM_PER_HARTREE).tolist(),
        "s1_level_parities": e1.parity[:8].tolist(),
        "phase_audit": mirror, "selection_and_coverage": selection,
        "fixed_one_dimensional_cut": cut, "relaxed_planar_s1_check": relaxed_planar,
        "profile_diagnostics": profile_report, "convergence": convergence, "spectra": spectra_summary,
        "basis_shape": BASIS_SHAPES[-1], "electronic_grid_nonnegative_x_shape": [len(x), len(y)],
        "electronic_plane_file": plane_file,
        "electronic_plane_q_range": [float(-x[-1] / lengths[0]), float(x[-1] / lengths[0])],
        "electronic_plane_r_range": [float(y_box[0] / lengths[1]), float(y_box[1] / lengths[1])],
        "cartesian_plane_rotation_metric_correction": rotation_metric_correction(path, directions),
        "sigma_cm_inverse": SIGMA_CM, "temperature_kelvin": 0,
        "emission_initial_state": "relaxed lowest S1 vibrational eigenstate (even)",
        "optical_weighting": "common reference is the archived AH 0-0; E/Eref for absorption and (E/Eref)^3 for emission",
    }
    np.savez_compressed(output / "h2co_inversion_spectra.npz", **data)
    (output / "h2co_inversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def plot(output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    with np.load(output / "h2co_inversion_spectra.npz") as archive:
        data = dict(archive)
    report = json.loads((output / "h2co_inversion_report.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.9), layout="constrained")
    styles = (
        ("coupled", "Anarmônico 2D · dipolo variável", "#17637f", "-", 1.65),
        ("control", "AH 2D · HT linear", "#a74a28", "--", 1.25),
        ("archived_ah", "AH 6D · HT e Duschinsky", "#64616c", ":", 1.35),
    )
    for ax, process, title in zip(axes, PROCESSES, ("Absorção", "Emissão após relaxação em S₁")):
        axis = data[f"{process}_axis"]
        scale = max(data[f"{process}_{name}_curve"].max() for name, *_ in styles)
        for name, text, color, line, width in styles:
            ax.plot(axis, data[f"{process}_{name}_curve"] / scale, color=color, ls=line, lw=width, label=text)
        dark_origin = report["anharmonic_00_cm_inverse"]
        if process == "absorption" and axis[0] <= dark_origin <= axis[-1]:
            ax.axvline(dark_origin, color="#496675", ls=(0, (1.8, 2.8)), lw=0.85)
            ax.annotate("0–0 escura\nmodelo 2D", (dark_origin, 0.94), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color="#496675")
        ax.set(xlim=(axis[0], axis[-1]), ylim=(0, 1.12), yticks=(0, 0.5, 1))
        ax.set_title(title, loc="left", fontsize=11.5, pad=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}".replace(",", ".")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}".replace(".", ",")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e8ebee", lw=0.7, zorder=0)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    sensitivity = report["convergence"]["smaller_relaxation_range"]["spectral_differences"]["absorption"]
    if sensitivity["relative_l1_broadened_intensity_change"] > 0.05:
        axes[0].text(0.98, 0.62, "A progressão em alta energia\né sensível ao domínio do recorte", transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=8, color="#4a5860")
    axes[1].set_title(f"RHF–CIS/6-31G(d) · 0 K · σ = {report['sigma_cm_inverse']:g} cm⁻¹",
                      loc="right", fontsize=9, pad=9)
    fig.suptitle("Formaldeído  |  Espectro vibrônico e inversão", fontsize=15, x=0.04, ha="left")
    fig.supxlabel("Número de onda (cm⁻¹)")
    fig.supylabel("Intensidade relativa · escala comum por painel")
    fig.savefig(output / "h2co_inversion_ah.png", dpi=230, facecolor="white")
    fig.savefig(output / "h2co_inversion_ah.pdf", facecolor="white")
    plt.close(fig)

    fig, (potential, dipole) = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
    q = data["profile_q"]
    profile = data["profile_effective_cm_inverse"]
    potential.plot(q, profile, color="#17637f", lw=1.7)
    peak = max(data["s1_levels_relative_to_profile_min_cm_inverse"])
    for parity, energy, color in zip(("par", "ímpar"), data["s1_levels_relative_to_profile_min_cm_inverse"], ("#6b452a", "#795474")):
        potential.axhline(energy, color=color, ls="--", lw=1, label=f"nível {parity}")
    potential.set_ylim(0, max(peak*1.32, profile[len(q)//2]*1.28))
    potential.set(ylabel="Energia relativa (cm⁻¹)", title="Perfil efetivo de S₁")
    potential.legend(loc="upper center", fontsize=8, frameon=False)
    dipole.axhline(0, color="#8b8f95", lw=0.75)
    dipole.plot(q, data["profile_dipole_au"], color="#a74a28", lw=1.7)
    dipole.set(ylabel="Dipolo assinado μᵧ (u.a.)", title="Dipolo ao longo do vale de S₁")
    for ax in (potential, dipole):
        ax.set(xlim=(q[0], q[-1]), xlabel="Coordenada de piramidalização q")
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "h2co_inversion_profile.png", dpi=230, facecolor="white")
    fig.savefig(output / "h2co_inversion_profile.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="prior AH benchmark directory")
    parser.add_argument("electronic", type=Path, help="new inversion single-point directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="render existing spectra")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot(args.output)
    else:
        assemble(args.reference, args.electronic, args.output)
