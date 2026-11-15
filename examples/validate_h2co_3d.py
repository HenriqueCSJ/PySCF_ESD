"""H2CO inversion with a 3D fitted PySCF surface and a tensor sine DVR."""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_h2co_analytic_ah import CM_PER_HARTREE, SIGMA_CM, broaden
from validate_h2co_inversion import load_plane, projected_modes, rotation_metric_correction


PROCESSES = ("absorption", "emission")
GRID_SMALL, GRID_FINE = (63, 47, 31), (79, 55, 39)


def electronic_fields(data, reference):
    return np.column_stack((data["rhf_energy_hartree"] - reference,
                            data["cis_energy_hartree"] - reference, data["dipole_au"][:, 1]))


class InversionSurface:
    """Interpolate 3D corrections to the retained, finer t=0 plane."""

    def __init__(self, path, space, previous, data, reference, *, coarser=False):
        from scipy.interpolate import CubicSpline

        _, _, _, _, self.plane, _, _, _ = load_plane(previous, path, reference)
        q, r, t = (data[name].round(11) for name in ("q", "r", "t"))
        fields = electronic_fields(data, reference)
        if np.any(q < 0) or not np.any(q == 0) or not np.any(t == 0):
            raise ValueError("expected nonnegative inversion samples with a t=0 reference slice")
        if np.max(np.abs(data["dipole_au"][:, (0, 2)])) > 1e-7:
            raise ValueError("unexpected transverse dipole on the inversion surface")
        axes = [np.unique(values) for values in (q, r, t)]
        lookup = {(qi, ri, ti): i for i, (qi, ri, ti) in enumerate(zip(q, r, t))}
        if len(lookup) != q.size or np.prod([len(axis) for axis in axes]) != len(lookup):
            raise ValueError("the electronic volume must be a complete rectangular grid")
        order = np.array([[[lookup[qi, ri, ti] for ti in axes[2]] for ri in axes[1]] for qi in axes[0]])
        volume = fields[order]
        x, y, z = axes[0] * space.lengths[0], axes[1] * space.lengths[1], axes[2]
        zero = np.flatnonzero(z == 0)[0]
        t0 = volume[:, :, zero]
        self.base_error = np.max(np.abs(t0 - np.stack([f(x, y) for f in self.plane], axis=-1)), axis=(0, 1))
        if np.max(self.base_error[:2]) > 2e-8 or self.base_error[2] > 1e-7:
            raise ValueError(f"volume and archived t=0 cut disagree: {self.base_error}")
        self.shape = (x.size, y.size, z.size)
        self.bounds = ((float(-x[-1]), float(x[-1])), (float(y[0]), float(y[-1])), (float(z[0]), float(z[-1])))
        delta = volume - t0[:, :, None, :]
        if coarser:
            ix = np.union1d(np.r_[0, np.arange(2,len(x)-1,2), len(x)-1], np.flatnonzero(axes[0] == 1))
            iy = np.union1d(np.r_[0, np.arange(2,len(y)-1,2), len(y)-1], np.flatnonzero((axes[1] == 0) | (axes[1] == 1)))
            x, y, delta = x[ix], y[iy], delta[np.ix_(ix, iy)]
        self.planar_dipole = float(np.max(np.abs(volume[0, :, :, 2])))
        if self.planar_dipole > 1e-7:
            raise ValueError("nonvanishing dipole on the planar geometries")
        delta[0, :, :, 2] = 0.0
        sign = np.array([1, 1, -1])
        self.correction = np.concatenate((sign * delta[:0:-1], delta))
        nodes = np.r_[-x[:0:-1], x], y, z
        self.weights = tuple(CubicSpline(axis, np.eye(len(axis)), extrapolate=False) for axis in nodes)

    def grid(self, axes):
        """Return S0, S1 and signed dipole on product axes in mass-weighted units."""
        weights = [spline(axis) for spline, axis in zip(self.weights, axes)]
        out = np.einsum("ia,jb,kc,abcp->ijkp", *weights, self.correction, optimize=True)
        for i, field in enumerate(self.plane):
            out[:, :, :, i] += field(*axes[:2])[:, :, None]
        if not np.isfinite(out).all():
            raise ValueError("DVR box exceeds the calculated electronic domain")
        return np.moveaxis(out, -1, 0)

    def points(self, points, order=(0, 0, 0)):
        """Evaluate at arbitrary points, optionally differentiating the splines."""
        xyz = np.atleast_2d(points).T
        weights = [spline(axis, nu) for spline, axis, nu in zip(self.weights, xyz, order)]
        out = np.einsum("ia,ib,ic,abcp->ip", *weights, self.correction, optimize=True)
        if order[2] == 0:
            out += np.column_stack([f.ev(*xyz[:2], dx=order[0], dy=order[1]) for f in self.plane])
        return out


def audit_samples(path, surface, reference, lengths):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        points = np.column_stack((saved["q"] * lengths[0], saved["r"] * lengths[1], saved["t"]))
        predicted = surface.points(points)
        errors = predicted - electronic_fields(saved, reference)
        if not np.isfinite(errors).all():
            raise ValueError("validation samples lie outside the interpolation domain")
    return {
        "points": len(points),
        "max_abs_s0_s1_energy_cm_inverse": (np.max(np.abs(errors[:, :2]), axis=0) * CM_PER_HARTREE).tolist(),
        "rms_s0_s1_energy_cm_inverse": (np.sqrt(np.mean(errors[:, :2]**2, axis=0)) * CM_PER_HARTREE).tolist(),
        "max_abs_vertical_gap_error_cm_inverse": float(np.max(np.abs(errors[:, 1]-errors[:, 0]))*CM_PER_HARTREE),
        "max_abs_mu_y_au": float(np.max(np.abs(errors[:, 2]))),
    }


def optimize_slice(surface, x, guess, state=1):
    from scipy.optimize import minimize

    def objective(transverse):
        coords = np.r_[x, transverse]
        gradient = [surface.points(coords, order)[0, state] for order in ((0, 1, 0), (0, 0, 1))]
        return surface.points(coords)[0, state], np.asarray(gradient)

    optimum = minimize(objective, guess, jac=True, method="L-BFGS-B", bounds=surface.bounds[1:],
                       options={"ftol": 1e-14, "gtol": 1e-11})
    if np.max(np.abs(optimum.jac)) > 1e-7:
        raise ValueError(f"poor stationarity in the slice x={x:.6g}: {optimum.message}")
    return optimum


def fitted_curvature(surface, point, state):
    from itertools import combinations_with_replacement

    hessian = np.empty((3, 3))
    for a, b in combinations_with_replacement(range(3), 2):
        order = np.bincount([a, b], minlength=3)
        hessian[a, b] = hessian[b, a] = surface.points(point, order)[0, state]
    return np.linalg.eigvalsh(hessian)


def fitted_saddle(surface, space, gap_el):
    guess = np.array(space.saddle_coordinates[1:]) * [space.lengths[1], 1.0]
    center = optimize_slice(surface, 0, guess)
    point = np.r_[0, center.x]
    curvatures = fitted_curvature(surface, point, 1)
    if np.count_nonzero(curvatures < 0) != 1 or not all(left < value < right for value, (left,right) in zip(center.x, surface.bounds[1:])):
        raise ValueError("the fitted planar geometry does not have exactly one unstable direction")
    return point, {
        "barrier_cm_inverse": float((center.fun - gap_el) * CM_PER_HARTREE),
        "anchor_barrier_predicted_by_interpolant_cm_inverse": float((surface.points(np.r_[0, guess])[0, 1] - gap_el) * CM_PER_HARTREE),
        "fitted_position_q_r_t": (point / np.r_[space.lengths, 1]).tolist(),
        "max_gradient_hartree_per_mass_weighted_bohr": float(np.max(np.abs(center.jac))),
        "curvature_eigenvalues_atomic_units": curvatures.tolist(),
    }


def nuclear_states(surface, bounds, shape, zpe):
    from pyscf_esd.anharmonic_volume import SineDVR3D

    dvr = SineDVR3D(bounds, shape)
    ground, excited, dipole = surface.grid(dvr.axes)
    excited += zpe
    states = [dvr.solve(energy, parity=parity, n_levels=1)
              for energy in (ground, excited) for parity in ("even", "odd")]
    if max(result.residuals[0] for result in states) > 1e-8:
        raise ValueError("unconverged nuclear eigenstate")
    return dvr, (ground, excited), dipole, states


def transitions(states, dipole):
    g0, g1, e0, e1 = states
    ground, excited = g0.vectors[0], e0.vectors[0]
    a0 = float(g0.energies[0])
    a1 = float(e0.energies[0])
    b0 = float(g1.energies[0])
    b1 = float(e1.energies[0])
    return {
        "lowest_s1_inversion_gap_cm_inverse": (b1-a1) * CM_PER_HARTREE,
        "dark_0_0_cm_inverse": (a1-a0) * CM_PER_HARTREE,
        "dark_0_0_strength_au2": float(np.vdot(excited, ground*dipole)**2),
        "first_bright_absorption_cm_inverse": (b1-a0) * CM_PER_HARTREE,
        "first_bright_absorption_strength_au2": float(np.vdot(e1.vectors[0], ground*dipole)**2),
        "first_bright_relaxed_emission_cm_inverse": (a1-b0) * CM_PER_HARTREE,
        "first_bright_relaxed_emission_strength_au2": float(np.vdot(g1.vectors[0], excited*dipole)**2),
        "max_eigenvector_residual_hartree": float(max(state.residuals[0] for state in states)),
    }


def transverse_profile(surface, dvr, saddle, space, states, zpe, gap_el):
    from scipy.fft import dstn
    from scipy.optimize import minimize_scalar
    from scipy.sparse.linalg import eigsh, LinearOperator

    y, t = dvr.axes[1:]
    shape = (len(y), len(t))
    ky, kt = (0.5 * (np.pi * np.arange(1, n + 1) / (right-left))**2
              for n, (left, right) in zip(shape, dvr.bounds[1:]))
    kinetic = ky[:, None] + kt[None, :]
    q = np.linspace(0, 1.65, 51)
    energies = surface.grid((q * space.lengths[0], y, t))[1]
    effective, electronic, previous_plane, mu, residuals = [], [], [], [], []
    guess, v0 = saddle[1:], np.ones(np.prod(shape))
    for x, potential in zip(q * space.lengths[0], energies):
        minimum = optimize_slice(surface, x, guess)
        guess = minimum.x
        electronic.append(minimum.fun)
        mu.append(float(surface.points(np.r_[x, guess])[0, 2]))
        planar = minimize_scalar(lambda value: float(surface.plane[1].ev(x, value)),
                                 bounds=surface.bounds[1], method="bounded")
        previous_plane.append(planar.fun)

        def act(vector):
            array = vector.reshape(shape)
            transformed = dstn(array, type=1, norm="ortho")
            return (dstn(kinetic * transformed, type=1, norm="ortho") + potential * array).ravel()

        size = int(np.prod(shape))
        level, vectors = eigsh(LinearOperator((size, size), matvec=act, dtype=float), k=1,
                               which="SA", tol=1e-11, v0=v0)
        v0 = vectors[:, 0]
        effective.append(level[0] + zpe)
        residuals.append(np.linalg.norm(act(v0) - level[0]*v0))

    energy_bottom = min(effective)
    e1 = np.array([float(state.energies[0]) for state in states[2:]])
    q_signed = np.r_[-q[:0:-1], q]

    def mirror(values, sign=1):
        return np.r_[sign * np.asarray(values)[:0:-1], values]

    sample = {"profile_q": q_signed,
              "profile_3d_electronic": (mirror(electronic) - gap_el) * CM_PER_HARTREE,
              "profile_2d_electronic": (mirror(previous_plane) - gap_el) * CM_PER_HARTREE,
              "profile_3d_effective": (mirror(effective) - energy_bottom) * CM_PER_HARTREE,
              "profile_dipole_y": mirror(mu, sign=-1),
              "levels_relative_to_effective_min": (e1 - energy_bottom) * CM_PER_HARTREE}
    info = {"barrier_effective_transverse_cm_inverse": float((effective[0] - energy_bottom)*CM_PER_HARTREE),
            "levels_relative_to_effective_barrier_cm_inverse": ((e1-effective[0])*CM_PER_HARTREE).tolist(),
            "electronic_barrier_previous_plane_cm_inverse": float((previous_plane[0] - gap_el)*CM_PER_HARTREE),
            "minimum_electronic_valley_minus_archived_well_cm_inverse": float((min(electronic) - gap_el)*CM_PER_HARTREE),
            "max_transverse_residual_hartree": float(max(residuals))}
    if info["max_transverse_residual_hartree"] > 1e-9:
        raise ValueError("transverse-profile eigenvectors did not converge")
    return sample, info


def positive_bins(measure, initial, sign):
    energy = sign * (measure.energies - initial) * CM_PER_HARTREE
    mask = energy > 0
    return np.column_stack((energy[mask], measure.weights[mask]))


def spectral_quadratures(nuclear, name):
    dvr, potentials, dipole, states = nuclear
    output = {}
    for process, start, target, sign in (
        ("absorption", states[0], potentials[1], 1),
        ("emission", states[2], potentials[0], -1),
    ):
        if name == "main":
            orders = (640, 1040, 1440) if process == "absorption" else (640, 960, 1200)
        else:
            orders = (780, 1120) if process == "absorption" else (640, 920)
        print(f"{name} {process}: Lanczos {orders}", flush=True)
        measures = dvr.spectral_measures(target, dipole * start.vectors[0], parity="odd", orders=orders)
        output[process] = measures, [positive_bins(measure, start.energies[0], sign) for measure in measures]
    return output


def spectra(optical_data, tests, base_ah, previous):
    """Compare the new spectral measure with the saved 2D and six-mode sticks."""
    result, diagnostics = {}, {}
    with np.load(previous / "spectra" / "h2co_inversion_spectra.npz") as old:
        for process, power in zip(PROCESSES, (1,3)):
            measures, quadratures = optical_data[process]
            last = quadratures[-1]
            models = {"volume": last, "plane": old[f"{process}_coupled_sticks"][:, :2],
                      "ah": old[f"{process}_archived_ah_sticks"][:, :2]}
            significant = []
            for sticks in models.values():
                weight = sticks[:, 1] * (sticks[:, 0] / base_ah)**power
                major = sticks[weight >= weight.max()*0.002, 0]
                significant.extend((major.min(), major.max()))
            axis = np.arange(min(significant)-280, max(significant)+282, 2.0)
            result[f"{process}_x"] = axis

            def broad(sticks):
                return broaden(sticks[:, 0], sticks[:, 1] * (sticks[:, 0] / base_ah)**power, axis, SIGMA_CM)

            for model, sticks in models.items():
                result[f"{process}_{model}"] = broad(sticks)
            last_curve = result[f"{process}_volume"]
            integrated = last_curve.sum()
            diagnostics[process] = {
                "lanczos_orders": [measure.order for measure in measures],
                "l1_relative_change_to_highest_order": [float(np.abs(last_curve - broad(bins)).sum()/integrated) for bins in quadratures[:-1]],
                "positive_photon_strength_au2": float(last[:, 1].sum()),
                "all_energy_quadrature_weight_au2": float(measures[-1].projected_norm_squared),
                "strongest_peak_cm_inverse": float(axis[np.argmax(last_curve)]),
            }
            convergence = {}
            near = axis <= last[last[:,1] >= last[:,1].max()*1e-4, 0].min() + 1200
            for probe_name, probe in tests.items():
                probe_measures, probe_bins = probe[process]
                probe_curve = broad(probe_bins[-1])
                difference = np.abs(last_curve - probe_curve)
                convergence[probe_name] = {
                    "lanczos_orders": [m.order for m in probe_measures],
                    "l1_relative_change_between_probe_orders": float(np.abs(probe_curve - broad(probe_bins[0])).sum()/probe_curve.sum()),
                    "l1_relative_change_from_main": float(difference.sum()/integrated),
                    "main_vs_probe_change_within_1200_cm_inverse_of_onset": (
                        float(difference[near].sum()/last_curve[near].sum()) if process == "absorption" else None
                    ),
                }
            diagnostics[process]["grid_or_box_probe"] = convergence
            result[f"{process}_volume_quadrature_bins"] = last
            print(f"{process}: {diagnostics[process]}", flush=True)
    return result, diagnostics


def assemble(ah, previous, electronic, output):
    from concurrent.futures import ThreadPoolExecutor
    from pyscf_esd.inversion_path import InversionPath
    from pyscf_esd.inversion_space import InversionSpace

    with np.load(ah / "electronic" / "electronic_results.npz") as store:
        endpoints = dict(store)
    path = InversionPath(endpoints["geometry_s0_bohr"], endpoints["geometry_s1_bohr"], endpoints["masses_amu"])
    old_electronic = previous / "electronic"
    old_planar = json.loads((old_electronic / "planar_relaxation.json").read_text(encoding="utf-8"))
    space = InversionSpace(path, old_planar["geometry_bohr"])
    reference = float(endpoints["ground_energy_hartree"])
    gap_el = float(endpoints["excited_energy_hartree"]) - reference
    refined = electronic / "electronic_volume_refined.npz"
    electronic_source = refined if refined.is_file() else electronic / "electronic_volume.npz"
    with np.load(electronic_source) as store:
        calculated = dict(store)
    actual_points = np.column_stack((calculated["q"] * space.lengths[0], calculated["r"] * space.lengths[1], calculated["t"]))
    geometry = space.origin + (actual_points @ space.directions.T).reshape(-1,4,3) / np.sqrt(space.masses_au)[None,:,None]
    np.testing.assert_allclose(calculated["geometry_bohr"], geometry, atol=1e-10, rtol=0)
    surface = InversionSurface(path, space, old_electronic, calculated, reference)
    coarse = InversionSurface(path, space, old_electronic, calculated, reference, coarser=True)
    holdouts = audit_samples(electronic / "electronic_volume_holdout.npz", surface, reference, space.lengths)
    new_holdouts = electronic / "electronic_volume_refined_holdout.npz"
    near_well_holdouts = audit_samples(new_holdouts, surface, reference, space.lengths) if new_holdouts.is_file() else None
    mirror = audit_samples(electronic / "electronic_volume_mirror_audit.npz", surface, reference, space.lengths)
    active_modes, spectator_modes = projected_modes(endpoints, path, space.directions)
    fit_frequencies = []
    for state, point in enumerate((np.zeros(3), np.r_[space.lengths, 0])):
        curvature = fitted_curvature(surface, point, state)
        if np.any(curvature <= 0):
            raise ValueError("the fitted surface does not have positive curvature at the archived minimum")
        fit_frequencies.append((np.sqrt(curvature)*CM_PER_HARTREE).tolist())
    zpe = float(0.5 * (spectator_modes[1]-spectator_modes[0]).sum())
    saddle, barrier = fitted_saddle(surface, space, gap_el)
    with np.load(electronic / "electronic_saddle.npz") as exact:
        direct = float(np.asarray(exact["cis_energy_hartree"]).item())
        np.testing.assert_allclose(exact["geometry_bohr"].reshape(4,3), space.geometry(*space.saddle_coordinates), atol=1e-10, rtol=0)
    barrier["direct_planar_3d_barrier_cm_inverse"] = (direct - reference - gap_el) * CM_PER_HARTREE
    barrier["prior_relaxed_planar_barrier_cm_inverse"] = old_planar["relaxed_planar_barrier_cm1"]
    print(f"Electronic barrier {barrier['barrier_cm_inverse']:.5f} cm^-1 (direct anchor {barrier['direct_planar_3d_barrier_cm_inverse']:.5f})", flush=True)
    print(f"Holdout errors {holdouts}", flush=True)

    nuclear = nuclear_states(surface, surface.bounds, GRID_FINE, zpe)
    dvr, potentials, dipole, states = nuclear
    low = transitions(states, dipole)
    if low["lowest_s1_inversion_gap_cm_inverse"] <= 0 or low["dark_0_0_strength_au2"] > 1e-12:
        raise ValueError("the low-state inversion ordering or dipole parity is inconsistent")
    print(f"3D low states {low}", flush=True)
    checks = {}
    optical_probes = {}
    tighter = (*surface.bounds[:2], (-32., 32.) if surface.bounds[2][1] > 32 else tuple(0.85*value for value in surface.bounds[2]))
    for label, fit, bounds in (("smaller_dvr", surface, surface.bounds), ("narrower_t_box", surface, tighter),
                               ("coarser_electronic_mesh", coarse, surface.bounds)):
        probe = nuclear_states(fit, bounds, GRID_SMALL, zpe)
        _, _, tested_dipole, tested_states = probe
        checks[label] = transitions(tested_states, tested_dipole)
        print(f"{label} gap {checks[label]['lowest_s1_inversion_gap_cm_inverse']:.6f} cm^-1", flush=True)
        if label != "coarser_electronic_mesh":
            optical_probes[label] = probe

    diagnostic_data, transverse = transverse_profile(surface, dvr, saddle, space, states, zpe, gap_el)
    previous_report = json.loads((previous / "spectra" / "h2co_inversion_report.json").read_text(encoding="utf-8"))
    if previous_report["sigma_cm_inverse"] != SIGMA_CM or previous_report["temperature_kelvin"] != 0:
        raise ValueError("the previous spectra used incompatible settings")
    old_ah = previous_report["archived_ah_6d_00_cm_inverse"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary = pool.submit(spectral_quadratures, nuclear, "main")
        futures = {label: pool.submit(spectral_quadratures, trial, label) for label, trial in optical_probes.items()}
        curves, optical = spectra(primary.result(), {label: future.result() for label, future in futures.items()}, old_ah, previous)
    report = {
        "method": "PySCF RHF–TDA/CIS/6-31G(d), three mass-weighted Cartesian coordinates, matrix-free sine DVR",
        "electronic_data_file": electronic_source.name,
        "saddle_q_r_t": space.saddle_coordinates, "electronic_box_mass_weighted_au": surface.bounds,
        "electronic_grid_positive_q_shape": surface.shape,"new_and_retained_grid_points": len(actual_points),
        "ground_reference_hartree": reference, "electronic_adiabatic_gap_cm_inverse": gap_el*CM_PER_HARTREE,
        "spectator_frequencies_cm_inverse": [(freq*CM_PER_HARTREE).tolist() for freq in spectator_modes],
        "analytic_projected_active_frequencies_cm_inverse": [(freq*CM_PER_HARTREE).tolist() for freq, _ in active_modes],
        "interpolated_active_frequencies_at_endpoints_cm_inverse": fit_frequencies,
        "three_spectator_zpe_difference_cm_inverse": zpe*CM_PER_HARTREE,
        "same_t0_plane_max_errors_hartree_hartree_au": surface.base_error.tolist(),
        "planar_mu_y_bound_au": surface.planar_dipole,
        "dipole_and_energy_holdouts": holdouts, "refined_near_well_holdouts": near_well_holdouts,
        "mirror_samples_vs_fit": mirror,
        "electronic_barrier": barrier, "transverse_diagnostics": transverse, "low_states": low,
        "previous_2d_lowest_gap_cm_inverse": previous_report["s1_lowest_gap_cm_inverse"],
        "q_r_t_dvr_shape": GRID_FINE, "convergence": checks, "spectra": optical,
        "temperature_kelvin": 0, "sigma_cm_inverse": SIGMA_CM,
        "relaxed_emission_initial_state": "lowest even vibrational S1 state",
        "optical_weighting": "E/Eref absorption and (E/Eref)^3 emission, Eref = prior AH 6D origin",
        "metric_rotation_term_size_at_endpoints": rotation_metric_correction(path, space.directions),
    }
    np.savez_compressed(output / "h2co_inversion_3d_spectra.npz", **diagnostic_data, **curves)
    (output / "h2co_inversion_3d_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def plot(output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    with np.load(output / "h2co_inversion_3d_spectra.npz") as archive:
        data = dict(archive)
    report = json.loads((output / "h2co_inversion_3d_report.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.9), layout="constrained")
    series = (("volume", "Anarmônico 3D", "#17637f", "-"),
              ("plane", "Anarmônico 2D", "#a74a28", "--"),
              ("ah", "AH 6D · HT e Duschinsky", "#666371", ":"))
    for ax, process, title in zip(axes, PROCESSES, ("Absorção", "Emissão após relaxação em S₁")):
        x = data[f"{process}_x"]
        scale = max(data[f"{process}_{model}"].max() for model, *_ in series)
        for name, label, color, line in series:
            ax.plot(x, data[f"{process}_{name}"] / scale, color=color, ls=line, lw=1.5, label=label)
        ax.set(xlim=(x[0], x[-1]), ylim=(0, 1.14), yticks=(0,0.5,1))
        ax.set_title(title, loc="left", fontsize=11.5, pad=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}".replace(",", ".")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}".replace(".", ",")))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#e8ebee", lw=0.7)
    dark = report["low_states"]["dark_0_0_cm_inverse"]
    if axes[0].get_xlim()[0] < dark < axes[0].get_xlim()[1]:
        axes[0].axvline(dark, color="#496675", ls=":", lw=1)
        axes[0].annotate("0–0 escura\nmodelo 3D", (dark, .94), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=8, color="#496675")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_title(f"RHF–CIS/6-31G(d) · 0 K · σ = {SIGMA_CM:g} cm⁻¹", loc="right", fontsize=9, pad=9)
    fig.suptitle("Formaldeído  |  Inversão anarmônica em 3D", fontsize=15, x=.04, ha="left")
    fig.supxlabel("Número de onda (cm⁻¹)")
    fig.supylabel("Intensidade relativa · escala comum por painel")
    fig.savefig(output / "h2co_inversion_3d_comparison.png", dpi=230, facecolor="white")
    fig.savefig(output / "h2co_inversion_3d_comparison.pdf", facecolor="white")
    plt.close(fig)

    fig, (electronic, effective) = plt.subplots(1, 2, figsize=(8.6, 3.4), layout="constrained")
    q = data["profile_q"]
    electronic.plot(q, data["profile_3d_electronic"], color="#17637f", lw=1.7, label="Recorte 3D")
    electronic.plot(q, data["profile_2d_electronic"], color="#a74a28", ls="--", lw=1.45, label="Recorte 2D")
    anchor = report["electronic_barrier"]["direct_planar_3d_barrier_cm_inverse"]
    electronic.scatter([0], [anchor], s=25, color="#332a38", zorder=4, label="Planar relaxado · cálculo direto")
    electronic.set(ylabel="Energia relativa (cm⁻¹)", title="Barreira eletrônica de S₁")
    electronic.legend(loc="upper center", frameon=False, fontsize=8)
    effective.plot(q, data["profile_3d_effective"], color="#17637f", lw=1.7)
    levels = data["levels_relative_to_effective_min"]
    for energy, parity, color in zip(levels, ("par", "ímpar"), ("#814e2c", "#795474")):
        effective.axhline(energy, color=color, ls="--", lw=1.05, label=f"Nível {parity}")
    effective.set(ylim=(0, max(max(levels)*1.35, np.max(data["profile_3d_effective"])*1.04)),
                  ylabel="Energia relativa (cm⁻¹)", title="Perfil efetivo e níveis de S₁")
    effective.legend(loc="upper center", frameon=False, fontsize=8)
    for ax in (electronic, effective):
        ax.set(xlim=(q[0],q[-1]), xlabel="Coordenada de piramidalização q")
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "h2co_inversion_3d_barrier.png", dpi=230, facecolor="white")
    fig.savefig(output / "h2co_inversion_3d_barrier.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ah", type=Path, help="archived analytic AH benchmark")
    parser.add_argument("previous", type=Path, help="archived 2D inversion benchmark")
    parser.add_argument("electronic", type=Path, help="new 3D PySCF grid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="plot already calculated arrays")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot(args.output)
    else:
        assemble(args.ah, args.previous, args.electronic, args.output)
