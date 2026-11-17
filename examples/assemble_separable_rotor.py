"""Separable thermal rotor–spectator spectra from qualified electronic cuts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.constants import physical_constants
from scipy.spatial.transform import Rotation

from pyscf_esd.anharmonic import HarmonicSpectators
from pyscf_esd.anharmonic_dvr import solve_dvr
from pyscf_esd.anharmonic_sampling import PhasedRotorDipole, evaluate_rotor_samples
from pyscf_esd.inversion_path import _AMU_TO_ELECTRON_MASS
from pyscf_esd.rotor_coordinates import map_transported_spectators, project_torsion
from pyscf_esd.spectral_properties import _EINSTEIN_A_FACTOR, HARTREE_TO_WAVENUMBER
from pyscf_esd.thermal_separable import thermal_separable_spectrum


PERIOD = 2 * np.pi
# Integrated cross section over wavenumber is π r_e f, with r_e in cm.
F_DENSITY_TO_CROSS_SECTION = np.pi * 100 * physical_constants["classical electron radius"][0]
PROCESSES = ("absorption", "emission")
ABSOLUTE_FACTORS = (2 / 3, _EINSTEIN_A_FACTOR)


def _body_frame(xyz, indices):
    first, middle, last = xyz[list(indices)]
    z = middle - first
    if np.linalg.norm(z) <= 1e-10:
        raise ValueError("the molecular frame has coincident axis atoms")
    z /= np.linalg.norm(z)
    x = last - middle
    x -= (x @ z) * z
    if np.linalg.norm(x) <= 1e-10:
        raise ValueError("the frame-defining atoms must be noncollinear")
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def _field(angles, values, targets, parity=1):
    return evaluate_rotor_samples(angles, values, np.atleast_1d(targets), closure_sign=parity)


@dataclass(frozen=True)
class Cut:
    angle: np.ndarray
    xyz: np.ndarray
    tangent: np.ndarray
    values: np.ndarray
    signs: tuple[int, ...]
    minimum_abs_overlaps: np.ndarray
    original: dict

    @property
    def boundary(self):
        return "periodic" if self.signs[0] == 1 else "antiperiodic"

    def geometry(self, angle):
        return _field(self.angle, self.xyz, angle)[0], _field(self.angle, self.tangent, angle)[0]


def _cut(record, name, masses, frame_indices, minimum):
    is_dipole = name == "midpoint"
    values_key = "dipole_au" if is_dipole else "energy_eh"
    overlap_keys = ("overlap_s0", "overlap_s1") if is_dipole else ("overlap",)
    points = record["points"]
    size = len(points)
    if size < 4 or any(row.get("tracking_passed") is not True for row in points):
        raise ValueError(f"{name} requires a complete tracked loop including an explicit duplicate")
    theta = np.deg2rad([row["angle_deg"] for row in points])
    xyz = np.asarray([row["coordinates_bohr"] for row in points], dtype=float)
    tangent = np.asarray([row["tangent_bohr_per_rad"] for row in points], dtype=float)
    values = np.asarray([row[values_key] for row in points], dtype=float)
    if xyz.shape != (size, len(masses), 3) or tangent.shape != xyz.shape:
        raise ValueError(f"{name} geometry and analytic tangent shapes disagree with the masses")
    if values.shape != ((size, 3) if is_dipole else (size,)):
        raise ValueError(f"{name} {values_key} has the wrong shape")
    if not all(np.isfinite(x).all() for x in (theta, xyz, tangent, values)):
        raise ValueError(f"{name} contains nonfinite data")
    if not np.allclose(theta, theta[0] + np.linspace(0, PERIOD, size), atol=1e-8, rtol=0):
        raise ValueError(f"{name} needs uniform and contiguous labeled 2π samples")
    if frame_indices is not None:
        frames = np.stack([_body_frame(row, frame_indices) for row in xyz])
        xyz, tangent = xyz @ frames, tangent @ frames
        if is_dipole:
            values = (values[:, None, :] @ frames)[:, 0, :]
    xyz -= np.average(xyz, axis=1, weights=masses)[:, None, :]
    tangent -= np.average(tangent, axis=1, weights=masses)[:, None, :]
    if max(np.max(np.abs(xyz[-1] - xyz[0])), np.max(np.abs(tangent[-1] - tangent[0]))) > 1e-6:
        raise ValueError(f"{name} closure lost the labeled geometry or analytic tangent")

    overlaps = np.asarray([[row[key] for key in overlap_keys] for row in points[1:]], dtype=float)
    duplicate = np.asarray([record[f"closure_{key}"] for key in overlap_keys], dtype=float)
    links = np.vstack((overlaps, duplicate))
    if not np.isfinite(links).all() or np.any(np.abs(links) > 1 + 1e-8) or np.any(np.abs(links) < minimum):
        raise ValueError(f"{name} has a normalized electronic overlap below the qualification threshold")
    signs = tuple(int(value) for value in np.prod(np.sign(links), axis=0))
    if is_dipole:
        if np.linalg.norm(values[-1] - np.prod(np.sign(duplicate)) * values[0]) > max(
            1e-7, 1e-3 * np.linalg.norm(values[0])
        ):
            raise ValueError("the duplicate midpoint transition dipole disagrees with its electronic closure gauge")
        phases = np.r_[1.0, np.cumprod(np.prod(np.sign(overlaps[:-1]), axis=1))]
        values = values * np.r_[phases, 1.0][:, None]
    elif abs(values[-1] - values[0]) > 1e-7:
        raise ValueError(f"{name} closure changed the absolute electronic energy")
    return Cut(theta[:-1], xyz[:-1], tangent[:-1], values[:-1], signs, np.abs(links).min(axis=0), record)


def _cut_inertia(cut, masses):
    moments = np.array([
        project_torsion(xyz, masses, tangent).inertia_me_bohr2
        for xyz, tangent in zip(cut.xyz, cut.tangent, strict=True)
    ])
    # Uniform-angle average of the inverse metric on each independent cut.
    effective = float(1 / np.mean(1 / moments))
    return dict(
        effective_me_bohr2=effective,
        min_sampled_me_bohr2=float(moments.min()),
        max_sampled_me_bohr2=float(moments.max()),
        sampled_span_fraction=float(np.ptp(moments) / effective),
    )


@dataclass(frozen=True)
class HarmonicCut:
    angle: float
    frequencies: np.ndarray
    modes: np.ndarray
    hessian_method: str
    is_same_surface: bool
    energy_anchor_difference_eh: float | None


def _harmonic_cut(cut, masses, frame_indices):
    stationary = cut.original["stationary"]
    angle = np.deg2rad(float(stationary["angle_deg"]))
    coordinates = np.array(stationary["coordinates_bohr"], dtype=float, copy=True)
    hessian = np.asarray(stationary["hessian_eh_bohr2"], dtype=float)
    natoms = len(masses)
    if hessian.shape == (natoms, natoms, 3, 3):
        hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natoms, 3 * natoms)
    if coordinates.shape != (natoms, 3) or hessian.shape != (3 * natoms, 3 * natoms):
        raise ValueError("stationary Cartesian geometry and Hessian dimensions disagree with masses")
    if not all(np.isfinite(row).all() for row in (angle, coordinates, hessian)):
        raise ValueError("stationary Cartesian geometry and Hessian must be finite")
    if not isinstance(stationary.get("hessian_method"), str) or not stationary["hessian_method"].strip():
        raise ValueError("declare the analytic Hessian method for each stationary geometry")
    if frame_indices is not None:
        frame = _body_frame(coordinates, frame_indices)
        transform = np.kron(np.eye(natoms), frame)
        coordinates = coordinates @ frame
        hessian = transform.T @ hessian @ transform
    coordinates -= np.average(coordinates, axis=0, weights=masses)
    expected, tangent = cut.geometry(angle)
    if np.max(np.abs(coordinates - expected)) > 1e-5:
        raise ValueError("the analytic Hessian anchor does not reproduce the sampled surface-cut geometry")
    anchor_delta = None
    if stationary.get("energy_eh") is not None:
        zero = cut.values[0]
        energy = zero + _field(cut.angle, cut.values - zero, angle)[0]
        anchor_delta = float(energy - stationary["energy_eh"])
        if not np.isfinite(anchor_delta) or abs(anchor_delta) > 1e-6:
            raise ValueError("the independent scan and stationary anchor disagree in absolute electronic energy")

    space = project_torsion(coordinates, masses, tangent).spectator_axes
    weights = np.repeat(np.sqrt(masses), 3)
    weighted_hessian = hessian / np.outer(weights, weights)
    if np.max(np.abs(weighted_hessian - weighted_hessian.T)) > 1e-9:
        raise ValueError("the mass-weighted analytic Hessian fails Cartesian symmetry")
    block = space.T @ (0.5 * (weighted_hessian + weighted_hessian.T)) @ space
    eigvals, vectors = np.linalg.eigh(block)
    if np.any(eigvals <= 0):
        raise ValueError(f"spectator-only projected Hessian has nonpositive curvature {eigvals.min():.8g} Eh")
    # The explicitly sampled rotor already supplies its curvature. The bath
    # takes exactly the 3N-7-dimensional restriction and independently tests it.
    return HarmonicCut(
        float(angle), np.sqrt(eigvals), space @ vectors, stationary["hessian_method"],
        stationary.get("hessian_is_same_surface") is True, anchor_delta,
    )


def _transport(modes, cut, masses, angle, target, axis, moving, sense, steps):
    weakest = 1.0
    previous_xyz, _ = cut.geometry(angle)
    increment = (target - angle) / steps
    for node in np.linspace(angle + increment, target, steps):
        xyz, tangent = cut.geometry(node)
        basis = project_torsion(xyz, masses, tangent).spectator_axes
        bond = previous_xyz[axis[1]] - previous_xyz[axis[0]]
        rotation = Rotation.from_rotvec(sense * increment * bond / np.linalg.norm(bond)).as_matrix()
        rotated = modes.reshape(len(masses), 3, -1).copy()
        rotated[moving] = np.einsum("ij,ajm->aim", rotation, rotated[moving])
        u, singular, vt = np.linalg.svd(basis.T @ rotated.reshape(modes.shape), full_matrices=False)
        weakest = min(weakest, float(singular[-1]))
        if weakest < 1e-6:
            raise ValueError("spectator transport loses an independent angular tangent direction")
        modes = basis @ u @ vt
        previous_xyz = xyz
    return modes, weakest


def _spectators(cuts, midpoint, masses, frame_indices, axis, moving, sense):
    charts = tuple(_harmonic_cut(cut, masses, frame_indices) for cut in cuts)
    start, finish = (row.angle for row in charts)
    finish = start + (finish - start + np.pi) % PERIOD - np.pi
    target = (start + finish) / 2
    xyz, tangent = midpoint.geometry(target)
    common = project_torsion(xyz, masses, tangent)
    displacement = cuts[0].geometry(target)[0] - cuts[1].geometry(target)[0]

    def mapping(steps):
        transported = [
            _transport(chart.modes, cut, masses, angle, target, axis, moving, sense, steps)
            for cut, chart, angle in zip(cuts, charts, (start, finish), strict=True)
        ]
        result = map_transported_spectators(transported[0][0], transported[1][0], common, displacement, masses)
        return result, [item[1] for item in transported]

    coarse, _ = mapping(32)
    result, steps_minima = mapping(64)
    j, k = result.rotation, result.shift_au
    w0, w1 = (item.frequencies for item in charts)
    models = (HarmonicSpectators(w0, w1, 0, 0, j, k), HarmonicSpectators(w1, w0, 0, 0, j.T, -j.T @ k))
    diagnostics = dict(
        model="harmonic spectator Hessian blocks at the independent endpoint geometries",
        stationary_angles_deg=np.rad2deg([start, finish]).tolist(),
        common_angle_deg=float(np.rad2deg(target)),
        hessian_methods=[item.hessian_method for item in charts],
        hessians_are_native_to_electronic_surfaces=[item.is_same_surface for item in charts],
        anchor_electronic_energy_difference_eh=[item.energy_anchor_difference_eh for item in charts],
        frequencies_cm_inverse=[(w * HARTREE_TO_WAVENUMBER).tolist() for w in (w0, w1)],
        zpe_cm_inverse=[float(0.5 * w.sum() * HARTREE_TO_WAVENUMBER) for w in (w0, w1)],
        zpe_excited_minus_ground_cm_inverse=float(0.5 * (w1.sum() - w0.sum()) * HARTREE_TO_WAVENUMBER),
        duschinsky_rotation=j.tolist(),
        duschinsky_shift_sqrt_me_bohr=k.tolist(),
        spectator_reorganization_eh=[
            float(0.5 * np.linalg.norm(w0 * (j.T @ k))**2), float(0.5 * np.linalg.norm(w1 * k)**2),
        ],
        common_projection_singular_values=[
            result.initial_projection_singular_values.tolist(), result.final_projection_singular_values.tolist(),
        ],
        minimum_step_transport_singular_values=steps_minima,
        rotation_drift_frobenius_32_to_64=float(np.linalg.norm(j - coarse.rotation)),
        displacement_drift_sqrt_me_bohr_32_to_64=float(np.linalg.norm(k - coarse.shift_au)),
    )
    return models, diagnostics


def _run_profiles(cuts, midpoint, baths, moments, grid, temperature, fwhm, step):
    theta = midpoint.angle[0] + PERIOD * np.arange(grid) / grid
    zero = float(cuts[0].values.min())
    potentials = [zero + _field(cut.angle, cut.values - zero, theta) for cut in cuts]
    if np.min(potentials[1] - potentials[0]) <= 0:
        raise ValueError("independently sampled absolute surface cuts reverse energy order")
    rotor = [
        solve_dvr(theta, energy, boundary=cut.boundary, n_levels=grid, kinetic_coefficient=1 / moment)
        for cut, energy, moment in zip(cuts, potentials, moments, strict=True)
    ]
    parity = int(np.prod(midpoint.signs))
    dipole = PhasedRotorDipole(theta, _field(midpoint.angle, midpoint.values, theta, parity), parity)
    return tuple(
        thermal_separable_spectrum(
            initial, final, transition_dipole_au=dipole, spectators=bath, process=process,
            temperature_kelvin=temperature, fwhm_cm_inverse=fwhm, step_cm_inverse=step,
        )
        for initial, final, bath, process in zip(rotor, rotor[::-1], baths, PROCESSES, strict=True)
    )


def _density(profile, factor):
    return factor * profile.photon_weighted_density_per_cm_inverse


def _stats(profile, factor):
    axis = profile.wavenumber_cm_inverse
    density = _density(profile, factor)
    active = profile.positive_photon_weighted_strength > 0
    return dict(
        origin_00_cm_inverse=profile.origin_00_cm_inverse,
        centroid_cm_inverse=profile.centroid_cm_inverse,
        peak_cm_inverse=float(axis[np.argmax(density)]) if active else None,
        strength_sum=profile.positive_photon_weighted_strength * factor,
        profile_area=float(density.sum() * profile.step_cm_inverse),
        raw_positive_photon_weighted_strength=profile.positive_photon_weighted_strength,
        signed_thermal_dipole_strength_au2=profile.signed_dipole_strength_au2,
        initial_population_coverage=profile.initial_population_coverage,
        final_dipole_strength_coverage=profile.final_dipole_strength_coverage,
        grid_negative_weight_fraction=profile.negative_weight_grid_fraction,
        grid_tail_probability_bound=profile.tail_probability_bound,
    )


def _convergence(coarse, fine):
    axis = fine.wavenumber_cm_inverse
    reference = fine.photon_weighted_density_per_cm_inverse
    guess = np.interp(axis, coarse.wavenumber_cm_inverse, coarse.photon_weighted_density_per_cm_inverse, left=0, right=0)
    return dict(
        origin_change_cm_inverse=float(fine.origin_00_cm_inverse - coarse.origin_00_cm_inverse),
        profile_l1_relative_change=None if reference.sum() == 0 else float(np.abs(guess - reference).sum() / reference.sum()),
    )


def _experiment(path, x_col, y_col):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = np.array([(float(row[x_col]), float(row[y_col])) for row in csv.DictReader(stream)])
    if rows.ndim != 2 or len(rows) < 2 or not np.isfinite(rows).all() or np.any(rows <= 0):
        raise ValueError("experiment needs finite positive photon wavenumbers and absolute cross sections")
    rows = rows[np.argsort(rows[:, 0])]
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("experiment wavenumbers must be distinct")
    return rows


def _write_profile(path, result, factor, is_absorption):
    axis = result.wavenumber_cm_inverse
    density = _density(result, factor)
    if is_absorption:
        values = np.column_stack((axis, density, F_DENSITY_TO_CROSS_SECTION * density))
        header = "photon_wavenumber_cm_inverse,thermal_f_density_per_cm_inverse,thermal_cross_section_cm2_per_molecule"
    else:
        values = np.column_stack((axis, density))
        header = "photon_wavenumber_cm_inverse,thermal_photon_rate_density_s_inverse_per_cm_inverse"
    np.savetxt(path, values, delimiter=",", comments="", header=header)


def _plot(folder, profiles, experiment, label, experiment_label, temperature, fwhm, approximate_bath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    figure, (absorb, emit) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True, layout="constrained")
    figure.suptitle(f"{label} separable thermal spectra", fontsize=14, fontweight="semibold")
    visible = []
    for axes, profile, factor, kind, color in zip(
        (absorb, emit), profiles, ABSOLUTE_FACTORS, ("Absorption", "Fluorescence"), ("#503c88", "#006b65"), strict=True
    ):
        axis = profile.wavenumber_cm_inverse
        density = _density(profile, factor)
        if axes is absorb:
            density *= F_DENSITY_TO_CROSS_SECTION
        axes.plot(axis, density, color=color, linewidth=1.5, label="Calculated")
        valid = density > density.max() * 2e-4
        if np.any(valid):
            visible.append((axis[valid][0] - fwhm, axis[valid][-1] + fwhm))
        axes.set_title(kind, loc="right", fontsize=10, fontweight="semibold")
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(alpha=0.20, linewidth=0.6)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        axes.yaxis.set_major_formatter(formatter)
    if experiment is not None:
        absorb.scatter(*experiment.T, s=15, facecolors="none", edgecolors="#342c2a", linewidths=0.7, label=experiment_label)
        absorb.legend(frameon=False, loc="best")
        visible.append((experiment[0, 0] - fwhm, experiment[-1, 0] + fwhm))
    absorb.set_ylim(bottom=0)
    emit.set_ylim(bottom=0)
    absorb.set_ylabel(r"Cross section (cm$^2$ molecule$^{-1}$)")
    emit.set_ylabel(r"Photon rate (s$^{-1}$ per cm$^{-1}$)")
    if visible:
        emit.set_xlim(min(item[0] for item in visible), max(item[1] for item in visible))
    emit.set_xlabel(r"Photon wavenumber (cm$^{-1}$)")
    footer = f"{temperature:g} K  ·  Gaussian FWHM {fwhm:g} cm⁻¹"
    if approximate_bath:
        footer += "  ·  Auxiliary spectator curvature"
    figure.supxlabel(footer, fontsize=9)
    for extension in ("svg", "png"):
        figure.savefig(folder / f"separable_thermal_spectra.{extension}", dpi=300)
    plt.close(figure)


def assemble(document, output, *, grids=(256, 512), temperature=298.0, fwhm=80.0, step=1.0,
             experiment=None, experiment_label="Experiment"):
    """Assemble an absolute absorption and emission candidate from one input document.

    Each of ground, excited and midpoint has an independent closed 360° angle
    grid (all angles in a common physical convention). ground and excited
    provide absolute energy_eh and scalar signed overlap per sequential link.
    midpoint provides signed dipole_au (Cartesian length-gauge), overlap_s0,
    overlap_s1. Each cut reports closure_overlap[_s0/_s1] for duplicate to
    original first gauge. Each node has angle_deg, coordinates_bohr,
    tangent_bohr_per_rad and a Boolean tracking_passed. Each energy cut has
    stationary coordinates, angle and Cartesian analytic hessian_eh_bohr2.
    The mode-transport axis and top are specified as zero-based atom indices.
    """
    masses = _AMU_TO_ELECTRON_MASS * np.asarray(document["masses_amu"], dtype=float)
    if masses.ndim != 1 or len(masses) < 3 or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("provide positive finite masses in the shared atom order")
    frame = document.get("frame_atom_indices")
    axis = tuple(document["axis_atoms"])
    moving = np.asarray(document["moving_atoms"])
    sense = document.get("rotation_sense", 1)
    if (len(axis) != 2 or moving.ndim != 1 or len(moving) == 0 or sense not in (-1, 1)
            or any(type(item) is not int or not 0 <= item < len(masses) for item in (*axis, *moving.tolist()))
            or len(set((*axis, *moving.tolist()))) != 2 + len(moving)):
        raise ValueError("provide two indexed bond atoms and distinct off-axis moving atoms")
    if frame is not None and (len(frame) != 3 or len(set(frame)) != 3
                              or any(type(i) is not int or not 0 <= i < len(masses) for i in frame)):
        raise ValueError("provide three distinct zero-based molecular-frame atom indices")
    minimum = float(document.get("minimum_overlap", 0.65))
    if not 0 < minimum <= 1:
        raise ValueError("minimum_overlap must be in (0, 1]")
    ground, excited, midpoint = [
        _cut(document[name], name, masses, frame, minimum) for name in ("ground", "excited", "midpoint")
    ]
    cuts = (ground, excited)
    if midpoint.signs != (ground.signs[0], excited.signs[0]):
        raise ValueError("independent state holonomies disagree between surface and dipole cuts")
    if len(grids) != 2 or not max(len(item.angle) for item in (*cuts, midpoint)) <= grids[0] < grids[1] <= 1024:
        raise ValueError("choose two increasing DVR sizes between the largest input grid and 1024")
    hamiltonians, bridge = _spectators(cuts, midpoint, masses, frame, axis, moving, sense)
    inertia_metrics = [_cut_inertia(cut, masses) for cut in cuts]
    moments = [item["effective_me_bohr2"] for item in inertia_metrics]
    coarse = _run_profiles(cuts, midpoint, hamiltonians, moments, grids[0], temperature, fwhm, step)
    fine = _run_profiles(cuts, midpoint, hamiltonians, moments, grids[1], temperature, fwhm, step)
    summary = dict(
        status="working_candidate",
        electronic_method=document["method"],
        geometry_path=document["geometry_path"],
        temperature_kelvin=temperature,
        gaussian_fwhm_cm_inverse=fwhm,
        profile_step_cm_inverse=step,
        population_model="separate conditional Boltzmann populations on S0 and S1 with unit statistical weights",
        source_grid_start_deg=[float(np.rad2deg(item.angle[0])) for item in (*cuts, midpoint)],
        electronic_nodes=[len(item.angle) for item in (*cuts, midpoint)],
        overlap_minimum_abs=[item.minimum_abs_overlaps.tolist() for item in (*cuts, midpoint)],
        phase_closure_signs=list(midpoint.signs),
        kinetic_model="surface-specific inverse-inertia averages with a common physical-angle quadrature",
        projected_inertia_ground_and_excited=inertia_metrics,
        spectator_model=bridge,
        dvr_grids=list(grids),
        dvr_convergence={process: _convergence(c, f) for process, c, f in zip(PROCESSES, coarse, fine, strict=True)},
        electronic_scan_refinement_status="independent denser electronic loop required to check sampling convergence",
    )
    for process, profile, factor in zip(PROCESSES, fine, ABSOLUTE_FACTORS, strict=True):
        summary[process] = _stats(profile, factor)
    if experiment is not None:
        axis = fine[0].wavenumber_cm_inverse
        cross_section = F_DENSITY_TO_CROSS_SECTION * _density(fine[0], ABSOLUTE_FACTORS[0])
        experimental_energy, experimental_cross_section = experiment.T
        measured_peak = int(np.argmax(experimental_cross_section))
        summary["experimental_absorption"] = dict(
            source=experiment_label,
            energy_range_cm_inverse=[float(experimental_energy[0]), float(experimental_energy[-1])],
            measured_peak_cm_inverse=float(experimental_energy[measured_peak]),
            calculated_at_measured_peak_cm2_per_molecule=float(np.interp(
                experimental_energy[measured_peak], axis, cross_section, left=0, right=0,
            )),
            measured_peak_cross_section_cm2_per_molecule=float(experimental_cross_section.max()),
            calculated_peak_cross_section_cm2_per_molecule=float(cross_section.max()),
        )
    output.mkdir(parents=True, exist_ok=True)
    for process, profile, factor in zip(PROCESSES, fine, ABSOLUTE_FACTORS, strict=True):
        _write_profile(output / f"{process}_profile.csv", profile, factor, process == "absorption")
    _plot(
        output, fine, experiment, document.get("label", "Molecular"), experiment_label, temperature, fwhm,
        approximate_bath=not all(bridge["hessians_are_native_to_electronic_surfaces"]),
    )
    (output / "spectra_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("input", type=Path)
    options.add_argument("--output", type=Path, required=True)
    options.add_argument("--dvr-grids", type=int, nargs=2, default=(256, 512))
    options.add_argument("--temperature", type=float, default=298.0)
    options.add_argument("--fwhm", type=float, default=80.0)
    options.add_argument("--step", type=float, default=1.0)
    options.add_argument("--experiment", type=Path)
    options.add_argument("--experiment-x", default="wavenumber_cm-1")
    options.add_argument("--experiment-y", default="cross_section_cm2_molecule-1")
    options.add_argument("--experiment-label", default="Experiment")
    args = options.parse_args()
    experiment = None if args.experiment is None else _experiment(args.experiment, args.experiment_x, args.experiment_y)
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    assemble(
        document, args.output, grids=args.dvr_grids, temperature=args.temperature, fwhm=args.fwhm, step=args.step,
        experiment=experiment, experiment_label=args.experiment_label,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
