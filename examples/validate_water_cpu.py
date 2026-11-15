"""Reproduce the small CPU benchmark in a dedicated working directory."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from scipy.optimize import root

SYMBOLS = ("O", "H", "H")
BASIS = "6-31+g*"
XC = "pbe0"
GRID_LEVEL = 5
ROOTS = 5


def water_geometry(internal):
    x, z = internal
    return np.array([[0.0, 0.0, 0.0], [-x, 0.0, z], [x, 0.0, z]])


def reference_rks(coordinates):
    molecule = gto.M(atom=list(zip(SYMBOLS, coordinates)), basis=BASIS, unit="Bohr", verbose=0)
    mean_field = dft.RKS(molecule, xc=XC)
    mean_field.grids.level = GRID_LEVEL
    mean_field.grids.prune = None
    mean_field.small_rho_cutoff = 0.0
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-9
    mean_field.direct_scf_tol = 1e-14
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent RKS reference did not converge")
    return mean_field


def optimize_water():
    def projected_gradient(internal):
        gradient = reference_rks(water_geometry(internal)).nuc_grad_method()
        gradient.grid_response = True
        value = gradient.kernel()
        return [value[2, 0] - value[1, 0], value[1, 2] + value[2, 2]]

    result = root(projected_gradient, [1.42, 1.11], tol=1e-8)
    if np.linalg.norm(result.fun, ord=np.inf) > 1e-7:
        raise RuntimeError(f"water optimization failed with residual {result.fun}")
    return water_geometry(result.x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stencil", action="store_true", help="validate the end-to-end CPU workflow")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    lib.num_threads(args.threads)
    args.workdir.mkdir(parents=True, exist_ok=True)
    existing_reference = args.workdir / "reference.json"
    if args.stencil and existing_reference.is_file():
        reference = json.loads(existing_reference.read_text(encoding="utf-8"))
        if reference.get("pyscf") != pyscf.__version__ or reference.get("settings") != reference_settings():
            raise ValueError("the saved water reference uses another version or scientific recipe")
        coordinates = np.asarray(reference["geometry_bohr"])
        validate_stencil(coordinates, reference, args.workdir)
        return
    coordinates = optimize_water()
    mf = reference_rks(coordinates)
    td = mf.TDA()
    td.nstates = ROOTS
    td.conv_tol = 1e-10
    td.kernel()
    if not np.all(td.converged):
        raise RuntimeError("independent TDA reference did not converge")
    _, _, internal, external = mf.stability(internal=True, external=True, return_status=True)
    result = {
        "pyscf": pyscf.__version__,
        "settings": reference_settings(),
        "geometry_bohr": coordinates.tolist(),
        "ground_energy_hartree": float(mf.e_tot),
        "rks_stability": [bool(internal), bool(external)],
        "excitation_energies_hartree": td.e.tolist(),
        "dipole_magnitudes_au": np.linalg.norm(td.transition_dipole(), axis=1).tolist(),
    }
    (args.workdir / "reference.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.stencil:
        validate_stencil(coordinates, result, args.workdir)


def reference_settings():
    return {"symbols": list(SYMBOLS), "basis": BASIS, "xc": XC, "grid_level": GRID_LEVEL, "roots": ROOTS}


def validate_stencil(coordinates, reference, workdir):
    from pyscf.data.nist import HARTREE2WAVENUMBER
    from pyscf_esd.cpu import Settings
    from pyscf_esd.cpu_workflow import run_cpu_vibronic

    settings = Settings(basis=BASIS, xc=XC, nstates=ROOTS, grid_level=GRID_LEVEL,
                        direct_scf_tol=1e-14, scf_conv_tol=1e-12, scf_residual_tol=1e-8,
                        td_residual_tol=1e-9)
    options = dict(target_index=0, include_ht=True, compute_excited_hessian=True)
    coarse = run_cpu_vibronic(SYMBOLS, coordinates, settings, checkpoint_dir=workdir / "step_0.010",
                              step_bohr=0.01, **options)
    fine = run_cpu_vibronic(SYMBOLS, coordinates, settings, checkpoint_dir=workdir / "step_0.005",
                            step_bohr=0.005, **options)
    resumed = run_cpu_vibronic(SYMBOLS, coordinates, settings, checkpoint_dir=workdir / "step_0.005",
                               step_bohr=0.005, **options)
    if len(fine.displacements) != 18 or not all(point.cached for point in resumed.displacements):
        raise RuntimeError("water benchmark did not complete and reuse all 18 displaced points")

    def max_error(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    def relative_phase(other):
        # This isolated, bright water root provides an independent phase test
        # from its perpendicular dipole. It is not the workflow's tracker.
        return np.sign(fine.center.transition_dipoles_au[0] @ other.center.transition_dipoles_au[0])

    absorption = fine.spectrum(final_max_quanta=12)
    emission = fine.spectrum(process="emission", final_max_quanta=12)
    # Independently solve the reverse transition from each thermally occupied
    # ground level to the emission initial v=0. Dipole strengths must agree.
    thermal = fine.spectrum(initial_max_quanta=12, final_max_quanta=0, temperature_kelvin=2000)
    reversed_sticks = {tuple(i): (energy, strength) for i, energy, strength in zip(
        thermal.initial_quanta, thermal.photon_energy_hartree, thermal.dipole_strength_au2)}
    reciprocity_error = max(max_error(
        [energy, strength], reversed_sticks[tuple(final)],
    ) for final, energy, strength in zip(emission.final_quanta, emission.photon_energy_hartree,
                                        emission.dipole_strength_au2))

    # An orthogonal projection into the ground vibrational subspace at R0 is
    # useful as a curvature diagnostic. It gives no excited minimum or AH surface.
    modes = fine.model.cartesian_mode_derivatives_au.reshape(3, 9)
    projected_hessian = modes @ fine.excited_hessian_hartree_per_bohr2 @ modes.T
    result = {
        "method": "all-electron direct CPU RKS/PBE0 + singlet TDA / 6-31+G*, unpruned grid 5",
        "target": "lowest singlet TDA root, length dipole perpendicular to the molecular plane",
        "ground_energy_hartree": fine.center.ground_energy_hartree,
        "vertical_excitation_hartree": float(fine.center.excitation_energies_hartree[0]),
        "transition_dipole_au": fine.center.transition_dipoles_au[0].tolist(),
        "vg_gap_00_hartree": fine.model.gap_00_hartree,
        "vg_reorganization_hartree": fine.model.reorganization_energy_hartree,
        "vg_frequencies_cm-1": (fine.model.frequencies_hartree * HARTREE2WAVENUMBER).tolist(),
        "local_s1_projected_curvature_eigenvalues_au": np.linalg.eigvalsh(projected_hessian).tolist(),
        "s0_max_gradient_hartree_per_bohr": float(np.max(np.abs(fine.ground_gradient_hartree_per_bohr))),
        "center_scf_residual_hartree": fine.center.scf_residual,
        "stencil_max_scf_residual_hartree": max(p.scf_residual_hartree for p in fine.displacements),
        "stencil_max_tda_residual_hartree": max(p.tda_max_residual_hartree for p in fine.displacements),
        "stencil_points": len(fine.displacements),
        "min_transition_density_similarity": min(p.match.overlap for p in fine.displacements),
        "min_overlap_margin": min(p.match.margin for p in fine.displacements),
        "min_sampled_root_gap_hartree": min(p.match.gap_hartree for p in fine.displacements),
        "matched_indices": sorted({p.match.index for p in fine.displacements}),
        "raw_phase_flips": sum(p.match.phase < 0 for p in fine.displacements),
        "dense_tda_recoveries": {
            name: sum(p.tda_route == "dense_same_operator" for p in case.displacements)
            for name, case in (("coarse", coarse), ("fine", fine))
        },
        "reused_points": sum(p.cached for p in resumed.displacements),
        "max_energy_error_hartree": max_error(fine.center.excitation_energies_hartree,
                                              reference["excitation_energies_hartree"]),
        "gs_energy_error_hartree": abs(fine.center.ground_energy_hartree - reference["ground_energy_hartree"]),
        "max_dipole_magnitude_error_au": max_error(np.linalg.norm(fine.center.transition_dipoles_au, axis=1),
                                                  reference["dipole_magnitudes_au"]),
        "coarse_gradient_check_hartree_per_bohr": max_error(coarse.excited_gradient_hartree_per_bohr,
                                                            coarse.energy_difference_gradient_hartree_per_bohr),
        "fine_gradient_check_hartree_per_bohr": max_error(fine.excited_gradient_hartree_per_bohr,
                                                          fine.energy_difference_gradient_hartree_per_bohr),
        "fine_hessian_asymmetry_hartree_per_bohr2": fine.hessian_asymmetry_hartree_per_bohr2,
        "ground_hessian_asymmetry_hartree_per_bohr2": fine.ground_hessian_asymmetry_hartree_per_bohr2,
        "hessian_step_change_hartree_per_bohr2": max_error(fine.excited_hessian_hartree_per_bohr2,
                                                            coarse.excited_hessian_hartree_per_bohr2),
        "ht_step_change_au_per_bohr": max_error(fine.dipole_jacobian_au_per_bohr,
                                                relative_phase(coarse) * coarse.dipole_jacobian_au_per_bohr),
        "resume_absorption_strength_change_au2": max_error(absorption.thermal_strength_au2,
                                                           resumed.spectrum(final_max_quanta=12).thermal_strength_au2),
        "vg_reciprocity_max_absolute_error_au": reciprocity_error,
        "absorption_dipole_strength_coverage": absorption.final_dipole_strength_coverage,
        "emission_dipole_strength_coverage": emission.final_dipole_strength_coverage,
    }
    result.update(independent_derivative_checks(fine, coordinates))
    thresholds = {
        "gs_energy_error_hartree": 1e-7,
        "max_energy_error_hartree": 1e-6,
        "max_dipole_magnitude_error_au": 1e-6,
        "fine_gradient_check_hartree_per_bohr": 1e-5,
        "hessian_step_change_hartree_per_bohr2": 2e-4,
        "ht_step_change_au_per_bohr": 2e-4,
        "resume_absorption_strength_change_au2": 1e-10,
        "vg_reciprocity_max_absolute_error_au": 1e-10,
        "independent_o_z_dipole_derivative_error_au_per_bohr": 1e-5,
        "o_z_ground_curvature_error_hartree_per_bohr2": 2e-4,
        "o_z_excited_curvature_error_hartree_per_bohr2": 2e-4,
    }
    failures = [key for key, limit in thresholds.items() if not 0 <= result[key] <= limit]
    result["acceptance_maxima"] = thresholds
    result["passed"] = not failures
    (workdir / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"water benchmark exceeded acceptance thresholds: {', '.join(failures)}")


def independent_derivative_checks(fine, coordinates):
    # Oxygen displacement along the symmetry axis retains the molecular plane.
    # Independently orient the selected transition using its large y dipole.
    step = 0.005
    independent = []
    for sign in (-1, 1):
        xyz = coordinates.copy()
        xyz[0, 2] += sign * step
        mf = reference_rks(xyz)
        td = mf.TDA()
        td.nstates = ROOTS
        td.conv_tol = 1e-10
        td.kernel()
        if not np.all(td.converged):
            raise RuntimeError("independent displaced TDA reference did not converge")
        vector = td.transition_dipole()[0]
        if abs(vector[1]) < 0.5 or td.e[1] - td.e[0] < 0.05:
            raise RuntimeError("the independent water dipole or root gap lost its identifying signature")
        vector *= np.sign(vector[1] * fine.center.transition_dipoles_au[0, 1])
        gradient = mf.nuc_grad_method()
        gradient.grid_response = True
        independent.append((mf.e_tot + td.e[0], vector, gradient.kernel()))
    minus, plus = independent
    center_energy = fine.center.ground_energy_hartree + fine.center.excitation_energies_hartree[0]
    numerical_s1 = (plus[0] + minus[0] - 2 * center_energy) / step**2
    numerical_s0 = (plus[2] - minus[2]) / (2 * step)
    numerical_ht = (plus[1] - minus[1]) / (2 * step)
    ground_hessian = fine.center.ground_hessian()
    ground_hessian = (ground_hessian + ground_hessian.transpose(1, 0, 3, 2)) * 0.5
    numerical_s0_reference = ground_hessian[:, 0, :, 2]
    return {
        "independent_o_z_dipole_derivative_error_au_per_bohr": float(np.max(np.abs(
            numerical_ht - fine.dipole_jacobian_au_per_bohr[2],
        ))),
        "o_z_ground_curvature_error_hartree_per_bohr2": float(np.max(np.abs(numerical_s0 - numerical_s0_reference))),
        "o_z_excited_curvature_error_hartree_per_bohr2": float(abs(
            numerical_s1 - fine.excited_hessian_hartree_per_bohr2[2, 2],
        )),
    }


if __name__ == "__main__":
    main()
