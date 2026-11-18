"""Analytic density-fitted GPU electronic inputs for the shared vibronic workflow."""

from dataclasses import replace

import numpy as np

from .cpu import ElectronicConvergenceError, OrbitalSeed, _vg_root_count
from .cpu_workflow import _prepare_vg_center
from .electronic_workflow import (
    _ElectronicSession, _prepare_electronic, _prepare_ground, _ROOT_GAP,
    _STATIONARY_TOLERANCE, _symmetric_analytic_hessian,
)
from .gpu import _fp64_host, _ground_hessian, _solve_gpu_reference, solve_point
from .molecular_tda_hessian import analytic_molecular_tda_hessian
from .tda_dipole import analytic_tda_transition_dipole
from .tda_xc_curvature import require_semilocal_xc_fourth


class _GpuElectronicSession(_ElectronicSession):
    # Solved references retain grids and DF integrals on the device.
    cache_size = 4
    quadrature_grid_response = True
    ground_hessian_route = "analytic_gpu_df_rks"

    def __init__(self, symbols, coordinates, settings, *, solvent, masses_amu=None):
        from pyscf.dft import libxc
        from pyscf.scf.dispersion import parse_dft

        xc, _, dispersion = parse_dft(settings.xc)
        if (dispersion is not None or libxc.is_nlc(xc) or libxc.xc_type(xc) not in ("LDA", "GGA")
                or libxc.rsh_coeff(xc)[0] != 0 or not libxc.is_hybrid_xc(xc)):
            raise ValueError("the analytic GPU workflow requires a global hybrid LDA/GGA without dispersion or NLC")
        super().__init__(symbols, coordinates, settings, masses_amu=masses_amu)
        self.solvent = solvent

    def _ground_reference(self, molecule, previous):
        _, reference, _, coefficients, _ = _solve_gpu_reference(
            molecule, self.settings, previous, density_fitting=True, solvent=self.solvent,
        )
        return reference, OrbitalSeed(molecule, coefficients, molecule.nelectron // 2, self.settings)

    def _ground_gradient(self, reference, shape):
        with reference.mo_coeff.device:
            gradient = reference.nuc_grad_method()
            gradient.grid_response = True
            return _fp64_host(gradient.kernel(), shape, "ground-state gradient")

    def ground_hessian(self, point):
        values = _ground_hessian(point.mean_field, self.settings, len(point.coordinates), grid_response=True)
        return _symmetric_analytic_hessian(values, "S0")

    def _calculate(self, coordinates, count, seed):
        settings = replace(self.settings, nstates=count)
        self.tracked_solves += 1
        return solve_point(
            self.symbols, coordinates, settings, previous=replace(seed, settings=settings),
            density_fitting=True, hessian_grid_response=True, solvent=self.solvent,
        )

    def ground_stability(self, ground):
        with ground.mean_field.mo_coeff.device:
            super().ground_stability(ground)

    def excited_stability(self, excited):
        with excited.point._mean_field.mo_coeff.device:
            super().excited_stability(excited)

    def excited_hessian(self, point):
        result = analytic_molecular_tda_hessian(
            point.point._tda, point.root, root_gap_tolerance=_ROOT_GAP,
            cphf_tolerance=self.settings.cphf_conv_tol, response_tolerance=self.settings.td_residual_tol,
            max_cycle=self.settings.cphf_max_cycle,
        )
        reference_gradient = point.point.ground_gradient()
        error = float(np.max(np.abs(reference_gradient + result.excitation_gradient - point.total_gradient())))
        gap = point.point.excitation_energies_hartree[point.root]
        if abs(result.excitation_hartree - gap) > 1e-8 or error > 2e-7:
            raise ElectronicConvergenceError("the analytic GPU Hessian and the tracked optimization gradient disagree")
        point.curvature_diagnostics = {
            "route": "analytic_gpu_df_lr_tda",
            "gradient_disagreement_hartree_per_bohr": error,
            "orbital_response_residual": result.orbital_response_residual,
            "orbital_adjoint_residual": result.orbital_adjoint_residual,
            "tda_response_residual": result.tda_response_residual,
        }
        return _symmetric_analytic_hessian(result.hessian, "S1")

    def dipole_derivatives(self, point, modes):
        result = analytic_tda_transition_dipole(
            point.point._tda, point.root, phase=point.phase, root_gap_tolerance=_ROOT_GAP,
            response_tolerance=self.settings.td_residual_tol,
            cphf_conv_tol=self.settings.cphf_conv_tol, cphf_max_cycle=self.settings.cphf_max_cycle,
        )
        if not np.allclose(result.dipole, point.dipole, atol=1e-8, rtol=0):
            raise ElectronicConvergenceError("the analytic GPU dipole disagrees with the tracked transition phase")
        return np.einsum("kax,axc->kc", modes, result.derivative), {
            "route": "analytic_gpu_lr_tda",
            "adjoint_residuals": result.adjoint_residuals.tolist(),
            "translation_error": result.translation_error,
            "rotation_error": result.rotation_error,
        }


def prepare_gpu_electronic(symbols, coordinates, settings, *, target_index, direct_dipole, solvent, masses_amu=None):
    session = _GpuElectronicSession(symbols, coordinates, settings, solvent=solvent, masses_amu=masses_amu)
    require_semilocal_xc_fourth(settings.xc)
    return _prepare_electronic(session, target_index=target_index, direct_dipole=direct_dipole)


def prepare_gpu_vg(symbols, coordinates, settings, *, target_index, solvent, masses_amu=None):
    session = _GpuElectronicSession(symbols, coordinates, settings, solvent=solvent, masses_amu=masses_amu)
    minima, _ = _prepare_ground(session)
    ground, hessian = minima[0]
    nocc = ground.seed.nocc
    dimension = nocc * (ground.seed.mo_coeff.shape[1] - nocc)
    _vg_root_count(settings.nstates, target_index, dimension)
    tracked = session.start_tracking(ground, target_index)
    if abs(tracked.point.ground_energy_hartree - ground.energy) > 1e-8:
        raise ElectronicConvergenceError("the GPU VG center differs from the optimized ground reference")
    vg = _prepare_vg_center(
        tracked.point, tracked.root, stationary_tolerance=_STATIONARY_TOLERANCE,
        minimum_gap_hartree=_ROOT_GAP, ground_hessian_symmetry_tolerance=1e-5, ground_hessian=hessian,
        masses_amu=session.masses,
    )
    return ground.coordinates, vg
