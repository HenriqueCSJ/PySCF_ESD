"""Analytic length transition-dipole derivatives for GPU RKS/TDA/C-PCM."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .cpu import ElectronicConvergenceError, _readonly


@dataclass(frozen=True)
class TDADipoleResult:
    """Length-gauge dipole and Cartesian derivative in the specified root phase."""

    dipole: NDArray[np.float64]  # (dipole Cartesian,), e * bohr
    derivative: NDArray[np.float64]  # (atom, displacement Cartesian, dipole Cartesian), e
    root_index: int
    separation_hartree: float
    root_residual_hartree: float
    adjoint_residuals: NDArray[np.float64]
    translation_error: float
    rotation_error: float


def analytic_tda_transition_dipole(
    td, root_index: int, *, phase: float = 1.0, root_gap_tolerance: float = 1e-5,
    response_tolerance: float = 1e-8, cphf_conv_tol: float = 1e-10,
    cphf_max_cycle: int = 100,
) -> TDADipoleResult:
    """Differentiate an isolated real singlet of converged equilibrium LR-TDA.

    This route uses GPU4PySCF RKS with a global hybrid LDA/GGA functional,
    the primary moving XC grid, and equilibrium linear-response C-PCM. Solve
    the TDA roots through at least one root above the requested root, or supply
    the complete occupied–virtual spectrum. ``phase`` multiplies the stored
    PySCF amplitude and the corresponding dipole and derivative together.
    """
    if phase not in (-1.0, 1.0) or isinstance(phase, (bool, np.bool_)):
        raise ValueError("phase must be +1 or -1")
    if any(not np.isfinite(value) or value <= 0 for value in (
        root_gap_tolerance, response_tolerance, cphf_conv_tol,
    )):
        raise ValueError("root-gap and response tolerances must be finite and positive")
    if (isinstance(cphf_max_cycle, (bool, np.bool_)) or not isinstance(cphf_max_cycle, Integral)
            or cphf_max_cycle < 1):
        raise ValueError("cphf_max_cycle must be a positive integer")

    import cupy
    from pyscf import lib

    from .gpu import _fp64_host
    from .tda_hessian import prepare_tda_root_response
    from .tda_xc_grid import moving_xc_grid_correction

    mf = _validate_reference(td)

    with mf.mo_coeff.device, lib.temporary_env(mf, conv_tol_cpscf=cphf_conv_tol), lib.temporary_env(
        mf.with_solvent, equilibrium_solvation=True,
    ):
        nao = mf.mol.nao_nr()
        coeff = _fp64_host(mf.mo_coeff, (nao, nao), "complete RKS orbital basis")
        occupations = _fp64_host(mf.mo_occ, (coeff.shape[1],), "RKS occupations")
        nocc = int(np.count_nonzero(occupations == 2))
        nvir = coeff.shape[1] - nocc
        if not nocc or not nvir or np.any(occupations[:nocc] != 2) or np.any(occupations[nocc:] != 0):
            raise ValueError("the TDA dipole derivative requires closed-shell integer occupations")
        shape = (nocc, nvir)
        x, omega, separation = _target_root(td, root_index, shape, root_gap_tolerance)
        operator, diagonal = td.gen_vind()

        def apply(vector):
            return _fp64_host(operator(cupy.asarray(vector).reshape(1, -1)), (1, vector.size),
                              "singlet TDA operator").reshape(shape)

        prepared = prepare_tda_root_response(
            apply, omega, np.sqrt(2) * phase * x,
            diagonal=_fp64_host(diagonal, (nocc * nvir,), "TDA diagonal").reshape(shape),
            root_separation_hartree=separation, root_tolerance=response_tolerance,
            root_gap_tolerance=root_gap_tolerance, response_atol=min(1e-10, response_tolerance),
            response_rtol=response_tolerance,
        )
        q = prepared.amplitude
        # Use an origin near the molecule for good conditioning. Its derivative
        # multiplies the identically zero occupied–virtual overlap.
        mol = mf.mol.copy()
        coordinates = mol.atom_coords()
        origin = coordinates.mean(axis=0)
        with mol.with_common_orig(origin):
            position = mol.intor_symmetric("int1e_r", comp=3)
            ket_gradient = mol.intor("int1e_irp", comp=9).reshape(3, 3, mol.nao, mol.nao)
        position_mo = coeff.T @ position @ coeff
        b = np.sqrt(2) * position_mo[:, :nocc, nocc:]
        dipole = np.einsum("dia,ia->d", b, q)
        native_dipole = _fp64_host(td.transition_dipole(), (len(td.xy), 3), "TDA transition dipoles")
        if not np.allclose(dipole, phase * native_dipole[root_index], atol=1e-8, rtol=1e-8):
            raise ElectronicConvergenceError("length dipole disagrees with the stored native TDA root")
        adjoints = prepared.solve(b)
        mo1 = _occupied_orbital_response(mf, cphf_max_cycle)
        derivative = _fixed_amplitude_dipole_derivative(
            coeff, q, position_mo, ket_gradient, -mol.intor("int1e_ipovlp", comp=3),
            mol.aoslice_by_atom()[:, 2:], mo1,
        )

        gradient = td.Gradients()
        gradient.cphf_max_cycle = cphf_max_cycle
        gradient.cphf_conv_tol = cphf_conv_tol

        def quadratic_gradient(amplitude):
            force = gradient.grad_elec((cupy.asarray(amplitude), 0), singlet=True)
            # The native gradient includes the implicit orbital response at
            # fixed grid nodes and weights. Use its relaxed response density
            # to restore the moving-grid terms of the same Lagrangian.
            transition = gradient._dmxpy
            correction = moving_xc_grid_correction(
                mf, gradient._dmz1doo, transition + transition.T,
            )
            return (_fp64_host(force, (mol.natm, 3), "TDA electronic gradient")
                    + _fp64_host(correction, (mol.natm, 3), "TDA XC grid response"))

        derivative += _polarized_gradient(quadratic_gradient, q, adjoints.vectors, mol.natm)
        derivative = _fp64_host(derivative, (mol.natm, 3, 3), "TDA transition-dipole derivative")

    translation_error, rotation_error = _rigid_motion_errors(coordinates - origin, dipole, derivative)
    return TDADipoleResult(
        dipole=_readonly(dipole), derivative=_readonly(derivative), root_index=int(root_index),
        separation_hartree=prepared.separation_hartree, root_residual_hartree=prepared.root_residual,
        adjoint_residuals=_readonly(np.asarray(adjoints.residual_norms)),
        translation_error=translation_error, rotation_error=rotation_error,
    )


def _validate_reference(td):
    from gpu4pyscf.dft.rks import RKS
    from gpu4pyscf.solvent.pcm import PCM
    from gpu4pyscf.tdscf.rks import TDA

    if not isinstance(td, TDA) or not td.singlet or td.wfnsym is not None:
        raise ValueError("supply a GPU4PySCF singlet RKS/TDA calculation without a symmetry filter")
    mf = td._scf
    if not isinstance(mf, RKS) or not mf.converged:
        raise ValueError("converge the GPU4PySCF RKS reference before the TDA dipole derivative")
    ground_solvent = getattr(mf, "with_solvent", None)
    td_solvent = getattr(td, "with_solvent", None)
    if (not all(isinstance(solvent, PCM) and solvent.method.upper() in ("C-PCM", "CPCM")
                and not solvent.frozen and solvent.frozen_dm0_for_finite_difference_without_response is None
                for solvent in (ground_solvent, td_solvent))
            or not td_solvent.equilibrium_solvation):
        raise ValueError("the TDA dipole derivative requires equilibrium linear-response C-PCM")
    if mf.mol.spin or mf.mol.has_ecp() or mf.mol.pseudo or mf.do_nlc():
        raise ValueError("the TDA dipole derivative requires a closed-shell all-electron reference without NLC")
    if getattr(td, "frozen", None) not in (None, 0):
        raise ValueError("the TDA dipole derivative requires all occupied and virtual orbitals")
    if mf.cphf_grids is not mf.grids or mf.small_rho_cutoff != 0:
        raise ValueError("the RKS and CPHF responses must use the same grid without density cutoff")
    ni = mf._numint
    omega, _, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    # The native TDA gradient includes the two induced PCM orbital terms in
    # its hybrid branch. The semilocal branch does not include them.
    if omega != 0 or hybrid == 0 or ni._xc_type(mf.xc) not in ("LDA", "GGA"):
        raise ValueError("the analytic C-PCM dipole derivative supports global hybrid LDA/GGA functionals")
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    return mf


def _target_root(td, root_index, shape, gap_tolerance):
    from .gpu import _fp64_host

    if isinstance(root_index, (bool, np.bool_)) or not isinstance(root_index, Integral):
        raise ValueError("root_index must be a zero-based integer")
    if td.xy is None or not 0 <= root_index < len(td.xy):
        raise ValueError("root_index is outside the solved singlet TDA roots")
    nroots = len(td.xy)
    if nroots > np.prod(shape):
        raise ValueError("the solved TDA roots exceed the occupied–virtual dimension")
    energies = _fp64_host(td.e, (nroots,), "TDA excitation energies")
    converged = td.converged.get() if hasattr(td.converged, "get") else np.asarray(td.converged)
    if converged.shape != (nroots,) or not np.all(converged):
        raise ElectronicConvergenceError("converge the solved singlet TDA roots before computing their response")
    if np.any(energies <= 0) or np.any(np.diff(energies) < 0):
        raise ElectronicConvergenceError("solved TDA energies must be positive and sorted")
    if nroots != np.prod(shape) and root_index == nroots - 1:
        raise ValueError("solve at least one TDA root above the requested root to check electronic isolation")
    omega = float(energies[root_index])
    separation = float(np.min(np.abs(np.delete(energies, root_index) - omega), initial=np.inf))
    if nroots != np.prod(shape) and omega - td.positive_eig_threshold < gap_tolerance:
        raise ValueError("the TDA positive eigenvalue threshold prevents checking electronic isolation")
    if separation < gap_tolerance:
        raise ValueError("the target TDA root is not electronically isolated")
    x, y = td.xy[root_index]
    if not np.isscalar(y) or y != 0:
        raise ValueError("the TDA solution must have zero de-excitation amplitude")
    x = _fp64_host(x, shape, "singlet TDA amplitude")
    if not np.isclose(np.vdot(x, x), 0.5, rtol=0, atol=1e-8):
        raise ElectronicConvergenceError("singlet TDA amplitude fails the native normalization check")
    return x, omega, separation


def _occupied_orbital_response(mf, cphf_max_cycle):
    from .gpu import _checked_hessian_response, _fp64_host

    hessian = mf.Hessian()
    hessian.grid_response = True
    hessian.max_cycle = cphf_max_cycle
    coeff, occupation, energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    h1 = hessian.make_h1(coeff, occupation)
    occupied_response, _ = _checked_hessian_response(
        hessian, hessian.solve_mo1, energy, coeff, occupation, h1,
    )
    return _fp64_host(occupied_response, h1.shape, "first-order occupied RKS orbitals")


def _fixed_amplitude_dipole_derivative(coeff, q, position_mo, ket_gradient, ip_overlap, slices, mo1):
    nocc = q.shape[0]
    transition = coeff[:, :nocc] @ q @ coeff[:, nocc:].T
    transition += transition.T.copy()
    derivative = np.empty((len(slices), 3, 3))
    for atom, (left, right) in enumerate(slices):
        overlap_bra = ip_overlap[:, left:right, :]
        overlap = coeff[left:right].T @ overlap_bra @ coeff
        overlap += overlap.transpose(0, 2, 1).copy()
        u = -0.5 * overlap
        u[:, nocc:, :nocc] = mo1[atom, :, nocc:, :nocc]
        u[:, :nocc, nocc:] = -overlap[:, :nocc, nocc:] - u[:, nocc:, :nocc].transpose(0, 2, 1)
        ao = -np.einsum("dxpq,pq->xd", ket_gradient[:, :, :, left:right],
                        transition[:, left:right], optimize=True)
        occ = np.einsum("xpi,dpa,ia->xd", u[:, :, :nocc], position_mo[:, :, nocc:], q, optimize=True)
        vir = np.einsum("dip,xpa,ia->xd", position_mo[:, :nocc, :], u[:, :, nocc:], q, optimize=True)
        derivative[atom] = np.sqrt(2) * (ao + occ + vir)
    return derivative


def _polarized_gradient(quadratic_gradient, q, adjoints, natm):
    """Return the three bilinear contractions zᵀ A′ q via same-center gradients."""
    derivative = np.zeros((natm, 3, len(adjoints)))
    for component, adjoint in enumerate(adjoints):
        scale = np.linalg.norm(adjoint)
        if scale:
            unit = adjoint / scale
            # Both trial amplitudes have the native TDA normalization.
            plus, minus = (q + unit) / 2, (q - unit) / 2
            derivative[:, :, component] = 0.5 * scale * (quadratic_gradient(plus) - quadratic_gradient(minus))
    return derivative


def _rigid_motion_errors(coordinates, dipole, derivative):
    axes = np.eye(3)
    rotation = np.cross(axes[:, None, :], coordinates)
    dipole_response = np.einsum("kax,axd->kd", rotation, derivative)
    return (float(np.max(np.abs(derivative.sum(axis=0)))),
            float(np.max(np.abs(dipole_response - np.cross(axes, dipole)))))
