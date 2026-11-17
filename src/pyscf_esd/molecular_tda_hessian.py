"""Analytic nuclear curvature of isolated molecular RKS/TDA/C-PCM roots."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .cis_integrals import IntegralDerivativeEngine
from .tda_df_derivatives import MolecularDFDerivatives
from .tda_hessian import assemble_tda_hessian, solve_tda_root_response
from .tda_orbital_curvature import (
    OrbitalDerivative, OrbitalResponse, fixed_amplitude_fock_density, tda_first_action,
    tda_mixed_contraction, tda_orbital_gradient, tda_transition_density, tda_transition_density_derivative,
)
from .tda_xc_operator import MovingXCOperator


_HESSIAN_SYMMETRY_TOLERANCE = 1e-8


def _inner(left, right):
    return np.einsum("ij,ij->", left, right)


@dataclass(frozen=True)
class MolecularTDAHessian:
    """Excited-state Hessian and same-geometry response diagnostics."""

    hessian: NDArray[np.float64]  # (atom, atom, Cartesian, Cartesian), Eh / bohr²
    excitation_gradient: NDArray[np.float64]  # (atom, Cartesian), Eh / bohr
    excitation_hartree: float
    orbital_response_residual: float
    orbital_adjoint_residual: float
    root_residual: float
    root_separation_hartree: float
    tda_response_residual: float


@dataclass(frozen=True)
class _First:
    direction: NDArray[np.float64]
    orbitals: OrbitalDerivative
    transition: NDArray[np.float64]
    linear_kernel_potential: NDArray[np.float64]
    relaxed_linear_potential: NDArray[np.float64]


def _response_asymmetry(response):
    derivatives = response.amplitude_derivatives.reshape(response.excitation_gradient.size, -1)
    relaxation = 2.0 * (response.projected_first_action @ derivatives.T)
    return float(np.max(np.abs(relaxation - relaxation.T)))


def _solve_symmetric_tda_response(
    operator, omega, amplitude, first_actions, *, response_atol, response_rtol, progress=None,
    **response_options,
):
    cached_actions = []

    def cache_first_actions():
        for chunk in first_actions:
            cached_actions.append(chunk)
            yield chunk

    response = solve_tda_root_response(
        operator, omega, amplitude, cache_first_actions(), response_atol=response_atol,
        response_rtol=response_rtol, **response_options,
    )
    if _response_asymmetry(response) <= _HESSIAN_SYMMETRY_TOLERANCE:
        return response

    # Separate RHS residuals can pass their tests while their cross contraction
    # still fails the Hessian's symmetry test. Refine only the projected solves.
    if progress is not None:
        progress("tda_root_response_retry", 0, response.excitation_gradient.size)
    response = solve_tda_root_response(
        operator, omega, amplitude, cached_actions, response_atol=response_atol * 0.1,
        response_rtol=response_rtol * 0.1, **response_options,
    )
    asymmetry = _response_asymmetry(response)
    if not np.isfinite(asymmetry) or asymmetry > _HESSIAN_SYMMETRY_TOLERANCE:
        raise ValueError(
            f"TDA eigenvector response contraction asymmetry {asymmetry:.3g} exceeds "
            f"{_HESSIAN_SYMMETRY_TOLERANCE:.3g} after one tighter projected solve"
        )
    if progress is not None:
        size = response.excitation_gradient.size
        progress("tda_root_response_retry", size, size)
    return response


class _MolecularTDAOperator:
    def __init__(self, td, q, *, cphf_tolerance, max_cycle):
        import cupy as cp
        from pyscf.df.df import DF

        mf = td._scf
        self.q = q
        self.mf = mf
        self.core = IntegralDerivativeEngine(mf.mol)
        native_df = DF(mf.mol, auxbasis=mf.with_df.auxbasis)
        native_df.auxmol = mf.with_df.auxmol
        native_df.max_memory = mf.max_memory
        self.df = MolecularDFDerivatives(native_df)
        self.xc = MovingXCOperator(mf)
        _, _, self.hybrid = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
        ground_response = mf.gen_response(hermi=1)
        transition_response = td.gen_response(singlet=True, hermi=0)

        def ground_kernel(density):
            return cp.asnumpy(ground_response(cp.asarray(density)))

        def transition_kernel(density):
            array = cp.asarray(density)
            if array.ndim == 2:
                return cp.asnumpy(transition_response(array[None]))[0]
            return cp.asnumpy(transition_response(array))

        self.transition_kernel = transition_kernel
        self.ground = cp.asnumpy(mf.make_rdm1())
        fock = cp.asnumpy(mf.get_fock(dm_or_wfn=mf.make_rdm1()))
        self.orbitals = OrbitalResponse(cp.asnumpy(mf.mo_coeff), q.shape[0], fock, ground_kernel,
                                        atol=cphf_tolerance, max_cycle=max_cycle)
        self.transition = tda_transition_density(self.orbitals.coefficients, q)
        self.potential = transition_kernel(self.transition)
        self.kernel_adjoint = self.xc.density_adjoint(self.ground, self.transition)
        source = tda_orbital_gradient(self.orbitals, q, self.potential, self.kernel_adjoint)
        self.adjoint, self.adjoint_residual = self.orbitals.solve(source)
        self.relaxed_density = fixed_amplitude_fock_density(self.orbitals, q, self.adjoint)
        self.mixed_density_potential = ground_kernel(self.relaxed_density) + self.kernel_adjoint
        self.first_df_densities = np.stack((self.ground, self.transition, self.relaxed_density))

    def check_df_reference(self):
        """Verify that the analytic Coulomb-metric fit matches the GPU reference."""
        import cupy as cp

        density = self.first_df_densities[:2]
        expected = self.mf.get_jk(self.mf.mol, cp.asarray(density), hermi=0)
        for matrix, native in zip(self.df.reference_jk(density), expected):
            if not np.allclose(matrix, cp.asnumpy(native), atol=2e-9, rtol=2e-9):
                error = np.max(np.abs(matrix - cp.asnumpy(native)))
                raise ValueError(f"the analytic DF metric does not reproduce native J/K (AO error {error:.3g})")

    def solvent_derivatives(self):
        """Get analytic fixed-density solvent terms in PySCF's native AO order."""
        import cupy as cp
        from pyscf import lib
        from gpu4pyscf.solvent.hessian.pcm import analytical_grad_vmat

        pcm = self.mf.with_solvent
        ground = cp.asarray(self.ground)
        relaxed = cp.asarray(self.relaxed_density)
        transition = cp.asarray((self.transition + self.transition.T) * 0.5)
        zero = cp.zeros_like(ground)
        nao = self.mf.mol.nao
        basis, mask = cp.eye(nao), cp.ones(nao)

        def gradient(density):
            return analytical_grad_vmat(pcm, density, basis, mask)

        # PCM energy is quadratic in the AO density and the nuclear surface
        # potential. Polarization here is exact at the fixed geometry.
        with lib.temporary_env(pcm, _intermediates=pcm._intermediates.copy()):
            base = gradient(zero)
            v1 = gradient(ground)
            t1 = gradient(transition) - base
            r1 = gradient(relaxed) - base
            ground_hessian = (pcm.hess(ground + relaxed) - pcm.hess(ground - relaxed)) * 0.5
            transition_hessian = pcm.hess(transition) + pcm.hess(-transition) - 2 * pcm.hess(zero)
        ncart = self.mf.mol.natm * 3
        fixed_hessian = (ground_hessian + 2 * transition_hessian).transpose(0, 2, 1, 3).reshape(ncart, ncart)
        gradients = tuple(cp.asnumpy(array.reshape(ncart, nao, nao)) for array in (v1, t1, r1))
        return gradients, cp.asnumpy(fixed_hessian)

    def density_fitting_hessian(self):
        r, t = self.relaxed_density, self.transition
        return self._jk(self.df.bilinear_hessian(r, self.ground)) + 2 * self._jk(self.df.bilinear_hessian(t, t))

    def _jk(self, components):
        j, k = components
        return j - 0.5 * self.hybrid * k

    def first(self, x, solvent_fock, solvent_transition, solvent_relaxed, xc_fock, occupied_response):
        df = self.df.directional(x)
        df_ground, df_transition, df_relaxed = self._jk(df.first_jk(self.first_df_densities))
        core = self.core.first_one_electron(x)
        fixed_fock = core.hcore + df_ground + solvent_fock + xc_fock
        orbitals = self.orbitals.first(core.overlap, fixed_fock, occupied_response=occupied_response)
        transition = tda_transition_density_derivative(self.orbitals, self.q, orbitals)
        return _First(x, orbitals, transition, df_transition + solvent_transition, df_relaxed + solvent_relaxed)

    def mixed(self, first_x, first_y, kernel_x, kernel_y, potential_x, potential_y):
        x, y = first_x.direction, first_y.direction
        dx, dy = first_x.orbitals.density, first_y.orbitals.density
        r = self.relaxed_density
        core = self.core.second_one_electron(x, y)
        orbital = self.orbitals.mixed(core.overlap, first_x.orbitals, first_y.orbitals,
                                      include_density_response=False)

        # The adjoint eliminates C_xy's unknown occupied–virtual rotation.
        # The clamped mixed Fock term can be contracted on the relaxed AO
        # density, and density response to the known part of C_xy is separate.
        scalar = (_inner(r, core.hcore) - _inner(self.adjoint, orbital.stationarity_source)
                  + _inner(self.mixed_density_potential, orbital.density))
        scalar += _inner(first_x.relaxed_linear_potential, dy) + _inner(first_y.relaxed_linear_potential, dx)
        return scalar + tda_mixed_contraction(
            self.orbitals, self.q, orbital, first_x.orbitals, first_y.orbitals,
            first_x.transition, first_y.transition, self.potential, potential_x,
            potential_y, kernel_x, kernel_y, 0.0,
        )


def analytic_molecular_tda_hessian(
    td, root_index, *, cphf_tolerance=1e-10, response_tolerance=1e-8,
    root_gap_tolerance=1e-5, max_cycle=150, progress=None,
) -> MolecularTDAHessian:
    """Differentiate a solved GPU4PySCF restricted equilibrium LR-TDA root.

    Uses global hybrid LDA/GGA, the unpruned primary grid, Coulomb-metric DF
    for both J and K, and the same equilibrium C-PCM cavity for SCF and TDA.
    First and mixed nuclear derivatives are evaluated at the supplied geometry.
    The Cartesian response and one adjoint account for orbital relaxation.
    """
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.df.df import DF

    from .gpu import _checked_hessian_response
    from .tda_dipole import _target_root, _validate_reference
    from .tda_xc_curvature import require_semilocal_xc_fourth

    mf = _validate_reference(td)
    if (not isinstance(getattr(mf, "with_df", None), DF) or mf.mol.cart or mf.mol._pseudo
            or getattr(mf, "only_dfj", False) or mf.disp):
        raise ValueError("analytic TDA curvature requires all-electron spherical AOs and DF J/K without dispersion")
    require_semilocal_xc_fourth(mf.xc)
    ground_pcm, transition_pcm = mf.with_solvent, td.with_solvent
    if (ground_pcm.get_eps() != transition_pcm.get_eps()
            or ground_pcm.lebedev_order != transition_pcm.lebedev_order
            or ground_pcm.vdw_scale != transition_pcm.vdw_scale
            or ground_pcm.r_probe != transition_pcm.r_probe
            or ground_pcm.surface_discretization_method.upper() != "SWIG"
            or transition_pcm.surface_discretization_method.upper() != "SWIG"):
        raise ValueError("the RKS and TDA responses must use the same equilibrium SWIG cavity")
    if any(not np.isfinite(tol) or tol <= 0 for tol in (cphf_tolerance, response_tolerance, root_gap_tolerance)):
        raise ValueError("response tolerances must be finite and positive")

    with mf.mo_coeff.device, lib.temporary_env(mf, conv_tol_cpscf=cphf_tolerance), lib.temporary_env(
        ground_pcm, equilibrium_solvation=True,
    ):
        for key in ("grid_coords", "charge_exp", "switch_fun", "gslice_by_atom"):
            if (key not in ground_pcm.surface or key not in transition_pcm.surface
                    or not bool(cp.array_equal(cp.asarray(ground_pcm.surface[key]), cp.asarray(transition_pcm.surface[key])))):
                raise ValueError("the stored RKS and TDA cavities differ")
        nocc = int(cp.count_nonzero(mf.mo_occ == 2))
        shape = nocc, mf.mo_coeff.shape[1] - nocc
        x, omega, separation = _target_root(td, root_index, shape, root_gap_tolerance)
        q = x / np.linalg.norm(x)
        model = _MolecularTDAOperator(td, q, cphf_tolerance=cphf_tolerance, max_cycle=max_cycle)
        model.check_df_reference()
        operator, diagonal = td.gen_vind()

        def apply(amplitude):
            return cp.asnumpy(operator(cp.asarray(amplitude).reshape(1, -1))).reshape(shape)

        def apply_batch(amplitudes):
            return cp.asnumpy(operator(cp.asarray(amplitudes).reshape(len(amplitudes), -1))).reshape(amplitudes.shape)

        fock_mo = model.orbitals.fock_mo
        base_action = (q @ fock_mo[nocc:, nocc:] - fock_mo[:nocc, :nocc] @ q
                       + 2 * model.orbitals.occupied.T @ model.potential.T @ model.orbitals.virtual)
        if np.linalg.norm(apply(q) - base_action) > response_tolerance:
            raise ValueError("the derivative model does not reproduce the native TDA action")
        native_hessian = mf.Hessian()
        native_hessian.grid_response = True
        native_hessian.auxbasis_response = 2
        native_hessian.max_cycle = max_cycle
        ncart = 3 * mf.mol.natm
        native_solve = native_hessian.solve_mo1

        def checked_ground_response(*args, **kwargs):
            first_mo, first_energy = _checked_hessian_response(native_hessian, native_solve, *args, **kwargs)
            first_occupied.append(cp.asnumpy(first_mo).reshape(ncart, mf.mo_coeff.shape[1], nocc))
            return first_mo, first_energy

        first_occupied = []
        with lib.temporary_env(native_hessian, solve_mo1=checked_ground_response):
            ground = cp.asnumpy(native_hessian.kernel()).transpose(0, 2, 1, 3).reshape(ncart, ncart)
        if progress is not None:
            progress("ground_hessian", 1, 1)

        solvent_gradients, solvent_hessian = model.solvent_derivatives()
        if progress is not None:
            progress("fixed_density_solvent", 1, 1)
        fixed_hessian = model.density_fitting_hessian() + solvent_hessian
        if progress is not None:
            progress("fixed_density_df", 1, 1)
        xc_fock = model.xc.first_derivatives(model.ground)
        if progress is not None:
            progress("xc_fock_derivatives", 1, 1)
        first = []
        for index, direction in enumerate(np.eye(ncart).reshape(-1, mf.mol.natm, 3)):
            first.append(model.first(direction, *(field[index] for field in solvent_gradients),
                                     xc_fock[index], first_occupied[0][index]))
            if progress is not None:
                progress("first_operator_derivatives", len(first), ncart)
        del solvent_gradients, xc_fock
        ground_derivatives = np.stack([point.orbitals.density for point in first])
        kernel_derivatives = model.xc.first_derivatives(model.ground, model.transition, ground_derivatives)
        potentials = np.empty_like(kernel_derivatives)
        for start in range(0, ncart, 5):
            stop = min(start + 5, ncart)
            for index in range(start, stop):
                kernel_derivatives[index] += first[index].linear_kernel_potential
            potentials[start:stop] = kernel_derivatives[start:stop] + model.transition_kernel(
                np.stack([point.transition for point in first[start:stop]])
            )
        if progress is not None:
            progress("xc_kernel_derivatives", 1, 1)
        fixed_hessian += model.xc.bilinear_hessian(model.ground, ground_derivatives,
                                                  model.relaxed_density, model.transition)
        if progress is not None:
            progress("mixed_xc_derivatives", 1, 1)
        def derivative_actions():
            for start in range(0, ncart, 5):
                if progress is not None:
                    progress("tda_root_response", start, ncart)
                yield np.stack([tda_first_action(model.orbitals, q, first[index].orbitals, model.potential,
                                                 potentials[index]) for index in range(start, min(start + 5, ncart))])
            if progress is not None:
                progress("tda_root_response", ncart, ncart)

        response = _solve_symmetric_tda_response(
            apply, omega, q, derivative_actions(), diagonal=cp.asnumpy(diagonal).reshape(shape),
            root_separation_hartree=separation, root_tolerance=response_tolerance,
            root_gap_tolerance=root_gap_tolerance, response_atol=min(1e-10, response_tolerance),
            response_rtol=response_tolerance, maxiter=max_cycle, operator_batch=apply_batch,
            lowest_root=root_index == 0, response_batch_size=5, progress=progress,
        )
        second = np.empty((ncart, ncart))
        for i, left in enumerate(first):
            for j in range(i + 1):
                second[i, j] = second[j, i] = model.mixed(left, first[j], kernel_derivatives[i], kernel_derivatives[j],
                                                         potentials[i], potentials[j])
            if progress is not None:
                progress("mixed_operator_derivatives", (i + 1) * (i + 2) // 2, ncart * (ncart + 1) // 2)
        hessian = assemble_tda_hessian(
            ground, second + fixed_hessian, response, symmetry_tolerance=_HESSIAN_SYMMETRY_TOLERANCE,
        )

    natm = mf.mol.natm
    return MolecularTDAHessian(
        hessian.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3),
        response.excitation_gradient.reshape(natm, 3), omega,
        max(point.orbitals.residual_norm for point in first), model.adjoint_residual,
        response.root_residual, response.separation_hartree, response.max_response_residual,
    )
