"""First-order spin–orbit intensity borrowing over the full RKS/TDA space."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .harmonic import _real_array
from .pyscf_soc import _breit_pauli_somf_ao, _read_roots, _validate_reference


@dataclass(frozen=True)
class TDASOCResponse:
    """Cartesian triplet components (rows) and electric dipole axes (columns)."""

    energy_hartree: float
    transition_dipole_au: NDArray[np.complex128]
    emitting_state_mixing_au: NDArray[np.complex128]
    ground_state_mixing_au: NDArray[np.complex128]
    relative_residuals: tuple[float, float]
    dipole_derivatives_au: NDArray[np.complex128] | None = None
    derivative_relative_residuals: tuple[float, float, float] | None = None


def _solve_response(action, diagonal, rhs, tolerance, max_cycles):
    """Apply independent preconditioned CG recurrences in one operator batch."""
    scale = np.linalg.norm(rhs, axis=1)
    valid = scale > 0
    solution = np.zeros_like(rhs)
    if not np.any(valid):
        return solution, 0.
    retained = rhs[valid] / scale[valid, None]
    inverse = 1 / np.maximum(np.abs(diagonal), 1e-6)
    result = np.zeros_like(retained)
    residual = retained.copy()
    direction = residual * inverse
    norm = np.einsum("ij,ij->i", residual, direction)
    for _ in range(max_cycles):
        active = np.linalg.norm(residual, axis=1) > tolerance / 2
        if not np.any(active):
            break
        image = action(direction[active])
        curvature = np.einsum("ij,ij->i", direction[active], image)
        if not np.isfinite(curvature).all() or np.any(curvature <= 0):
            raise ValueError("SOC response encountered a nonpositive TDA shifted operator")
        step = (norm[active] / curvature)[:, None]
        result[active] += step * direction[active]
        residual[active] -= step * image
        preconditioned = residual[active] * inverse
        new_norm = np.einsum("ij,ij->i", residual[active], preconditioned)
        direction[active] = preconditioned + (new_norm / norm[active])[:, None] * direction[active]
        norm[active] = new_norm
    error = float(np.linalg.norm(action(result) - retained, axis=1).max())
    if not isfinite(error) or error > tolerance:
        raise ValueError(f"SOC response relative residual {error:.3g} exceeds tolerance {tolerance:.3g}")
    solution[valid] = result * scale[valid, None]
    return solution, error


def _excited_operator_action(operator, q):
    nocc = q.shape[0]
    return q @ operator[:, nocc:, nocc:].swapaxes(-1, -2) - operator[:, :nocc, :nocc].swapaxes(-1, -2) @ q


def _response_inputs(triplet_tda, root):
    from pyscf.tdscf.rks import TDA

    mf = getattr(triplet_tda, "_scf", None)
    occupied, virtual = _validate_reference(mf)
    shape = occupied.shape[1], virtual.shape[1]
    _, energy, amplitudes = _read_roots(triplet_tda, (root,), shape, singlet=False)
    if getattr(triplet_tda, "wfnsym", None) is not None:
        raise ValueError("full-space SOC response requires TDA without a symmetry filter")
    q = amplitudes[0]
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-8, rtol=0.):
        raise ValueError("the selected triplet TDA root must be normalized")
    coeff = np.column_stack((occupied, virtual))
    mol = mf.mol
    with mol.with_common_origin(np.zeros(3)):
        dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
    dipole = coeff.T @ dipole_ao @ coeff
    somf = (coeff.T @ _breit_pauli_somf_ao(mol, 2 * occupied @ occupied.T) @ coeff).imag
    singlet_tda = TDA(mf)
    singlet_tda.singlet = True
    singlet_action, singlet_diag = singlet_tda.gen_vind()
    triplet_action, triplet_diag = triplet_tda.gen_vind()
    return (coeff, dipole_ao, float(energy[0]), q, dipole, somf,
            singlet_action, singlet_diag, triplet_action, triplet_diag)


def _sources(q, dipole, somf):
    nocc = q.shape[0]
    d = np.sqrt(2.) * dipole[:, :nocc, nocc:].reshape(3, -1)
    g = somf[:, :nocc, nocc:].reshape(3, -1) / np.sqrt(2.)
    b = _excited_operator_action(somf, q).reshape(3, -1) / 2
    t = _excited_operator_action(dipole, q).reshape(3, -1)
    return d, g, b, t


def _full_space_moments(energy, q, dipole, somf, singlet_action, singlet_diag, triplet_action, triplet_diag,
                        tolerance, max_cycles):
    d, g, b, t = _sources(q, dipole, somf)
    x, sr = _solve_response(lambda v: singlet_action(v) - energy * v, singlet_diag - energy,
                            d, tolerance, max_cycles)
    y, tr = _solve_response(triplet_action, triplet_diag, g, tolerance, max_cycles)
    return -1j * (b @ x.T), -1j * (y @ t.T), (sr, tr), x, y


def _dipole_derivatives(mol, coefficients, coefficient_derivatives, dipole_ao, directions):
    with mol.with_common_origin(np.zeros(3)):
        # r_alpha nabla_beta acts on the ket. Moving its center changes
        # the sign of nabla_beta, which cancels the electron charge in mu.
        ket_gradient = mol.intor("int1e_irp", comp=9).reshape(3, 3, mol.nao, mol.nao)
    atom_of_ao = np.repeat(np.arange(mol.natm), np.diff(mol.aoslice_by_atom()[:, 2:], axis=1).ravel())
    ket = np.einsum("xapq,naq->nxpq", ket_gradient, directions[:, atom_of_ao].transpose(0, 2, 1))
    integral_part = coefficients.T @ (ket + ket.swapaxes(-1, -2)) @ coefficients
    ket_orbital = coefficients.T @ dipole_ao @ coefficient_derivatives[:, None]
    return integral_part + ket_orbital + ket_orbital.swapaxes(-1, -2)


def _derivative_adjoint(energy, q, dipole, somf, x, y,
                        singlet_action, singlet_diag, triplet_action, triplet_diag, tolerance, max_cycles):
    _, _, b, t = _sources(q, dipole, somf)
    w, sr = _solve_response(lambda v: singlet_action(v) - energy * v, singlet_diag - energy,
                            b, tolerance, max_cycles)
    v, tr = _solve_response(triplet_action, triplet_diag, t, tolerance, max_cycles)
    # Differentiate the normalized spatial TDA root in its orthogonal
    # complement. One adjoint per output component replaces a solve per
    # nuclear coordinate. The pure-imaginary SOC spatial operator has an
    # antisymmetric particle/hole action, so the transpose changes its sign.
    c = np.stack([-_excited_operator_action(somf, local.reshape(q.shape)) / 2 for local in x], axis=1)
    c += np.stack([_excited_operator_action(dipole, local.reshape(q.shape)) for local in y])
    flat_q = q.ravel()

    def project(vectors):
        return vectors - (vectors @ flat_q)[:, None] * flat_q

    def projected_action(vectors):
        orthogonal = project(vectors)
        return project(triplet_action(orthogonal) - energy * orthogonal) + (vectors - orthogonal)

    u, qr = _solve_response(projected_action, triplet_diag - energy,
                            project(c.reshape(9, -1)), tolerance, max_cycles)
    return w, v, project(u).reshape(3, 3, -1), (sr, tr, qr)


def _moment_derivative(q, x, y, w, v, dipole_derivatives, somf_derivatives,
                       singlet_response, triplet_response, root_response, energy_derivative):
    fixed_sources = [_sources(q, d, h) for d, h in zip(dipole_derivatives, somf_derivatives, strict=True)]
    dd, dg, db, dt = (np.stack(items) for items in zip(*fixed_sources, strict=True))
    derivative = (
        db @ x.T + w @ dd.swapaxes(1, 2) + dg @ v.T + y @ dt.swapaxes(1, 2)
        + energy_derivative[:, None, None] * (w @ x.T)
        - singlet_response - triplet_response - root_response
    )
    return -1j * derivative


def tda_soc_response(triplet_tda, root=0, *, response_tolerance=1e-8, max_cycles=128,
                     directions=None) -> TDASOCResponse:
    """Evaluate first-order SOMF emission moments without selecting upper roots.

    The RKS reference and target triplet must be solved. Both TDA resolvents
    must be positive (stable triplet and a target below the singlet manifold).
    The transition is evaluated in a Cartesian basis of the degenerate
    spin-free triplet. Spin–orbit shifts and spin–spin splittings are outside
    this first-order dipole model.
    Supply nuclear directions (coordinate, atom, Cartesian displacement in
    bohr) to differentiate both intensity-borrowing paths analytically.
    The analytic derivative uses the lowest isolated spatial triplet root.
    """
    if (isinstance(response_tolerance, (bool, np.bool_)) or not isinstance(response_tolerance, Real)
            or not isfinite(response_tolerance)
            or response_tolerance <= 0.):
        raise ValueError("response_tolerance must be finite and positive")
    if isinstance(max_cycles, (bool, np.bool_)) or not isinstance(max_cycles, Integral) or max_cycles < 1:
        raise ValueError("max_cycles must be a positive integer")
    if directions is not None:
        directions = _real_array(directions, "nuclear directions")
        mol = getattr(triplet_tda, "mol", None)
        if (mol is None or directions.ndim != 3 or not len(directions)
                or directions.shape[1:] != (mol.natm, 3)):
            raise ValueError("nuclear directions must have shape (coordinate, atom, 3)")
        if root != 0:
            raise ValueError("analytic SOC dipole derivatives require the lowest spatial triplet")
    coefficients, dipole_ao, energy, q, dipole, somf, sa, sd, ta, td = _response_inputs(triplet_tda, root)
    if directions is not None:
        if q.size > 1:
            _, energies, _ = _read_roots(triplet_tda, (0, 1), q.shape, singlet=False)
            if energies[1] - energies[0] <= 1e-5:
                raise ValueError("analytic SOC dipole derivatives require an isolated spatial triplet")
        if np.linalg.norm(ta(q.reshape(1, -1)) - energy * q.ravel()) > 1e-7:
            raise ValueError("the triplet eigenpair residual exceeds 1e-7 Hartree")
    emitting, ground, residuals, x, y = _full_space_moments(
        energy, q, dipole, somf, sa, sd, ta, td, response_tolerance, max_cycles,
    )
    if directions is None:
        return TDASOCResponse(energy, emitting + ground, emitting, ground, residuals)

    from pyscf.lib import logger

    mf = triplet_tda._scf
    # MO coefficients supplied by CPU PySCF are in occupied/virtual order.
    if not np.array_equal(coefficients, mf.mo_coeff):
        raise ValueError("SOC dipole derivatives require occupied-first MO ordering")
    logger.note(mf, "Analytic SOC dipole response: adjoint TDA solves")
    w, v, u, derivative_residuals = _derivative_adjoint(
        energy, q, dipole, somf, x, y, sa, sd, ta, td, response_tolerance, max_cycles,
    )
    from .somf_derivatives import breit_pauli_somf_mo_derivatives
    from .tda_nuclear_response import analytic_tda_nuclear_response

    def progress(stage, done, total):
        if done == 1 or done == total or done % 6 == 0:
            logger.note(mf, "%s %d/%d", stage, done, total)

    logger.note(mf, "Analytic SOC dipole response: orbital derivatives")
    response = analytic_tda_nuclear_response(mf, directions=directions, progress=progress)
    dc = response.mo_coeff_derivatives
    dmu = _dipole_derivatives(mf.mol, coefficients, dc, dipole_ao, directions)
    logger.note(mf, "Analytic SOC dipole response: Breit–Pauli SOMF derivatives")
    dh = breit_pauli_somf_mo_derivatives(mf.mol, coefficients, mf.mo_occ, dc, directions=directions).imag
    rows = lambda z: np.repeat(z, 3, axis=0).reshape(9, *q.shape)
    columns = lambda z: np.tile(z, (3, 1)).reshape(9, *q.shape)
    logger.note(mf, "Analytic SOC dipole response: singlet TDA operator derivatives")
    s_contract = response.operator_derivative_bilinears(rows(w), columns(x), singlet=True).reshape(-1, 3, 3)
    logger.note(mf, "Analytic SOC dipole response: triplet TDA operator derivatives")
    t_contract = response.operator_derivative_bilinears(
        np.concatenate((columns(v), u.reshape(9, *q.shape), q[None])),
        np.concatenate((rows(y), np.broadcast_to(q, (10, *q.shape)))), singlet=False,
    )
    derivatives = _moment_derivative(
        q, x, y, w, v, dmu, dh, s_contract, t_contract[:, :9].reshape(-1, 3, 3),
        t_contract[:, 9:18].reshape(-1, 3, 3), t_contract[:, 18],
    )
    return TDASOCResponse(energy, emitting + ground, emitting, ground, residuals, derivatives, derivative_residuals)
