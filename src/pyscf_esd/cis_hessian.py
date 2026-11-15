"""Analytic nuclear derivatives of isolated singlet RHF–TDA (CIS) roots."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .cis_integrals import IntegralDerivativeEngine


@dataclass(frozen=True)
class CISHessianResult:
    """Total excited-state curvature and response diagnostics."""

    hessian: NDArray[np.float64]  # (atom, atom, Cartesian, Cartesian), Eh/bohr²
    gradient: NDArray[np.float64]  # (atom, Cartesian), Eh/bohr
    total_energy_hartree: float
    excitation_hartree: float
    root_index: int
    overlap: float
    separation_hartree: float
    max_response_residual: float


@dataclass(frozen=True)
class CISDipoleResult:
    """Length transition dipole and its derivative in the supplied root phase."""

    dipole: NDArray[np.float64]  # Cartesian component, e·bohr
    derivative: NDArray[np.float64]  # (atom, displacement Cartesian, dipole Cartesian), e
    root_index: int
    overlap: float
    separation_hartree: float
    max_response_residual: float


def _contract(equation, *jets):
    """Contract jets with coefficients in the order 1, x, y, xy."""
    optimize = np.einsum_path(equation, *(jet[0] for jet in jets), optimize="greedy")[0]
    result = [None] * 4
    choices = [tuple(i for i, term in enumerate(jet) if term is not None) for jet in jets]
    for degrees in product(*choices):
        mask = 0
        for degree in degrees:
            if mask & degree:
                break
            mask |= degree
        else:
            value = np.einsum(
                equation, *(jet[degree] for jet, degree in zip(jets, degrees)), optimize=optimize
            )
            if result[mask] is None:
                result[mask] = value
            else:
                result[mask] += value
    return result


def _generator(kappa, nmo, nocc):
    matrix = np.zeros((nmo, nmo))
    matrix[nocc:, :nocc] = kappa.T
    matrix[:nocc, nocc:] = -kappa
    return matrix


def _orbitals(coeff, overlap_x, overlap_y, overlap_xy, kx, ky, kxy):
    # The orthogonalizing frame is C0 (C0.T S C0)^(-1/2) exp(K).
    # Expand both factors at identity, retaining the full mixed coefficient.
    ex = -0.5 * (coeff.T @ overlap_x @ coeff)
    ox = ex + kx
    if overlap_y is None:
        return coeff, coeff @ ox, None, None
    ey = -0.5 * (coeff.T @ overlap_y @ coeff)
    oxy = (
        -0.5 * (coeff.T @ overlap_xy @ coeff)
        + 1.5 * (ex @ ey + ey @ ex)
        + ex @ ky + ey @ kx
        + 0.5 * (kx @ ky + ky @ kx)
        + kxy
    )
    return coeff, coeff @ ox, coeff @ (ey + ky), coeff @ oxy


def _fock(core, eri, nocc):
    coulomb = np.einsum("pqii->pq", eri[:, :, :nocc, :nocc])
    exchange = np.einsum("piiq->pq", eri[:, :nocc, :nocc, :])
    return core + 2 * coulomb - exchange


def _cis_matrix(fock, eri, nocc):
    occupied, virtual = slice(None, nocc), slice(nocc, None)
    nvir = fock.shape[0] - nocc
    result = (
        np.einsum("ij,ab->iajb", np.eye(nocc), fock[virtual, virtual])
        - np.einsum("ij,ab->iajb", fock[occupied, occupied], np.eye(nvir))
        + 2 * eri[occupied, virtual, occupied, virtual]
        - eri[occupied, occupied, virtual, virtual].transpose(0, 2, 1, 3)
    ).reshape(nocc * nvir, nocc * nvir)
    return (result + result.T) * 0.5


def _transform(integrals, orbitals, order):
    hcore = tuple(term.hcore if term is not None else None for term in integrals)
    eri = tuple(term.eri if term is not None else None for term in integrals)
    core_mo = _contract("up,uv,vq->pq", orbitals, hcore, orbitals)[order]
    eri_mo = _contract("up,vq,uvwx,wr,xs->pqrs", orbitals, orbitals, eri, orbitals, orbitals)[order]
    return core_mo, eri_mo


def _energy_coefficient(core, eri, enuc, nocc):
    occ = slice(None, nocc)
    eri_occ = eri[occ, occ, occ, occ]
    return (
        2 * np.trace(core[occ, occ])
        + 2 * np.einsum("iijj->", eri_occ)
        - np.einsum("ijji->", eri_occ)
        + enuc
    )


def analytic_cis_hessian(
    mean_field,
    target_amplitude,
    *,
    root_gap_tolerance=1e-5,
    minimum_overlap=0.99,
    response_tolerance=1e-10,
):
    """Differentiate the total energy of a tracked molecular RHF–TDA root.

    Supply a converged conventional all-electron molecular PySCF RHF reference
    and the target singlet TDA amplitudes in exactly its MO basis. All occupied
    and virtual orbitals participate. Intended for small dense benchmarks.
    Degenerate/near-degenerate targets and singular RHF response are rejected.

    First and mixed second orbital responses enforce the differentiated
    occupied–virtual Fock equations. The CIS eigenvalue response is evaluated
    in the full TDA space. AO integral derivatives are analytic.
    """
    return _analytic_cis_derivatives(
        mean_field, target_amplitude, root_gap_tolerance, minimum_overlap, response_tolerance,
        dipole_only=False,
    )


def analytic_cis_transition_dipole(
    mean_field,
    target_amplitude,
    *,
    root_gap_tolerance=1e-5,
    minimum_overlap=0.99,
    response_tolerance=1e-10,
):
    """Return the analytic length dipole and nuclear derivative of a singlet CIS root.

    The reference and amplitude follow the Hessian interface. The returned
    dipole and derivative use the sign of the supplied tracked amplitude.
    First-order orbital and CIS eigenvector response are solved at one geometry.
    """
    return _analytic_cis_derivatives(
        mean_field, target_amplitude, root_gap_tolerance, minimum_overlap, response_tolerance,
        dipole_only=True,
    )


def _analytic_cis_derivatives(
    mean_field, target_amplitude, root_gap_tolerance, minimum_overlap, response_tolerance, *, dipole_only,
):
    from pyscf import ao2mo, lib
    from pyscf.scf.hf import RHF

    if type(mean_field) is not RHF or not mean_field.converged:
        raise ValueError("a converged conventional PySCF RHF reference is required")
    if mean_field.disp or vars(mean_field).keys() & {
        "get_hcore", "get_ovlp", "get_veff", "get_jk", "get_fock", "energy_elec", "energy_nuc", "energy_tot",
    }:
        raise ValueError("the analytic CIS route requires the unmodified RHF Hamiltonian")
    mol = mean_field.mol
    if mol.has_ecp() or mol.cart:
        raise ValueError("the analytic CIS route requires all-electron spherical AO integrals")
    coeff = np.asarray(mean_field.mo_coeff)
    occupation = np.asarray(mean_field.mo_occ)
    if np.iscomplexobj(coeff) or not np.all(np.isfinite(coeff)) or coeff.shape != (mol.nao, mol.nao):
        raise ValueError("expected a complete real RHF orbital basis")
    nmo = coeff.shape[1]
    nocc = int(np.count_nonzero(occupation == 2))
    nvir = nmo - nocc
    if not nocc or not nvir or not np.array_equal(occupation, np.repeat([2, 0], [nocc, nvir])):
        raise ValueError("expected contiguous doubly-occupied RHF orbitals followed by virtuals")
    amplitude = np.asarray(target_amplitude)
    if np.iscomplexobj(amplitude) or amplitude.shape != (nocc, nvir) or not np.all(np.isfinite(amplitude)):
        raise ValueError("target amplitude must be a real (occupied, virtual) array")
    amplitude = amplitude.ravel()
    norm = np.linalg.norm(amplitude)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("target amplitude has invalid norm")
    for value in (root_gap_tolerance, response_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("root-gap and response tolerances must be finite and positive")
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be in (0, 1]")

    # Reserve the integral engine's dense kernels, cached nuclear first
    # derivatives and a margin for concurrent AO/MO contraction workspaces.
    work_arrays = 17 if dipole_only else 44 + 3 * mol.natm
    estimated_mb = work_arrays * np.dtype(float).itemsize * mol.nao**4 / 1e6
    if estimated_mb + lib.current_memory()[0] > mean_field.max_memory:
        raise MemoryError(f"the dense CIS derivatives need about {estimated_mb:.0f} MB of additional memory")
    engine = IntegralDerivativeEngine(mol)
    base = engine.base
    if mean_field._eri is not None and not np.allclose(
        ao2mo.restore(8, mean_field._eri, mol.nao), ao2mo.restore(8, base.eri, mol.nao),
        atol=1e-9, rtol=0,
    ):
        raise ValueError("reference ERIs do not match the native molecular Hamiltonian")
    if not np.allclose(coeff.T @ base.overlap @ coeff, np.eye(nmo), atol=1e-8, rtol=0):
        raise ValueError("RHF orbitals are not orthonormal in the AO metric")
    bare_orbitals = (coeff, None, None, None)
    core, eri = _transform((base, None, None, None), bare_orbitals, 0)
    reference_energy = _energy_coefficient(core, eri, base.enuc, nocc)
    if not np.isclose(reference_energy, mean_field.e_tot, atol=1e-8, rtol=0):
        raise ValueError("reference energy does not match the native RHF Hamiltonian")
    fock = _fock(core, eri, nocc)
    reference_residual = float(np.max(np.abs(fock[:nocc, nocc:])))
    if reference_residual > response_tolerance:
        raise ValueError("RHF occupied–virtual Fock residual exceeds the response tolerance")
    cis = _cis_matrix(fock, eri, nocc)
    energies, vectors = np.linalg.eigh(cis)
    overlaps = vectors.T @ (amplitude / norm)
    root = int(np.argmax(np.abs(overlaps)))
    score = abs(overlaps[root])
    if score < minimum_overlap:
        raise ValueError("CIS eigenvector does not match the tracked target amplitude")
    mask = np.arange(energies.size) != root
    gaps = energies[root] - energies[mask]
    separation = float(np.min(np.abs(gaps), initial=np.inf))
    if energies[root] <= 0 or separation < root_gap_tolerance:
        raise ValueError("target root is nonpositive or is not electronically isolated")

    iajb = eri[:nocc, nocc:, :nocc, nocc:]
    beta = (2 * iajb - iajb.transpose(0, 3, 2, 1)).reshape(cis.shape)
    response = cis + beta
    values, basis = np.linalg.eigh((response + response.T) * 0.5)
    if np.min(np.abs(values)) < response_tolerance:
        raise ValueError("singular RHF occupied–virtual response")
    residuals = [reference_residual]

    def check_response(fock_derivative):
        error = float(np.max(np.abs(fock_derivative[:nocc, nocc:])))
        if not np.isfinite(error) or error > response_tolerance:
            raise ValueError("RHF orbital response residual exceeds tolerance")
        residuals.append(error)

    def solve(rhs):
        flat = -rhs[:nocc, nocc:].ravel()
        solution = basis @ ((basis.T @ flat) / values)
        return _generator(solution.reshape(nocc, nvir), nmo, nocc)

    directions = np.eye(3 * mol.natm).reshape(-1, mol.natm, 3)
    zero = np.zeros_like(coeff)
    first_response = []
    gradient = []
    vector = np.copysign(1.0, overlaps[root]) * vectors[:, root]
    other_vectors = vectors[:, mask]
    if dipole_only:
        operator_mol = mol.copy()
        with operator_mol.with_common_orig(np.zeros(3)):
            position = operator_mol.intor_symmetric("int1e_r", comp=3)
            # The component indices are (position, ket gradient).
            ket_gradient = operator_mol.intor("int1e_irp", comp=9).reshape(3, 3, nmo, nmo)
        ao_atoms = np.repeat(np.arange(mol.natm), np.diff(mol.aoslice_by_atom()[:, 2:], axis=1).ravel())
        occupied = slice(None, nocc)
        virtual = slice(nocc, None)
        position_ov = np.einsum("up,duv,vq->dpq", coeff[:, occupied], position, coeff[:, virtual], optimize=True)
        dipole = np.sqrt(2) * (position_ov.reshape(3, -1) @ vector)

    for direction in directions:
        derivative = engine.first(direction)
        jets = (base, derivative, None, None)
        cv = _orbitals(coeff, derivative.overlap, None, None, zero, None, None)
        rx = _fock(*_transform(jets, cv, 1), nocc)
        kx = solve(rx)
        cv = _orbitals(coeff, derivative.overlap, None, None, kx, None, None)
        core_x, eri_x = _transform(jets, cv, 1)
        fx = _fock(core_x, eri_x, nocc)
        check_response(fx)
        ax = _cis_matrix(fx, eri_x, nocc)
        projected = other_vectors.T @ ax @ vector
        if dipole_only:
            # A ket center displacement is minus its electronic derivative;
            # add the corresponding bra derivative by Hermiticity.
            ket = -np.einsum("dxpq,qx->dpq", ket_gradient, direction[ao_atoms])
            position_x = (position, ket + ket.transpose(0, 2, 1), None, None)
            bra = tuple(term[:, occupied] if term is not None else None for term in cv)
            ket_mo = tuple(term[:, virtual] if term is not None else None for term in cv)
            position_ov_x = _contract("up,duv,vq->dpq", bra, position_x, ket_mo)[1]
            vector_x = other_vectors @ (projected / gaps)
            gradient.append(np.sqrt(2) * (position_ov_x.reshape(3, -1) @ vector + position_ov.reshape(3, -1) @ vector_x))
        else:
            first_response.append((derivative, kx, projected))
            gradient.append(_energy_coefficient(core_x, eri_x, derivative.enuc, nocc) + vector @ ax @ vector)

    if dipole_only:
        return CISDipoleResult(
            dipole=dipole,
            derivative=np.asarray(gradient).reshape(mol.natm, 3, 3),
            root_index=root,
            overlap=min(1.0, float(score)),
            separation_hartree=separation,
            max_response_residual=max(residuals),
        )

    hessian = np.empty((len(directions), len(directions)))
    for i, (dx, kx, px) in enumerate(first_response):
        for j in range(i + 1):
            dy, ky, py = first_response[j]
            dxy = engine.second(directions[i], directions[j])
            jets = (base, dx, dy, dxy)
            cv = _orbitals(coeff, dx.overlap, dy.overlap, dxy.overlap, kx, ky, zero)
            rxy = _fock(*_transform(jets, cv, 3), nocc)
            kxy = solve(rxy)
            cv = _orbitals(coeff, dx.overlap, dy.overlap, dxy.overlap, kx, ky, kxy)
            core_xy, eri_xy = _transform(jets, cv, 3)
            fxy = _fock(core_xy, eri_xy, nocc)
            check_response(fxy)
            axy = _cis_matrix(fxy, eri_xy, nocc)
            value = (_energy_coefficient(core_xy, eri_xy, dxy.enuc, nocc)
                     + vector @ axy @ vector + 2 * np.sum(px * py / gaps))
            hessian[i, j] = hessian[j, i] = value

    if not np.all(np.isfinite(hessian)):
        raise ValueError("analytic CIS Hessian contains nonfinite values")
    native_shape = hessian.reshape(mol.natm, 3, mol.natm, 3).transpose(0, 2, 1, 3)
    return CISHessianResult(
        hessian=native_shape,
        gradient=np.asarray(gradient).reshape(mol.natm, 3),
        total_energy_hartree=float(reference_energy + energies[root]),
        excitation_hartree=float(energies[root]),
        root_index=root,
        overlap=min(1.0, float(score)),
        separation_hartree=separation,
        max_response_residual=max(residuals),
    )
