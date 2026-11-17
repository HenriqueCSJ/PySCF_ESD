"""Effective one-electron operators in a restricted LR-TDA state basis."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic import _real_array


_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
_SPIN = _PAULI / 2
_SPIN_EXCITATIONS = np.concatenate((np.eye(2)[None], _PAULI)) / np.sqrt(2)
# The electron operator multiplies the ket spin excitation on the left. The
# missing occupied electron multiplies on the right and enters with a minus.
_PARTICLE_SPIN = np.einsum("Auv,kuw,Bwv->kAB", _SPIN_EXCITATIONS.conj(), _SPIN, _SPIN_EXCITATIONS)
_HOLE_SPIN = np.einsum("Auv,Buw,kwv->kAB", _SPIN_EXCITATIONS.conj(), _SPIN_EXCITATIONS, _SPIN)
_GROUND_SOC_SPIN = np.einsum("kuv,Avu->kA", _SPIN, _SPIN_EXCITATIONS)


@dataclass(frozen=True)
class TDAStateInteraction:
    """One-body matrices ordered as S0, S1, …, T1x, T1y, T1z, T2x, … ."""

    dipole: NDArray[np.float64]  # (Cartesian, state bra, state ket), e * bohr
    soc: NDArray[np.complex128]  # (state bra, state ket), Hartree


def assemble_tda_state_interaction(
    singlet_amplitudes: ArrayLike,
    triplet_amplitudes: ArrayLike,
    dipole_mo: ArrayLike,
    soc_mo: ArrayLike,
    *,
    dipole_reference: ArrayLike | None = None,
) -> TDAStateInteraction:
    """Project effective one-body operators onto restricted TDA singles.

    Each spin manifold supplies real, orthonormal spatial roots with shape
    (roots, occupied, virtual), including an explicit zero-length root axis
    for an empty manifold. For raw restricted PySCF TDA amplitudes, use
    ``sqrt(2) * X``. The normalization of the spin excitation matrices is
    I/sqrt(2) and sigma_x,y,z/sqrt(2), with matrix row on the created spin
    and column on the annihilated spin, in alpha, beta order.

    Both MO operators have shape (Cartesian, MO bra, MO ket), with the
    occupied orbitals preceding the virtual orbitals. ``dipole_mo`` is real
    symmetric in e*bohr, with the desired electric-dipole sign. ``soc_mo`` is
    Hermitian in Hartree. Its kth MO block multiplies the spin operator
    sigma_k/2 and includes any imaginary unit and physical prefactors. The
    optional ``dipole_reference`` sets the total ground dipole, including any
    common nuclear contribution. The default is twice the occupied trace.

    This constructs matrix elements of the supplied effective operators. It
    uses the unrelaxed singles representation of the LR-TDA roots.
    """
    singlets = _orthonormal_roots(singlet_amplitudes, "singlet amplitudes")
    triplets = _orthonormal_roots(triplet_amplitudes, "triplet amplitudes")
    if singlets.shape[1:] != triplets.shape[1:]:
        raise ValueError("the singlet and triplet amplitudes must share one spatial orbital basis")
    nocc, nvir = singlets.shape[1:]
    dipole_mo = _hermitian_mo(_real_array(dipole_mo, "dipole MO operator"), nocc + nvir, "dipole MO operator")
    soc_mo = _hermitian_mo(np.asarray(soc_mo, dtype=complex), nocc + nvir, "SOC MO operator")

    if dipole_reference is None:
        reference = 2 * np.trace(dipole_mo[:, :nocc, :nocc], axis1=1, axis2=2)
    else:
        reference = _real_array(dipole_reference, "reference dipole")
        if reference.shape != (3,):
            raise ValueError("reference dipole must have shape (3,)")

    nsinglet, ntriplet = len(singlets), len(triplets)
    roots = np.concatenate((singlets, triplets))
    # Index the computed spatial blocks in the Cartesian spin basis.
    root = np.r_[np.arange(nsinglet), np.repeat(np.arange(nsinglet, len(roots)), 3)]
    spin = np.r_[np.zeros(nsinglet, dtype=int), np.tile(np.arange(1, 4), ntriplet)]
    root_bra = root[:, None]
    spin_bra = spin[:, None]
    nstate = len(root) + 1

    dipole_particle, dipole_hole, dipole_ground = _spatial_blocks(roots, dipole_mo, nocc)
    dipole = reference[:, None, None] * np.eye(nstate)
    dipole[:, 1:, 1:] += (dipole_particle - dipole_hole)[:, root_bra, root] * (spin_bra == spin)
    dipole[:, 0, 1:nsinglet + 1] = np.sqrt(2) * dipole_ground[:, :nsinglet]
    dipole[:, 1:nsinglet + 1, 0] = dipole[:, 0, 1:nsinglet + 1]

    soc_particle, soc_hole, soc_ground = _spatial_blocks(roots, soc_mo, nocc)
    soc = np.zeros((nstate, nstate), dtype=complex)
    soc[1:, 1:] = np.einsum("krs,krs->rs", soc_particle[:, root_bra, root], _PARTICLE_SPIN[:, spin_bra, spin])
    soc[1:, 1:] -= np.einsum("krs,krs->rs", soc_hole[:, root_bra, root], _HOLE_SPIN[:, spin_bra, spin])
    soc[0, 1:] = np.einsum("kr,kr->r", soc_ground[:, root], _GROUND_SOC_SPIN[:, spin])
    soc[1:, 0] = soc[0, 1:].conj()

    dipole = 0.5 * dipole + 0.5 * dipole.swapaxes(1, 2)
    soc = 0.5 * soc + 0.5 * soc.conj().T
    dipole.setflags(write=False)
    soc.setflags(write=False)
    return TDAStateInteraction(dipole=dipole, soc=soc)


def _orthonormal_roots(value, name):
    roots = _real_array(value, name)
    if roots.ndim != 3 or any(size == 0 for size in roots.shape[1:]):
        raise ValueError(f"{name} must have shape (roots, occupied, virtual) with occupied and virtual orbitals")
    roots_flat = roots.reshape(len(roots), roots.shape[1] * roots.shape[2])
    if not np.allclose(roots_flat @ roots_flat.T, np.eye(len(roots)), atol=1e-8, rtol=0):
        raise ValueError(f"{name} must contain orthonormal spatial roots")
    return roots


def _hermitian_mo(matrix, nmo, name):
    if matrix.shape != (3, nmo, nmo) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape (3, {nmo}, {nmo})")
    adjoint = matrix.conj().swapaxes(1, 2)
    if not np.allclose(matrix, adjoint, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    return 0.5 * matrix + 0.5 * adjoint


def _spatial_blocks(roots, operator, nocc):
    """Contract vv for the particle, transposed oo for the hole, and ov for S0."""
    particle = np.einsum("ria,kab,sib->krs", roots, operator[:, nocc:, nocc:], roots, optimize=True)
    hole = np.einsum("ria,kji,sja->krs", roots, operator[:, :nocc, :nocc], roots, optimize=True)
    ground = np.einsum("kia,ria->kr", operator[:, :nocc, nocc:], roots, optimize=True)
    return particle, hole, ground
