"""Track real restricted TDA states by particle–hole transition densities."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class StateMatch:
    """Assignment in candidate order, with the phase relative to the reference."""

    index: int
    phase: int
    overlap: float
    overlaps: NDArray[np.float64]
    margin: float | None
    gap_hartree: float | None


class StateTrackingError(ValueError):
    """No well-isolated candidate could be assigned to the target state."""


def _real_array(data: ArrayLike, ndim: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(data)
    if array.ndim != ndim or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real rank-{ndim} array")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def orbital_overlap_blocks(
    reference_mo_coeff: ArrayLike,
    candidate_mo_coeff: ArrayLike,
    ao_cross_overlap: ArrayLike,
    reference_mo_occ: ArrayLike,
    candidate_mo_occ: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return occupied and virtual MO overlaps between two geometries.

    The AO overlap has reference rows and candidate columns, as returned by
    ``pyscf.gto.intor_cross("int1e_ovlp", reference_mol, candidate_mol)``.
    The geometries share a Cartesian frame. Input MO coefficient columns are
    orthonormal in their respective AO metrics. Occupations must be 2 or 0,
    with X amplitude axes in the same order as the selected MO columns.
    """
    ref = _real_array(reference_mo_coeff, 2, "reference_mo_coeff")
    new = _real_array(candidate_mo_coeff, 2, "candidate_mo_coeff")
    cross = _real_array(ao_cross_overlap, 2, "ao_cross_overlap")
    ref_occ = _real_array(reference_mo_occ, 1, "reference_mo_occ")
    new_occ = _real_array(candidate_mo_occ, 1, "candidate_mo_occ")
    if cross.shape != (ref.shape[0], new.shape[0]):
        raise ValueError("ao_cross_overlap does not match the MO coefficient AO dimensions")
    for coefficients, occupations in ((ref, ref_occ), (new, new_occ)):
        if occupations.size != coefficients.shape[1]:
            raise ValueError("occupations and MO coefficient columns must correspond")
        if not np.all((occupations == 0.0) | (occupations == 2.0)):
            raise ValueError("restricted TDA requires occupations of 0 or 2")
        if not np.any(occupations == 2.0) or not np.any(occupations == 0.0):
            raise ValueError("tracking requires both occupied and virtual orbitals")
    ref_filled, new_filled = ref_occ == 2.0, new_occ == 2.0
    if np.count_nonzero(ref_filled) != np.count_nonzero(new_filled):
        raise ValueError("the two geometries must have the same electron count")
    mo_cross = ref.T @ cross @ new
    return mo_cross[np.ix_(ref_filled, new_filled)], mo_cross[np.ix_(~ref_filled, ~new_filled)]


def track_state(
    reference_x: ArrayLike,
    candidate_x: ArrayLike,
    occupied_overlap: ArrayLike,
    virtual_overlap: ArrayLike,
    *,
    candidate_energies_hartree: ArrayLike | None = None,
    minimum_overlap: float = 0.65,
    minimum_margin: float = 0.10,
    minimum_gap_hartree: float = 0.0,
) -> StateMatch:
    """Select one candidate and determine its sign using the AO metric.

    The real X matrices are indexed by occupied and virtual orbitals.
    ``candidate_x`` adds a leading root axis. The two overlap blocks have
    reference rows and candidate columns. The signed scores are normalized
    Hilbert–Schmidt overlaps of the unsymmetrized particle–hole transition
    densities. They measure state character and resolve the relative sign
    of its transition density, without representing many-electron overlaps.

    The best absolute score must exceed zero and reach ``minimum_overlap``.
    Competing candidates need a score separation of at least
    ``minimum_margin``; exact ties always fail. Margin or gap thresholds
    require at least two candidates. With candidate energies, the gap is to
    the closest sampled root in energy. A positive ``minimum_gap_hartree``
    rejects nearby roots, since assignments to individual eigenvectors can
    be unreliable within a near-degenerate subspace. The supplied roots
    must span the states relevant to the match. Multiply the chosen raw X
    and transition dipole by ``phase`` to align their signs to the reference.
    """
    reference = _real_array(reference_x, 2, "reference_x")
    candidates = _real_array(candidate_x, 3, "candidate_x")
    occ = _real_array(occupied_overlap, 2, "occupied_overlap")
    vir = _real_array(virtual_overlap, 2, "virtual_overlap")
    if not candidates.shape[0]:
        raise ValueError("at least one candidate is required")
    nroot, nocc, nvir = candidates.shape
    if occ.shape != (reference.shape[0], nocc) or vir.shape != (reference.shape[1], nvir):
        raise ValueError("the MO overlap blocks do not match the X occupied and virtual dimensions")
    for name, threshold, upper in (
        ("minimum_overlap", minimum_overlap, 1.0),
        ("minimum_margin", minimum_margin, 1.0),
        ("minimum_gap_hartree", minimum_gap_hartree, float("inf")),
    ):
        if not isfinite(threshold) or not 0.0 <= threshold <= upper:
            raise ValueError(f"{name} is out of range")
    if nroot < 2 and (minimum_margin > 0.0 or minimum_gap_hartree > 0.0):
        raise ValueError("a positive margin or gap threshold requires at least two candidates")
    energies = None
    if candidate_energies_hartree is not None:
        energies = _real_array(candidate_energies_hartree, 1, "candidate_energies_hartree")
        if energies.shape != (nroot,):
            raise ValueError("one excitation energy is required per candidate")
    elif minimum_gap_hartree > 0.0:
        raise ValueError("candidate energies are required for a positive gap threshold")

    reference_norm = np.linalg.norm(reference)
    candidate_norms = np.linalg.norm(candidates, axis=(1, 2))
    if (
        reference_norm == 0.0 or not np.isfinite(reference_norm)
        or np.any(candidate_norms == 0.0) or not np.all(np.isfinite(candidate_norms))
    ):
        raise ValueError("X amplitudes must have finite nonzero Frobenius norms")
    transported = occ.T @ (reference / reference_norm) @ vir
    overlaps = np.einsum("ia,kia->k", transported, candidates / candidate_norms[:, None, None])
    if not np.all(np.isfinite(overlaps)) or np.any(np.abs(overlaps) > 1.0 + 1e-8):
        raise ValueError("the supplied orbital overlaps produce invalid normalized scores")
    overlaps = np.clip(overlaps, -1.0, 1.0)
    overlaps.setflags(write=False)
    scores = np.abs(overlaps)
    index = int(np.argmax(scores))
    best = float(scores[index])
    if best == 0.0 or best < minimum_overlap:
        raise StateTrackingError(
            f"target state has insufficient overlap: best root {index + 1} scored {best:.6f}"
        )
    margin = gap = None
    if nroot > 1:
        other_roots = np.arange(nroot) != index
        second = float(scores[other_roots].max())
        margin = best - second
        if margin <= np.finfo(float).eps or margin < minimum_margin:
            raise StateTrackingError(
                f"ambiguous state identity: best and second scores are {best:.6f} and {second:.6f}"
            )
        if energies is not None:
            gap = float(np.min(np.abs(energies[other_roots] - energies[index])))
            if minimum_gap_hartree > 0.0 and gap < minimum_gap_hartree:
                raise StateTrackingError(
                    f"near-degenerate state: root {index + 1} has a gap of {gap:.6g} Hartree"
                )
    return StateMatch(index, 1 if overlaps[index] > 0.0 else -1, best, overlaps, margin, gap)
