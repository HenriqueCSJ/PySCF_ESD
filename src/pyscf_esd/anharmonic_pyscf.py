"""Prepare a common-coordinate anharmonic model from completed PySCF points."""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .cpu import ElectronicPoint
from .state_tracking import StateMatch, StateTrackingError, orbital_overlap_blocks, track_state


@dataclass(frozen=True)
class TrackedAnharmonicInputs:
    """State-resolved energies and dipoles in the order of the supplied points.

    Both potentials use the first point's ground energy as their common zero.
    The first transition dipole defines the arbitrary global electronic sign.
    ``links`` contains matches for consecutive points, followed by the closing
    match if ``periodic=True`` was requested.
    """

    ground_potential_hartree: NDArray[np.float64]
    excited_potential_hartree: NDArray[np.float64]
    transition_dipole_au: NDArray[np.float64]
    root_indices: NDArray[np.intp]
    phases: NDArray[np.intp]
    links: tuple[StateMatch, ...]


def track_anharmonic_scan(
    points: Sequence[ElectronicPoint], *, target_index: int, periodic: bool,
    minimum_overlap: float = 0.65, minimum_margin: float = 0.10,
    minimum_gap_hartree: float = 1e-4,
) -> TrackedAnharmonicInputs:
    """Follow an isolated real singlet TDA root on an already evaluated scan.

    Supply points from one electronic recipe and one atom order in a common
    Cartesian frame. Use the same nuclear geometry at each coordinate for
    both states. For a periodic coordinate, omit the repeated endpoint. The
    closing link must match the original state and sign for an ordinary
    periodic nuclear model. A failed sign check can indicate inadequate
    sampling or a nontrivial electronic phase around a loop.

    Transition-density overlaps are a local state-character and phase proxy.
    Review the initial root assignment and the coordinate mapping before use.
    Calling this function evaluates cross-geometry AO overlaps and launches no
    SCF, TDA, gradient, or Hessian calculations.
    """
    from pyscf import gto

    points = tuple(points)
    if len(points) < 3:
        raise ValueError("at least three evaluated points are required")
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError("periodic must be a boolean")
    first = points[0]
    root_count = len(first.excitation_energies_hartree)
    if (isinstance(target_index, (bool, np.bool_)) or not isinstance(target_index, Integral)
        or not 0 <= target_index < root_count - 1):
        raise ValueError("provide a valid zero-based target and at least one higher solved root")

    atom_map = tuple(first.mol.atom_symbol(i) for i in range(first.mol.natm))
    occupations = 2 * (np.arange(first.mo_coeff.shape[1]) < first.nocc)
    for point in points:
        if (point.settings != first.settings or point.nocc != first.nocc
            or point.mo_coeff.shape != first.mo_coeff.shape
            or tuple(point.mol.atom_symbol(i) for i in range(point.mol.natm)) != atom_map
            or len(point.excitation_energies_hartree) != root_count):
            raise ValueError("scan points must share the electronic recipe, atom order and root count")

    thresholds = dict(minimum_overlap=minimum_overlap, minimum_margin=minimum_margin,
                      minimum_gap_hartree=minimum_gap_hartree)

    def match(source, destination, root_index, phase):
        occ, vir = orbital_overlap_blocks(
            source.mo_coeff, destination.mo_coeff,
            gto.intor_cross("int1e_ovlp", source.mol, destination.mol),
            occupations, occupations,
        )
        result = track_state(
            phase * source.amplitudes[root_index], destination.amplitudes, occ, vir,
            candidate_energies_hartree=destination.excitation_energies_hartree, **thresholds,
        )
        if result.index == root_count - 1:
            raise StateTrackingError("the match is the highest solved root; request additional roots")
        return result

    anchor = match(first, first, target_index, 1)
    if anchor.index != target_index or anchor.phase != 1:
        raise StateTrackingError("could not validate the reference root")
    indices, phases, links = [int(target_index)], [1], []
    for index, (source, destination) in enumerate(zip(points, points[1:])):
        try:
            result = match(source, destination, indices[-1], phases[-1])
        except StateTrackingError as exc:
            raise StateTrackingError(f"scan link {index}→{index + 1}: {exc}") from exc
        indices.append(result.index)
        phases.append(result.phase)
        links.append(result)
    if periodic:
        try:
            closing = match(points[-1], first, indices[-1], phases[-1])
        except StateTrackingError as exc:
            raise StateTrackingError(f"periodic closing link: {exc}") from exc
        if closing.index != target_index or closing.phase != 1:
            raise StateTrackingError("periodic scan does not close in the original state and phase")
        links.append(closing)

    ground = np.array([point.ground_energy_hartree for point in points]) - first.ground_energy_hartree
    excited = ground + np.array([point.excitation_energies_hartree[root]
                                 for point, root in zip(points, indices)])
    dipoles = np.stack([phase * point.transition_dipoles_au[root]
                        for point, root, phase in zip(points, indices, phases)])
    if any(array.dtype.kind not in "fiu" or not np.all(np.isfinite(array))
           for array in (ground, excited, dipoles)):
        raise ValueError("scan energies and dipoles must be finite and real")
    if dipoles.shape != (len(points), 3):
        raise ValueError("each selected transition dipole must have three components")
    return TrackedAnharmonicInputs(
        ground, excited, dipoles, np.array(indices, dtype=np.intp),
        np.array(phases, dtype=np.intp), tuple(links),
    )
