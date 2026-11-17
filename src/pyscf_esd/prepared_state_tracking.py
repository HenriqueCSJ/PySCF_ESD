"""Qualify and phase two isolated real electronic states from prepared overlaps."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .state_tracking import StateMatch, StateTrackingError, _real_array, match_state_overlaps


@dataclass(frozen=True)
class PreparedStatePath:
    """Selected energies, raw-state indices and phases in the initial label order.

    The first axis indexes the points and the second indexes the two labeled
    states. Both first-point phases are +1. ``links`` contains the forward
    match diagnostics in the raw input gauges, including the seam for a closed
    path. ``closure_signs`` records the two individual holonomies, or ``None``
    for an open path.
    """

    root_indices: NDArray[np.intp]
    phases: NDArray[np.intp]
    energies_hartree: NDArray[np.float64]
    links: tuple[tuple[StateMatch, StateMatch], ...]
    closure_signs: tuple[int, int] | None

    @property
    def signed_link_overlaps(self) -> NDArray[np.float64]:
        """Raw selected overlaps for each link and state, including any seam."""
        result = np.array([[match.overlaps[match.index] for match in link] for link in self.links])
        result.setflags(write=False)
        return result

    def align_transition_dipoles(self, raw_dipoles_au: ArrayLike) -> NDArray[np.float64]:
        """Apply both phases to raw Cartesian dipoles for the selected root pairs."""
        dipoles = _real_array(raw_dipoles_au, 2, "raw_dipoles_au")
        if dipoles.shape != (len(self.phases), 3):
            raise ValueError("supply one raw Cartesian transition dipole per path point")
        return self.phases.prod(axis=1)[:, None] * dipoles


def track_prepared_states(
    energies_hartree: Sequence[ArrayLike],
    signed_overlaps: Sequence[ArrayLike],
    *,
    initial_indices: tuple[int, int],
    periodic: bool = False,
    minimum_overlap: float = 0.65,
    minimum_margin: float = 0.10,
    minimum_gap_hartree: float = 1e-4,
) -> PreparedStatePath:
    """Track two labeled states using backend-supplied physical-state overlaps.

    Supply all relevant competing roots at each point with energies on a
    common zero. Each overlap matrix has raw orthonormal physical states as
    rows at the source point and columns at the destination, in the same root
    order as the energies. For a closed path, omit the repeated endpoint and
    supply an extra matrix from the last point to the original first point
    in its original gauge. The caller establishes the physical overlap
    capability, matching electronic recipe, geometry and atom correspondence,
    and relevant candidate coverage. The routine qualifies only the supplied
    samples. Transition-density similarities and unsigned magnitudes cannot
    establish the phases of the physical states.

    The absolute-overlap winner must agree in both directions, pass the
    overlap and margin cutoffs, and be separated from the closest supplied
    competitor in energy at each endpoint. The two selected roots must be
    distinct, and a closed path must return to each initial root identity.
    A return with reversed phase is permitted and recorded separately for
    each state. Provide ``signed_link_overlaps`` and raw dipoles to
    ``phase_periodic_scan`` with the same overlap cutoff to construct a rotor
    model.
    """
    energies = tuple(_real_array(values, 1, "root energies") for values in energies_hartree)
    point_count = len(energies)
    if not isinstance(periodic, (bool, np.bool_)):
        raise ValueError("periodic must be a boolean")
    if point_count < 2 or any(len(values) < 2 for values in energies):
        raise ValueError("supply at least two points with at least two roots per point")
    if periodic and point_count < 3:
        raise ValueError("a closed path requires at least three sampled points")
    if len(initial_indices) != 2 or any(
        isinstance(root, (bool, np.bool_)) or not isinstance(root, Integral) or not 0 <= root < len(energies[0])
        for root in initial_indices
    ) or initial_indices[0] == initial_indices[1]:
        raise ValueError("initial_indices must contain two distinct valid zero-based roots")
    if not isfinite(minimum_gap_hartree) or minimum_gap_hartree <= 0:
        raise ValueError("minimum_gap_hartree must be finite and positive")

    link_count = point_count if periodic else point_count - 1
    if len(signed_overlaps) != link_count:
        raise ValueError("supply one signed overlap matrix per link, including the seam for a closed path")
    thresholds = dict(minimum_overlap=minimum_overlap, minimum_margin=minimum_margin,
                      minimum_gap_hartree=minimum_gap_hartree)

    root_indices = np.empty((point_count, 2), dtype=np.intp)
    phases = np.empty_like(root_indices)
    root_indices[0] = initial_indices
    phases[0] = 1
    links = []
    closure_signs = None
    for source, raw_overlap in enumerate(signed_overlaps):
        destination = (source + 1) % point_count
        context = f"prepared state link {source} to {destination}"
        overlap = _real_array(raw_overlap, 2, f"{context} overlaps")
        if overlap.shape != (len(energies[source]), len(energies[destination])):
            raise ValueError(f"{context}: overlap matrix dimensions must match the root counts")
        if np.any(np.abs(overlap) > 1.0 + 1e-8) or np.linalg.norm(overlap, ord=2) > 1.0 + 1e-8:
            raise ValueError(f"{context}: overlap matrix is incompatible with normalized orthogonal states")

        try:
            source_roots = root_indices[source]
            matches = tuple(
                match_state_overlaps(overlap[root], candidate_energies_hartree=energies[destination], **thresholds)
                for root in source_roots
            )
            destination_roots = tuple(match.index for match in matches)
            if destination_roots[0] == destination_roots[1]:
                raise StateTrackingError("both tracked states match the same destination root")
            if destination == 0 and not np.array_equal(destination_roots, root_indices[0]):
                raise StateTrackingError("seam does not return to both initial root identities")
            for root, match in zip(source_roots, matches):
                reverse = match_state_overlaps(
                    overlap[:, match.index], candidate_energies_hartree=energies[source], **thresholds,
                )
                if reverse.index != root:
                    raise StateTrackingError(f"reverse assignment does not recover source root {root + 1}")
        except StateTrackingError as exc:
            raise StateTrackingError(f"{context}: {exc}") from exc

        links.append(matches)
        destination_phases = phases[source] * np.array([match.phase for match in matches])
        if destination == 0:
            closure_signs = (int(destination_phases[0]), int(destination_phases[1]))
        else:
            root_indices[destination] = destination_roots
            phases[destination] = destination_phases

    selected_energies = np.stack([values[roots] for values, roots in zip(energies, root_indices)])
    for array in (root_indices, phases, selected_energies):
        array.setflags(write=False)
    return PreparedStatePath(root_indices, phases, selected_energies, tuple(links), closure_signs)
