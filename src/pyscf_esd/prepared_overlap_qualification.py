"""Check independent calibration evidence for signed physical-root overlaps."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .state_tracking import _real_array


_ATOL = 1e-8


@dataclass(frozen=True)
class RawRootBasis:
    """A geometry and raw checkpoint with root labels in energy-vector order."""

    geometry_id: str
    checkpoint_id: str
    root_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.geometry_id, self.checkpoint_id)):
            raise ValueError("supply geometry and raw checkpoint identifiers")
        if (
            not isinstance(self.root_order, tuple)
            or len(self.root_order) < 2
            or any(isinstance(root, (bool, np.bool_)) or not isinstance(root, Integral) or root < 0
                   for root in self.root_order)
            or len(set(self.root_order)) != len(self.root_order)
        ):
            raise ValueError("supply at least two distinct nonnegative root labels in energy-vector order")


@dataclass(frozen=True)
class PhysicalRootOverlap:
    """One evaluation with source roots as rows and destination roots as columns.

    A declared flip changes the sign of a physical state in the evaluator's
    input, with its raw checkpoint and root order unchanged. Flip labels are
    root identifiers in ``root_order``. ``evaluator`` identifies the overlap
    implementation, including the state representation it evaluates.
    """

    source: RawRootBasis
    destination: RawRootBasis
    matrix: ArrayLike
    evaluator: str
    phase_transport_applied: bool
    flipped_source_root: int | None = None
    flipped_destination_root: int | None = None


def qualify_prepared_overlap(
    source: RawRootBasis,
    destination: RawRootBasis,
    *,
    forward: PhysicalRootOverlap,
    reverse: PhysicalRootOverlap,
    source_self: PhysicalRootOverlap,
    destination_self: PhysicalRootOverlap,
    source_flip: PhysicalRootOverlap,
    destination_flip: PhysicalRootOverlap,
    independent_forward: PhysicalRootOverlap,
) -> NDArray[np.float64]:
    """Check a calibration pair and return its read-only raw forward matrix.

    Choose a distinct-geometry pair whose matrix can distinguish the two
    orientations. Supply separately evaluated forward, swapped, self and
    single-root-flipped probes from the candidate implementation, plus a
    forward matrix from an independent physical-state implementation. The
    caller verifies actual independence, the raw checkpoint identities and
    corresponding root energies, and consistency of the electronic recipe
    and atom correspondence. Evidence derived by transforming the candidate
    output cannot establish these properties. Evaluator labels alone cannot
    verify independence. Use this contract on informative calibration pairs,
    including a sensitive seam when relevant. It imposes no second-evaluator
    requirement on subsequent production links.
    """
    if source.geometry_id == destination.geometry_id or source.checkpoint_id == destination.checkpoint_id:
        raise ValueError("a calibration pair must use distinct geometries and raw checkpoints")
    probes = (forward, reverse, source_self, destination_self, source_flip, destination_flip)
    for probe in (*probes, independent_forward):
        if not isinstance(probe.evaluator, str) or not probe.evaluator.strip():
            raise ValueError("identify the evaluator of each overlap matrix")
    if any(probe.evaluator != forward.evaluator for probe in probes):
        raise ValueError("the forward, swapped, self and gauge probes must use the same candidate evaluator")
    if independent_forward.evaluator == forward.evaluator:
        raise ValueError("supply a physical forward matrix from an independent evaluator")

    def matrix(
        probe: PhysicalRootOverlap,
        left: RawRootBasis,
        right: RawRootBasis,
        name: str,
        *,
        flipped_on: str | None = None,
    ) -> NDArray[np.float64]:
        if probe.source != left or probe.destination != right:
            raise ValueError(f"{name}: checkpoint identities and root order must match the requested axes")
        if not isinstance(probe.phase_transport_applied, (bool, np.bool_)) or probe.phase_transport_applied:
            raise ValueError(f"{name}: overlaps must be evaluated without phase transport")
        flips = (("source", probe.flipped_source_root, left), ("destination", probe.flipped_destination_root, right))
        for axis, label, basis in flips:
            if axis == flipped_on:
                if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or label not in basis.root_order:
                    raise ValueError(f"{name}: supply one explicit {axis} root flip")
            elif label is not None:
                raise ValueError(f"{name}: unexpected {axis} root flip")
        values = _real_array(probe.matrix, 2, name)
        if values.shape != (len(left.root_order), len(right.root_order)):
            raise ValueError(f"{name}: matrix dimensions must match the root counts")
        if np.any(np.abs(values) > 1.0 + _ATOL) or np.linalg.norm(values, ord=2) > 1.0 + _ATOL:
            raise ValueError(f"{name}: overlap matrix is incompatible with orthonormal states")
        return values

    def agrees(actual: NDArray[np.float64], expected: NDArray[np.float64], *, atol: float = _ATOL) -> bool:
        return bool(np.allclose(actual, expected, rtol=0, atol=atol))

    raw = matrix(forward, source, destination, "forward overlap")
    for probe, basis, name in ((source_self, source, "source"), (destination_self, destination, "destination")):
        if not agrees(matrix(probe, basis, basis, f"{name} self overlap"), np.eye(len(basis.root_order))):
            raise ValueError(f"{name} self overlap must be the identity in the same checkpoint gauge")

    swapped = matrix(reverse, destination, source, "swapped overlap")
    if raw.shape[0] == raw.shape[1] and agrees(raw, raw.T, atol=2 * _ATOL):
        raise ValueError("use a calibration pair whose overlap distinguishes the two orientations")
    if not agrees(swapped, raw.T):
        raise ValueError("swapped overlap must equal the transpose of the raw forward overlap")

    flipped_source = matrix(source_flip, source, destination, "source gauge probe", flipped_on="source")
    flipped_destination = matrix(
        destination_flip, source, destination, "destination gauge probe", flipped_on="destination",
    )
    expected_source = raw.copy()
    expected_destination = raw.copy()
    expected_source[source.root_order.index(source_flip.flipped_source_root), :] *= -1
    expected_destination[:, destination.root_order.index(destination_flip.flipped_destination_root)] *= -1
    if (
        agrees(expected_source, raw, atol=2 * _ATOL)
        or agrees(expected_destination, raw, atol=2 * _ATOL)
        or agrees(expected_source, expected_destination, atol=2 * _ATOL)
    ):
        raise ValueError("choose detectable root flips that distinguish the source and destination gauge probes")
    if not agrees(flipped_source, expected_source):
        raise ValueError("source gauge probe must flip only the specified row")
    if not agrees(flipped_destination, expected_destination):
        raise ValueError("destination gauge probe must flip only the specified column")

    reference = matrix(independent_forward, source, destination, "independent physical forward overlap")
    if not agrees(raw, reference):
        raise ValueError("raw forward overlap disagrees with the independent physical forward overlap")

    result = raw.copy()
    result.setflags(write=False)
    return result
