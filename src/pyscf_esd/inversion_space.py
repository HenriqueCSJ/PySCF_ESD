"""Mass-weighted inversion coordinates including planar relaxation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .inversion_path import InversionPath


@dataclass(frozen=True, init=False)
class InversionSpace:
    """Cartesian space through S0, both S1 wells and the planar S1 saddle.

    The first two coordinates keep the original plane's unit displacements
    q and r. The third coordinate t has units of bohr times sqrt(electron
    mass). ``lengths`` gives the corresponding mass-weighted scales of q and
    r. The columns of ``directions`` are orthonormal mass-weighted directions
    in flattened atom/Cartesian order. Reflection through the plane changes
    q to -q and leaves r and t unchanged.
    """

    origin: NDArray[np.float64]
    lengths: NDArray[np.float64]
    directions: NDArray[np.float64]
    masses_au: NDArray[np.float64]
    saddle_coordinates: tuple[float, float, float]

    def __init__(self, path: InversionPath, planar_geometry_bohr):
        origin = path.geometry(0.0)
        plus, minus = path.geometry(1.0), path.geometry(-1.0)
        masses = path.masses_au
        roots = np.sqrt(masses)[:, None]
        odd_even = np.stack(((plus - minus) / 2, (plus + minus) / 2 - origin))
        old_plane = (odd_even * roots).reshape(2, -1).T
        lengths = np.linalg.norm(old_plane, axis=0)
        if np.any(lengths <= 1e-10):
            raise ValueError("both inversion-plane directions must be nonzero")
        old_plane /= lengths

        raw = np.asarray(planar_geometry_bohr)
        if raw.shape != origin.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError("planar geometry must provide finite Cartesian coordinates in the same atom order")
        centered = raw - np.average(raw, weights=masses, axis=0)
        left, _, right = np.linalg.svd((centered * masses[:, None]).T @ origin)
        if np.linalg.det(left @ right) < 0:
            left[:, -1] *= -1
        aligned = centered @ (left @ right)
        if np.max(np.abs(aligned[:, 2] - origin[:, 2])) > 1e-8:
            raise ValueError("the relaxed reference geometry must lie in the ground-state plane")

        delta = ((aligned - origin) * roots).ravel()
        in_plane = old_plane.T @ delta
        residual = delta - old_plane @ in_plane
        distance = np.linalg.norm(residual)
        if distance <= 1e-8:
            raise ValueError("the planar reference does not define a third coordinate")
        directions = np.column_stack((old_plane, residual / distance))
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("the inversion-space axes are not orthonormal")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "masses_au", masses)
        object.__setattr__(self, "saddle_coordinates", (0.0, float(in_plane[1] / lengths[1]), float(distance)))

    def geometry(self, q: float, r: float, t: float) -> NDArray[np.float64]:
        """Return the centered Cartesian coordinates in bohr."""
        point = np.array([q * self.lengths[0], r * self.lengths[1], t], dtype=float)
        if not np.all(np.isfinite(point)):
            raise ValueError("inversion-space coordinates must be finite")
        displacement = (self.directions @ point).reshape(-1, 3) / np.sqrt(self.masses_au)[:, None]
        return self.origin + displacement
