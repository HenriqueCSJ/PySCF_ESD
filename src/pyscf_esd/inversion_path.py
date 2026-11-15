"""Reflection-symmetric, one-dimensional coordinate through two H2CO minima."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pyscf_adapter import _as_real_array


# Match pyscf.data.nist.AMU2AU in the endpoint Hessian calculation (PySCF 2.15).
_AMU_TO_ELECTRON_MASS = 1822.8884858012984


class InversionPath:
    """Use an atom-ordered planar reference in the xy plane and a pyramidal well.

    Both endpoints are centered at their centers of mass. The excited well is
    properly aligned to the ground endpoint, preserving the supplied well
    and atom order. The dimensionless q equals 0 at the planar endpoint, +1
    at the aligned well, and -1 at its reflection through the xy plane. Along
    this constrained path, out-of-plane displacements are linear in q and
    in-plane relaxation is even in q.
    """

    def __init__(
        self,
        ground_coords_bohr: ArrayLike,
        excited_coords_bohr: ArrayLike,
        masses_amu: ArrayLike,
    ) -> None:
        # Four atoms in a common order are needed for this H2CO coordinate.
        masses = _as_real_array(masses_amu, (4,), "masses in amu")
        if np.any(masses <= 0):
            raise ValueError("masses in amu must be positive")
        ground = _as_real_array(ground_coords_bohr, (4, 3), "ground coordinates")
        excited = _as_real_array(excited_coords_bohr, (4, 3), "excited coordinates")
        self._ground = ground - np.average(ground, axis=0, weights=masses)
        excited_centered = excited - np.average(excited, axis=0, weights=masses)
        if np.max(np.abs(self._ground[:, 2])) > 1e-8:
            raise ValueError("the ground endpoint must lie in the xy plane within 1e-8 bohr")
        self._ground[:, 2] = 0.0

        left, singular, right = np.linalg.svd(excited_centered.T @ (masses[:, None] * self._ground))
        if singular[1] <= 1e-10 * singular[0]:
            raise ValueError("the endpoints must define a noncollinear alignment")
        if np.linalg.det(left @ right) < 0:
            left[:, -1] *= -1.0
        self._excited_rotation = left @ right
        displacement = excited_centered @ self._excited_rotation - self._ground
        self._even = displacement.copy()
        self._even[:, 2] = 0.0
        self._odd = displacement.copy()
        self._odd[:, :2] = 0.0
        self._masses_au = masses * _AMU_TO_ELECTRON_MASS
        self._sqrt_mass = np.sqrt(self._masses_au)[:, None]
        for q in (0.0, 1.0):
            self.metric(q)

    @property
    def excited_rotation(self) -> NDArray[np.float64]:
        """Rotation A for row-vector coordinates, ``centered_excited @ A``.

        Rotate each atom-pair Cartesian Hessian block H as ``A.T @ H @ A``.
        The translation to the center-of-mass frame leaves the Hessian unchanged.
        """
        return self._excited_rotation.copy()

    @property
    def masses_au(self) -> NDArray[np.float64]:
        """Atom-ordered masses in units of electron mass."""
        return self._masses_au.copy()

    def geometry(self, q: float) -> NDArray[np.float64]:
        """Evaluate the centered Cartesian coordinates in bohr."""
        q = float(_as_real_array(q, (), "q"))
        fraction = q / np.hypot(1.0, q)
        return self._ground + q * self._odd + 2.0 * fraction**2 * self._even

    def tangent(self, q: float) -> NDArray[np.float64]:
        """Return the Cartesian derivative dR/dq, with shape (4, 3)."""
        q = float(_as_real_array(q, (), "q"))
        inverse_scale = 1.0 / np.hypot(1.0, q)
        return self._odd + 4.0 * (q * inverse_scale) * inverse_scale**3 * self._even

    def _weighted_tangent(self, q: float) -> NDArray[np.float64]:
        geometry = self.geometry(q)
        tangent = self.tangent(q)
        # Remove translation and the angular motion omega x R. The projection
        # gives the inertia-metric correction -L.T @ inv(I) @ L as a sum of squares.
        tangent -= np.average(tangent, axis=0, weights=self._masses_au)
        weighted_geometry = self._sqrt_mass * geometry
        inertia = np.sum(weighted_geometry**2) * np.eye(3) - weighted_geometry.T @ weighted_geometry
        angular_momentum = np.sum(np.cross(geometry, self._masses_au[:, None] * tangent), axis=0)
        try:
            angular_velocity = np.linalg.solve(inertia, angular_momentum)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"rotational inertia is singular at q={q}") from error
        weighted_tangent = self._sqrt_mass * (tangent - np.cross(angular_velocity, geometry))
        internal_speed = np.linalg.norm(weighted_tangent)
        rigid_speed_tolerance = 1e-12 * max(1.0, np.linalg.norm(self._sqrt_mass * tangent))
        if not np.isfinite(internal_speed) or internal_speed <= rigid_speed_tolerance:
            raise ValueError(f"the inversion coordinate has zero or undefined internal speed at q={q}")
        return weighted_tangent

    def metric(self, q: float) -> float:
        """Return the square of ds/dq in units of electron mass times bohr²."""
        return float(np.sum(self._weighted_tangent(q)**2))

    def arclength(self, q: float) -> float:
        """Integrate signed s(q) from q=0, in units of bohr times sqrt(electron mass)."""
        from scipy.integrate import quad

        q = float(_as_real_array(q, (), "q"))
        length, _ = quad(lambda point: np.sqrt(self.metric(point)), 0.0, abs(q), epsabs=1e-10, epsrel=1e-10)
        return float(np.copysign(length, q))

    def local_tangent(self, q: float) -> NDArray[np.float64]:
        """Return the unit internal tangent in mass-weighted Cartesian coordinates (4, 3)."""
        weighted_tangent = self._weighted_tangent(q)
        return weighted_tangent / np.linalg.norm(weighted_tangent)
