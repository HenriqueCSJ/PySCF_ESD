"""Determinant representation of prepared restricted MRSF singlet roots."""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .determinant_overlap import _spin_minors
from .state_tracking import _real_array


@dataclass(frozen=True)
class MRSFSingletExpansion:
    """Restricted root amplitudes from which signed determinants can be materialized."""

    _amplitudes: NDArray[np.float64]

    @property
    def n_core(self) -> int:
        return self._amplitudes.shape[1] - 2

    @property
    def n_orbitals(self) -> int:
        return self._amplitudes.shape[2] + self.n_core

    @cached_property
    def _occupations(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        occupied = np.arange(self.n_core + 2)
        holes = np.broadcast_to(occupied, (occupied.size, occupied.size))[
            ~np.eye(occupied.size, dtype=bool)
        ].reshape(occupied.size, occupied.size - 1)
        particles = np.arange(self.n_core, self.n_orbitals)
        core = np.broadcast_to(np.arange(self.n_core), (particles.size, self.n_core))
        return holes, np.column_stack((core, particles))

    @cached_property
    def _determinants(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        holes, particles = self._occupations
        terms: dict[tuple[tuple[int, ...], tuple[int, ...]], NDArray[np.float64]] = {}
        for index, occupied in enumerate(holes):
            for column, attached in enumerate(particles):
                weights = self._amplitudes[:, index, column] / np.sqrt(2)
                if np.any(weights):
                    pair = (tuple(occupied), tuple(attached))
                    for key in (pair, pair[::-1]):
                        if key in terms:
                            terms[key] += weights
                        else:
                            terms[key] = weights.copy()
        alpha, beta = (np.asarray(occupations, dtype=np.int64) for occupations in zip(*terms))
        coefficients = np.column_stack(tuple(terms.values()))
        for array in (coefficients, alpha, beta):
            array.setflags(write=False)
        return coefficients, alpha, beta

    @property
    def coefficients(self) -> NDArray[np.float64]:
        return self._determinants[0]

    @property
    def alpha(self) -> NDArray[np.int64]:
        return self._determinants[1]

    @property
    def beta(self) -> NDArray[np.int64]:
        return self._determinants[2]

    def _response_density_mo(self, bra_root: int, ket_root: int) -> NDArray[np.float64]:
        """Return the unrelaxed difference density or transition density in MO order."""
        phases = (-1) ** (np.arange(self.n_core + 2) + 1)[:, None]
        bra = self._amplitudes[bra_root] * phases
        ket = self._amplitudes[ket_root] * phases
        # When a one-body operator connects the paired and open-open
        # sectors, its spin-summed contribution carries an extra sqrt(2).
        # The signed overlap amplitudes already contain 1/sqrt(2) in the
        # open-open block. Split the sectors so each term gets its weight.
        bra_open = np.zeros_like(bra)
        ket_open = np.zeros_like(ket)
        bra_open[-2:, :2] = bra[-2:, :2]
        ket_open[-2:, :2] = ket[-2:, :2]
        bra -= bra_open
        ket -= ket_open
        density = np.zeros((self.n_orbitals, self.n_orbitals))
        density[self.n_core:, self.n_core:] = (
            ket.T @ bra + 2 * (ket_open.T @ bra + ket.T @ bra_open + ket_open.T @ bra_open)
        )
        n_alpha = self.n_core + 2
        density[:n_alpha, :n_alpha] -= (
            bra @ ket.T + 2 * (bra_open @ ket.T + bra @ ket_open.T + bra_open @ ket_open.T)
        )
        return density

    def overlap(self, ket: "MRSFSingletExpansion", cross_mo_overlap: ArrayLike) -> NDArray[np.float64]:
        """Contract the signed response expansions with the raw bra-by-ket MO metric."""
        metric = _real_array(cross_mo_overlap, 2, "cross_mo_overlap")
        if self.n_core != ket.n_core:
            raise ValueError("bra and ket must have the same electron count")
        if metric.shape != (self.n_orbitals, ket.n_orbitals):
            raise ValueError("cross_mo_overlap shape must match the two orbital spaces")
        if np.linalg.norm(metric, ord=2) > 1.0 + 1e-8:
            raise ValueError("cross_mo_overlap is incompatible with orthonormal orbitals")
        bra_holes, bra_particles = self._occupations
        ket_holes, ket_particles = ket._occupations
        hh = _spin_minors(metric, bra_holes, ket_holes)
        pp = _spin_minors(metric, bra_particles, ket_particles)
        hp = _hole_particle_minors(metric, self.n_core)
        ph = _hole_particle_minors(metric.T, self.n_core).T

        # Spin inversion gives the same two contributions twice, cancelling
        # the 1/sqrt(2) on each side. The two mixed blocks retain their own
        # orientations because the cross-geometry metric is not symmetric.
        bra = self._amplitudes.reshape(len(self._amplitudes), -1)
        overlap = np.empty((len(bra), len(ket._amplitudes)))
        for root, amplitudes in enumerate(ket._amplitudes):
            transformed = hh @ amplitudes @ pp.T + hp @ amplitudes.T @ ph.T
            overlap[:, root] = bra @ transformed.ravel()
        return overlap


def _hole_particle_minors(metric: NDArray[np.float64], n_core: int) -> NDArray[np.float64]:
    """Return the minors after removing one bra orbital and attaching a ket particle."""
    n_alpha = n_core + 2
    core = metric[:n_alpha, :n_core]
    particle_columns = metric[:n_alpha, n_core:]
    left, singular_values, right = np.linalg.svd(core, full_matrices=True)
    orientation = np.linalg.slogdet(left)[0] * np.linalg.slogdet(right)[0]
    u, v = left[:, -2:].T
    # Append a particle column and unit column e_i to the shared core block.
    # The determinant in the two-dimensional orthogonal complement is the
    # desired minor times the cofactor sign of e_i in the last column.
    cofactors = v[:, None] * (u @ particle_columns) - u[:, None] * (v @ particle_columns)
    signs = (-1) ** (np.arange(n_alpha) + n_alpha - 1)
    return (orientation * np.prod(singular_values) * signs[:, None]) * cofactors


def expand_restricted_mrsf_singlets(
    response_vectors: ArrayLike, *, n_core: int, n_orbitals: int,
) -> MRSFSingletExpansion:
    """Expand the restricted OpenQP MRSF singlet response-vector representation.

    Rows of the input index the beta particle and alpha hole, with hole
    index varying fastest. Columns index the roots in their input gauges.
    The orbital order is the doubly occupied core, the two SOMOs and then
    the virtual orbitals. The first diagonal SOMO slot stores the open-open
    coefficient and its partner slot is unused.

    The expansion is for the two-reference restricted MRSF model and its
    unrelaxed transition properties. The caller verifies the backend and
    checkpoint representation before using this map. Energies and
    gradients come from the original DFT response calculation.
    """
    if (isinstance(n_core, (bool, np.bool_)) or not isinstance(n_core, Integral)
            or isinstance(n_orbitals, (bool, np.bool_)) or not isinstance(n_orbitals, Integral)
            or n_core < 0 or n_orbitals < n_core + 2):
        raise ValueError("supply a nonnegative core and an orbital space containing both SOMOs")
    n_core, n_orbitals = int(n_core), int(n_orbitals)
    n_alpha = n_core + 2
    expected_rows = (n_orbitals - n_core) * n_alpha
    vectors = _real_array(response_vectors, 2, "response_vectors")
    if vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
        raise ValueError("expected one particle-major spin-flip vector per root")
    if np.any(np.abs(vectors[n_alpha + n_core + 1]) > 1e-10):
        raise ValueError("the unused diagonal SOMO slot is populated")

    roots = vectors.T.reshape(vectors.shape[1], n_orbitals - n_core, n_alpha)
    roots = roots.transpose(0, 2, 1).copy()
    roots[:, -1, 1] = 0
    packed = roots.reshape(len(roots), -1)
    if not np.allclose(packed @ packed.T, np.eye(len(roots)), atol=1e-8, rtol=0):
        raise ValueError("the expanded roots are not orthonormal in their common MO basis")

    # The two diagonal SOMO slots represent one normalized open-open CSF.
    scale = 1 / np.sqrt(2)
    roots[:, -2, 0] *= scale
    roots[:, -1, 1] = -roots[:, -2, 0]
    # Annihilating an alpha orbital and inserting a beta orbital into the
    # ordered alpha-then-beta determinant gives this hole-dependent sign.
    roots *= (-1) ** (np.arange(n_alpha) + 1)[None, :, None]
    # Spin inversion already occurs within the open-open block. Use the
    # extra factor there so the explicit inversion sum counts each term once.
    roots[:, -2:, :2] *= scale
    roots.setflags(write=False)
    return MRSFSingletExpansion(roots)
