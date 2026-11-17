"""Mass-metric coordinates for one periodic internal rotor and its spectators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic import _real_array


_Real = NDArray[np.float64]


@dataclass(frozen=True)
class RotorProjection:
    """Orthonormal mass-weighted columns and an angular inertia in m_e bohr²."""

    rigid_axes: _Real
    unit_torsion: _Real
    spectator_axes: _Real
    inertia_me_bohr2: float


@dataclass(frozen=True)
class RotorSpectatorModes:
    """Harmonic spectators projected at one stationary geometry."""

    frequencies_hartree: _Real
    modes: _Real
    frozen_torsion_curvature_hartree: float
    conditional_torsion_curvature_hartree: float
    full_harmonic_zpe_hartree: float


@dataclass(frozen=True)
class RotorSpectatorMap:
    """Affine spectator coordinates obey Q_f = rotation @ Q_i + shift_au."""

    rotation: _Real
    shift_au: _Real
    initial_projection_singular_values: _Real
    final_projection_singular_values: _Real


def _geometry_and_masses(coordinates_bohr, masses_me):
    coordinates = _real_array(coordinates_bohr, "rotor coordinates in bohr")
    masses = _real_array(masses_me, "rotor masses in electron masses")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 3:
        raise ValueError("provide Cartesian coordinates for at least three atoms")
    if masses.shape != (len(coordinates),) or np.any(masses <= 0):
        raise ValueError("provide one positive mass in electron masses per atom")
    return coordinates, masses


def project_torsion(
    coordinates_bohr: ArrayLike, masses_me: ArrayLike, tangent_bohr_per_rad: ArrayLike,
) -> RotorProjection:
    """Project a declared angular tangent off overall translation and rotation.

    Coordinates and tangent share a Cartesian frame and atom order. The
    effective inertia is the squared mass-weighted tangent norm after
    rigid-motion removal. This partition is defined for nonlinear geometries.
    """
    coordinates, masses = _geometry_and_masses(coordinates_bohr, masses_me)
    tangent = _real_array(tangent_bohr_per_rad, "angular tangent")
    if tangent.shape != coordinates.shape:
        raise ValueError("the angular tangent must match the Cartesian geometry")
    center = coordinates - np.average(coordinates, axis=0, weights=masses)
    root_mass = np.sqrt(masses)[:, None, None]
    translations = (root_mass * np.broadcast_to(np.eye(3), (len(coordinates), 3, 3))).reshape(-1, 3)
    rotation_vectors = np.stack([np.cross(axis, center) for axis in np.eye(3)], axis=-1)
    rotations = (root_mass * rotation_vectors).reshape(-1, 3)
    rigid, singular, _ = np.linalg.svd(np.column_stack((translations, rotations)), full_matrices=False)
    if singular[-1] <= 1e-8 * singular[0]:
        raise ValueError("the angular partition requires six independent rigid motions")
    raw = (np.sqrt(masses)[:, None] * tangent).ravel()
    internal = raw - rigid @ (rigid.T @ raw)
    inertia = float(internal @ internal)
    if inertia <= 1e-16 * max(1., float(raw @ raw)):
        raise ValueError("the angular tangent has no independent internal motion")
    unit = internal / np.sqrt(inertia)
    frame, _ = np.linalg.qr(np.column_stack((rigid, unit)), mode="complete")
    return RotorProjection(rigid, unit, frame[:, 7:], inertia)


def project_spectator_modes(
    hessian_hartree_per_bohr2: ArrayLike, masses_me: ArrayLike, projection: RotorProjection,
) -> RotorSpectatorModes:
    """Project a stationary Hessian onto the angular and spectator subspaces.

    Either a Cartesian matrix or PySCF's atom, atom, xyz, xyz layout can be
    supplied. The conditional curvature includes local spectator relaxation
    through the Schur complement of the harmonic Hessian.
    """
    masses = _real_array(masses_me, "rotor masses in electron masses")
    if masses.ndim != 1 or np.any(masses <= 0):
        raise ValueError("provide positive masses in electron masses")
    natoms = len(masses)
    hessian = _real_array(hessian_hartree_per_bohr2, "Cartesian Hessian")
    if hessian.shape == (natoms, natoms, 3, 3):
        hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natoms, 3 * natoms)
    if hessian.shape != (3 * natoms, 3 * natoms) or len(projection.unit_torsion) != 3 * natoms:
        raise ValueError("Hessian, masses and rotor projection have inconsistent dimensions")
    mass_weights = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(mass_weights, mass_weights)
    if np.max(np.abs(weighted - weighted.T)) > 1e-9:
        raise ValueError("mass-weighted Hessian is not symmetric")
    weighted = 0.5 * (weighted + weighted.T)
    axes, torsion = projection.spectator_axes, projection.unit_torsion
    bath = axes.T @ weighted @ axes
    eigenvalues, eigenvectors = np.linalg.eigh(bath)
    vibrational = np.column_stack((torsion, axes))
    all_vibrational = np.linalg.eigvalsh(vibrational.T @ weighted @ vibrational)
    if min(eigenvalues[0], all_vibrational[0]) <= 0:
        raise ValueError("stationary geometry has nonpositive torsional or spectator curvature")
    torsional = float(torsion @ weighted @ torsion)
    coupling = axes.T @ weighted @ torsion
    conditional = torsional - float(coupling @ np.linalg.solve(bath, coupling))
    inertia = projection.inertia_me_bohr2
    # The weighted torsion vector has unit norm. Multiplying by the inertia
    # converts curvature along that vector to d²E/d(angle)².
    return RotorSpectatorModes(
        np.sqrt(eigenvalues), axes @ eigenvectors, torsional * inertia,
        conditional * inertia, 0.5 * float(np.sqrt(all_vibrational).sum()),
    )


def _bond_group(coordinates, axis_atoms, moving_atoms):
    indices = tuple(axis_atoms) + tuple(moving_atoms)
    if len(axis_atoms) != 2 or not moving_atoms or any(
        isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(coordinates)
        for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("provide a distinct pair of axis atoms and distinct off-axis moving atom indices")
    first, last = axis_atoms
    difference = coordinates[last] - coordinates[first]
    length = np.linalg.norm(difference)
    if length <= 1e-12:
        raise ValueError("the internal-rotation axis has zero length")
    return coordinates[first], difference / length, np.asarray(moving_atoms, dtype=np.intp)


def _axis_rotation(axis, angle):
    cross_columns = np.cross(axis, np.eye(3)).T
    cosine = np.cos(angle)
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + np.sin(angle) * cross_columns


def bond_rotation(
    coordinates_bohr: ArrayLike, axis_atoms: tuple[int, int], moving_atoms: tuple[int, ...], angle_rad: float,
) -> tuple[_Real, _Real]:
    """Return a rigid top rotation and its exact Cartesian angular tangent.

    Angles use the right-hand rule about the axis from the first atom to the
    second. Atom indices are zero-based. The top excludes the two pivot atoms.
    """
    geometry = _real_array(coordinates_bohr, "rotor geometry in bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3 or not np.isfinite(angle_rad):
        raise ValueError("provide Cartesian coordinates and a finite angle in radians")
    origin, axis, moving = _bond_group(geometry, tuple(axis_atoms), tuple(moving_atoms))
    rotated = geometry.copy()
    rotated[moving] = (geometry[moving] - origin) @ _axis_rotation(axis, angle_rad).T + origin
    tangent = np.zeros_like(geometry)
    tangent[moving] = np.cross(axis, rotated[moving] - origin)
    return rotated, tangent


def _orthogonal_part(matrix):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right, singular


def transport_bond_spectators(
    coordinates_bohr: ArrayLike, masses_me: ArrayLike, modes: ArrayLike, axis_atoms: tuple[int, int],
    moving_atoms: tuple[int, ...], angle_rad: float, *, steps: int = 64,
) -> tuple[_Real, float]:
    """Transport mass-weighted spectator vectors along a rigid top rotation.

    Successive rotations and orthogonal projections remove induced rigid and
    torsional motion. The returned minimum step singular value measures loss
    between the transported and local tangent spaces. Repeat with a smaller
    step to check convergence for the application.
    """
    xyz, masses = _geometry_and_masses(coordinates_bohr, masses_me)
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("transport steps must be a positive integer")
    modes = _real_array(modes, "mass-weighted spectator modes")
    size = xyz.size - 7
    if modes.shape != (xyz.size, size) or not np.allclose(modes.T @ modes, np.eye(size), atol=1e-8, rtol=0):
        raise ValueError("supply all orthonormal mass-weighted spectator mode columns")
    _, axis, moving = _bond_group(xyz, tuple(axis_atoms), tuple(moving_atoms))
    initial_xyz, initial_tangent = bond_rotation(xyz, axis_atoms, moving_atoms, 0.)
    initial_space = project_torsion(initial_xyz, masses, initial_tangent).spectator_axes
    if not np.allclose(modes, initial_space @ (initial_space.T @ modes), rtol=0, atol=1e-8):
        raise ValueError("input modes do not span the initial spectator tangent space")
    increment = angle_rad / steps
    block = _axis_rotation(axis, increment)
    minimum_overlap = 1.
    for total in np.linspace(increment, angle_rad, steps):
        rotated, tangent = bond_rotation(xyz, axis_atoms, moving_atoms, total)
        space = project_torsion(rotated, masses, tangent).spectator_axes
        trial = modes.reshape(len(xyz), 3, size).copy()
        trial[moving] = np.einsum("ij,ajk->aik", block, trial[moving])
        # The polar factor gives the closest orthonormal frame in the new
        # spectator space. The projected frame's singular values measure
        # overlap before reorthogonalization.
        polar, singular = _orthogonal_part(space.T @ trial.reshape(xyz.size, size))
        minimum_overlap = min(minimum_overlap, float(singular[-1]))
        modes = space @ polar
    return modes, minimum_overlap


def map_transported_spectators(
    initial_modes: ArrayLike, final_modes: ArrayLike, common: RotorProjection,
    initial_minus_final_bohr: ArrayLike, masses_me: ArrayLike,
) -> RotorSpectatorMap:
    """Make an affine spectator map in a declared common angular frame.

    Supply both mode sets transported to the chosen angle, and the difference
    of the geometries at that angle. The returned singular values diagnose
    how much of each mode subspace the common projection retains.
    """
    axes = common.spectator_axes
    initial = _real_array(initial_modes, "transported initial spectator modes")
    final = _real_array(final_modes, "transported final spectator modes")
    delta, masses = _geometry_and_masses(initial_minus_final_bohr, masses_me)
    if initial.shape != axes.shape or final.shape != axes.shape or delta.size != axes.shape[0]:
        raise ValueError("the common projection, displaced geometry and transported modes must agree")
    first, first_singular = _orthogonal_part(axes.T @ initial)
    last, last_singular = _orthogonal_part(axes.T @ final)
    if min(first_singular[-1], last_singular[-1]) <= 1e-8:
        raise ValueError("the common reference loses an independent spectator direction")
    displacement = (np.sqrt(masses)[:, None] * delta).ravel()
    return RotorSpectatorMap(last.T @ first, last.T @ (axes.T @ displacement), first_singular, last_singular)
