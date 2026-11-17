"""Electronic phosphorescence from a prepared spin–orbit state interaction."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic import _KB_HARTREE_PER_K, _real_array
from .spectral_properties import _EINSTEIN_A_FACTOR, _lifetime_s


@dataclass(frozen=True)
class PhosphorescenceResult:
    """Electronic line strengths and vacuum rates at a single geometry.

    Mixed state vectors are columns in the supplied spin-free basis. Sublevel
    arrays follow increasing mixed energy. An ensemble rate requires the
    caller to request Boltzmann populations by supplying a temperature.
    """

    mixed_energies_from_ground_hartree: NDArray[np.float64]
    mixed_state_vectors: NDArray[np.complex128]
    ground_character: float
    triplet_state_indices: NDArray[np.intp]
    triplet_character: NDArray[np.float64]
    triplet_subspace_minimum_overlap_squared: float
    photon_energy_hartree: NDArray[np.float64]
    sublevel_splittings_hartree: NDArray[np.float64]
    sublevel_rates_resolved: NDArray[np.bool_]
    transition_dipole_au: NDArray[np.complex128]
    dipole_strength_au2: NDArray[np.float64]
    sublevel_vacuum_rates_s_inverse: NDArray[np.float64]
    sublevel_radiative_lifetimes_s: tuple[float | None, ...]
    temperature_kelvin: float | None
    sublevel_populations: NDArray[np.float64] | None
    vacuum_radiative_rate_s_inverse: float | None
    vacuum_radiative_lifetime_s: float | None


def _hermitian(array: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.complex128]:
    matrix = np.asarray(array, dtype=np.complex128)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    adjoint = matrix.swapaxes(0, 1).conj()
    if not np.allclose(matrix, adjoint, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be Hermitian in its two electronic state indices")
    return 0.5 * matrix + 0.5 * adjoint


def spin_orbit_phosphorescence(
    spin_free_energies_hartree: ArrayLike,
    spin_orbit_hartree: ArrayLike,
    dipole_operator_au: ArrayLike,
    *,
    ground_index: int,
    triplet_indices: tuple[int, int, int],
    temperature_kelvin: float | None = None,
    degeneracy_tolerance_hartree: float = 1e-10,
) -> PhosphorescenceResult:
    """Mix a singlet/triplet basis and evaluate electric-dipole T₁ → S₀ rates.

    Supply spin-free energies on one zero, a Hermitian spin-dependent
    Hamiltonian contribution (Hartree), and Cartesian length-gauge dipole
    matrices (e·bohr) in the same orthonormal electronic basis, geometry,
    frame and phases. Include permanent and excited-to-excited dipoles when
    present. The labeled ground state must remain the isolated lowest
    eigenstate. The target three-dimensional triplet subspace must retain
    majority overlap in each principal direction and be isolated from the
    omitted eigenstates. The solver returns the electronic rates at the
    supplied geometry. An input temperature gives a rate conditional on
    thermal equilibration among the three emitting sublevels.
    """
    energies = _real_array(spin_free_energies_hartree, "spin-free energies")
    if energies.ndim != 1 or energies.size < 4:
        raise ValueError("supply at least four finite real spin-free energies")
    triplet = tuple(triplet_indices)
    labels = (ground_index, *triplet)
    if (len(triplet) != 3 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                                or not 0 <= i < energies.size for i in labels)
            or len(set(labels)) != 4):
        raise ValueError("supply a ground label and three distinct triplet labels as zero-based basis indices")
    if (isinstance(degeneracy_tolerance_hartree, bool)
            or not isinstance(degeneracy_tolerance_hartree, Real)
            or not isfinite(degeneracy_tolerance_hartree) or degeneracy_tolerance_hartree <= 0):
        raise ValueError("degeneracy tolerance must be finite and positive")
    if temperature_kelvin is not None and (isinstance(temperature_kelvin, bool)
            or not isinstance(temperature_kelvin, Real) or not isfinite(temperature_kelvin)
            or temperature_kelvin < 0):
        raise ValueError("temperature must be finite and nonnegative")
    if np.ptp(energies[list(triplet)]) > degeneracy_tolerance_hartree:
        raise ValueError("the three spin-free components of the target triplet must share one energy")

    n = energies.size
    hamiltonian = _hermitian(spin_orbit_hartree, (n, n), "spin-dependent Hamiltonian")
    hamiltonian[np.diag_indices(n)] += energies - energies[ground_index]
    dipole = _hermitian(dipole_operator_au, (n, n, 3), "electric dipole")
    triplet_dipole = dipole[np.ix_(triplet, triplet)]
    if (not np.allclose(dipole[ground_index, list(triplet)], 0., atol=1e-12, rtol=0)
            or not np.allclose(triplet_dipole, np.eye(3)[:, :, None] * triplet_dipole[0, 0],
                               atol=1e-12, rtol=0)):
        raise ValueError("the supplied ground and triplet dipoles must be spin-free")
    mixed, vectors = np.linalg.eigh(hamiltonian)
    mixed -= mixed[0]
    ground_character = float(abs(vectors[ground_index, 0])**2)
    if ground_character <= 0.5 or mixed[1] <= degeneracy_tolerance_hartree:
        raise ValueError("the labeled singlet no longer defines an isolated mixed ground state")

    projector = vectors[np.asarray(triplet), :]
    character = np.sum(np.abs(projector)**2, axis=0)
    states = np.sort(np.argsort(-character[1:], kind="stable")[:3] + 1)
    overlaps = np.linalg.svd(projector[:, states], compute_uv=False)
    weakest = float(overlaps[-1]**2)
    excluded = np.setdiff1d(np.arange(n), states)
    if (weakest <= 0.5 or np.any(np.abs(mixed[states, None] - mixed[excluded]) <= degeneracy_tolerance_hartree)):
        raise ValueError("the labeled triplet is not an isolated, predominantly triplet subspace")

    photon_energy = mixed[states]
    transition = np.einsum("i,ijc,jk->kc", vectors[:, 0].conj(), dipole, vectors[:, states], optimize=True)
    strengths = np.sum(np.abs(transition)**2, axis=1)
    rates = _EINSTEIN_A_FACTOR * photon_energy**3 * strengths
    if not np.isfinite(rates).all():
        raise ValueError("the electronic emission rates exceed the numerical range")
    splittings = photon_energy - photon_energy[0]
    resolved = np.sum(np.abs(splittings[:, None] - splittings) <= degeneracy_tolerance_hartree, axis=1) == 1
    population = average_rate = None
    if temperature_kelvin is not None:
        temperature_kelvin = float(temperature_kelvin)
        thermal_energy = _KB_HARTREE_PER_K * temperature_kelvin
        if thermal_energy == 0:
            population = (splittings <= degeneracy_tolerance_hartree).astype(float)
        else:
            with np.errstate(over="ignore", under="ignore"):
                population = np.exp(-splittings / thermal_energy)
        population /= population.sum()
        average_rate = float(np.dot(population, rates))

    return PhosphorescenceResult(
        mixed_energies_from_ground_hartree=mixed, mixed_state_vectors=vectors,
        ground_character=ground_character, triplet_state_indices=states, triplet_character=character[states],
        triplet_subspace_minimum_overlap_squared=weakest, photon_energy_hartree=photon_energy,
        sublevel_splittings_hartree=splittings, sublevel_rates_resolved=resolved, transition_dipole_au=transition,
        dipole_strength_au2=strengths, sublevel_vacuum_rates_s_inverse=rates,
        sublevel_radiative_lifetimes_s=tuple(_lifetime_s(rate) for rate in rates),
        temperature_kelvin=temperature_kelvin, sublevel_populations=population,
        vacuum_radiative_rate_s_inverse=average_rate,
        vacuum_radiative_lifetime_s=None if average_rate is None else _lifetime_s(average_rate),
    )
