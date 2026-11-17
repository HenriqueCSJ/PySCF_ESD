"""Electronic and vibronic phosphorescence from prepared spin–orbit matrices."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic import HarmonicSpectrum, _KB_HARTREE_PER_K, _real_array, harmonic_spectrum
from .pyscf_adapter import AdiabaticHessianParameters
from .spectral_properties import _EINSTEIN_A_FACTOR, _lifetime_s, summarize_spectral_properties
from .vibronic_properties import condon_ht_properties
from .workflow import DEFAULT_MAX_QUANTA, WorkflowIncomplete, _adaptive_spectrum, _validate_spectral_options


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
    degeneracy_tolerance_hartree: float = 1e-10


def _hermitian(array: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.complex128]:
    matrix = np.asarray(array, dtype=np.complex128)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    adjoint = matrix.swapaxes(0, 1).conj()
    if not np.allclose(matrix, adjoint, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be Hermitian in its two electronic state indices")
    return 0.5 * matrix + 0.5 * adjoint


def _triplet_populations(splittings, temperature_kelvin, degeneracy_tolerance_hartree):
    thermal_energy = _KB_HARTREE_PER_K * temperature_kelvin
    if thermal_energy == 0:
        weights = (splittings <= degeneracy_tolerance_hartree).astype(float)
    else:
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(-splittings / thermal_energy)
    return weights / weights.sum()


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
        population = _triplet_populations(splittings, temperature_kelvin, degeneracy_tolerance_hartree)
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
        degeneracy_tolerance_hartree=float(degeneracy_tolerance_hartree),
    )


def spin_orbit_dipole_derivatives(
    electronic: PhosphorescenceResult,
    dipole_operator_au: ArrayLike,
    hamiltonian_derivatives: ArrayLike,
    dipole_operator_derivatives: ArrayLike,
) -> NDArray[np.complex128]:
    """Differentiate the prepared mixed emission moments at one geometry.

    Derivatives of the complete Hamiltonian and dipole matrices have shapes
    (state, state, coordinate) and (state, state, coordinate, dipole Cartesian).
    They are derivatives of the matrix entries in one smoothly transported
    orthonormal spin-free basis. Include all basis and orbital response in
    those inputs. Units are Hartree and e·bohr per supplied coordinate.
    The result has axes (emitting sublevel, coordinate, dipole Cartesian).
    Each selected state must be individually isolated. The phases use
    parallel transport in the supplied moving frame.
    """
    if not isinstance(electronic, PhosphorescenceResult):
        raise TypeError("supply a PhosphorescenceResult from spin_orbit_phosphorescence")
    energies, vectors = electronic.mixed_energies_from_ground_hartree, electronic.mixed_state_vectors
    n = len(energies)
    hprime = np.asarray(hamiltonian_derivatives)
    if hprime.ndim != 3 or hprime.shape[:2] != (n, n) or hprime.shape[-1] == 0:
        raise ValueError("Hamiltonian derivatives must have shape (state, state, coordinate)")
    coordinates = hprime.shape[-1]
    hprime = _hermitian(hprime, (n, n, coordinates), "Hamiltonian derivatives")
    dprime = _hermitian(dipole_operator_derivatives, (n, n, coordinates, 3), "electric dipole derivatives")
    dipole = _hermitian(dipole_operator_au, (n, n, 3), "electric dipole")
    indices = np.r_[0, electronic.triplet_state_indices]
    selected = vectors[:, indices]
    reconstructed = np.einsum("a,abc,bj->jc", selected[:, 0].conj(), dipole, selected[:, 1:], optimize=True)
    if not np.allclose(reconstructed, electronic.transition_dipole_au, rtol=1e-9, atol=1e-12):
        raise ValueError("dipole operator and spin–orbit result disagree at the expansion geometry")

    gaps = energies[indices][None, :] - energies[:, None]
    off_diagonal = np.arange(n)[:, None] != indices[None, :]
    if np.any(np.abs(gaps[off_diagonal]) <= electronic.degeneracy_tolerance_hartree):
        raise ValueError("individual mixed-state dipole derivatives require isolated electronic sublevels")
    perturbation = np.einsum("ai,abq,bj->ijq", vectors.conj(), hprime, selected, optimize=True)
    response = np.divide(perturbation, gaps[:, :, None], out=np.zeros_like(perturbation),
                         where=off_diagonal[:, :, None])
    derivative = np.einsum("ai,ijq->ajq", vectors, response, optimize=True)
    g, t = selected[:, 0].conj(), selected[:, 1:]
    moment_derivative = (
        np.einsum("aq,abd,bj->jqd", derivative[:, 0].conj(), dipole, t, optimize=True)
        + np.einsum("a,abqd,bj->jqd", g, dprime, t, optimize=True)
        + np.einsum("a,abd,bjq->jqd", g, dipole, derivative[:, 1:], optimize=True)
    )
    if not np.isfinite(moment_derivative).all():
        raise ValueError("mixed-state dipole derivatives exceed the numerical range")
    return moment_derivative


@dataclass(frozen=True)
class VibronicPhosphorescenceResult:
    """Prepared AH emission with separately assigned triplet sublevel spectra."""

    temperature_kelvin: float
    sublevel_populations: NDArray[np.float64]
    sublevel_00_hartree: NDArray[np.float64]
    sublevel_rates_resolved: NDArray[np.bool_]
    sublevel_spectra: tuple[HarmonicSpectrum, ...]
    basis_attempts: tuple[list[dict], ...]

    @property
    def physical_properties(self) -> dict:
        """Report intrinsic sublevel results and the population-weighted rate."""
        sublevels = []
        total_rate = 0.0
        for index, (population, origin, resolved, spectrum) in enumerate(zip(
            self.sublevel_populations, self.sublevel_00_hartree, self.sublevel_rates_resolved,
            self.sublevel_spectra, strict=True,
        )):
            summary = summarize_spectral_properties(
                {"emission": spectrum}, spectral_status={"emission": "complete"},
                temperature_kelvin=self.temperature_kelvin, assignment_scheme="harmonic_quanta",
            )["spectra"]["emission"]
            channel_rate = float(population) * summary["vacuum_radiative_rate_s_inverse"]
            total_rate += channel_rate
            summary.update(
                sublevel_index=index, sublevel_population=float(population), origin_00_hartree=float(origin),
                rate_resolved=bool(resolved), population_weighted_vacuum_rate_s_inverse=channel_rate,
            )
            decomposition = condon_ht_properties(spectrum, "emission")
            if decomposition is not None:
                summary["condon_herzberg_teller"] = decomposition
            sublevels.append(summary)
        return {
            "model_used": "prepared_phosphorescence_ah",
            "temperature_kelvin": self.temperature_kelvin,
            "sublevels": sublevels,
            "vacuum_radiative_rate_s_inverse": total_rate,
            "vacuum_radiative_lifetime_s": _lifetime_s(total_rate),
        }


def run_prepared_phosphorescence(
    parameters: AdiabaticHessianParameters,
    electronic: PhosphorescenceResult,
    *,
    lowest_sublevel_00_hartree: float,
    dipole_derivatives_au: ArrayLike | None = None,
    temperature_kelvin: float = 0.0,
    coverage: float = 0.995,
    max_quanta: int = DEFAULT_MAX_QUANTA,
    max_overlaps: int = 250_000,
) -> VibronicPhosphorescenceResult:
    """Use a shared triplet harmonic surface and prepared SOC moments for emission.

    The moments must be evaluated at the triplet minimum, in the same
    Cartesian frame as the harmonic model. Supply the adiabatic 0–0 origin of
    the lowest spin sublevel, with zero-point energies, on an independent
    electronic energy reference. The vertical SOC gap is not used as a 0–0
    origin. Optional complex HT slopes have axes (sublevel, T normal mode,
    dipole Cartesian). Match the ordering and phases of the electronic
    moments and the order and signs of the prepared T normal modes.
    All sublevels share the same geometry and curvature and differ by
    constant spin splittings. Sublevel and vibrational populations are each
    Boltzmann-distributed at the requested temperature in the emitting state.
    """
    _validate_spectral_options("emission", temperature_kelvin, coverage, max_quanta, max_overlaps)
    if not isinstance(parameters, AdiabaticHessianParameters) or not isinstance(electronic, PhosphorescenceResult):
        raise TypeError("supply prepared AH parameters and a PhosphorescenceResult")
    if (isinstance(lowest_sublevel_00_hartree, bool) or not isinstance(lowest_sublevel_00_hartree, Real)
            or not isfinite(lowest_sublevel_00_hartree) or lowest_sublevel_00_hartree <= 0):
        raise ValueError("supply a finite, positive adiabatic 0–0 origin in Hartree")
    slopes = None
    if dipole_derivatives_au is not None:
        slopes = np.asarray(dipole_derivatives_au, dtype=np.complex128)
        if slopes.shape != (3, parameters.excited_frequencies_hartree.size, 3) or not np.isfinite(slopes).all():
            raise ValueError("supply finite complex HT slopes with shape (sublevel, T normal mode, 3)")

    rotation = parameters.rotation.T
    shift = -(rotation @ parameters.shift_au)
    origin = float(lowest_sublevel_00_hartree) + electronic.sublevel_splittings_hartree
    population = _triplet_populations(electronic.sublevel_splittings_hartree, temperature_kelvin,
                                      electronic.degeneracy_tolerance_hartree)
    spectra, checks = [], []
    for i, gap in enumerate(origin):
        def spectrum(**options):
            return harmonic_spectrum(
                parameters.excited_frequencies_hartree, parameters.ground_frequencies_hartree,
                gap_00_hartree=-gap, rotation=rotation, shift_au=shift,
                transition_dipole_au=electronic.transition_dipole_au[i],
                dipole_derivatives_au=None if slopes is None else slopes[i], **options,
            )
        try:
            sticks, attempts = _adaptive_spectrum(
                spectrum, process="emission", temperature=temperature_kelvin, coverage=coverage,
                max_quanta=max_quanta, max_overlaps=max_overlaps,
            )
        except WorkflowIncomplete as error:
            raise WorkflowIncomplete(
                f"phosphorescence sublevel {i + 1}: {error}",
                {**error.diagnostics, "sublevel_index": i}, error.partial_spectra,
            ) from error
        spectra.append(sticks)
        checks.append(attempts)
    return VibronicPhosphorescenceResult(
        float(temperature_kelvin), population, origin, electronic.sublevel_rates_resolved.copy(),
        tuple(spectra), tuple(checks),
    )
