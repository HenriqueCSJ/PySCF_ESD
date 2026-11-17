"""Automatic stationary-state electronic inputs for the CPU spectroscopy workflow."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cis_hessian import analytic_cis_hessian, analytic_cis_transition_dipole
from .cpu import (
    ElectronicConvergenceError, ElectronicPoint, OrbitalSeed, ReferenceStabilityUnresolvedError, Settings,
    UnstableReferenceError,
    _checked_array, _scf_residual, _solve_reference, build_molecule, check_restricted_reference, solve_point,
)
from .pyscf_adapter import AdiabaticHessianParameters, adiabatic_hessian
from .state_tracking import StateTrackingError, orbital_overlap_blocks, track_state
from .vibronic_properties import electronic_relaxation_properties, harmonic_model_properties


_Real = NDArray[np.float64]
_STATIONARY_TOLERANCE = 1e-5  # Maximum per-atom gradient, Hartree/bohr
_ROOT_GAP = 1e-4  # Hartree
_SUBSPACE_LOSS_LIMIT = 0.10
_CACHE_SIZE = 64  # Native PySCF objects can retain large integral and grid arrays
_DIPOLE_ABSOLUTE_TOLERANCE_AU = 2e-6
_DIPOLE_RELATIVE_TOLERANCE = 0.065


@dataclass(frozen=True)
class ElectronicPreparation:
    """Stationary inputs and a state- and phase-tracked electronic evaluator.

    Coordinates and dipole components use the common ground-minimum Cartesian
    frame. Energies returned by `evaluate` are absolute total energies in
    Hartree. The dipole derivative rows use the signs and order of the ground
    normal coordinates in `model`.
    """

    ground_coordinates_bohr: _Real
    excited_coordinates_bohr: _Real
    model: AdiabaticHessianParameters
    transition_dipole_au: _Real
    dipole_derivatives_au: _Real | None
    evaluate: Callable[[ArrayLike], tuple[float, float, _Real]] = field(repr=False)
    diagnostics: dict
    ground_energy_hartree: float
    excited_energy_hartree: float
    alternate_excited_coordinates_bohr: tuple[_Real, ...]
    alternate_ground_coordinates_bohr: tuple[_Real, ...]


@dataclass
class _GroundPoint:
    coordinates: _Real
    seed: OrbitalSeed
    energy: float
    gradient: _Real
    mean_field: object


@dataclass
class _TrackedPoint:
    coordinates: _Real
    point: ElectronicPoint
    root: int
    phase: int
    gradient: _Real | None = None
    curvature_diagnostics: dict = field(default_factory=dict)

    @property
    def energy(self) -> float:
        return float(self.point.ground_energy_hartree + self.point.excitation_energies_hartree[self.root])

    @property
    def dipole(self) -> _Real:
        return self.phase * self.point.transition_dipoles_au[self.root]

    def total_gradient(self) -> _Real:
        if self.gradient is None:
            self.gradient = self.point.excited_gradient(self.root)
        return self.gradient


def _maximum_motion(vector: _Real) -> float:
    return float(np.max(np.linalg.norm(vector.reshape(-1, 3), axis=1)))


def _center(coordinates: _Real, masses: _Real) -> _Real:
    return coordinates - np.average(coordinates, axis=0, weights=masses)


def _align(coordinates: _Real, reference: _Real, masses: _Real) -> _Real:
    centered = _center(coordinates, masses)
    origin = np.average(reference, axis=0, weights=masses)
    left, _, right = np.linalg.svd(centered.T @ (masses[:, None] * (reference - origin)))
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return centered @ (left @ right) + origin


def _cartesian_frame(coordinates: _Real) -> _Real:
    """Columns span the vibrational tangent space in Cartesian coordinates."""
    centered = coordinates - coordinates.mean(axis=0)
    rotations = np.cross(np.eye(3), centered[:, None, :]).transpose(0, 2, 1)
    axes, values, _ = np.linalg.svd(rotations.reshape(-1, 3), full_matrices=False)
    rank = int(np.count_nonzero(values > 1e-10 * values[0]))
    if rank < 2:
        raise ValueError("the molecular geometry has undefined rotations")
    translations = np.broadcast_to(np.eye(3), (len(centered), 3, 3)).reshape(-1, 3)
    basis, _ = np.linalg.qr(np.column_stack((translations, axes[:, :rank])), mode="complete")
    return basis[:, 3 + rank:]


def _linear_candidate(coordinates: _Real, masses: _Real) -> _Real | None:
    if len(coordinates) <= 2:
        return None
    origin = np.average(coordinates, weights=masses, axis=0)
    centered = coordinates - origin
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0]
    linear = (centered @ axis)[:, None] * axis
    residual = _maximum_motion(linear - centered)
    if 0 < residual < 2e-6:
        return linear + origin
    return None


def _matrix_hessian(hessian: _Real) -> _Real:
    return hessian.transpose(0, 2, 1, 3).reshape(hessian.shape[0] * 3, -1)


def _atomic_hessian(hessian: _Real) -> _Real:
    natm = hessian.shape[0] // 3
    return hessian.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)


def _symmetric_analytic_hessian(hessian: _Real, state: str) -> _Real:
    matrix = _matrix_hessian(hessian)
    error = np.max(np.abs(matrix - matrix.T))
    if error > 1e-5:
        raise ElectronicConvergenceError(f"{state} analytic Hessian is asymmetric by {error:.3g} Hartree/bohr²")
    return _atomic_hessian((matrix + matrix.T) * 0.5)


class _ElectronicSession:
    cache_size = _CACHE_SIZE
    quadrature_grid_response = False
    ground_hessian_route = "analytic_pyscf_rks"

    def __init__(self, symbols: Sequence[str], coordinates: ArrayLike, settings: Settings):
        from pyscf import dft
        from pyscf.scf.dispersion import parse_dft

        if isinstance(symbols, str):
            raise ValueError("symbols must contain one element symbol per atom")
        self.symbols = tuple(symbols)
        if len(self.symbols) < 2:
            raise ValueError("vibronic spectroscopy requires a molecule with at least two atoms")
        molecule = build_molecule(self.symbols, coordinates, settings)
        self.masses = molecule.atom_mass_list(isotope_avg=True)
        self.initial_coordinates = np.asarray(molecule.atom_coords(), dtype=float)
        occupied = molecule.nelectron // 2
        self.root_dimension = occupied * (molecule.nao_nr() - occupied)
        self.root_search_limit = self.root_dimension
        xc, _, dispersion = parse_dft(settings.xc)
        self.pure_hf = (
            dispersion is None and dft.libxc.xc_type(xc) == "HF"
            and dft.libxc.rsh_coeff(xc) == (0, 1, 0)
        )
        self.settings = settings
        if self.pure_hf:
            self.settings = replace(
                settings,
                scf_conv_tol=min(settings.scf_conv_tol, 1e-13),
                scf_residual_tol=min(settings.scf_residual_tol, 0.2 * settings.cphf_conv_tol),
                direct_scf_tol=min(settings.direct_scf_tol, 1e-13),
            )
        self.ground_cache: OrderedDict[bytes, _GroundPoint] = OrderedDict()
        self.tracked_cache: OrderedDict[bytes, _TrackedPoint] = OrderedDict()
        self.anchors: list[_TrackedPoint] = []
        self.ground_solves = 0
        self.tracked_solves = 0
        self.ground_cache_hits = 0
        self.tracked_cache_hits = 0
        self.overlaps: list[float] = []
        self.gaps: list[float] = []
        self.visited_roots: set[int] = set()

    def _remember(self, cache: OrderedDict, point):
        cache[point.coordinates.tobytes()] = point
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return point

    def ground(self, coordinates: _Real, previous: _GroundPoint | None = None) -> _GroundPoint:
        key = coordinates.tobytes()
        known = self.ground_cache.get(key)
        if known is not None:
            self.ground_cache.move_to_end(key)
            self.ground_cache_hits += 1
            return known
        molecule = build_molecule(self.symbols, coordinates, self.settings)
        reference, seed = self._ground_reference(molecule, None if previous is None else previous.seed)
        force = self._ground_gradient(reference, coordinates.shape)
        self.ground_solves += 1
        return self._remember(self.ground_cache, _GroundPoint(coordinates, seed, float(reference.e_tot), force, reference))

    def _ground_reference(self, molecule, previous):
        reference, _, _ = _solve_reference(molecule, self.settings, previous)
        return reference, OrbitalSeed(molecule, reference.mo_coeff, molecule.nelectron // 2, self.settings)

    def _ground_gradient(self, reference, shape):
        gradient = reference.nuc_grad_method()
        gradient.grid_response = True
        return _checked_array(gradient.kernel(), shape, "ground-state gradient")

    def ground_hessian(self, point: _GroundPoint) -> _Real:
        solver = point.mean_field.Hessian()
        solver.grid_response = False  # Native CPU RKS Hessian has no grid response
        solver.max_cycle = self.settings.cphf_max_cycle
        size = len(point.coordinates)
        values = _checked_array(solver.kernel(), (size, size, 3, 3), "ground-state Hessian")
        return _symmetric_analytic_hessian(values, "S0")

    def _calculate(self, coordinates: _Real, count: int, seed: OrbitalSeed | ElectronicPoint) -> ElectronicPoint:
        settings = replace(self.settings, nstates=count)
        projected = OrbitalSeed(seed.mol, seed.mo_coeff, seed.nocc, settings)
        self.tracked_solves += 1
        return solve_point(self.symbols, coordinates, settings, previous=projected)

    def start_tracking(self, ground: _GroundPoint, root: int) -> _TrackedPoint:
        self.root_dimension = ground.seed.nocc * (ground.seed.mo_coeff.shape[1] - ground.seed.nocc)
        if not 0 <= root < self.root_dimension:
            raise ValueError(f"target index {root} is outside this basis's {self.root_dimension} singlet TDA pairs")
        requested = min(self.root_dimension, max(root + 2, self.settings.nstates))
        self.root_search_limit = min(self.root_dimension, max(requested + 12, 2 * requested))
        point = self._calculate(ground.coordinates, requested, ground.seed)
        gaps = np.abs(np.delete(point.excitation_energies_hartree, root) - point.excitation_energies_hartree[root])
        if gaps.size and float(gaps.min()) < _ROOT_GAP:
            raise StateTrackingError(
                f"S0 minimum: target root {root + 1} is separated by only {gaps.min():.4g} Hartree. "
                "An isolated-state model cannot assign a unique phase to this manifold."
            )
        if gaps.size:
            self.gaps.append(float(gaps.min()))
        self.visited_roots.add(root)
        tracked = _TrackedPoint(ground.coordinates, point, root, 1)
        self.anchors.append(tracked)
        return self._remember(self.tracked_cache, tracked)

    @staticmethod
    def _match(previous: _TrackedPoint, candidate: ElectronicPoint):
        """Match a new eigenvector against the phase carried by the previous point."""
        from pyscf import gto

        reference = previous.point
        cross = gto.intor_cross("int1e_ovlp", reference.mol, candidate.mol)
        occupied, virtual = orbital_overlap_blocks(
            reference.mo_coeff, candidate.mo_coeff, cross,
            2 * (np.arange(reference.mo_coeff.shape[1]) < reference.nocc),
            2 * (np.arange(candidate.mo_coeff.shape[1]) < candidate.nocc),
        )
        multiple = len(candidate.excitation_energies_hartree) > 1
        return track_state(
            previous.phase * reference.amplitudes[previous.root],
            candidate.amplitudes, occupied, virtual,
            candidate_energies_hartree=candidate.excitation_energies_hartree,
            minimum_margin=0.1 if multiple else 0,
            minimum_gap_hartree=_ROOT_GAP if multiple else 0,
        )

    def isolated_point(self, coordinates: _Real, previous: _TrackedPoint) -> _TrackedPoint:
        return self.tracked(coordinates, previous, preserve_rank=True)

    def tracked(
        self, coordinates: _Real, previous: _TrackedPoint | None = None, *, preserve_rank: bool = False,
    ) -> _TrackedPoint:
        key = coordinates.tobytes()
        known = self.tracked_cache.get(key)
        if known is None:
            known = next((anchor for anchor in self.anchors if np.array_equal(anchor.coordinates, coordinates)), None)
        if known is not None:
            if previous is not None and previous is not known:
                match = self._match(previous, known.point)
                if preserve_rank and match.index != previous.root:
                    raise StateTrackingError("trial geometry changes the energy rank of the isolated target state")
                if (match.index, match.phase) != (known.root, known.phase):
                    raise StateTrackingError(
                        "independent paths to a cached geometry disagree on the target state or its phase"
                    )
            self.tracked_cache_hits += 1
            if key in self.tracked_cache:
                self.tracked_cache.move_to_end(key)
                return known
            return self._remember(self.tracked_cache, known)
        if previous is None:
            candidates = tuple(self.tracked_cache.values()) + tuple(self.anchors)
            previous = min(candidates, key=lambda saved: float(np.sum((coordinates - saved.coordinates) ** 2)))
        reference = previous.point
        count = len(reference.excitation_energies_hartree)
        while True:
            candidate = self._calculate(coordinates, count, reference)
            try:
                match = self._match(previous, candidate)
            except StateTrackingError as error:
                if count == self.root_search_limit or "insufficient overlap" not in str(error):
                    raise
            else:
                # A continuous isolated adiabatic eigenvalue keeps its
                # energy rank. Endpoint overlap alone can jump branches.
                if preserve_rank and match.index != previous.root:
                    raise StateTrackingError("trial geometry changes the energy rank of the isolated target state")
                # A match at the last computed root leaves the gap to the
                # next higher root untested unless the full space was solved.
                if match.index < count - 1 or count == self.root_dimension:
                    break
                if count == self.root_search_limit:
                    raise StateTrackingError(
                        "the target reached the upper end of the computed root window. "
                        "Increase the requested root count to establish whether the state is isolated."
                    )
            count = min(self.root_search_limit, max(count + 2, 2 * count))
        self.overlaps.append(match.overlap)
        if match.gap_hartree is not None:
            self.gaps.append(match.gap_hartree)
        self.visited_roots.add(match.index)
        point = _TrackedPoint(coordinates, candidate, match.index, match.phase)
        return self._remember(self.tracked_cache, point)

    def evaluate(self, coordinates: ArrayLike) -> tuple[float, float, _Real]:
        values = _checked_array(coordinates, self.initial_coordinates.shape, "geometry in bohr").copy()
        candidates = tuple(self.tracked_cache.values()) + tuple(self.anchors)
        previous = min(candidates, key=lambda saved: float(np.sum((values - saved.coordinates) ** 2)))
        origin = previous.coordinates
        # Transport the target through nearby geometries when a scan jumps
        # farther than one tracking step from the nearest saved point.
        segments = max(1, int(np.ceil(_maximum_motion(values - origin) / 0.20)))
        for fraction in np.arange(1, segments, dtype=float) / segments:
            previous = self.isolated_point(origin + fraction * (values - origin), previous)
        point = self.isolated_point(values, previous)
        return point.point.ground_energy_hartree, point.energy, point.dipole

    def ground_stability(self, ground: _GroundPoint) -> None:
        check_restricted_reference(ground.mean_field, "Reference at an S0 stationary geometry")

    def excited_stability(self, excited: _TrackedPoint) -> None:
        check_restricted_reference(excited.point._mean_field, "Reference at a tracked S1 stationary geometry")

    def _conventional_rhf(self, point: _TrackedPoint):
        native = point.point._mean_field
        hf = native.to_hf()
        # to_hf clears convergence when changing class. The pure-HF
        # Hamiltonian here is unchanged, and we check its native stationarity.
        residual = _scf_residual(hf)
        if residual > self.settings.cphf_conv_tol:
            raise ElectronicConvergenceError(
                f"native RHF occupied–virtual residual {residual:.3g} exceeds the CIS response tolerance"
            )
        hf.converged = True
        return hf

    def excited_hessian(self, point: _TrackedPoint) -> _Real:
        reason = None
        if self.pure_hf:
            try:
                result = analytic_cis_hessian(
                    self._conventional_rhf(point), point.point.amplitudes[point.root],
                    root_gap_tolerance=_ROOT_GAP, response_tolerance=self.settings.cphf_conv_tol,
                )
            except MemoryError as error:
                reason = str(error)
            except ValueError as error:
                raise ElectronicConvergenceError(f"analytic CIS Hessian could not be validated: {error}") from error
            else:
                gradient_error = _maximum_motion(result.gradient - point.total_gradient())
                if abs(result.total_energy_hartree - point.energy) > 1e-8 or gradient_error > 2e-7:
                    raise ElectronicConvergenceError("native HF/TDA and analytic CIS disagree on the tracked state")
                point.curvature_diagnostics = {
                    "route": "analytic_rhf_cis",
                    "max_response_residual": result.max_response_residual,
                    "native_gradient_disagreement_hartree_per_bohr": gradient_error,
                }
                return _symmetric_analytic_hessian(result.hessian, "S1")
        hessian, diagnostics = _estimate_curvature(self, point)
        if reason is not None:
            diagnostics["analytic_unavailable"] = reason
        point.curvature_diagnostics = diagnostics
        return hessian

    def dipole_derivatives(self, point: _TrackedPoint, modes: _Real) -> tuple[_Real, dict]:
        reason = None
        if self.pure_hf:
            try:
                result = analytic_cis_transition_dipole(
                    self._conventional_rhf(point), point.point.amplitudes[point.root],
                    root_gap_tolerance=_ROOT_GAP, response_tolerance=self.settings.cphf_conv_tol,
                )
            except MemoryError as error:
                reason = str(error)
            except ValueError as error:
                raise ElectronicConvergenceError(f"analytic CIS dipole derivative could not be validated: {error}") from error
            else:
                if not np.allclose(point.phase * result.dipole, point.dipole, atol=1e-8, rtol=0):
                    raise ElectronicConvergenceError("analytic CIS dipole disagrees with the tracked native dipole phase")
                derivatives = point.phase * np.einsum("kax,axc->kc", modes, result.derivative)
                return derivatives, {"route": "analytic_rhf_cis", "max_response_residual": result.max_response_residual}
        derivatives, diagnostics = _estimate_dipole(self, point, modes)
        if reason is not None:
            diagnostics["analytic_unavailable"] = reason
        return derivatives, diagnostics


def _design(dimensions: int, generator: np.random.Generator) -> tuple[_Real, _Real]:
    orthogonal, _ = np.linalg.qr(generator.normal(size=(dimensions, dimensions)))
    extra = generator.normal(size=(max(3, dimensions), dimensions))
    extra /= np.linalg.norm(extra, axis=1)[:, None]
    extra *= generator.uniform(0.44, 0.84, len(extra))[:, None]
    base = np.vstack((orthogonal, extra))
    fit = np.vstack((base, -base))
    holdout = generator.normal(size=(max(4, dimensions // 2 + 1), dimensions))
    holdout /= np.linalg.norm(holdout, axis=1)[:, None]
    holdout *= generator.uniform(0.49, 0.91, len(holdout))[:, None]
    return fit, holdout


def _physical_scale(rows: _Real, frame: _Real, radius: float) -> _Real:
    # Use one common scaling so that fit and holdout use the same coordinates.
    motions = np.linalg.norm((rows @ frame.T).reshape(len(rows), -1, 3), axis=2)
    return rows * (radius / np.max(motions))


def _fit_symmetric_curvature(displacements: _Real, gradients: _Real) -> _Real:
    # Symmetry-constrained gradient regression. The normal equations form a
    # Sylvester equation; solve it in the eigenbasis of X.T @ X.
    singular, axes = np.linalg.eigh(displacements.T @ displacements)
    if singular[0] <= 1e-10 * singular[-1]:
        raise ElectronicConvergenceError("the mixed-displacement curvature design is rank-deficient")
    projection = displacements.T @ gradients
    rhs = axes.T @ (projection + projection.T) @ axes
    return axes @ (rhs / (singular[:, None] + singular[None, :])) @ axes.T


def _raise_repeated_reference_failure(failures) -> None:
    # A failed stability computation cannot establish a physical instability.
    unresolved = next(
        (error for error in reversed(failures) if isinstance(error, ReferenceStabilityUnresolvedError)), None,
    )
    if unresolved is not None:
        raise unresolved
    if not failures or not all(isinstance(error, UnstableReferenceError) for error in failures):
        return
    if all(error.channel == failures[0].channel for error in failures):
        raise failures[-1]


def _estimate_curvature(session: _ElectronicSession, center: _TrackedPoint) -> tuple[_Real, dict]:
    """Fit vibrational curvature to analytic gradients and check direct energies."""
    frame = _cartesian_frame(center.coordinates)
    gradient = center.total_gradient().ravel() @ frame
    generator = np.random.default_rng(57301)
    fit, holdout = _design(frame.shape[1], generator)
    failure = ""
    rejected = []
    for radius in (0.048, 0.030, 0.018):
        x = _physical_scale(np.vstack((fit, holdout)), frame, radius)
        try:
            samples = [
                session.isolated_point(center.coordinates + (frame @ row).reshape(center.coordinates.shape), center)
                for row in x
            ]
            gradients = np.stack([sample.total_gradient().ravel() @ frame for sample in samples])
        except (StateTrackingError, ElectronicConvergenceError) as error:
            failure = str(error)
            rejected.append(error)
            continue
        split = len(fit)
        hessian = _fit_symmetric_curvature(x[:split], gradients[:split] - gradient)
        predicted_gradients = gradient + x[split:] @ hessian
        delta_gradient = gradients[split:] - gradient
        energy_shifts = np.array([sample.energy - center.energy for sample in samples[split:]])
        predicted_energies = x[split:] @ gradient + 0.5 * np.einsum("ni,ij,nj->n", x[split:], hessian, x[split:])
        gradient_error = float(np.sqrt(np.mean((gradients[split:] - predicted_gradients) ** 2)))
        energy_error = float(np.sqrt(np.mean((energy_shifts - predicted_energies) ** 2)))
        gradient_limit = 8e-6 + 0.055 * float(np.sqrt(np.mean(delta_gradient**2)))
        energy_limit = 1.5e-7 + 0.075 * float(np.sqrt(np.mean(energy_shifts**2)))
        diagnostics = {
            "route": "symmetric_local_analytic_gradient_regression",
            "fit_points": split,
            "holdout_points": len(holdout),
            "maximum_atom_displacement_bohr": radius,
            "holdout_gradient_rmse_hartree_per_bohr": gradient_error,
            "holdout_gradient_limit_hartree_per_bohr": gradient_limit,
            "holdout_direct_energy_rmse_hartree": energy_error,
            "holdout_direct_energy_limit_hartree": energy_limit,
        }
        if gradient_error <= gradient_limit and energy_error <= energy_limit:
            return _atomic_hessian(frame @ hessian @ frame.T), diagnostics
        rejected.append(None)
        failure = f"Gradient RMSE {gradient_error:.3g} Hartree/bohr, energy RMSE {energy_error:.3g} Hartree"
    _raise_repeated_reference_failure(rejected)
    raise ElectronicConvergenceError(
        f"S1 local curvature could not be validated against independent gradients and direct energies. {failure}. "
        "Check electronic convergence, grid resolution, and local state isolation."
    )


def _estimate_dipole(session: _ElectronicSession, center: _TrackedPoint, modes: _Real) -> tuple[_Real, dict]:
    """Fit the phase-aligned dipole slope in ground normal coordinates."""
    scales = np.max(np.linalg.norm(modes, axis=2), axis=1)
    frame = (modes.reshape(len(modes), -1) / scales[:, None]).T
    generator = np.random.default_rng(7951)
    fit, holdout = _design(frame.shape[1], generator)
    failure = ""
    rejected = []
    for radius in (0.046, 0.028, 0.017):
        x = _physical_scale(np.vstack((fit, holdout)), frame, radius)
        try:
            shifts = np.stack([
                session.isolated_point(center.coordinates + (frame @ row).reshape(center.coordinates.shape), center).dipole
                - center.dipole for row in x
            ])
        except (StateTrackingError, ElectronicConvergenceError) as error:
            failure = str(error)
            rejected.append(error)
            continue
        split = len(fit)
        slope, _, rank, _ = np.linalg.lstsq(x[:split], shifts[:split], rcond=None)
        if rank != frame.shape[1]:
            raise ElectronicConvergenceError("the mixed-displacement HT design is rank-deficient")
        mismatch = shifts[split:] - x[split:] @ slope
        error = float(np.sqrt(np.mean(mismatch**2)))
        limit = _DIPOLE_ABSOLUTE_TOLERANCE_AU + _DIPOLE_RELATIVE_TOLERANCE * float(
            np.sqrt(np.mean(shifts[split:]**2))
        )
        if error <= limit:
            # The regression columns used dR/dQ divided by scales.
            # Restore the derivative with respect to Q.
            return scales[:, None] * slope, {
                "route": "local_phase_aligned_dipole_regression",
                "fit_points": split,
                "holdout_points": len(holdout),
                "maximum_atom_displacement_bohr": radius,
                "holdout_dipole_rmse_au": error,
                "holdout_dipole_limit_au": limit,
            }
        failure = f"holdout dipole RMSE {error:.3g} e·bohr"
        rejected.append(None)
    _raise_repeated_reference_failure(rejected)
    raise ElectronicConvergenceError(
        f"local HT regression could not be validated against independent dipoles ({failure}). "
        "Check the electronic convergence and isolation of the target state."
    )


def _check_ht_zero_point(session, center, modes, frequencies, derivatives) -> dict:
    """Check every material HT mode whose vibrational quantum spans the electronic gap."""
    if not frequencies.size:
        return {"status": "not_triggered"}
    excitation = center.point.excitation_energies_hartree
    neighbors = np.delete(excitation, center.root)
    gap = float(np.min(abs(neighbors - excitation[center.root]), initial=np.inf))
    amplitude = 1.0 / np.sqrt(2.0 * frequencies)
    dipole_changes = amplitude[:, None] * derivatives
    change_norms = np.linalg.norm(dipole_changes, axis=1)
    eligible = np.flatnonzero((frequencies >= gap) & (change_norms >= _DIPOLE_ABSOLUTE_TOLERANCE_AU))
    if not eligible.size:
        return {"status": "not_triggered"}

    # Check the largest predicted changes first so failures can stop the probes early.
    checked_modes = eligible[np.argsort(-change_norms[eligible], kind="stable")].tolist()
    diagnostics = {
        "status": "passed",
        "ground_mode_index": checked_modes[0],
        "checked_ground_mode_indices": checked_modes,
        "electronic_gap_hartree": gap,
    }
    for mode in checked_modes:
        step = amplitude[mode] * modes[mode]
        change = dipole_changes[mode]
        observed = np.stack([
            session.isolated_point(center.coordinates + sign * step, center).dipole - center.dipole
            for sign in (-1, 1)
        ])
        error = float(np.max(np.linalg.norm(observed - np.stack((-change, change)), axis=1)))
        scale = float(np.max(np.linalg.norm(observed, axis=1)))
        limit = _DIPOLE_ABSOLUTE_TOLERANCE_AU + _DIPOLE_RELATIVE_TOLERANCE * scale
        if error > limit:
            raise ElectronicConvergenceError(
                f"linear HT fails the zero-point displacement check for mode {mode + 1} "
                f"(dipole error {error:.3g} e·bohr, limit {limit:.3g} e·bohr, "
                f"neighboring root gap {gap:.3g} Hartree). A coordinate-dependent dipole or "
                "coupled electronic states are needed to qualify the spectrum and radiative rate."
            )
        if mode == checked_modes[0]:
            diagnostics.update(max_dipole_error_au=error, dipole_error_limit_au=limit)
    return diagnostics


def _gradient(point: _GroundPoint | _TrackedPoint) -> _Real:
    return point.gradient if isinstance(point, _GroundPoint) else point.total_gradient()


def _inverse_hessian_guess(frame: _Real, initial_hessian: _Real) -> _Real:
    stiffness, modes = np.linalg.eigh(frame.T @ _matrix_hessian(initial_hessian) @ frame)
    # Limit the conditioning of a borrowed Hessian so soft modes cannot
    # overwhelm its step. Its rigid-motion complement retains unit scale.
    floor = max(1e-8, 1e-3 * float(np.max(np.abs(stiffness))))
    modes = frame @ modes
    return np.eye(frame.shape[0]) + (modes * (1 / np.maximum(stiffness, floor) - 1)) @ modes.T


def _optimize(evaluate, initial, masses: _Real, state: str, *, initial_hessian: _Real | None = None):
    """Projected Cartesian BFGS using direct energies and analytic gradients."""
    point = initial
    inverse_hessian = np.eye(point.coordinates.size)
    counts = Counter(
        bfgs_iterations=0,
        optimization_trial_evaluations=0,
        optimization_energy_rejections=0,
        optimization_electronic_rejections=0,
        optimization_steepest_descent_restarts=0,
    )
    for cycle in range(121):
        raw_gradient = _gradient(point)
        maximum_gradient = _maximum_motion(raw_gradient)
        if maximum_gradient <= 0.5 * _STATIONARY_TOLERANCE:
            return point, counts
        if cycle == 120:
            break
        frame = _cartesian_frame(point.coordinates)
        if cycle == 0 and initial_hessian is not None:
            inverse_hessian = _inverse_hessian_guess(frame, initial_hessian)
        gradient = frame @ (frame.T @ raw_gradient.ravel())
        quasi_newton = -frame @ (frame.T @ (inverse_hessian @ gradient))
        steepest = -gradient
        quasi_newton /= max(1., _maximum_motion(quasi_newton) / 0.22)
        steepest /= max(1., _maximum_motion(steepest) / 0.22)
        last_error = None
        rejected = []
        accepted = None
        for attempt, direction in enumerate((quasi_newton, steepest)):
            if attempt:
                if np.allclose(direction, quasi_newton, atol=1e-14, rtol=1e-12):
                    break
                counts["optimization_steepest_descent_restarts"] += 1
                inverse_hessian = np.eye(point.coordinates.size)
            if not float(gradient @ direction) < 0:
                continue
            for power in range(16):
                trial = point.coordinates + 0.5**power * direction.reshape(raw_gradient.shape)
                proposal = _align(trial, point.coordinates, masses)
                # Use the realized displacement after re-alignment in
                # the Armijo energy test.
                slope = float(gradient @ (proposal - point.coordinates).ravel())
                if slope >= 0:
                    continue
                counts["optimization_trial_evaluations"] += 1
                try:
                    candidate = evaluate(proposal, point)
                    if not candidate.energy <= point.energy + 1e-4 * slope + 2e-11:
                        counts["optimization_energy_rejections"] += 1
                        rejected.append(None)
                        continue
                    candidate_gradient = _gradient(candidate).ravel()
                except (ElectronicConvergenceError, StateTrackingError) as error:
                    last_error = error
                    rejected.append(error)
                    counts["optimization_electronic_rejections"] += 1
                    continue
                accepted = candidate, candidate_gradient
                break
            if accepted is not None:
                break
        if accepted is None:
            if maximum_gradient <= _STATIONARY_TOLERANCE:
                return point, counts
            _raise_repeated_reference_failure(rejected)
            detail = "" if last_error is None else f" Last electronic failure: {last_error}"
            raise ElectronicConvergenceError(
                f"{state} optimization could not find a tracked downhill step at iteration {cycle + 1} "
                f"(maximum gradient {maximum_gradient:.3g} Hartree/bohr).{detail}"
            )
        candidate, candidate_gradient = accepted
        step = frame @ (frame.T @ (candidate.coordinates - point.coordinates).ravel())
        gradient_change = frame @ (frame.T @ (candidate_gradient - raw_gradient.ravel()))
        curvature = float(step @ gradient_change)
        if curvature > np.sqrt(np.finfo(float).eps) * np.linalg.norm(step) * np.linalg.norm(gradient_change):
            transformed = inverse_hessian @ gradient_change
            inverse_hessian += (
                (1 + float(gradient_change @ transformed) / curvature) * np.outer(step, step)
                - np.outer(step, transformed) - np.outer(transformed, step)
            ) / curvature
        point = candidate
        counts["bfgs_iterations"] += 1
    raise ElectronicConvergenceError(
        f"{state} geometry did not converge within 120 analytic-gradient iterations "
        f"(maximum gradient {maximum_gradient:.3g} Hartree/bohr)"
    )


def _find_minima(
    evaluate, raw_hessian, initial, masses: _Real, state: str, *,
    qualify_stationary, initial_hessian: _Real | None = None,
):
    checked = []
    optimization_counts = Counter()

    def descend(seed, depth):
        point, counts = _optimize(evaluate, seed, masses, state, initial_hessian=initial_hessian if depth == 0 else None)
        optimization_counts.update(counts)
        linear = _linear_candidate(point.coordinates, masses)
        if linear is not None:
            linear_point = evaluate(linear, point)
            if _maximum_motion(_gradient(linear_point)) <= _STATIONARY_TOLERANCE:
                point = linear_point
        for known in checked:
            residual = _maximum_motion(_align(point.coordinates, known.coordinates, masses) - known.coordinates)
            if residual < 2e-4:
                return []
        checked.append(point)
        # An unstable determinant can invalidate the curvature response
        # before the workflow otherwise gets a chance to retry the reference.
        qualify_stationary(point)
        hessian = raw_hessian(point)
        frame = _cartesian_frame(point.coordinates)
        eigenvalues, directions = np.linalg.eigh(frame.T @ _matrix_hessian(hessian) @ frame)
        if eigenvalues[0] > 1e-8:
            return [(point, hessian)]
        if eigenvalues[0] >= -1e-7:
            raise ElectronicConvergenceError(
                f"{state} geometry has unresolved vibrational curvature near zero "
                f"({eigenvalues[0]:.3g} Hartree/bohr²). A local harmonic minimum is undefined."
            )
        if depth == 3:
            raise ElectronicConvergenceError(f"{state} relaxed to a saddle after three unstable-mode descents")
        # Follow both signs of the most negative internal mode. A broken
        # symmetry can yield distinct local minima from the same saddle.
        direction = (frame @ directions[:, 0]).reshape(point.coordinates.shape)
        direction /= _maximum_motion(direction)
        minima = []
        for sign in (-1, 1):
            last_error = None
            rejected = []
            for displacement in (0.20, 0.10, 0.05, 0.025):
                trial = point.coordinates + sign * displacement * direction
                try:
                    branch = evaluate(_align(trial, point.coordinates, masses), point)
                except (ElectronicConvergenceError, StateTrackingError) as error:
                    last_error = error
                    rejected.append(error)
                    continue
                if branch.energy < point.energy - 2e-10:
                    minima.extend(descend(branch, depth + 1))
                    break
                rejected.append(None)
            else:
                _raise_repeated_reference_failure(rejected)
                detail = "" if last_error is None else f" Last electronic failure: {last_error}"
                raise ElectronicConvergenceError(
                    f"{state} stationary point has negative curvature but a downhill branch could not be followed.{detail}"
                )
        return minima

    minima = descend(initial, 0)
    if not minima:
        raise ElectronicConvergenceError(f"{state} search did not locate a checked local minimum")
    minima.sort(key=lambda item: item[0].energy)
    return minima, {
        "checked_stationary_geometries": len(checked),
        "local_minima_found": len(minima),
        "inverse_hessian_initialization": "identity" if initial_hessian is None else "ground_state_hessian",
        **optimization_counts,
    }


def _prepare_ground(session: _ElectronicSession):
    initial = session.ground(session.initial_coordinates)
    return _find_minima(
        session.ground, session.ground_hessian, initial, session.masses, "S0",
        qualify_stationary=session.ground_stability,
    )


def optimize_ground(symbols: Sequence[str], coordinates_bohr: ArrayLike, settings: Settings) -> _Real:
    """Find and verify a stable closed-shell S0 local minimum from a starting geometry."""
    session = _ElectronicSession(symbols, coordinates_bohr, settings)
    minima, _ = _prepare_ground(session)
    return minima[0][0].coordinates.copy()


def prepare_electronic(
    symbols: Sequence[str], coordinates_bohr: ArrayLike, settings: Settings, *, target_index: int,
    direct_dipole: bool = False,
) -> ElectronicPreparation:
    """Relax S0 and the singlet chosen by zero-based root index at the S0 minimum.

    Relaxation follows analytic gradients. The CPU S0 Hessian is analytic.
    For pure HF, the same Hamiltonian's conventional CIS derivatives are used
    when the dense analytic response fits memory. Other cases use mixed local
    regression with independent electronic holdouts for the S1 curvature and
    the signed transition dipole slope. An ambiguous target, unresolved
    stationary point, or poorly matched vibrational subspaces stops preparation.
    Set ``direct_dipole`` when the nuclear solver uses the sampled dipole field.
    """
    return _prepare_electronic(
        _ElectronicSession(symbols, coordinates_bohr, settings),
        target_index=target_index, direct_dipole=direct_dipole,
    )


def _prepare_electronic(session, *, target_index, direct_dipole):
    if isinstance(target_index, (bool, np.bool_)) or not isinstance(target_index, Integral):
        raise ValueError("target_index must be a zero-based integer")
    if not 0 <= target_index < session.root_dimension:
        raise ValueError(f"target index {target_index} is outside this basis's {session.root_dimension} singlet TDA pairs")
    grounds, ground_diagnostics = _prepare_ground(session)
    ground, ground_hessian = grounds[0]
    tracker = session.start_tracking(ground, int(target_index))
    if abs(tracker.point.ground_energy_hartree - ground.energy) > 1e-8:
        raise ElectronicConvergenceError("the S0 and TDA preprocessing converged to different ground references")
    exciteds, excited_diagnostics = _find_minima(
        session.isolated_point, session.excited_hessian, tracker, session.masses, "S1",
        qualify_stationary=session.excited_stability, initial_hessian=ground_hessian,
    )
    excited, excited_hessian = exciteds[0]
    session.anchors.extend(point for point, _ in exciteds)
    if _cartesian_frame(ground.coordinates).shape[1] != _cartesian_frame(excited.coordinates).shape[1]:
        raise ElectronicConvergenceError(
            "S0 and the tracked excited state have different linear/nonlinear vibrational dimensions. "
            "This requires a nuclear model with explicit treatment of the changing rotational subspace."
        )
    try:
        model = adiabatic_hessian(
            ground.seed.mol, excited.point.mol, ground_hessian, excited_hessian,
            ground_energy_hartree=ground.energy, excited_energy_hartree=excited.energy,
            ground_gradient=ground.gradient, excited_gradient=excited.total_gradient(),
            stationary_tolerance=_STATIONARY_TOLERANCE, subspace_tolerance=_SUBSPACE_LOSS_LIMIT,
        )
    except ValueError as error:
        raise ElectronicConvergenceError(f"the two stationary minima do not qualify for an AH model: {error}") from error
    aligned_excited = excited.coordinates @ model.excited_alignment_rotation + model.excited_alignment_translation_bohr
    if _maximum_motion(aligned_excited - excited.coordinates) > 1e-8:
        aligned_point = session.isolated_point(aligned_excited, excited)
        if abs(aligned_point.energy - excited.energy) > 1e-7:
            raise ElectronicConvergenceError("realigning the excited minimum changed its tracked electronic energy")
        session.anchors.append(aligned_point)
    if direct_dipole:
        derivatives = None
        dipole_diagnostics = {"route": "direct_coordinate_dependent"}
    else:
        modes = model.ground_cartesian_mode_derivatives_au
        derivatives, dipole_diagnostics = session.dipole_derivatives(tracker, modes)
        dipole_diagnostics["zero_point_linearity"] = _check_ht_zero_point(
            session, tracker, modes, model.ground_frequencies_hartree, derivatives,
        )
    return ElectronicPreparation(
        ground_coordinates_bohr=ground.coordinates.copy(),
        excited_coordinates_bohr=aligned_excited,
        model=model,
        transition_dipole_au=tracker.dipole,
        dipole_derivatives_au=derivatives,
        evaluate=session.evaluate,
        diagnostics={
            "ground_optimization": ground_diagnostics,
            "excited_optimization": excited_diagnostics,
            "electronic_relaxation": electronic_relaxation_properties(
                ground_at_ground_hartree=ground.energy,
                ground_at_excited_hartree=excited.point.ground_energy_hartree,
                excited_at_ground_hartree=tracker.energy,
                excited_at_excited_hartree=excited.energy,
                transition_dipole_at_ground_au=tracker.dipole,
                transition_dipole_at_excited_au=excited.dipole,
            ),
            "harmonic_reference": harmonic_model_properties(
                model, dipole_derivatives_au=derivatives,
                transition_dipole_at_ground_au=tracker.dipole,
                direct_transition_dipole_at_excited_au=excited.dipole @ model.excited_alignment_rotation,
            ),
            "ground_hessian": {"route": session.ground_hessian_route,
                               "quadrature_grid_response": session.quadrature_grid_response},
            "excited_gradient_quadrature_grid_response": session.quadrature_grid_response,
            "excited_reference_internal_stability": True,
            "excited_reference_external_stability": True,
            "excited_hessian": excited.curvature_diagnostics,
            "ht": dipole_diagnostics,
            "ground_scf_solves": session.ground_solves,
            "tracked_tda_solves": session.tracked_solves,
            "ground_cache_hits": session.ground_cache_hits,
            "tracked_cache_hits": session.tracked_cache_hits,
            "tracked_root_indices": sorted(session.visited_roots),
            "minimum_tracking_overlap": min(session.overlaps, default=None),
            "minimum_tracking_gap_hartree": min(session.gaps, default=None),
            "minimum_ground_gradient_max_hartree_per_bohr": _maximum_motion(ground.gradient),
            "minimum_excited_gradient_max_hartree_per_bohr": _maximum_motion(excited.total_gradient()),
            "maximum_duschinsky_subspace_loss": model.rotation_correction_norm,
            "duschinsky_subspace_loss_limit": _SUBSPACE_LOSS_LIMIT,
            "ground_alternate_energies_hartree": [point.energy for point, _ in grounds[1:]],
            "excited_alternate_energies_hartree": [point.energy for point, _ in exciteds[1:]],
        },
        ground_energy_hartree=ground.energy,
        excited_energy_hartree=excited.energy,
        alternate_excited_coordinates_bohr=tuple(
            _align(point.coordinates, ground.coordinates, session.masses) for point, _ in exciteds[1:]
        ),
        alternate_ground_coordinates_bohr=tuple(
            _align(point.coordinates, ground.coordinates, session.masses) for point, _ in grounds[1:]
        ),
    )
