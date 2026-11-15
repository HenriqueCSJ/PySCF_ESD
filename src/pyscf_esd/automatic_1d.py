"""Common-coordinate DVR from state- and phase-tracked direct electronic points."""

from __future__ import annotations

import numpy as np

from .anharmonic_dvr import solve_dvr
from .automatic_anharmonic import (
    AnharmonicUnavailableError, _ElectronicSurfaces, _LEVEL_TOLERANCE, _populations,
)
from .cpu import ElectronicConvergenceError
from .harmonic import HarmonicSpectrum
from .state_tracking import StateTrackingError


_MAX_SAMPLES = 176
_MAX_LEVELS = 64
_MAX_DOMAIN_ATTEMPTS = 10
_ENERGY_ATOL = 0.75 * _LEVEL_TOLERANCE
_ENERGY_RTOL = 0.0015
_DIPOLE_RTOL = 0.002
_BOUNDARY_POPULATION = 1e-4
_STRENGTH_TOLERANCE = 1e-4


class DirectOneMode:
    """Reuse samples on an open axis through both mass-aligned minima.

    The supplied normal vector is dR/dQ. With x represented in this Q,
    the exact Cartesian vibrational kinetic coefficient is one in atomic
    units. For a diatomic this is the reduced-mass bond kinetic operator.
    """

    def __init__(self, electronic: _ElectronicSurfaces, diagnostics: dict):
        self.electronic = electronic
        self.diagnostics = diagnostics
        model = electronic.model
        self.origin = electronic.ground_coordinates_bohr
        self.mode = model.ground_cartesian_mode_derivatives_au[0]
        self.mode_sign = float(model.rotation[0, 0])
        self.displacement = -self.mode_sign * float(model.shift_au[0])
        self.frequencies = np.array([
            model.ground_frequencies_hartree[0], model.excited_frequencies_hartree[0],
        ])
        modes_match = np.allclose(
            self.mode_sign * self.mode, model.excited_cartesian_mode_derivatives_au[0],
            rtol=1e-5, atol=1e-8,
        )
        axis_center = self.origin + self.displacement * self.mode
        if (not modes_match or np.any(~np.isfinite(self.frequencies)) or np.any(self.frequencies <= 0)
                or not np.allclose(axis_center, electronic.excited_coordinates_bohr, rtol=0, atol=1e-6)):
            raise AnharmonicUnavailableError("the stationary geometries lack a shared stable one-dimensional axis")
        self.centers = np.array([0.0, self.displacement])
        self.samples: dict[float, np.ndarray] = {}
        self.knots: set[float] = set()
        self._last_converged: dict[str, tuple[dict, HarmonicSpectrum]] = {}
        self.axis_bounds = self._collision_limits()
        self.knot_step = 0.9 / np.sqrt(max(self.frequencies))
        diagnostics["minimum_coordinates_sqrt_electron_mass_bohr"] = self.centers.tolist()
        diagnostics["kinetic_coefficient_au"] = 1.0

    def _collision_limits(self):
        # Stop the axis before it crosses the short-distance region of any
        # atom pair. The reference distance comes from the two actual minima.
        lower, upper = -np.inf, np.inf
        ia, ja = np.triu_indices(len(self.origin), k=1)
        vectors = self.origin[ia] - self.origin[ja]
        slopes = self.mode[ia] - self.mode[ja]
        for vector, slope in zip(vectors, slopes):
            square = slope @ slope
            if square <= 1e-24:
                continue
            minimum_distance = 0.35 * min(np.linalg.norm(vector), np.linalg.norm(vector + self.displacement * slope))
            center = -(vector @ slope) / square
            distance_residual = vector + center * slope
            radius_squared = (minimum_distance**2 - distance_residual @ distance_residual) / square
            if radius_squared <= 0:
                continue
            radius = np.sqrt(radius_squared)
            if center < self.centers.min():
                lower = max(lower, center + radius)
            elif center > self.centers.max():
                upper = min(upper, center - radius)
            else:
                raise AnharmonicUnavailableError("the common axis passes through a short atom-pair distance")
        return lower, upper

    def _sample(self, coordinates):
        for coordinate in sorted(coordinates, key=lambda q: min(abs(q - self.centers))):
            if coordinate in self.samples:
                continue
            if len(self.samples) >= _MAX_SAMPLES:
                raise AnharmonicUnavailableError("direct one-mode interpolation exhausted its electronic sample budget")
            self.diagnostics["energy_evaluations"] += 1
            try:
                ground, excited, dipole = self.electronic.evaluate(self.origin + coordinate * self.mode)
            except (ElectronicConvergenceError, StateTrackingError) as error:
                raise AnharmonicUnavailableError(
                    f"direct one-mode electronic sample failed at Q={coordinate:.5g}: {error}"
                ) from error
            values = np.r_[ground, excited, dipole]
            if values.shape != (5,) or not np.all(np.isfinite(values)):
                raise AnharmonicUnavailableError("a direct one-mode point returned invalid energies or dipole")
            self.samples[coordinate] = values
        return np.stack([self.samples[coordinate] for coordinate in coordinates])

    def _interpolate(self, lower, upper):
        from scipy.interpolate import CubicSpline, PPoly

        step = self.knot_step
        first, last = np.ceil(lower / step), np.floor(upper / step)
        if not np.isfinite(first + last) or last - first + 3 > _MAX_SAMPLES:
            raise AnharmonicUnavailableError("the requested one-mode domain exceeds the electronic sample budget")
        ordinates = np.arange(first, last + 1) * step
        self.knots.update(np.round(np.r_[lower, ordinates, self.centers, upper], 11))
        knots = sorted(q for q in self.knots if lower - 1e-10 <= q <= upper + 1e-10)
        # Clip roundoff duplicates across boundaries without adding spline
        # intervals too small for the precision of electronic energies.
        knots = np.array([q for i, q in enumerate(knots) if i == 0 or q - knots[i - 1] > 1e-8])
        reference = self._sample(np.round(self.centers, 11))[[0, 1], [0, 1]]
        while True:
            values = self._sample(knots)
            values[:, :2] -= reference
            if np.min(values[:, :2]) < -10 * _ENERGY_ATOL:
                raise AnharmonicUnavailableError("the direct one-mode scan samples below a verified minimum")
            fit = CubicSpline(knots, values)
            midpoints = np.round((knots[1:] + knots[:-1]) / 2, 11)
            true_values = self._sample(midpoints)
            true_values[:, :2] -= reference
            if np.min(true_values[:, :2]) < -10 * _ENERGY_ATOL:
                raise AnharmonicUnavailableError("the direct one-mode scan samples below a verified minimum")
            residual = fit(midpoints) - true_values
            energy_limit = _ENERGY_ATOL + _ENERGY_RTOL * abs(true_values[:, :2])
            dipole_limit = 1e-6 + _DIPOLE_RTOL * np.linalg.norm(true_values[:, 2:], axis=1)
            energy_failed = np.any(abs(residual[:, :2]) > energy_limit, axis=1)
            dipole_failed = np.linalg.norm(residual[:, 2:], axis=1) > dipole_limit
            failed = energy_failed | dipole_failed
            if not np.any(failed):
                for index in range(2):
                    potential = PPoly(fit.c[:, :, index], fit.x)
                    stationary = potential.derivative().roots(discontinuity=False, extrapolate=False)
                    if np.any(potential(stationary[np.isfinite(stationary)]) < -10 * _ENERGY_ATOL):
                        raise AnharmonicUnavailableError(
                            "the interpolated one-mode potential introduces an unsupported deeper minimum"
                        )
                self.diagnostics["interpolation"] = {
                    "knots": len(knots), "holdout_points": len(midpoints),
                    "max_energy_residual_hartree": float(np.max(abs(residual[:, :2]))),
                    "max_energy_tolerance_fraction": float(np.max(abs(residual[:, :2]) / energy_limit)),
                    "max_dipole_residual_au": float(np.max(np.linalg.norm(residual[:, 2:], axis=1))),
                    "max_dipole_tolerance_fraction": float(
                        np.max(np.linalg.norm(residual[:, 2:], axis=1) / dipole_limit)
                    ),
                }
                return fit, reference, (float(knots[0]), float(knots[-1]))
            new_knots = midpoints[failed]
            if np.any(np.diff(knots)[failed] < 2e-8):
                raise AnharmonicUnavailableError("direct one-mode samples disagree at the interpolation precision")
            self.knots.update(new_knots)
            knots = np.sort(np.r_[knots, new_knots])

    def _solve(self, fit, bounds, counts, grid_size):
        from scipy.special import eval_hermite

        grid = np.linspace(*bounds, grid_size + 2)[1:-1]
        samples = fit(grid)
        energies, wavefunctions = [], []
        for index, count in enumerate(counts):
            result = solve_dvr(grid, samples[:, index], boundary="dirichlet", n_levels=count)
            vectors = result.coefficients.copy()
            x = (grid - self.centers[index]) * np.sqrt(self.frequencies[index])
            if index:
                x *= self.mode_sign
            anchor = eval_hermite(np.arange(count)[None, :], x[:, None]) * np.exp(-0.5 * x[:, None]**2)
            vectors *= np.where(np.sum(anchor * vectors, axis=0) < 0, -1.0, 1.0)
            energies.append(result.energies_hartree)
            wavefunctions.append(vectors)
        return energies, wavefunctions, samples[:, 2:]

    def spectrum(self, *, process, temperature, initial_cutoff, final_cutoff, max_overlaps):
        attempt = {
            "temperature_kelvin": float(temperature),
            "initial_max_quanta": int(initial_cutoff) if temperature else 0,
            "final_max_quanta": int(final_cutoff),
            "domain_attempts": [],
        }
        self.diagnostics.setdefault("one_mode_attempts", {}).setdefault(process, []).append(attempt)
        try:
            result = self._spectrum(
                process, temperature, initial_cutoff, final_cutoff, max_overlaps, attempt["domain_attempts"],
            )
        except AnharmonicUnavailableError as error:
            attempt.update(status="incomplete", reason=str(error))
            last_converged = self._last_converged.get(process)
            if last_converged is not None:
                previous_attempt, previous = last_converged
                self.diagnostics.setdefault("last_converged_sticks", {})[process] = {
                    "status": "partial",
                    "temperature_kelvin": previous_attempt["temperature_kelvin"],
                    "initial_max_quanta": previous_attempt["initial_max_quanta"],
                    "final_max_quanta": previous_attempt["final_max_quanta"],
                    "photon_energy_hartree": previous.photon_energy_hartree.tolist(),
                    "initial_quanta": previous.initial_quanta.tolist(),
                    "final_quanta": previous.final_quanta.tolist(),
                    "population": previous.population.tolist(),
                    "dipole_strength_au2": previous.dipole_strength_au2.tolist(),
                    "transition_dipole_au": previous.transition_dipole_au.tolist(),
                    "initial_population_coverage": previous.initial_population_coverage,
                    "final_dipole_strength_coverage": previous.final_dipole_strength_coverage,
                }
            raise
        attempt.update(
            status="converged",
            initial_population_coverage=result.initial_population_coverage,
            final_dipole_strength_coverage=result.final_dipole_strength_coverage,
        )
        self._last_converged[process] = (attempt, result)
        self.diagnostics.get("last_converged_sticks", {}).pop(process, None)
        return result

    def _spectrum(self, process, temperature, initial_cutoff, final_cutoff, max_overlaps, domain_attempts):
        initial_surface, final_surface = (0, 1) if process == "absorption" else (1, 0)
        retained = [int(initial_cutoff) + 1 if temperature else 1, int(final_cutoff) + 1]
        if max(retained) > _MAX_LEVELS or retained[0] * retained[1] > max_overlaps:
            raise AnharmonicUnavailableError("the requested one-mode levels exceed the DVR or max_overlaps budget")
        counts = retained if process == "absorption" else retained[::-1]
        width = (np.sqrt(2 * np.array(counts) - 1) + 2.1) / np.sqrt(self.frequencies)
        raw_bounds = np.array([min(self.centers - width), max(self.centers + width)])
        # A shared step preserves both knots and independent midpoints when a
        # later basis request needs a longer continuation of one edge.
        bounds = self._outward_bounds(raw_bounds)
        previous_spectrum = self.diagnostics.get("spectrum", {}).get(process)
        if previous_spectrum is not None:
            lower, upper = previous_spectrum["domain_sqrt_electron_mass_bohr"]
            bounds = (min(bounds[0], lower), max(bounds[1], upper))
        for _ in range(_MAX_DOMAIN_ATTEMPTS):
            record = {"attempted_domain_sqrt_electron_mass_bohr": list(bounds)}
            domain_attempts.append(record)
            try:
                fit, reference, grid_bounds = self._interpolate(*bounds)
            except AnharmonicUnavailableError as error:
                record.update(status="sampling_incomplete", reason=str(error))
                raise
            record["sampled_domain_sqrt_electron_mass_bohr"] = list(grid_bounds)
            wall = fit(np.asarray(grid_bounds))[:, :2]
            previous = None
            for grid_size in (111, 223, 447):
                energies, vectors, dipole = self._solve(fit, grid_bounds, counts, grid_size)
                start, end = vectors[initial_surface], vectors[final_surface]
                amplitudes = np.stack([start.T @ (dipole[:, k, None] * end) for k in range(3)], axis=-1)
                strengths = np.sum(amplitudes**2, axis=-1)
                required_strength = np.sum(start**2 * np.sum(dipole**2, axis=1)[:, None], axis=0)
                if previous is not None:
                    previous_energies, previous_strengths = previous
                    energy_error = max(float(np.max(abs(a - b))) for a, b in zip(energies, previous_energies))
                    strength_error = float(np.max(
                        np.sum(abs(strengths - previous_strengths), axis=1) / np.maximum(required_strength, 1e-25)
                    ))
                    record.update(
                        grid_size=grid_size, max_grid_energy_shift_hartree=energy_error,
                        max_grid_strength_error_fraction=strength_error,
                    )
                    if energy_error <= _LEVEL_TOLERANCE and strength_error <= _STRENGTH_TOLERANCE:
                        break
                previous = (energies, strengths)
            else:
                record["status"] = "grid_incomplete"
                raise AnharmonicUnavailableError("the one-mode DVR energy or intensity grid has not converged")
            probabilities = self._boundary_probabilities(vectors, grid_bounds, grid_size)
            margins = wall - np.array([e[-1] for e in energies])
            record.update(
                grid_size=grid_size,
                boundary_probability_by_edge_and_surface=probabilities.tolist(),
                wall_margin_hartree_by_edge_and_surface=margins.tolist(),
            )
            failed = (probabilities > _BOUNDARY_POPULATION) | (margins <= 0)
            if not np.any(failed):
                record["status"] = "confined"
                break
            record["status"] = "extension_required"
            next_bounds = self._extend_bounds(bounds, failed, probabilities, margins)
            if next_bounds == bounds:
                record["status"] = "short_distance_limit"
                raise AnharmonicUnavailableError("the direct one-mode levels reach the short-distance domain limit")
            bounds = next_bounds
        else:
            raise AnharmonicUnavailableError("the direct one-mode levels are not confined by the sampled domain")

        omega = self.frequencies[[initial_surface]]
        initial_labels = np.arange(retained[0], dtype=np.intp)[:, None]
        populations = _populations(energies[initial_surface], initial_labels, omega, temperature)
        denominator = float(populations @ required_strength)
        coverage = min(1.0, float(populations @ strengths.sum(axis=1) / denominator)) if denominator else None
        photons = (reference[final_surface] - reference[initial_surface]
                   + energies[final_surface][None, :] - energies[initial_surface][:, None])
        if process == "emission":
            photons *= -1
        i, j = np.nonzero((photons > 0) & (populations[:, None] > 0))
        order = np.argsort(photons[i, j], kind="stable")
        i, j = i[order], j[order]
        self.diagnostics.setdefault("spectrum", {})[process] = {
            "domain_sqrt_electron_mass_bohr": list(grid_bounds),
            "grid_size": grid_size,
            "max_grid_energy_shift_hartree": energy_error,
            "max_grid_strength_error_fraction": strength_error,
            "max_boundary_probability": float(probabilities.max()),
            "minimum_wall_margin_hartree": float(margins.min()),
            "interpolation": self.diagnostics["interpolation"],
            "partition_tail": "harmonic estimate" if temperature else None,
        }
        return HarmonicSpectrum(
            photon_energy_hartree=photons[i, j], initial_quanta=initial_labels[i], final_quanta=j[:, None],
            population=populations[i], dipole_strength_au2=strengths[i, j],
            initial_population_coverage=float(populations.sum()), final_dipole_strength_coverage=coverage,
            transition_dipole_au=amplitudes[i, j],
        )

    def _outward_bounds(self, bounds):
        step = self.knot_step
        lower_step = np.floor(bounds[0] / step + 1e-10) * step
        upper_step = np.ceil(bounds[1] / step - 1e-10) * step
        return float(max(lower_step, self.axis_bounds[0])), float(min(upper_step, self.axis_bounds[1]))

    def _boundary_probabilities(self, vectors, bounds, grid_size):
        spacing = (bounds[1] - bounds[0]) / (grid_size + 1)
        widths = np.maximum(1, np.ceil(0.5 / np.sqrt(self.frequencies) / spacing)).astype(int)
        return np.array([
            [np.max(np.sum(v[:n]**2, axis=0)) for v, n in zip(vectors, widths)],
            [np.max(np.sum(v[-n:]**2, axis=0)) for v, n in zip(vectors, widths)],
        ])

    def _extend_bounds(self, bounds, failed, probabilities, margins):
        oscillator_lengths = 1 / np.sqrt(self.frequencies)
        increments = np.zeros(2)
        for edge, surface in zip(*np.nonzero(failed)):
            length = oscillator_lengths[surface]
            margin = margins[edge, surface]
            if margin <= 0:
                # Reach a classical turning point before inspecting decay.
                advance = 1.5 * length
            else:
                # Once outside the turning point, estimate the extra interval
                # from exponential probability decay at the sampled wall.
                excess = probabilities[edge, surface] / _BOUNDARY_POPULATION
                decay_length = 1 / np.sqrt(2 * margin)
                advance = np.clip(0.5 * decay_length * np.log(4 * excess), 0.75 * length, 2.5 * length)
            increments[edge] = max(increments[edge], advance)
        proposal = self._outward_bounds((bounds[0] - increments[0], bounds[1] + increments[1]))
        return tuple(new if shift else old for new, old, shift in zip(proposal, bounds, increments))
