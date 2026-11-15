"""Contract direct S1 breathing × spectator cuts into the benzene 3D model.

All Hamiltonians and fit coefficients are in cm^-1, with the dimensionless
analytic S0 normal coordinates of the original scan. One-dimensional spectator
cuts resolve diagonal couplings to breathing; spectator-pair and spectator–
active transverse couplings require additional electronic information.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.special import eval_laguerre, roots_hermite, roots_laguerre

from validate_benzene_3d import (
    CM_PER_HARTREE,
    HarmonicBasis,
    RadialCorrection,
    calculation,
    harmonic_parameters,
    read_scan,
    solve,
)
from validate_benzene_absorption import (
    absorption_progression,
    base_surface_fit,
    with_transverse_correction,
)


@dataclass(frozen=True)
class Spectator:
    label: str
    frequency: float
    multiplicity: int
    powers: tuple[int, ...]
    q1: np.ndarray
    qj: np.ndarray
    energies: np.ndarray
    train: np.ndarray
    electronic_diagnostics: dict | None = None


def read_spectators(metadata_path, points_path, holdout, *, method=None, tail_points_path=None, training_q1=None):
    with np.load(metadata_path, allow_pickle=False) as source:
        labels = source["labels"].tolist()
        frequencies = source["frequencies_cm1"].tolist()
        multiplicities = source["multiplicities"].tolist()
        irreps = source["irreps"].tolist()
        sign_symmetric = source["sign_symmetric"].tolist()
        mode_vectors = dict(zip(labels, source["mode_vectors_bohr"], strict=True))
        radial_vector = source["q1_vector_bohr"]
    if len(set(labels)) != len(labels) or sum(multiplicities) != 27:
        raise ValueError("require unique cut representatives accounting for 27 omitted coordinates")
    points = defaultdict(list)
    root_checks = defaultdict(list)
    references = {}
    geometrical_center = None
    method_settings = None
    direct_points = sorted(points_path.glob("*.json"))
    external_holdout = sorted(tail_points_path.glob("*.json")) if tail_points_path is not None else []
    for path in (*direct_points, *external_holdout):
        row = json.loads(path.read_text(encoding="utf-8"))
        group, q1, qj = row["group"], float(row["q1"]), float(row["qj"])
        energy, radial = float(row["energy_s1_hartree"]), float(row["radial_s1_hartree"])
        if group not in labels or qj == 0 or not np.isfinite([q1, qj, energy, radial]).all():
            raise ValueError(f"invalid spectator cut point in {path}")
        if abs(energy - radial - row["delta_s1_hartree"]) > 1e-10:
            raise ValueError(f"unmatched direct radial energy in {path}")
        if abs(references.setdefault(q1, radial) - radial) > 1e-8:
            raise ValueError(f"inconsistent direct radial S1 reference at q1={q1}")
        geometry = np.asarray(row["positions_bohr"])
        center = geometry - q1 * radial_vector - qj * mode_vectors[group]
        if geometrical_center is None:
            geometrical_center = center
        if not np.allclose(center, geometrical_center, rtol=0, atol=1e-10):
            raise ValueError(f"inconsistent direct geometry on the declared coordinate system in {path}")
        direct_method = row["method"]
        settings = {"response": "tddft" if direct_method["response"] == "full singlet TDDFT" else direct_method["response"],
                    "basis": direct_method["basis"], "grid": direct_method["grid"], "density_fit": direct_method["density_fit"],
                    "auxiliary_basis": direct_method["auxbasis"]}
        if method_settings is None:
            method_settings = settings if method is None else method
        if settings != method_settings:
            raise ValueError(f"incompatible direct electronic method in {path}")
        residual, overlap = row["td_residual_hartree"], row["xy_overlap_normalized"]
        if not row["converged"] or not np.isfinite([residual, overlap]).all():
            raise ValueError(f"unconverged or untracked S1 energy in {path}")
        points[group].append((q1, qj, (energy - radial) * CM_PER_HARTREE, path in external_holdout))
        root_checks[group].append((residual, overlap, row["independent_roots_energy_order"] is not None))
    if set(points) != set(labels):
        raise ValueError(f"missing direct groups: {sorted(set(labels) - points.keys())}")

    spectators = []
    for label, frequency, multiplicity, irrep, even in zip(labels, frequencies, multiplicities, irreps, sign_symmetric, strict=True):
        if frequency <= 0 or multiplicity not in (1, 2):
            raise ValueError(f"invalid frequency or multiplicity for {label}")
        q1, qj, energies, tail_only = np.array(points[label]).T
        if len(np.unique(np.column_stack((q1, qj)), axis=0)) != len(q1):
            raise ValueError(f"duplicate cut geometry for {label}")
        signed = np.any(qj < 0)
        if not signed and not even:
            raise ValueError(f"signed direct cuts are required for {label}")
        if irrep.lower() == "a1g":
            powers = (1, 2, 3, 4)
        elif not even:
            powers = (2, 3, 4)
        else:
            powers = (2, 4)
        train = ~np.isin(q1, holdout) & (tail_only == 0)
        if training_q1 is not None:
            train &= np.isin(q1, training_q1)
        if even:
            train &= qj > 0
        root_checks_array = np.asarray(root_checks[label])
        diagnostics = {
            "point_count": len(q1),
            "max_td_residual_hartree": float(root_checks_array[:, 0].max()),
            "min_xy_overlap_normalized": float(root_checks_array[:, 1].min()),
            "independent_lowest_root_audit_points": int(root_checks_array[:, 2].sum()),
        }
        spectators.append(Spectator(label, frequency, multiplicity, powers, q1, qj, energies, train, diagnostics))
    train_nodes = np.unique(spectators[0].q1[spectators[0].train])
    if train_nodes.size < 4 or 0 not in train_nodes:
        raise ValueError("require the direct q1=0 cut among at least four training slices")
    if training_q1 is not None and set(training_q1) != set(train_nodes):
        raise ValueError(f"incomplete requested radial grid: {train_nodes.tolist()}")
    for spectator in spectators:
        if not np.array_equal(np.unique(spectator.q1[spectator.train]), train_nodes):
            raise ValueError(f"mismatched radial training grid for {spectator.label}")
        amplitudes = np.unique(spectator.qj[spectator.train])
        for coordinate in train_nodes:
            if not np.array_equal(np.sort(spectator.qj[spectator.train & (spectator.q1 == coordinate)]), amplitudes):
                raise ValueError(f"incomplete spectator cut for {spectator.label} at q1={coordinate}")
    return spectators


@dataclass(frozen=True)
class CutFit:
    label: str
    frequency: float
    multiplicity: int
    powers: tuple[int, ...]
    coefficients: np.ndarray
    radial_bounds: tuple[float, float]
    spectator_bound: float
    report: dict
    spline: CubicSpline | None = None

    def polynomial_segments(self, limits):
        polynomial = np.polynomial.Polynomial
        if self.spline is None:
            yield limits, [polynomial(row) for row in self.coefficients]
            return
        edges = np.r_[limits[0], self.spline.x[1:-1], limits[1]]
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            origin = polynomial([-self.spline.x[index], 1])
            yield (left, right), [polynomial(row[::-1])(origin) for row in self.spline.c[:, index].T]

    def curves(self, q1, extension):
        clipped = np.clip(q1, *self.radial_bounds)
        derivative = (q1 == clipped).astype(float)
        if extension:
            saturation = np.tanh((q1 - clipped) / extension)
            clipped = clipped + extension * saturation
            derivative = 1 - saturation**2
        if self.spline is not None:
            return self.spline(clipped), self.spline(clipped, 1) * derivative[:, None]
        # Each row contains one univariate polynomial coefficient series.
        polys = [np.polynomial.Polynomial(row) for row in self.coefficients]
        values = np.column_stack([poly(clipped) for poly in polys])
        slopes = np.column_stack([poly.deriv()(clipped) for poly in polys]) * derivative[:, None]
        return values, slopes


def fit_cut(spectator, degree, max_power):
    powers = tuple(power for power in spectator.powers if power <= max_power)
    training_q = spectator.q1[spectator.train]
    training_s = spectator.qj[spectator.train]
    scale_q = np.max(np.abs(training_q))
    scale_s = np.max(np.abs(training_s))
    orders = np.arange(degree + 1) if degree != "spline" else np.array([0])
    q_design = (spectator.q1[:, None] / scale_q)**orders
    s_design = (spectator.qj[:, None] / scale_s)**np.asarray(powers)
    spline = None

    def solve_coefficients(design, mask):
        fit, _, rank, singular = np.linalg.lstsq(design[mask], spectator.energies[mask], rcond=None)
        if rank != len(fit):
            raise ValueError(f"underdetermined q1 × spectator fit for {spectator.label} ({rank}/{len(fit)})")
        return fit, float(singular[0] / singular[-1])

    if degree == "spline":
        radial_nodes = np.unique(training_q)
        fits = [solve_coefficients(s_design, spectator.train & (spectator.q1 == q)) for q in radial_nodes]
        values = np.stack([value for value, _ in fits]) / scale_s**np.asarray(powers)
        spline = CubicSpline(radial_nodes, values, axis=0)
        coefficients = spline(0.0)[:, None]
        prediction = np.einsum("qp,qp->q", spline(spectator.q1), spectator.qj[:, None]**np.asarray(powers))
        condition = max(item[1] for item in fits)
        fit_report = {"q1_knots": radial_nodes.tolist(), "coefficients_at_q1_knots": values.tolist(),
                      "radial_interpolant": "not-a-knot cubic spline"}
    else:
        design = (q_design[:, None, :] * s_design[:, :, None]).reshape(len(spectator.q1), -1)
        fit, condition = solve_coefficients(design, spectator.train)
        prediction = design @ fit
        coefficients = fit.reshape(len(powers), len(orders)) / scale_q**orders / scale_s**np.asarray(powers)[:, None]
        fit_report = {"coefficients_cm_inverse_by_spectator_and_q1_powers": coefficients.tolist()}
    residuals = prediction - spectator.energies
    report = {
        "spectator_mode_reference_cm_inverse": spectator.frequency,
        "coordinate_multiplicity": spectator.multiplicity,
        "spectator_powers": powers,
        "design_condition_number": condition,
        **fit_report,
    }
    amplitude_holdout = np.abs(spectator.qj) > scale_s
    radial_holdout = ~np.isin(spectator.q1, training_q) & ~amplitude_holdout
    for label, mask in (("training", spectator.train), ("radial_holdout", radial_holdout),
                        ("symmetry_holdout", ~spectator.train & ~radial_holdout & ~amplitude_holdout)):
        errors = residuals[mask]
        if len(errors):
            report[label] = {
                "count": len(errors),
                "rms_cm_inverse": float(np.sqrt(np.mean(errors**2))),
                "max_abs_cm_inverse": float(np.max(np.abs(errors))),
                "residuals_by_q1_qj_cm_inverse": np.column_stack((spectator.q1[mask], spectator.qj[mask], errors)).tolist(),
            }
    return CutFit(spectator.label, spectator.frequency, spectator.multiplicity, powers, coefficients,
                  (float(training_q.min()), float(training_q.max())), float(scale_s), report, spline)


def oscillator_values(nodes, size):
    basis = np.empty((len(nodes), size))
    basis[:, 0] = np.pi**(-0.25)
    basis[:, 1] = np.sqrt(2) * nodes * basis[:, 0]
    for n in range(1, size - 1):
        basis[:, n + 1] = (np.sqrt(2) * nodes * basis[:, n] - np.sqrt(n) * basis[:, n - 1]) / np.sqrt(n + 1)
    return basis


@dataclass(frozen=True)
class SpectatorOscillator:
    kinetic: np.ndarray
    moments: np.ndarray
    in_domain: np.ndarray
    isotropic_doublet: bool
    extension: float
    quartic_tail: str
    cross_moments: np.ndarray | None = None


def potential_powers(q, powers, bound, extension, quartic_tail):
    capped = np.clip(q, -bound, bound)
    excess = q - capped
    if extension:
        capped += extension * np.tanh(excess / extension)
    profiles = [(q if power <= 2 else capped)**power for power in powers]
    for power in (term for term in powers if term >= 4 and term % 2 == 0):
        if quartic_tail == "polynomial":
            positive_tail = q**power
        else:
            beyond = np.abs(excess)
            slope = power * bound**(power - 1)
            half_curvature = power * (power - 1) * bound**(power - 2) / 2
            positive_tail = np.minimum(np.abs(q), bound)**power + slope * beyond + half_curvature * beyond**2
        profiles.append(positive_tail - capped**power)
    return np.asarray(profiles)


def spectator_oscillator(fit, size, extension, *, isotropic_doublets=True, quartic_tail="matched"):
    # Adapt the reference frequency to the positive vertical fit curvature.
    # Use an unmodified harmonic kinetic matrix in the dimensionless q coordinate.
    quadratic = float(fit.coefficients[fit.powers.index(2), 0])
    omega = np.sqrt(2 * fit.frequency * quadratic) if quadratic > 0 else 0.7 * fit.frequency
    length = np.sqrt(fit.frequency / omega)
    if isotropic_doublets and fit.multiplicity == 2:
        if any(power % 2 for power in fit.powers):
            raise ValueError(f"odd powers require an anisotropic doublet treatment for {fit.label}")
        return doublet_oscillator(fit, size, extension, omega, length, quartic_tail)
    nodes, weights = roots_hermite(max(120, size * 3))
    basis = oscillator_values(nodes, size) * np.sqrt(weights[:, None])
    coordinate = length * nodes
    profiles = potential_powers(coordinate, fit.powers, fit.spectator_bound, extension, quartic_tail)
    moments = np.einsum("qi,pq,qj->pij", basis, profiles, basis)
    # The wider pure-mode points do not extend the support of the fitted mixed products.
    cross_bound = min(fit.spectator_bound, 1.7)
    capped = np.clip(coordinate, -cross_bound, cross_bound)
    if extension:
        capped += extension * np.tanh((coordinate - capped) / extension)
    cross_moments = np.einsum("qi,pq,qj->pij", basis, np.stack((capped, capped**2, coordinate, capped**3)), basis)
    kinetic = np.diag(omega * (np.arange(size) + 0.5))
    kinetic -= 0.5 * omega * (basis.T @ (nodes[:, None]**2 * basis))
    legendre, legendre_weights = np.polynomial.legendre.leggauss(72)
    radius = fit.spectator_bound / length
    inside = oscillator_values(radius * legendre, size)
    inside *= np.exp(-0.5 * (radius * legendre[:, None])**2)
    in_domain = inside.T @ ((radius * legendre_weights)[:, None] * inside)
    return SpectatorOscillator(kinetic, moments, in_domain, False, extension, quartic_tail, cross_moments)


def doublet_oscillator(fit, size, extension, omega, length, quartic_tail):
    # m=0 radial oscillator with u=(r/length)^2. Laguerre polynomials are
    # orthonormal against exp(-u)du, including the angular normalization.
    nodes, weights = roots_laguerre(max(120, size * 3))
    numbers = np.arange(size)
    basis = eval_laguerre(numbers[None, :], nodes[:, None]) * np.sqrt(weights[:, None])
    radius = length * np.sqrt(nodes)
    profiles = potential_powers(radius, fit.powers, fit.spectator_bound, extension, quartic_tail)
    moments = np.einsum("qi,pq,qj->pij", basis, profiles, basis)
    kinetic = np.diag(omega * (2 * numbers + 1)) - 0.5 * omega * (basis.T @ (nodes[:, None] * basis))
    legendre, legendre_weights = np.polynomial.legendre.leggauss(72)
    half_extent = 0.5 * (fit.spectator_bound / length)**2
    u = half_extent * (1 + legendre)
    inside = eval_laguerre(numbers[None, :], u[:, None]) * np.exp(-0.5 * u[:, None])
    in_domain = inside.T @ ((half_extent * legendre_weights)[:, None] * inside)
    return SpectatorOscillator(kinetic, moments, in_domain, True, extension, quartic_tail)


@dataclass(frozen=True)
class Contraction:
    fit: CutFit
    oscillator: SpectatorOscillator
    radial_extension: float

    @property
    def replicated_cuts(self):
        return 1 if self.oscillator.isotropic_doublet else self.fit.multiplicity

    @property
    def breathing_active_step(self):
        return 1 if self.oscillator.isotropic_doublet or 1 in self.fit.powers else 2

    @property
    def continued_even_powers(self):
        return [power for power in self.fit.powers if power >= 4 and power % 2 == 0]

    def curves(self, q1):
        coefficients, derivatives = self.fit.curves(q1, self.radial_extension)
        indices = [self.fit.powers.index(power) for power in self.continued_even_powers]
        if indices:
            positive = coefficients[:, indices] > 0
            coefficients = np.column_stack((coefficients, coefficients[:, indices] * positive))
            derivatives = np.column_stack((derivatives, derivatives[:, indices] * positive))
        return coefficients, derivatives

    def confinement(self, coefficients):
        quadratic = coefficients[:, self.fit.powers.index(2)]
        powers = np.array(self.continued_even_powers)
        if not len(powers):
            return quadratic
        continued = coefficients[:, -len(powers):]
        if self.oscillator.quartic_tail == "polynomial":
            return np.where(np.any(continued > 0, axis=1), np.inf, quadratic)
        factors = powers * (powers - 1) * self.fit.spectator_bound**(powers - 2) / 2
        return quadratic + continued @ factors

    def confinement_on_effective_radial_domain(self):
        lo, hi = self.fit.radial_bounds
        limits = lo - self.radial_extension, hi + self.radial_extension
        minima = [self.confinement_in_segment(bounds, polynomials) for bounds, polynomials in self.fit.polynomial_segments(limits)]
        finite = [value for value in minima if value is not None]
        return min(finite) if finite else None

    def confinement_in_segment(self, limits, polynomials):
        q2 = polynomials[self.fit.powers.index(2)]
        positive_polynomials = [polynomials[self.fit.powers.index(power)] for power in self.continued_even_powers]

        def inside(values, bounds):
            return values.real[(np.abs(values.imag) < 1e-9) & (values.real > bounds[0]) & (values.real < bounds[1])]

        edges = np.unique(np.concatenate([limits] + [inside(poly.roots(), limits) for poly in positive_polynomials]))
        points = edges.tolist()
        for left, right in zip(edges[:-1], edges[1:]):
            midpoint = (left + right) / 2
            envelope = q2
            if self.oscillator.quartic_tail != "polynomial":
                for power, poly in zip(self.continued_even_powers, positive_polynomials, strict=True):
                    if poly(midpoint) > 0:
                        envelope += (power * (power - 1) * self.fit.spectator_bound**(power - 2) / 2) * poly
            points.extend(inside(envelope.deriv().roots(), (left, right)))
        points = np.asarray(points)
        if self.oscillator.quartic_tail == "polynomial":
            finite_mask = (np.logical_and.reduce([poly(points) <= 1e-10 for poly in positive_polynomials])
                           if positive_polynomials else np.ones(len(points), dtype=bool))
            returns = q2(points[finite_mask])
            return float(returns.min()) if returns.size else None
        asymptotic = q2(points)
        for power, poly in zip(self.continued_even_powers, positive_polynomials, strict=True):
            asymptotic += (power * (power - 1) * self.fit.spectator_bound**(power - 2) / 2) * np.maximum(poly(points), 0)
        return float(np.min(asymptotic))

    def resolve(self, q1, *, details=False):
        coefficients, derivatives = self.curves(q1)
        if np.any(self.confinement(coefficients) <= 0):
            raise ValueError(f"{self.fit.label} extrapolates to a nonpositive asymptotic curvature")
        matrices = self.oscillator.kinetic + np.einsum("qp,pij->qij", coefficients, self.oscillator.moments)
        energies, vectors = np.linalg.eigh(matrices)
        # The true ground functions are positive. The positive reference HO
        # ground fixes the phase consistently across independent q1 eigensolves.
        vectors[:, :, 0] *= np.where(vectors[:, :1, 0] < 0, -1.0, 1.0)
        jacobian = np.einsum("qp,pij->qij", derivatives, self.oscillator.moments)
        ground = vectors[:, :, 0]
        amplitudes = np.einsum("qki,qkl,ql->qi", vectors[:, :, 1:], jacobian, ground)
        gaps = energies[:, 1:] - energies[:, :1]
        born_huang_metric = np.sum((amplitudes / gaps)**2, axis=1)
        if not details:
            return energies[:, 0], born_huang_metric
        tails = 1 - np.einsum("qi,ij,qj->q", ground, self.oscillator.in_domain, ground)
        return energies, vectors, born_huang_metric, np.clip(tails, 0, 1)

    def audit(self, spectator):
        confinement = self.confinement_on_effective_radial_domain()
        if confinement is not None and confinement <= 0:
            raise ValueError(f"{self.fit.label} has a nonpositive asymptotic curvature in its effective q1 domain")
        nodes = np.linspace(self.fit.radial_bounds[0] - 4, self.fit.radial_bounds[1] + 4, 161)
        coefficients, _ = self.curves(nodes)
        energies, _, metric, tails = self.resolve(nodes, details=True)
        measured = (nodes >= self.fit.radial_bounds[0]) & (nodes <= self.fit.radial_bounds[1])
        narrow = (nodes >= max(0.0, self.fit.radial_bounds[0])) & (nodes <= min(3.0, self.fit.radial_bounds[1]))
        bound = self.fit.spectator_bound
        s = np.linspace(-3 * bound, 3 * bound, 601)
        potential = coefficients @ potential_powers(s, self.fit.powers, bound, self.oscillator.extension,
                                                     self.oscillator.quartic_tail)
        result = {
            "spectator_model": "isotropic doublet, m=0" if self.oscillator.isotropic_doublet else "one-dimensional cut",
            "spectator_bound_abs_qj": bound,
            "positive_quartic_tail": self.oscillator.quartic_tail,
            "min_quadratic_confinement_coefficient_cm_inverse": confinement,
            "min_fitted_origin_coefficient_c2_cm_inverse": float(coefficients[:, self.fit.powers.index(2)].min()),
            "min_local_excitation_in_basis_sector_in_training_q1_cm_inverse": float(np.diff(energies[measured, :2], axis=1).min()),
            "local_first_excitation_at_q1_1_5_cm_inverse": float(self.local_gaps()[0]),
            "local_first_radially_allowed_excitation_at_q1_1_5_cm_inverse": float(self.local_gaps()[self.breathing_active_step - 1]),
            "max_abs_sampled_potential_minimum_qj": float(np.abs(s[np.argmin(potential, axis=1)]).max()),
            "max_spectator_ground_probability_outside_cut_in_training_q1": float(tails[measured].max()),
            "max_spectator_ground_probability_outside_cut_in_q1_0_to_3": float(tails[narrow].max()),
            "max_born_huang_metric_in_training_q1": float(metric[measured].max()),
        }
        external = np.abs(spectator.qj) > bound
        if external.any():
            q1, qj = spectator.q1[external], spectator.qj[external]
            cut_coefficients, _ = self.curves(q1)
            basis = potential_powers(qj, self.fit.powers, bound, self.oscillator.extension, self.oscillator.quartic_tail)
            residuals = np.einsum("qi,iq->q", cut_coefficients, basis) - spectator.energies[external]
            result["spectator_tail_holdout_rms_cm_inverse"] = float(np.sqrt(np.mean(residuals**2)))
            result["spectator_tail_holdout_residuals_by_q1_qj_cm_inverse"] = np.column_stack((q1, qj, residuals)).tolist()
        return result

    def local_gaps(self):
        energy = self.resolve(np.array([1.5]), details=True)[0][0, :3]
        return energy[1:] - energy[0]


def confining_contraction(fit, levels, spectator_extension, radial_extension, *, isotropic, tail):
    oscillator = spectator_oscillator(fit, levels, spectator_extension, isotropic_doublets=isotropic, quartic_tail=tail)
    contraction = Contraction(fit, oscillator, radial_extension)
    confinement = contraction.confinement_on_effective_radial_domain()
    if tail == "matched" and confinement is not None and confinement <= 0 and contraction.continued_even_powers:
        # A locally negative quadratic may be stabilized by the measured positive
        # high-order power. Keep it polynomial if its bounded-curvature tail fails.
        oscillator = spectator_oscillator(fit, levels, spectator_extension, isotropic_doublets=isotropic, quartic_tail="polynomial")
        contraction = Contraction(fit, oscillator, radial_extension)
    return contraction


@dataclass(frozen=True)
class MixedFit:
    coefficients: np.ndarray
    bounds: tuple[float, float]
    report: dict
    odd_quartic: bool = False

    def curves(self, q1, extension):
        clipped = np.clip(q1, *self.bounds)
        outward = np.tanh((q1 - clipped) / extension)
        q = clipped + extension * outward
        linear = self.coefficients[1]
        return self.coefficients[0] + q[:, None] * linear, (1 - outward[:, None]**2) * linear


def read_b2g_mixed(path, odd_quartic=False):
    data = []
    for file in sorted(path.glob("nu4_nu5_*.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        cross = row["delta_s1_hartree"] - row["axis_j_delta_s1_hartree"] - row["axis_k_delta_s1_hartree"]
        if not row["converged"] or abs(cross - row["mixed_s1_hartree"]) > 1e-10:
            raise ValueError(f"invalid matched mixed energy in {file}")
        data.append([row["q1"], row["qj"], row["qk"], cross * CM_PER_HARTREE])
    if not data:
        raise ValueError("no direct ν4×ν5 mixed cuts")
    q1, qj, qk, cross = np.asarray(data).T
    train = (qj == .85) & (np.abs(qk) == .85)
    if not {0.0, 1.5, 3.0} <= set(q1[train]):
        raise ValueError("require mixed ν4×ν5 training slices at q1=0, 1.5 and 3")
    xy = qj * qk
    design = np.column_stack((xy, xy**2, q1 * xy, q1 * xy**2))
    training = train
    terms = ["q4*q5", "q4^2*q5^2"]
    if odd_quartic:
        have = set(zip(qj[~train], qk[~train], strict=True))
        if not {(1.7, -.85), (1.7, .85), (.85, -1.7), (.85, 1.7)} <= have:
            raise ValueError("odd quartic sensitivity needs both signed wide-amplitude mixed cuts")
        training = np.ones(len(q1), dtype=bool)
        design = np.column_stack((design, qj**2 * xy, qk**2 * xy))
        terms += ["q4^3*q5", "q4*q5^3"]
    values, _, rank, _ = np.linalg.lstsq(design[training], cross[training], rcond=None)
    if rank != design.shape[1]:
        raise ValueError("mixed ν4×ν5 cut does not resolve odd and even coefficients")
    residuals = design @ values - cross
    coefficients = (np.array([[*values[:2], *values[4:]], [*values[2:4], 0.0, 0.0]]) if odd_quartic
                    else values.reshape(2, 2))
    held_out = ~training
    return MixedFit(coefficients, (float(q1[train].min()), float(q1[train].max())), {
        "terms_in_order": terms,
        "coefficients_cm_inverse_by_q1_power_and_mixed_term": coefficients.tolist(),
        "training_points": int(training.sum()),
        "training_rms_cm_inverse": float(np.sqrt(np.mean(residuals[training]**2))),
        "holdout_residuals_by_q1_q4_q5_cm_inverse": np.column_stack((q1[held_out], qj[held_out], qk[held_out], residuals[held_out])).tolist(),
    }, odd_quartic)


@dataclass(frozen=True)
class JointContraction(Contraction):
    left: Contraction
    right: Contraction
    mixed: MixedFit
    uncapped_bilinear: bool
    quadratic_determinant_lower_bound: float | None

    @property
    def replicated_cuts(self):
        return 1

    @property
    def breathing_active_step(self):
        return 1

    def curves(self, q1):
        left, left_prime = self.left.curves(q1)
        right, right_prime = self.right.curves(q1)
        mixed, mixed_prime = self.mixed.curves(q1, self.radial_extension)
        return np.column_stack((left, right, mixed)), np.column_stack((left_prime, right_prime, mixed_prime))

    def confinement(self, coefficients):
        # The cross products are bounded outside the measured individual cuts.
        # The separately verified diagonal terms supply the joint confinement.
        return np.ones(len(coefficients))

    def audit(self, spectator=None):
        nodes = np.linspace(*self.fit.radial_bounds, 61)
        eigenvalues, _, metric, tails = self.resolve(nodes, details=True)
        return {
            "spectator_model": "joint ν4/ν5 product basis in even total parity",
            "spectator_bounds_abs_qj": [self.left.fit.spectator_bound, self.right.fit.spectator_bound],
            "min_local_excitation_in_basis_sector_in_training_q1_cm_inverse": float(np.diff(eigenvalues[:, :2], axis=1).min()),
            "local_first_radially_allowed_excitation_at_q1_1_5_cm_inverse": float(self.local_gaps()[0]),
            "max_probability_outside_joint_spectator_cut": float(tails.max()),
            "max_born_huang_metric_in_training_q1": float(metric.max()),
            "mixed_fit": self.mixed.report,
            "bilinear_spectator_tail": "unbounded" if self.uncapped_bilinear else "saturated",
            "lower_bound_on_asymptotic_quadratic_determinant_cm_inverse_squared": self.quadratic_determinant_lower_bound,
        }


def make_b2g_contraction(left, right, mixed, levels, *, uncapped_bilinear=False):
    def primitive(item):
        return confining_contraction(item.fit, levels, item.oscillator.extension, item.radial_extension,
                                     isotropic=False, tail=item.oscillator.quartic_tail)

    left, right = primitive(left), primitive(right)
    determinant = None
    if uncapped_bilinear:
        limits = np.asarray(mixed.bounds) + np.array([-0.5, 0.5])
        max_bilinear = np.max(np.abs(mixed.coefficients[0, 0] + mixed.coefficients[1, 0] * limits))
        diagonal_minima = [item.confinement_on_effective_radial_domain() for item in (left, right)]
        if all(value is not None for value in diagonal_minima):
            determinant = float(4 * np.prod(diagonal_minima) - max_bilinear**2)
            if min(diagonal_minima) <= 0 or determinant <= 0:
                raise ValueError("uncapped mixed bilinear lacks provable positive asymptotic confinement")
    a, b = left.oscillator, right.oscillator
    identity = np.eye(levels)
    mask = (np.add.outer(np.arange(levels), np.arange(levels)).ravel() % 2) == 0
    index = np.ix_(mask, mask)
    kinetic = (np.kron(a.kinetic, identity) + np.kron(identity, b.kinetic))[index]
    moments = [np.kron(moment, identity)[index] for moment in a.moments]
    moments.extend(np.kron(identity, moment)[index] for moment in b.moments)
    for order in (2 if uncapped_bilinear else 0, 1):
        moments.append(np.kron(a.cross_moments[order], b.cross_moments[order])[index])
    if mixed.odd_quartic:
        moments.extend((np.kron(a.cross_moments[3], b.cross_moments[0])[index],
                        np.kron(a.cross_moments[0], b.cross_moments[3])[index]))
    joint = SpectatorOscillator(kinetic, np.array(moments), np.kron(a.in_domain, b.in_domain)[index], False, a.extension, "joint")
    fit = replace(left.fit, label="nu4_nu5", multiplicity=2)
    return JointContraction(fit, joint, 0.5, left, right, mixed, uncapped_bilinear, determinant)


@dataclass(frozen=True)
class ContractedRadial:
    base: RadialCorrection
    contractions: tuple[Contraction, ...]
    active_q1_frequency: float
    born_huang: bool

    @property
    def limits(self):
        return self.base.limits

    def harmonic_reference(self, surface):
        return self.base.harmonic_reference(surface)

    def shifts(self, q1):
        q1 = np.atleast_1d(q1)
        shifts = []
        for contraction in self.contractions:
            energies, metric = contraction.resolve(np.r_[q1, 0.0])
            correction = energies - energies[-1]
            if self.born_huang:
                correction += 0.5 * self.active_q1_frequency * (metric - metric[-1])
            shifts.append(correction[:-1] * contraction.replicated_cuts)
        return np.asarray(shifts)

    def points(self, q1):
        return self.base.points(q1) + self.shifts(q1).sum(axis=0)


def replay_baseline(scan, baseline):
    surfaces, base_degree, _ = base_surface_fit(scan)
    if base_degree != baseline["base_total_degree"]:
        raise ValueError("the TDA scan does not reproduce the retained baseline fit degree")
    transverse = baseline["transverse_correction_fit"]
    if transverse:
        surfaces = with_transverse_correction(surfaces, np.asarray(transverse["terms_exponents_in_q1_q6_q8_order"]),
                                              np.asarray(transverse["coefficients_cm_inverse_by_term_and_surface"]))
    fit = baseline["radial_correction_fits"][f"degree_{baseline['primary_radial_correction_degree']}"]
    coefficients = fit["coefficients_cm_inverse_by_surface_and_q1_power_starting_at_one"]
    corrections = tuple(RadialCorrection(np.asarray(item), tuple(fit["training_domain_q1"]), fit["q1_bounded_extension"])
                        for item in coefficients)
    return surfaces, corrections, baseline["vertical_corrected_model_gap_cm_inverse"]


def active_s1(surfaces, frequencies, radial, size, order, levels):
    excited = surfaces[1]
    basis = HarmonicBasis(frequencies, excited.lengths, size, order, "capped")
    harmonic = radial.harmonic_reference(excited)
    harmonic_ground = solve(basis.kinetic + basis.field(harmonic), 1)[1][:, 0]
    refs = basis.rotated_references(harmonic_ground, *harmonic_parameters(harmonic, frequencies))
    energies, vectors, _ = solve(basis.kinetic + basis.field(excited) + basis.radial_correction(radial), levels)
    projections = (refs[:, [2, 1]].T @ vectors)**2
    _, chosen = linear_sum_assignment(-projections)
    return basis, energies, vectors, np.r_[0, chosen]


def radial_density_matrices(basis, vector, *, derivative=False):
    transverse = basis.quantum[:, 1:]
    spectator_blocks, block_indices = np.unique(transverse, axis=0, return_inverse=True)
    expansion = np.zeros((basis.nmax + 1, len(spectator_blocks), vector.shape[1]))
    expansion[basis.quantum[:, 0], block_indices] = vector
    weighted = np.einsum("qn,nia->qia", basis.quadrature_basis, expansion)
    density = np.einsum("qia,qib->qab", weighted, weighted)
    if not derivative:
        return density
    orders = np.arange(basis.nmax + 1)
    previous = np.column_stack((np.zeros(len(basis.nodes)), basis.quadrature_basis[:, :-1]))
    slopes = np.sqrt(2 * orders) * previous - basis.nodes[:, None] * basis.quadrature_basis
    weighted_slopes = np.einsum("qn,nia->qia", slopes, expansion)
    return density, np.einsum("qia,qib->qab", weighted_slopes, weighted)


def explicit_coupling(contractions, basis, energies, vectors, chosen, active_limits):
    reports = {}
    for contraction in contractions:
        local_energies, local_vectors, _, _ = contraction.resolve(basis.nodes, details=True)
        coefficients, _ = contraction.curves(basis.nodes)
        spectator_levels = contraction.oscillator.kinetic.shape[0]
        versions = {}
        for retained in active_limits if contraction.fit.label == "nu16" else active_limits[:2]:
            radial = radial_density_matrices(basis, vectors[:, :retained])
            matrix = np.kron(np.eye(retained), contraction.oscillator.kinetic)
            for values, moment in zip(coefficients.T, contraction.oscillator.moments, strict=True):
                matrix += np.kron(np.einsum("q,qab->ab", values, radial), moment)
            reference_shift = np.einsum("q,qab->ab", local_energies[:, 0], radial)
            matrix -= np.kron(reference_shift, np.eye(spectator_levels))
            matrix[np.diag_indices_from(matrix)] += np.repeat(energies[:retained], spectator_levels)
            coupled_energies, coupled_vectors, residual = solve(matrix, min(60, len(matrix)))
            projected = np.einsum("qab,qs->bas", radial[:, :, chosen], local_vectors[:, :, 0])
            weights = (projected.reshape(len(chosen), -1) @ coupled_vectors)**2
            _, assigned = linear_sum_assignment(-weights)
            terms = coupled_energies[assigned[1:]] - coupled_energies[assigned[0]]
            versions[str(retained)] = {
                "assigned_levels_from_vibrational_ground": assigned.tolist(),
                "adiabatic_channel_overlap_weights": weights[np.arange(len(chosen)), assigned].tolist(),
                "term_values_cm_inverse": terms.tolist(),
                "spacing_cm_inverse": float(np.diff(terms).item()),
                "spacing_change_from_adiabatic_cm_inverse": float(np.diff(terms).item() - energies[chosen[2]] + energies[chosen[1]]),
                "max_nuclear_eigenpair_residual_cm_inverse": residual,
            }
        reports[contraction.fit.label] = {
            "explicit_components": 2 if contraction.oscillator.isotropic_doublet else 1,
            "oscillator_sector": "isotropic m=0" if contraction.oscillator.isotropic_doublet else "one-dimensional",
            "active_level_limits": versions,
        }
    return reports


def local_coupling(contraction, q1, max_channel):
    eigenvalues, frame, metric, _ = contraction.resolve(q1, details=True)
    # Smooth each excited-state sign by adjacent overlaps. χ0 has already
    # been fixed by a positive overlap with the reference oscillator ground.
    overlaps = np.einsum("qij,qij->qj", frame[:-1], frame[1:])
    frame[1:] *= np.cumprod(np.where(overlaps < 0, -1.0, 1.0), axis=0)[:, None, :]
    _, derivatives = contraction.curves(q1)
    perturbation = np.einsum("qp,pij->qij", derivatives, contraction.oscillator.moments)
    restricted = frame[:, :, :max_channel + 1]
    gaps = eigenvalues[:, None, :max_channel + 1] - eigenvalues[:, :, None]
    coupling = np.einsum("qki,qkl,qlj->qij", frame, perturbation, restricted)
    np.divide(coupling, gaps, out=coupling, where=gaps != 0)
    diag = np.arange(max_channel + 1)
    coupling[:, diag, diag] = 0
    derivative_metric = np.einsum("qki,qkj->qij", coupling, coupling)
    return eigenvalues[:, :max_channel + 1] - eigenvalues[:, :1], coupling[:, :max_channel + 1], derivative_metric, metric


def collective_absorption(gap, s0_energy, eigenvalues, amplitude, active_weights):
    origin = float(gap + eigenvalues[0] - s0_energy)
    terms = eigenvalues - eigenvalues[0]
    photon = origin + terms
    strength = np.sum(amplitude**2, axis=1)
    spectrum = np.column_stack((terms, terms, photon, strength, strength * photon / origin, active_weights.T))
    return origin, {
        "spectrum_column_order": ["vibrational_energy_cm_inverse", "offset_from_calculated_origin_cm_inverse",
                                  "photon_cm_inverse", "dipole_strength_au2", "area_in_origin_scaled_units",
                                  "projection_weight_nu6", "projection_weight_nu1+nu6"],
        "spectrum_all_retained_levels_in_increasing_photon_order": spectrum.tolist(),
        "projection_definition": "Overlap with the assigned adiabatic active state in the common spectator ground channel.",
        "dipole_definition": "Project the original TDA active-coordinate dipole into the adiabatic ground spectator channel; its coupling to excited channels redistributes optical strength.",
    }


def simultaneous_coupling(contractions, candidates, basis, energies, vectors, chosen, channel_sizes, *, optical):
    """Solve the space of at most one excited spectator group at a time.

    In the q1-local spectator eigenbasis, potential terms are diagonal.
    The first-derivative and metric matrices follow by differentiating the
    local eigenvalue problem. The kinetic bilinear form needs only first
    derivatives and includes cross terms between singly excited groups.
    """
    local = {item.fit.label: local_coupling(item, basis.nodes, channel_sizes[-1][1]) for item in contractions}
    ground_metric = sum(values[-1] for values in local.values())
    scale = basis.frequencies[0] / 2
    zero_point_metric = scale * sum(item.resolve(np.array([0.0]))[1].item() for item in contractions)
    vertical_gap, initial_energy, active_dipoles = optical
    reports = {}
    for retained, max_channel in channel_sizes:
        density, slope_density = radial_density_matrices(basis, vectors[:, :retained], derivative=True)
        antisymmetric = slope_density - slope_density.transpose(0, 2, 1)

        def integrate(values, matrix=density):
            return np.einsum("q,qab->ab", values, matrix)

        channels = [(None, 0)]
        for item in candidates:
            label = item.fit.label
            step = item.breathing_active_step
            channels.extend((label, order) for order in range(step, max_channel + 1, step))
        shape = len(channels) * retained
        matrix = np.zeros((shape, shape))
        ground = np.diag(energies[:retained]) + scale * integrate(ground_metric)
        for i, (left_label, left) in enumerate(channels):
            rows = slice(i * retained, (i + 1) * retained)
            for j, (right_label, right) in enumerate(channels[:i + 1]):
                columns = slice(j * retained, (j + 1) * retained)
                if i == j == 0:
                    block = ground
                elif j == 0 or left_label == right_label:
                    local_gap, connection, local_metric, local_ground = local[left_label]
                    block = scale * (integrate(local_metric[:, left, right]) + integrate(connection[:, left, right], antisymmetric))
                    if i == j:
                        block += ground + integrate(local_gap[:, left] - scale * local_ground)
                else:
                    cross = local[left_label][1][:, left, 0] * local[right_label][1][:, right, 0]
                    block = 2 * scale * integrate(cross)
                matrix[rows, columns] = block
                if i != j:
                    matrix[columns, rows] = block.T
        coupled_energies, coupled_vectors, residual = solve(matrix, min(60, len(matrix)))
        overlaps = coupled_vectors[chosen]**2
        _, assigned = linear_sum_assignment(-overlaps)
        terms = coupled_energies[assigned[1:]] - coupled_energies[assigned[0]]
        dipoles = coupled_vectors[:retained].T @ active_dipoles[:retained]
        origin, absorption = collective_absorption(vertical_gap, initial_energy, coupled_energies - zero_point_metric,
                                                  dipoles, overlaps[1:])
        channel_probabilities = (coupled_vectors.reshape(len(channels), retained, -1)**2).sum(axis=1)
        profiles = []
        for index in np.flatnonzero(overlaps[2] >= 0.01):
            weights = defaultdict(float)
            for (label, _), probability in zip(channels[1:], channel_probabilities[1:, index], strict=True):
                weights[label] += float(probability)
            profiles.append({
                "level_index": int(index),
                "same_level_as_selected_upper_band": bool(index == assigned[2]),
                "term_value_cm_inverse": float(coupled_energies[index] - coupled_energies[0]),
                "spacing_from_selected_nu6_cm_inverse": float(coupled_energies[index] - coupled_energies[assigned[1]]),
                "adiabatic_nu1_plus_nu6_overlap_weight": float(overlaps[2, index]),
                "dipole_strength_au2": float(dipoles[index] @ dipoles[index]),
                "excited_spectator_channel_probabilities_by_group": dict(weights),
            })
        reports[f"{retained}_active_max_{max_channel}_local"] = {
            "channels_in_basis_order": channels,
            "assigned_level_indices": assigned.tolist(),
            "adiabatic_channel_overlap_weights": overlaps[np.arange(len(chosen)), assigned].tolist(),
            "term_values_cm_inverse": terms.tolist(),
            "spacing_cm_inverse": float(np.diff(terms).item()),
            "spacing_change_from_adiabatic_cm_inverse": float(np.diff(terms).item() - energies[chosen[2]] + energies[chosen[1]]),
            "max_nuclear_eigenpair_residual_cm_inverse": residual,
            "calculated_active_space_origin_cm_inverse": origin,
            "absorption": absorption,
            "upper_band_and_borrowed_projection_lines": profiles,
        }
    return reports


def widened_cuts(spectators, amplitude, maximum_power):
    widened = []
    for mode in spectators:
        q1_nodes = np.unique(mode.q1[mode.train])
        expanded = np.isin(mode.q1, q1_nodes) & np.isclose(mode.qj, amplitude)
        # Extend only balanced, even cuts with an outer training point on each q1 slice.
        if mode.powers == (2, 4) and np.all(np.isin(q1_nodes, mode.q1[expanded])) and amplitude > np.max(mode.qj[mode.train]):
            powers = (2, 4, 6) if maximum_power == 6 else (2, 4)
            widened.append(replace(mode, train=mode.train | expanded, powers=powers))
        else:
            widened.append(mode)
    return widened


def run(scan, baseline, key, spectators, measured_spacing, spectator_levels, coupled_groups, *, prefer_wide=False,
        prefer_spline=False, tail_amplitude=2.5, mixed_fit=None, joint_levels=16, max_active_levels=120,
        uncapped_bilinear=False):
    surfaces, base_corrections, gap = replay_baseline(scan, baseline)
    frequencies = scan["frequencies_cm1"]
    retained = baseline["models"][key]
    size, order = retained["max_total_harmonic_quanta"], retained["quadrature_order_per_axis"]

    def spectrum(base, correction, max_quanta=size, quadrature=order):
        result = calculation(surfaces, frequencies, gap, max_quanta, quadrature, "capped", "anharmonic",
                             radial_corrections=(base, correction))
        result["absorption_progression"] = absorption_progression(result)
        return result

    baseline_spectrum = spectrum(*base_corrections)
    baseline_spacing = baseline_spectrum["absorption_progression"]["spacing_cm_inverse"]
    if abs(baseline_spacing - retained["absorption_progression"]["spacing_cm_inverse"]) > 1e-5:
        raise ValueError(f"the replay spacing {baseline_spacing:g} disagrees with the retained baseline")
    fit_inputs = {"radial_degree_2_spectator_degree_4": (2, 4, spectators),
                  "radial_degree_3_spectator_degree_4": (3, 4, spectators),
                  "radial_degree_2_spectator_degree_2": (2, 2, spectators),
                  "spline_radial": ("spline", 4, spectators)}
    wide_quartic = widened_cuts(spectators, tail_amplitude, 4)
    has_wide = any(np.any(before.train != after.train) for before, after in zip(spectators, wide_quartic, strict=True))
    if has_wide:
        fit_inputs["tail_refit_quartic"] = (2, 4, wide_quartic)
        fit_inputs["tail_refit_sextic"] = (2, 6, widened_cuts(spectators, tail_amplitude, 6))
        fit_inputs["tail_refit_sextic_spline_radial"] = ("spline", 6, widened_cuts(spectators, tail_amplitude, 6))
    fits = {label: [fit_cut(mode, degree, power) for mode in modes] for label, (degree, power, modes) in fit_inputs.items()}
    models = {"baseline_replay": baseline_spectrum}
    fit_audits = {}
    contracts = {}
    variants = [
        ("adiabatic", 2, 4, 0.5, 0.25, spectator_levels, False, size, order),
        ("cubic_radial", 3, 4, 0.5, 0.25, spectator_levels, False, size, order),
        ("spline_radial", "spline", 4, 0.5, 0.25, spectator_levels, False, size, order),
        ("quadratic_spectator", 2, 2, 0.5, 0.25, spectator_levels, False, size, order),
        ("hard_tails", 2, 4, 0.0, 0.0, spectator_levels, False, size, order),
        ("born_huang", 2, 4, 0.5, 0.25, spectator_levels, True, size, order),
        ("separable_doublets", 2, 4, 0.5, 0.25, spectator_levels, False, size, order),
        ("positive_quartic_unbounded", 2, 4, 0.5, 0.25, spectator_levels, False, size, order),
        ("larger_spectator_basis", 2, 4, 0.5, 0.25, spectator_levels + 8, False, size, order),
        ("larger_active_basis", 2, 4, 0.5, 0.25, spectator_levels, False, size + 2, order + 8),
    ]
    if has_wide:
        variants.extend((
            ("tail_refit_quartic", 2, 4, 0.5, 0.25, spectator_levels, False, size, order),
            ("tail_refit_sextic", 2, 6, 0.5, 0.25, spectator_levels, False, size, order),
            ("tail_refit_sextic_born_huang", 2, 6, 0.5, 0.25, spectator_levels, True, size, order),
            ("tail_refit_sextic_larger_spectator_basis", 2, 6, 0.5, 0.25, spectator_levels + 8, False, size, order),
            ("tail_refit_sextic_spline_radial", "spline", 6, 0.5, 0.25, spectator_levels, False, size, order),
        ))
    for name, degree, power, radial_extension, spectator_extension, levels, bh, max_quanta, quadrature in variants:
        if degree == "spline":
            fit_key = name
        elif name.startswith("tail_refit"):
            fit_key = "tail_refit_sextic" if power == 6 else "tail_refit_quartic"
        else:
            fit_key = f"radial_degree_{degree}_spectator_degree_{power}"
        try:
            tail = "polynomial" if name == "positive_quartic_unbounded" else "matched"
            contractions = tuple(confining_contraction(fit, levels, spectator_extension, radial_extension,
                                                       isotropic=name != "separable_doublets", tail=tail)
                                 for fit in fits[fit_key])
            contracted = ContractedRadial(base_corrections[1], contractions, frequencies[0], bh)
            audit = {item.fit.label: item.audit(mode) for item, mode in zip(contractions, spectators, strict=True)}
            model = spectrum(base_corrections[0], contracted, max_quanta, quadrature)
        except ValueError as failure:
            models[name] = {"status": "unavailable", "reason": str(failure)}
        else:
            fit_audits[name], models[name], contracts[name] = audit, model, contracted

    attribution = {}
    coupled = {}
    collective = {}
    primary_key = ("tail_refit_sextic_spline_radial" if prefer_spline else "tail_refit_sextic") if prefer_wide else (
        "spline_radial" if prefer_spline else "adiabatic")
    primary_axis_key = primary_key
    if (prefer_wide or prefer_spline or mixed_fit is not None) and primary_key not in contracts:
        raise ValueError(f"requested primary fit {primary_key} is unavailable: {models.get(primary_key)}")
    if mixed_fit is not None:
        separate = contracts[primary_key]
        modes = {item.fit.label: item for item in separate.contractions}
        rest = [item for item in separate.contractions if item.fit.label not in ("nu4", "nu5")]
        for name, levels, uncapped in (("joint_b2g_smaller_basis", joint_levels - 4, uncapped_bilinear),
                                       ("joint_b2g", joint_levels, uncapped_bilinear),
                                       ("joint_b2g_alternative_bilinear_tail", joint_levels, not uncapped_bilinear)):
            joint = make_b2g_contraction(modes["nu4"], modes["nu5"], mixed_fit, levels, uncapped_bilinear=uncapped)
            contracted = replace(separate, contractions=tuple([*rest, joint]))
            fit_audits[name] = {**{label: fit_audits[primary_key][label] for label in (item.fit.label for item in rest)},
                                joint.fit.label: joint.audit()}
            models[name] = spectrum(base_corrections[0], contracted)
            contracts[name] = contracted
        primary_key = "joint_b2g"
    if primary_key in contracts:
        primary = contracts[primary_key]
        basis, _, vectors, chosen = active_s1(surfaces, frequencies, base_corrections[1], size, order, 40)
        density = radial_density_matrices(basis, vectors[:, chosen]).diagonal(axis1=1, axis2=2)
        shifts = primary.shifts(basis.nodes) @ density
        for item, shift in zip(primary.contractions, shifts, strict=True):
            attribution[item.fit.label] = {
                "coordinate_multiplicity": item.fit.multiplicity,
                "relative_term_changes_cm_inverse": (shift[1:] - shift[0]).tolist(),
                "spacing_change_cm_inverse": float(np.diff(shift[1:]).item()),
            }
        basis, energies, vectors, chosen = active_s1(surfaces, frequencies, primary, size, order, max_active_levels)
        state_density = radial_density_matrices(basis, vectors[:, chosen]).diagonal(axis1=1, axis2=2)
        for item in primary.contractions:
            _, _, _, tails = item.resolve(basis.nodes, details=True)
            fit_audits[primary_key][item.fit.label]["spectator_tail_probability_for_s1_ground_nu6_and_nu1_plus_nu6"] = (tails @ state_density).tolist()
        for name, contract in contracts.items():
            auxiliary_basis, _, auxiliary_vectors, auxiliary_chosen = active_s1(
                surfaces, frequencies, contract, models[name]["max_total_harmonic_quanta"],
                models[name]["quadrature_order_per_axis"], 12)
            models[name]["probability_outside_spectator_q1_training_domain"] = auxiliary_basis.axis_coverage(
                auxiliary_vectors[:, auxiliary_chosen], 0, spectators_domain(spectators[0])).tolist()
        if coupled_groups:
            candidates = {item.fit.label: item for item in primary.contractions}
            if coupled_groups == ["auto"]:
                softness = sorted(candidates, key=lambda key: fit_audits[primary_key][key]["min_local_excitation_in_basis_sector_in_training_q1_cm_inverse"])
                contribution = sorted(candidates, key=lambda key: abs(attribution[key]["spacing_change_cm_inverse"]), reverse=True)
                resonance = sorted(candidates, key=lambda key: abs(fit_audits[primary_key][key]["local_first_radially_allowed_excitation_at_q1_1_5_cm_inverse"] - baseline_spacing))
                coupled_groups = list(dict.fromkeys((*resonance[:3], *softness[:2], *contribution[:2])))
            elif mixed_fit is not None:
                coupled_groups = list(dict.fromkeys("nu4_nu5" if key in ("nu4", "nu5") else key for key in coupled_groups))
            selected = [candidates[key] for key in coupled_groups]
            active_limits = tuple(dict.fromkeys((28, 48, max_active_levels)))
            coupled = explicit_coupling([item for item in selected if not isinstance(item, JointContraction)],
                                        basis, energies, vectors, chosen, active_limits)
            ground_energy, ground_vector, _ = solve(basis.kinetic + basis.field(surfaces[0]) +
                                                    basis.radial_correction(base_corrections[0]), 1)
            initial_dipoles = np.column_stack([mu @ ground_vector[:, 0] for mu in basis.field(surfaces[2])])
            optical = gap, ground_energy.item(), vectors.T @ initial_dipoles
            collective_levels = tuple(dict.fromkeys(((28, 2), (48, 2), (48, 4), (48, 6),
                                                     (min(80, max_active_levels), 6), (max_active_levels, 6))))
            collective = simultaneous_coupling(primary.contractions, selected, basis, energies, vectors, chosen,
                                               collective_levels, optical=optical)
            all_groups = simultaneous_coupling(primary.contractions, primary.contractions, basis, energies, vectors, chosen,
                                              ((48, 2),), optical=optical)
            collective["all_groups_48_active_max_2_local"] = all_groups["48_active_max_2_local"]
            for version, model in collective.items():
                models[f"collective_{version}"] = {
                    "calculated_active_space_origin_cm_inverse": model["calculated_active_space_origin_cm_inverse"],
                    "absorption": model["absorption"],
                    "absorption_progression": absorption_progression(model),
                    "upper_band_and_borrowed_projection_lines": model["upper_band_and_borrowed_projection_lines"],
                }

    summaries = {}
    for name, model in models.items():
        if "absorption_progression" in model:
            summary = model["absorption_progression"]
            spacing = summary["spacing_cm_inverse"]
            summaries[name] = {
                **summary,
                "spacing_change_from_baseline_cm_inverse": spacing - baseline_spacing,
                "spacing_minus_experiment_cm_inverse": None if measured_spacing is None else spacing - measured_spacing,
            }
    radial_holdout_quality = {}
    for label, values in fits.items():
        retained = [row[2] for cut in values for row in cut.report.get("radial_holdout", {}).get("residuals_by_q1_qj_cm_inverse", [])]
        if retained:
            radial_holdout_quality[label] = {
                "points": len(retained),
                "rms_cm_inverse": float(np.sqrt(np.mean(np.square(retained)))),
                "max_abs_cm_inverse": float(np.max(np.abs(retained))),
            }
    return {
        "direct_target_electronic_method": baseline["direct_target_electronic_method"],
        "s1_spectator_coordinates_represented": sum(spectator.multiplicity for spectator in spectators),
        "independent_spectator_cuts": len(spectators),
        "spectator_q1_training_domain": spectators_domain(spectators[0]),
        "spectator_q1_training_nodes": np.unique(spectators[0].q1[spectators[0].train]).tolist(),
        "electronic_quality_by_group": {item.label: item.electronic_diagnostics for item in spectators},
        "coordinate_definition": "q_i = sqrt(omega_i) Q_i, with analytic S0 frequencies omega_i in Hartree",
        "transition_dipole_surface": "Original phase-tracked TDA fit in three active coordinates",
        "origin_convention": "Subtract each contracted S1 ground eigenenergy (and each Born–Huang term when included) at q1=0. Calculated origins are three-mode, with the spectator-induced radial relaxation correction but without a full S1–S0 spectator ZPE difference.",
        "primary_spectator_model": "Complete each full E doublet into its isotropic radial form and solve its m=0 sector. Continue positive higher powers through their boundary value, slope and curvature, using polynomial continuation when needed for confinement. Saturate negative higher powers and cubic terms smoothly.",
        "primary_single_axis_fit": primary_axis_key,
        "spectrum_intensity_convention": "Use the active-space TDA dipole in the common ground spectator channel. Excited channels borrow intensity through nuclear coupling. Spectator Franck–Condon factors and spectator-dependent transition dipoles are absent.",
        "progression_labels": ["nu6", "nu1+nu6"],
        "experimental_spacing_cm_inverse": measured_spacing,
        "spacing_summary": summaries,
        "radial_holdout_quality_by_fit": radial_holdout_quality,
        "polynomial_fits": {label: {fit.label: fit.report for fit in values} for label, values in fits.items()},
        "spectator_confinement_and_support_audits": fit_audits,
        "group_attribution_first_order_in_uncontracted_active_states": attribution,
        "mixed_b2g_joint_electronic_fit": None if mixed_fit is None else mixed_fit.report,
        "model_used_for_attribution_and_coupled_probes": primary_key,
        "explicit_single_spectator_group_probes": coupled,
        "simultaneous_single_excitation_group_channel_probe": collective,
        "models": models,
    }


def spectators_domain(spectator):
    training = spectator.q1[spectator.train]
    return float(training.min()), float(training.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-key", default="large_basis_tddft")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--tail-points", type=Path)
    parser.add_argument("--training-q1", type=float, nargs="+")
    parser.add_argument("--train-tail-amplitude", type=float, default=2.5)
    parser.add_argument("--prefer-wide", action="store_true")
    parser.add_argument("--prefer-spline", action="store_true")
    parser.add_argument("--mixed-points", type=Path)
    parser.add_argument("--uncapped-bilinear", action="store_true")
    parser.add_argument("--mixed-odd-quartic", action="store_true")
    parser.add_argument("--joint-levels", type=int, default=16)
    parser.add_argument("--max-active-levels", type=int, default=120)
    parser.add_argument("--holdout-q1", type=float, nargs="*", default=(0.75, 2.25))
    parser.add_argument("--experimental-spacing", type=float)
    parser.add_argument("--spectator-levels", type=int, default=20)
    parser.add_argument("--coupled-groups", nargs="+", default=["auto"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.spectator_levels < 10:
        parser.error("--spectator-levels must be at least 10")
    if args.joint_levels < 12 or args.max_active_levels < 48:
        parser.error("require --joint-levels >= 12 and --max-active-levels >= 48")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    spectators = read_spectators(args.metadata, args.points, args.holdout_q1,
                                 method=baseline["direct_target_electronic_method"],
                                 tail_points_path=args.tail_points, training_q1=args.training_q1)
    coupled = [] if args.coupled_groups == ["none"] else args.coupled_groups
    valid_groups = {item.label for item in spectators} | ({"nu4_nu5"} if args.mixed_points else set())
    if coupled != ["auto"] and not set(coupled) <= valid_groups:
        parser.error("specify auto, none or the exact group labels for --coupled-groups")
    result = run(read_scan(args.scan), baseline, args.baseline_key, spectators, args.experimental_spacing,
                 args.spectator_levels, coupled, prefer_wide=args.prefer_wide, prefer_spline=args.prefer_spline,
                 tail_amplitude=args.train_tail_amplitude,
                 mixed_fit=read_b2g_mixed(args.mixed_points, args.mixed_odd_quartic) if args.mixed_points else None,
                 joint_levels=args.joint_levels, max_active_levels=args.max_active_levels, uncapped_bilinear=args.uncapped_bilinear)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
