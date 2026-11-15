"""Solve the benzene ν1, ν6, ν8 Wilson cut from independent PySCF data.

The scan uses q_i = sqrt(omega_i) Q_i, where Q_i is mass weighted and the
analytic S0 frequencies omega_i in the coordinate definition are in Hartree.
The nuclear Hamiltonian uses cm^-1 and the same dimensionless q throughout.
"""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import roots_hermite


CM_PER_HARTREE = 219474.63136320
LABELS = ("nu1", "nu6", "nu8")
DYAD_LABELS = ("nu8", "nu1+nu6")
REFERENCE_LABELS = (*DYAD_LABELS, "nu6")
FERMI_TERM = (1, 1, 1)
SCAN_ROW_KEYS = ("q", "energies", "dipole_au", "positions_bohr", "tracked_root_overlap",
                 "td_residual_hartree", "scf_ao_commutator_norm_hartree")


class UnderdeterminedSurface(ValueError):
    """The scan coordinates cannot determine every requested surface coefficient."""


class UnstableQuadratic(ValueError):
    """A fitted quadratic has a nonpositive nuclear curvature."""


def read_scan(path):
    with np.load(path) as source:
        scan = dict(source)
    q = scan["q"]
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must contain one (ν1, ν6, ν8) point per row")
    expected = {"q": (len(q), 3), "energies": (len(q), 2),
                "dipole_au": (len(q), 3), "frequencies_cm1": (3,)}
    for key, shape in expected.items():
        if np.shape(scan[key]) != shape or not np.isfinite(scan[key]).all():
            raise ValueError(f"{key} must have shape {shape} and finite values")
    if np.any(scan["frequencies_cm1"] <= 0):
        raise ValueError("all three analytic S0 frequencies must be positive")
    return scan


def select_scan_rows(scan, mask):
    return {key: value[mask] if key in SCAN_ROW_KEYS else value for key, value in scan.items() if key != "holdout"}


def scan_quality(scan):
    geometry_keys = ("positions_bohr", "center_positions_bohr", "oscillator_vectors_bohr")
    quality = {}
    if all(key in scan for key in geometry_keys):
        positions, center, vectors = (scan[key] for key in geometry_keys)
        expected = center + np.einsum("ni,ijk->njk", scan["q"], vectors)
        if positions.shape != expected.shape or not np.isfinite(positions).all():
            raise ValueError("retained positions must contain finite geometry for each sample")
        error = np.abs(positions - expected).max()
        if error > 1e-10:
            raise ValueError("retained geometry does not match its listed dimensionless q coordinates")
        quality["max_geometry_reconstruction_error_bohr"] = float(error)
    for key, reduction in (("tracked_root_overlap", np.min), ("td_residual_hartree", np.max),
                           ("scf_ao_commutator_norm_hartree", np.max)):
        if key in scan:
            values = scan[key]
            if values.shape != (len(scan["q"]),) or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain one finite value per scan row")
            quality[f"{reduction.__name__}_{key}"] = float(reduction(values))
    return quality


def monomials(q, exponents):
    return np.prod(q[:, None, :] ** exponents[None, :, :], axis=-1)


def fit_residual(q, values, terms, lengths):
    matrix = monomials(q / lengths, terms)
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, values, rcond=None)
    if rank != len(terms):
        raise UnderdeterminedSurface(f"scan does not determine all polynomial terms ({rank}/{len(terms)})")
    scales = np.prod(lengths ** terms, axis=1)
    coefficients /= scales.reshape((-1,) + (1,) * (values.ndim - 1))
    return coefficients, float(singular_values[0] / singular_values[-1])


@dataclass(frozen=True)
class Surface:
    terms: np.ndarray
    values: np.ndarray
    lengths: np.ndarray
    linear_order: int

    def points(self, q):
        return monomials(np.atleast_2d(q), self.terms) @ self.values

    def harmonic(self):
        return self.select(self.terms.sum(axis=1) <= 2)

    def without_fermi(self):
        return self.select(np.any(self.terms != FERMI_TERM, axis=1))

    def select(self, mask):
        return Surface(self.terms[mask], self.values[mask], self.lengths, self.linear_order)

    def coefficient(self, term):
        match = np.all(self.terms == term, axis=1)
        return self.values[match][0] if match.any() else 0.0


@dataclass(frozen=True)
class RadialCorrection:
    """An origin-referenced energy correction along the breathing coordinate."""

    coefficients: np.ndarray
    limits: tuple[float, float]
    extension: float

    def points(self, q1):
        q1 = np.asarray(q1, dtype=float)
        clipped = np.clip(q1, *self.limits)
        if self.extension:
            clipped += self.extension * np.tanh((q1 - clipped) / self.extension)
        return np.polynomial.polynomial.polyval(clipped, np.r_[0.0, self.coefficients])

    def harmonic_reference(self, surface):
        terms = {tuple(term): value for term, value in zip(surface.terms, surface.values)
                 if term.sum() <= 2}
        for exponent, value in enumerate(self.coefficients[:2], 1):
            term = (exponent, 0, 0)
            terms[term] = terms.get(term, 0.0) + value
        return Surface(np.array(list(terms)), np.array(list(terms.values())), surface.lengths, 2)


def fit_surfaces(scan, degree, *, anchor_ground_quadratic=False):
    q = scan["q"]
    center = np.flatnonzero(np.all(q == 0, axis=1))
    if center.size != 1 or np.unique(q, axis=0).shape != q.shape:
        raise ValueError("the scan must have unique coordinates and one exact origin")
    lengths = np.minimum(q.max(axis=0), -q.min(axis=0))
    if np.any(lengths <= 0):
        raise ValueError("the electronic scan must cover positive and negative displacements")

    # A five-node axis cannot distinguish an unrestricted fifth power from
    # its lower powers. Mixed total-degree-five terms can be determined.
    limits = [min(degree, np.unique(axis).size - 1) for axis in q.T]
    terms = np.array([exponents for exponents in product(*(range(m + 1) for m in limits))
                      if sum(exponents) <= degree], dtype=int)
    order = terms.sum(axis=1)
    relative = (scan["energies"] - scan["energies"][center]) * CM_PER_HARTREE
    frequencies = scan["frequencies_cm1"]
    quadratic_terms = 2 * np.eye(3, dtype=int)
    # A scalar energy cannot contain a single E2g factor at any A1g distortion.
    # ν6 and ν8 have the same symmetry, so quadratic E2g cross terms are allowed.
    symmetry_allowed = terms[:, 1:].sum(axis=1) != 1
    if anchor_ground_quadratic:
        g_terms = terms[(order >= 3) & symmetry_allowed]
        g_base = 0.5 * monomials(q, quadratic_terms) @ frequencies
        g_values, g_cond = fit_residual(q, relative[:, 0] - g_base, g_terms, lengths)
        g_terms = np.vstack((quadratic_terms, g_terms))
        g_values = np.r_[frequencies / 2, g_values]
    else:
        g_terms = terms[(order >= 2) & symmetry_allowed]
        g_values, g_cond = fit_residual(q, relative[:, 0], g_terms, lengths)
    ground = Surface(g_terms, g_values, lengths, 2)

    e_allowed = (order > 0) & symmetry_allowed
    gradient = scan.get("s1_gradient_hartree_per_q")
    if gradient is not None and gradient.shape == (3,) and np.isnan(gradient).all():
        gradient = None
    if gradient is not None:
        if gradient.shape != (3,) or not np.isfinite(gradient).all() or np.max(np.abs(gradient[1:])) > 5e-5:
            raise ValueError("S1 gradient must be finite, length three and A1g in this cut")
        e_allowed &= order > 1
        e_base = q[:, 0] * gradient[0] * CM_PER_HARTREE
    else:
        e_base = 0.0
    e_terms = terms[e_allowed]
    e_values, e_cond = fit_residual(q, relative[:, 1] - e_base, e_terms, lengths)
    if gradient is not None:
        e_terms = np.vstack(((1, 0, 0), e_terms))
        e_values = np.r_[gradient[0] * CM_PER_HARTREE, e_values]
    excited = Surface(e_terms, e_values, lengths, 2)

    pure_a1g = np.all(q[:, 1:] == 0, axis=1)
    dark_defect = float(np.max(np.linalg.norm(scan["dipole_au"][pure_a1g], axis=1)))
    if dark_defect > 5e-4:
        raise ValueError(f"pure A1g geometries should remain dark, defect {dark_defect:g} a.u.")
    d_terms = terms[np.any(terms[:, 1:] != 0, axis=1)]
    d_values, d_cond = fit_residual(q, scan["dipole_au"], d_terms, lengths)
    dipole = Surface(d_terms, d_values, lengths, 1)

    return ground, excited, dipole, {
        "condition_numbers_for_scaled_fit": [g_cond, e_cond, d_cond],
        "s0_quadratic_source": "native analytic RKS Hessian" if anchor_ground_quadratic else "joint fit to direct S0 energies",
        "energy_symmetry_constraint": "Exclude terms with exactly one E2g power at every A1g displacement.",
        "raw_pure_a1g_max_dipole_au": dark_defect,
        "s1_q1_gradient_source": "analytic center gradient" if gradient is not None else "fit to signed energies",
        "raw_s1_gradient_hartree_per_q": gradient.tolist() if gradient is not None else None,
        "vertical_gap_cm_inverse": float(np.diff(scan["energies"][center][0]).item() * CM_PER_HARTREE),
    }


def fit_errors(surfaces, scan, held_out, center_energies):
    q = scan["q"]
    reach = float(np.max(np.abs(q) / surfaces[0].lengths))
    if held_out and reach > 1.0 + 1e-12:
        raise ValueError("holdout points must lie within the fitted scan limits")
    energy = (scan["energies"] - center_energies) * CM_PER_HARTREE
    energy_error = np.column_stack([surface.points(q) for surface in surfaces[:2]]) - energy
    gap_error = energy_error[:, 1] - energy_error[:, 0]
    dipole_error = surfaces[2].points(q) - scan["dipole_au"]
    dipole_norm = np.linalg.norm(dipole_error, axis=1)
    return {
        "points": len(q),
        "independent_samples": held_out,
        "max_abs_coordinate_fraction_of_axis_limit": reach,
        "max_abs_energy_cm_inverse": np.max(np.abs(energy_error), axis=0).tolist(),
        "rms_energy_cm_inverse": np.sqrt(np.mean(energy_error**2, axis=0)).tolist(),
        "max_abs_gap_cm_inverse": float(np.max(np.abs(gap_error))),
        "max_dipole_vector_error_au": float(dipole_norm.max()),
        "rms_dipole_vector_error_au": float(np.sqrt(np.mean(dipole_norm**2))),
    }


def harmonic_parameters(surface, frequencies):
    terms = np.eye(3, dtype=int)
    gradient = np.array([surface.coefficient(term) for term in terms])
    hessian = np.array([[surface.coefficient(a + b) * (1 + int(i == j))
                        for j, b in enumerate(terms)] for i, a in enumerate(terms)])
    root_frequencies = np.sqrt(frequencies)
    eigenvalues, rotation = eigh(root_frequencies[:, None] * hessian * root_frequencies[None, :])
    if eigenvalues.min() <= 0:
        raise UnstableQuadratic("fitted vertical quadratic is unstable in the three-mode subspace")
    rows, columns = linear_sum_assignment(-rotation**2)
    order = columns[np.argsort(rows)]
    rotation = rotation[:, order]
    rotation *= np.where(rotation.diagonal() < 0, -1, 1)
    return np.sqrt(eigenvalues[order]), rotation, -np.linalg.solve(hessian, gradient)


class HarmonicBasis:
    def __init__(self, frequencies, lengths, nmax, quadrature_order, tail):
        self.frequencies = frequencies
        self.nmax = nmax
        self.quantum = np.array([n for n in product(range(nmax + 1), repeat=3) if sum(n) <= nmax])
        self.indices = {tuple(n): i for i, n in enumerate(self.quantum)}
        nodes, weights = roots_hermite(quadrature_order)
        basis = np.empty((quadrature_order, nmax + 1))
        basis[:, 0] = np.pi**(-0.25)
        basis[:, 1] = np.sqrt(2) * nodes * basis[:, 0]
        for n in range(1, nmax):
            basis[:, n + 1] = (np.sqrt(2) * nodes * basis[:, n] - np.sqrt(n) * basis[:, n - 1]) / np.sqrt(n + 1)
        basis *= np.sqrt(weights[:, None])
        self.nodes = nodes
        self.quadrature_basis = basis
        self.moment = {}
        for axis, length in enumerate(lengths):
            capped = np.clip(nodes, -length, length)
            if tail == "smooth":
                beyond = np.maximum(np.abs(nodes) - length, 0)
                capped += np.sign(nodes) * 0.5 * np.tanh(beyond / 0.5)
            for power in range(6):
                self.moment[axis, power, False] = basis.T @ (nodes[:, None]**power * basis)
                self.moment[axis, power, True] = basis.T @ (capped[:, None]**power * basis)
        self.rows = [tuple(np.ix_(self.quantum[:, axis], self.quantum[:, axis])) for axis in range(3)]
        self.harmonic_diagonal = (self.quantum + 0.5) @ self.frequencies
        self.kinetic = np.diag(self.harmonic_diagonal)
        for axis, frequency in enumerate(frequencies):
            term = np.eye(3, dtype=int)[axis] * 2
            self.kinetic -= 0.5 * frequency * self.operator(term, False)

    def operator(self, term, cap):
        factors = [self.moment[axis, int(power), cap][index] for axis, (power, index) in enumerate(zip(term, self.rows))]
        return factors[0] * factors[1] * factors[2]

    def field(self, surface):
        if surface.values.ndim == 2:
            return np.stack([self.field(Surface(surface.terms, column, surface.lengths, surface.linear_order))
                             for column in surface.values.T])
        matrix = np.zeros_like(self.kinetic)
        for term, coefficient in zip(surface.terms, surface.values):
            matrix += coefficient * self.operator(term, term.sum() > surface.linear_order)
        return 0.5 * (matrix + matrix.T)

    def radial_correction(self, correction):
        values = correction.points(self.nodes)
        matrix = self.quadrature_basis.T @ (values[:, None] * self.quadrature_basis)
        return matrix[self.rows[0]] * (self.rows[1][0] == self.rows[1][1]) * (self.rows[2][0] == self.rows[2][1])

    def rotated_references(self, harmonic_ground, frequencies, rotation, displacement):
        derivative = np.zeros((3, len(self.quantum), len(self.quantum)))
        for source, quantum in enumerate(self.quantum):
            for axis, quanta in enumerate(quantum):
                if quanta:
                    target = quantum.copy()
                    target[axis] -= 1
                    target_index = self.indices[tuple(target)]
                    derivative[axis, target_index, source] = np.sqrt(quanta / 2)
                    derivative[axis, source, target_index] = -np.sqrt(quanta / 2)
        coordinates = [self.operator(term, False) - offset * np.eye(len(self.quantum))
                       for term, offset in zip(np.eye(3, dtype=int), displacement)]
        ladders = []
        for index, frequency in enumerate(frequencies):
            scale = np.sqrt(frequency / self.frequencies)
            ladders.append(sum(rotation[axis, index] * (scale[axis] * coordinate - deriv / scale[axis])
                               for axis, (coordinate, deriv) in enumerate(zip(coordinates, derivative))) / np.sqrt(2))
        states = np.column_stack((ladders[2] @ harmonic_ground, ladders[1] @ (ladders[0] @ harmonic_ground),
                                  ladders[1] @ harmonic_ground))
        return states / np.linalg.norm(states, axis=0)

    def coverage(self, vectors, lengths):
        return np.asarray([self.axis_coverage(vectors, axis, (-length, length))
                           for axis, length in enumerate(lengths)]).T.tolist()

    def axis_coverage(self, vectors, axis, limits):
        # Gauss–Hermite quadrature resolves matrix integrals. For the sharp
        # scan boundary, integrate the Gaussian marginal tails separately.
        from numpy.polynomial.legendre import leggauss
        from scipy.special import eval_hermite, factorial

        nodes, weights = leggauss(72)
        midpoint = (limits[0] + limits[1]) / 2
        radius = (limits[1] - limits[0]) / 2
        x = midpoint + radius * nodes
        numbers = np.arange(self.nmax + 1)
        basis = eval_hermite(numbers[None, :], x[:, None])
        basis *= np.exp(-0.5 * x[:, None]**2) / np.sqrt(np.sqrt(np.pi) * 2.0**numbers * factorial(numbers))
        interior = basis.T @ (radius * weights[:, None] * basis)
        marginal = interior[self.rows[axis]]
        for other in range(3):
            if other != axis:
                left, right = self.rows[other]
                marginal *= left == right
        integrated = np.sum(vectors * (marginal @ vectors), axis=0)
        return np.clip(1.0 - integrated, 0, 1)


def solve(matrix, levels):
    energies, vectors = eigh(matrix, subset_by_index=(0, levels - 1))
    defect = np.linalg.norm(matrix @ vectors - vectors * energies, axis=0).max()
    if defect > 1e-8:
        raise ValueError(f"nuclear eigenpair residual {defect:g} cm^-1 exceeds tolerance")
    return energies, vectors, float(defect)


def lines_and_block(start, target, reference, dipole, origin, sign, basis, lengths):
    energies, vectors, _ = target
    all_projections = reference.T @ vectors
    projections = all_projections[:2]
    weight = np.sum(projections**2, axis=0)
    chosen = np.sort(np.argsort(weight)[-2:])
    final = vectors[:, chosen]
    bare_amplitudes = reference[:, :2].T @ dipole
    all_amplitudes = vectors.T @ dipole
    amplitudes = all_amplitudes[chosen]
    vibrational_terms = energies - energies[0]
    offsets = sign * vibrational_terms[chosen]
    photon = origin + offsets
    strengths = np.sum(amplitudes**2, axis=1)
    areas = strengths * (photon / origin)**(3 if sign < 0 else 1)
    polar_left, singular_values, polar_right = np.linalg.svd(projections[:, chosen])
    polar = polar_left @ polar_right
    effective_matrix = (polar * vibrational_terms[chosen]) @ polar.T
    effective_dipoles = polar @ amplitudes
    line_order = np.argsort(photon)
    contributions = polar.T[:, :, None] * effective_dipoles[None, :, :]
    separate_strengths = np.sum(contributions**2, axis=2)
    interference_strength = 2 * np.einsum("ij,ij->i", contributions[:, 0], contributions[:, 1])
    lines = [{
        "level_index": int(chosen[i]),
        "vibrational_energy_cm_inverse": float(abs(offsets[i])),
        "offset_from_calculated_origin_cm_inverse": float(offsets[i]),
        "photon_cm_inverse": float(photon[i]),
        "projection_weights_in_dyad_order": (projections[:, chosen[i]]**2).tolist(),
        "dipole_strength_au2": float(strengths[i]),
        "effective_basis_component_strengths_au2": separate_strengths[i].tolist(),
        "effective_basis_interference_strength_au2": float(interference_strength[i]),
        "area_in_origin_scaled_units": float(areas[i]),
        "probability_outside_each_sampled_axis": coverage,
    } for i, coverage in zip(line_order, basis.coverage(final[:, line_order], lengths))]
    all_offsets = sign * vibrational_terms
    all_photons = origin + all_offsets
    all_strengths = np.sum(all_amplitudes**2, axis=1)
    all_areas = all_strengths * (all_photons / origin)**(3 if sign < 0 else 1)
    spectrum = np.column_stack((vibrational_terms, all_offsets, all_photons, all_strengths, all_areas, all_projections.T**2))
    indices = np.argsort(all_photons)
    spectrum = spectrum[indices[all_photons[indices] > 0]]
    # The μ phases depend on the reference basis. Their product with W does not.
    def interference(matrix, amplitudes):
        return float(matrix[0, 1] * np.dot(*amplitudes))

    return {
        "lines_in_increasing_photon_order": lines,
        "spectrum_all_retained_levels_in_increasing_photon_order": spectrum.tolist(),
        "spectrum_column_order": ["vibrational_energy_cm_inverse", "offset_from_calculated_origin_cm_inverse",
                                  "photon_cm_inverse", "dipole_strength_au2", "area_in_origin_scaled_units",
                                  *(f"projection_weight_{name}" for name in REFERENCE_LABELS)],
        "splitting_cm_inverse": float(np.ptp(offsets)),
        "area_ratio_low_over_high_photon": float(areas[line_order[0]] / areas[line_order[1]]) if areas[line_order[1]] > 0 else None,
        "sum_dipole_strength_au2": float(strengths.sum()),
        "initial_probability_outside_each_sampled_axis": basis.coverage(start[1][:, :1], lengths)[0],
        "reference_basis_order": DYAD_LABELS,
        "weakest_overlap_singular_value": float(singular_values.min()),
        "direct_reference_dipoles_au": bare_amplitudes.tolist(),
        "effective_block": {
            "hamiltonian_relative_to_vibrational_ground_cm_inverse": effective_matrix.tolist(),
            "abs_coupling_cm_inverse": float(abs(effective_matrix[0, 1])),
            "signed_cartesian_dipoles_au": effective_dipoles.tolist(),
            "gauge_invariant_interference_cm_inverse_au2": interference(effective_matrix, effective_dipoles),
        },
    }


def calculation(surfaces, frequencies, gap, nmax, quadrature, tail, model, *, radial_corrections=None):
    ground, excited, dipole = surfaces
    lengths = ground.lengths
    basis = HarmonicBasis(frequencies, lengths, nmax, quadrature, tail)
    if model == "native_analytic_harmonic":
        ground = Surface(2 * np.eye(3, dtype=int), frequencies / 2, lengths, 2)
    references = [ground.harmonic(), excited.harmonic()]
    if radial_corrections is not None:
        if model != "anharmonic":
            raise ValueError("radial corrections require the anharmonic model")
        references = [correction.harmonic_reference(surface)
                      for correction, surface in zip(radial_corrections, references, strict=True)]
    parameters = [harmonic_parameters(surface, frequencies) for surface in references]
    refs = [basis.rotated_references(solve(basis.kinetic + basis.field(surface), 1)[1][:, 0], *params)
            for surface, params in zip(references, parameters)]
    if model in ("harmonic", "native_analytic_harmonic"):
        ground, excited = ground.harmonic(), excited.harmonic()
    elif model == "without_cubic_fermi":
        ground, excited = ground.without_fermi(), excited.without_fermi()

    matrices = [basis.kinetic + basis.field(surface) for surface in (ground, excited)]
    if radial_corrections is not None:
        for matrix, correction in zip(matrices, radial_corrections, strict=True):
            matrix += basis.radial_correction(correction)
    states = [solve(matrix, min(40, len(basis.quantum))) for matrix in matrices]
    optical = basis.field(dipole)
    origin = float(gap + states[1][0][0] - states[0][0][0])
    result = {
        "basis_size": len(basis.quantum),
        "max_total_harmonic_quanta": nmax,
        "quadrature_order_per_axis": quadrature,
        "high_order_tail": tail,
        "calculated_active_space_origin_cm_inverse": origin,
        "max_nuclear_eigenpair_residual_cm_inverse": max(state[2] for state in states),
        "zero_zero_dipole_strength_au2": float(sum((states[1][1][:, 0] @ mu @ states[0][1][:, 0])**2 for mu in optical)),
        "ground_vibrational_energy_from_s0_vertical_cm_inverse": float(states[0][0][0]),
        "ground_vibrational_energy_from_s1_vertical_cm_inverse": float(states[1][0][0]),
    }
    for state, (freq, rotation, displacement) in zip(("s0", "s1"), parameters):
        result[f"harmonic_{state}_frequencies_cm_inverse"] = freq.tolist()
        result[f"harmonic_{state}_modes_as_analytic_mass_weighted_s0_modes"] = rotation.tolist()
        result[f"harmonic_{state}_displacement_in_q"] = displacement.tolist()
    for name, source, target, sign in (("emission", 1, 0, -1), ("absorption", 0, 1, 1)):
        source_vectors = np.column_stack([mu @ states[source][1][:, 0] for mu in optical])
        result[name] = lines_and_block(states[source], states[target], refs[target], source_vectors, origin, sign, basis, lengths)
        reference_dyad = refs[target][:, :2]
        direct_matrix = reference_dyad.T @ (matrices[target] - states[target][0][0] * np.eye(len(basis.quantum))) @ reference_dyad
        reference_dipoles = np.asarray(result[name]["direct_reference_dipoles_au"])
        result[name]["direct_harmonic_subspace_block_cm_inverse"] = direct_matrix.tolist()
        result[name]["direct_gauge_invariant_interference_cm_inverse_au2"] = float(direct_matrix[0, 1] * np.dot(*reference_dipoles))
        direct_areas = np.sum(reference_dipoles**2, axis=1) * ((origin + sign * direct_matrix.diagonal()) / origin)**(3 if sign < 0 else 1)
        result[name]["no_off_diagonal_projected_control_areas_in_basis_order"] = direct_areas.tolist()
        if radial_corrections is not None:
            result[name]["initial_probability_outside_q1_correction_domain"] = float(basis.axis_coverage(
                states[source][1][:, :1], 0, radial_corrections[source].limits)[0])
            energies, vectors, _ = states[target]
            photons = origin + sign * (energies - energies[0])
            states_in_spectrum = np.argsort(photons)
            states_in_spectrum = states_in_spectrum[photons[states_in_spectrum] > 0]
            result[name]["probability_outside_q1_correction_domain_by_spectrum_line"] = basis.axis_coverage(
                vectors[:, states_in_spectrum], 0, radial_corrections[target].limits).tolist()
    return result


def minimum_audit(surface, initial):
    derivatives = [Surface(surface.terms[mask] - np.eye(3, dtype=int)[i],
                           surface.values[mask] * surface.terms[mask, i], surface.lengths, 1)
                   for i in range(3) if (mask := surface.terms[:, i] > 0).any()]

    def value_and_gradient(q):
        return surface.points(q).item(), np.array([function.points(q).item() for function in derivatives])

    lengths = surface.lengths
    bounds = list(zip(-lengths, lengths))
    starts = np.vstack((np.zeros(3), np.clip(initial, -lengths, lengths), np.array(list(product((-0.75, 0.75), repeat=3))) * lengths))
    minima = [minimize(value_and_gradient, q, jac=True, method="L-BFGS-B", bounds=bounds) for q in starts]
    best = min(minima, key=lambda result: result.fun)
    return {
        "best_minimum_q": best.x.tolist(),
        "minimum_cm_inverse_relative_to_vertical_geometry": float(best.fun),
        "on_scan_boundary": bool(np.any(np.isclose(np.abs(best.x), lengths))),
        "spread_over_search_starts_cm_inverse": float(np.ptp([result.fun for result in minima])),
        "max_abs_gradient_cm_inverse_per_q": float(np.abs(best.jac).max()),
    }


def describe_fit(scan, holdouts, surfaces, diagnostics):
    center = scan["energies"][np.all(scan["q"] == 0, axis=1)]
    frequencies = scan["frequencies_cm1"]
    harmonic = [harmonic_parameters(surface, frequencies) for surface in surfaces[:2]]
    diagnostics["training_errors"] = fit_errors(surfaces, scan, False, center)
    if holdouts is not None:
        diagnostics["holdout_errors"] = fit_errors(surfaces, holdouts, True, center)
    diagnostics["harmonic_frequencies_cm_inverse_by_surface"] = [values[0].tolist() for values in harmonic]
    coefficients = [float(surface.coefficient(FERMI_TERM)) for surface in surfaces[:2]]
    diagnostics["q1_q6_q8_coefficients_cm_inverse"] = coefficients
    diagnostics["ground_pure_cubic_harmonic_coupling_in_native_reference_cm_inverse"] = float(coefficients[0] / (2 * np.sqrt(2)))
    diagnostics["minimum_search_on_measured_box"] = [minimum_audit(surface, params[2]) for surface, params in zip(surfaces[:2], harmonic)]
    return diagnostics


def run(scan, holdouts, nmax, quadrature, *, baseline_q1_limit=None):
    if nmax < 8 or quadrature < nmax + 8:
        raise ValueError("require nmax >= 8 and quadrature >= nmax + 8")
    if "holdout" in scan:
        flags = scan["holdout"]
        if flags.dtype != bool or flags.shape != (len(scan["q"]),):
            raise ValueError("holdout must be a boolean flag per input point")
        if flags.any():
            if holdouts is not None:
                raise ValueError("supply either holdout flags in the scan or a separate holdout file")
            holdouts = select_scan_rows(scan, flags)
            scan = select_scan_rows(scan, ~flags)
    if holdouts is not None and not set(map(tuple, holdouts["q"])).isdisjoint(map(tuple, scan["q"])):
        raise ValueError("training and holdout coordinates must be distinct")
    frequencies = scan["frequencies_cm1"]
    result = {
        "mode_order": LABELS,
        "coordinate_definition": "q_i = sqrt(omega_i) Q_i with omega_i in Hartree and Q_i mass weighted in electron-mass bohr units",
        "analytic_ground_frequencies_cm_inverse": frequencies.tolist(),
        "scan_points": len(scan["q"]),
        "scan_axes_q": [np.unique(qi).tolist() for qi in scan["q"].T],
        "spectator_modes_included": False,
        "selected_components_per_e2g_mode": 1,
        "symmetry_of_cut": "This cut retains one component of each E2g mode and can give spurious intensity to the symmetry-forbidden 0-0 line.",
        "electronic_scan_quality": scan_quality(scan),
        "tail_models": {
            "capped": "Use the boundary coordinate in higher-order terms outside the measured scan limits.",
            "smooth": "Continue the boundary slope and saturate the coordinate used in higher-order terms 0.5 q beyond the measured limits.",
        },
        "models": {},
        "surface_fits": {},
    }
    if holdouts is not None:
        result["electronic_holdout_quality"] = scan_quality(holdouts)
    fitted = {}
    for degree in (4, 5):
        try:
            *surfaces, diagnostics = fit_surfaces(scan, degree)
            diagnostics = describe_fit(scan, holdouts, surfaces, diagnostics)
        except (UnderdeterminedSurface, UnstableQuadratic) as failure:
            if degree == 4:
                raise
            result["surface_fits"][f"degree_{degree}"] = {"status": "unavailable", "reason": str(failure)}
            continue
        fitted[degree] = surfaces
        result["surface_fits"][f"degree_{degree}"] = diagnostics

    def energy_holdout_error(degree):
        return np.linalg.norm(result["surface_fits"][f"degree_{degree}"]["holdout_errors"]["rms_energy_cm_inverse"])

    primary = min(fitted, key=energy_holdout_error) if holdouts is not None else 4
    alternative = 5 if primary == 4 else 4
    result["primary_fit_degree"] = primary
    result["fit_degree_selection"] = ("Lowest combined RMS S0 and S1 energies on the independently computed holdout geometries."
                                      if holdouts is not None else "Quartic default for a scan without independent holdout geometries.")
    gap = result["surface_fits"]["degree_4"]["vertical_gap_cm_inverse"]
    *anchored, anchored_info = fit_surfaces(scan, primary, anchor_ground_quadratic=True)
    result["surface_fits"]["analytic_s0_anchored"] = describe_fit(scan, holdouts, anchored, anchored_info)
    for model, degree, size, order, tail in (
        ("anharmonic", primary, nmax, quadrature, "capped"),
        ("alternative_fit_degree", alternative, nmax, quadrature, "capped"),
        ("larger_basis", primary, nmax + 2, quadrature + 4, "capped"),
        ("larger_quadrature", primary, nmax, quadrature + 8, "capped"),
        ("smooth_tail", primary, nmax, quadrature, "smooth"),
        ("harmonic", primary, nmax, quadrature, "capped"),
        ("native_analytic_harmonic", primary, nmax, quadrature, "capped"),
        ("without_cubic_fermi", primary, nmax, quadrature, "capped"),
    ):
        if degree in fitted:
            result["models"][model] = calculation(fitted[degree], frequencies, gap, size, order, tail, model)
    result["models"]["analytic_s0_anchored"] = calculation(anchored, frequencies, gap, nmax, quadrature, "capped", "anharmonic")
    if baseline_q1_limit is not None:
        mask = np.abs(scan["q"][:, 0]) <= baseline_q1_limit
        if mask.all() or not mask.any():
            raise ValueError("baseline q1 range must retain only a subset of the scanned training points")
        narrow = select_scan_rows(scan, mask)
        narrow_holdouts = None
        if holdouts is not None and (mask := np.abs(holdouts["q"][:, 0]) <= baseline_q1_limit).any():
            narrow_holdouts = select_scan_rows(holdouts, mask)
        *narrow_surfaces, narrow_info = fit_surfaces(narrow, primary)
        result["surface_fits"]["narrower_q1_scan"] = describe_fit(narrow, narrow_holdouts, narrow_surfaces, narrow_info)
        result["models"]["narrower_q1_scan"] = calculation(narrow_surfaces, frequencies, gap, nmax, quadrature, "capped", "anharmonic")
    main_model = result["models"]["anharmonic"]
    result["probe_differences_from_main"] = {}
    for name in ("alternative_fit_degree", "larger_basis", "larger_quadrature", "smooth_tail", "narrower_q1_scan"):
        if name not in result["models"]:
            continue
        model = result["models"][name]
        result["probe_differences_from_main"][name] = {
            process: {key: model[process][key] - main_model[process][key]
                      for key in ("splitting_cm_inverse", "area_ratio_low_over_high_photon", "weakest_overlap_singular_value")
                      if model[process][key] is not None and main_model[process][key] is not None}
            for process in ("emission", "absorption")
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-quanta", type=int, default=12)
    parser.add_argument("--quadrature", type=int, default=42)
    parser.add_argument("--baseline-q1-limit", type=float, help="also solve a narrower scan with this absolute q1 limit")
    args = parser.parse_args()
    data = run(read_scan(args.scan), read_scan(args.holdout) if args.holdout else None, args.max_quanta,
               args.quadrature, baseline_q1_limit=args.baseline_q1_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
