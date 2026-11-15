"""Evaluate same-symmetry spectator mixing from direct two-mode benzene cuts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from validate_benzene_3d import CM_PER_HARTREE, calculation, read_scan
from validate_benzene_absorption import absorption_progression, with_transverse_correction
from validate_benzene_spectators import (
    ContractedRadial, confining_contraction, fit_cut, read_spectators, replay_baseline,
)


def fit_mixed_cuts(paths):
    paired = defaultdict(list)
    for path in sorted(paths.glob("*.json")):
        point = json.loads(path.read_text(encoding="utf-8"))
        pair = point["group_j"], point["group_k"]
        residual = (point["energy_s1_hartree"] - point["radial_s1_hartree"]
                    - point["axis_j_delta_s1_hartree"] - point["axis_k_delta_s1_hartree"])
        if abs(residual - point["mixed_s1_hartree"]) > 5e-11:
            raise ValueError(f"inconsistent direct cut references in {path}")
        paired[pair].append((point["q1"], point["qj"], point["qk"], residual * CM_PER_HARTREE))
    if not paired:
        raise ValueError("no mixed spectator cuts found")
    result, checks = {}, {}
    for pair, rows in paired.items():
        q1, qj, qk, value = np.array(rows).T
        signed_product = qj * qk
        # At one transverse amplitude the bilinear term is effective: odd
        # higher powers can enter it. Validate at distinct product amplitudes.
        q_degree = 2
        design = np.stack([q1**n * signed_product**power for power in (1, 2)
                           for n in range(q_degree + 1)], axis=1)
        train = np.isclose(np.abs(qj), .85) & np.isclose(np.abs(qk), .85)
        fit, _, rank, _ = np.linalg.lstsq(design[train], value[train], rcond=None)
        if rank != len(fit):
            raise ValueError(f"insufficient signed mixed cuts for {pair}")
        result[pair] = np.polynomial.Polynomial(fit[:q_degree + 1])
        checks["/".join(pair)] = {
            "bilinear_coefficients_by_q1_power_cm_inverse": fit[:q_degree + 1].tolist(),
            "holdout_residuals_cm_inverse": (design[~train] @ fit - value[~train]).tolist(),
        }
    return result, checks


def mixing_profile(nodes, fits, irreps, pair_fits, bounds):
    # Use a constant extension of the complete harmonic block outside the
    # mixed-cut domain, including its diagonal elements.
    clipped = np.clip(nodes, *bounds)
    curvature = {
        label: 2 * fits[label].curves(clipped, extension=0)[0][:, fits[label].powers.index(2)]
        for label in irreps
    }
    by_sector = defaultdict(list)
    for label, irrep in irreps.items():
        by_sector[irrep].append(label)
    contribution = {}
    active_softening = np.zeros((len(nodes), 2, 2))
    for irrep, sector in by_sector.items():
        pair_indices = {(sector.index(j), sector.index(k)): potential(clipped)
                        for (j, k), potential in pair_fits.items() if j in sector and k in sector}
        if not pair_indices:
            continue
        size = len(sector)
        hessian = np.zeros((len(nodes), size, size))
        for i, label in enumerate(sector):
            hessian[:, i, i] = curvature[label]
        for (j, k), value in pair_indices.items():
            hessian[:, j, k] = hessian[:, k, j] = value
        mass_metric = np.sqrt([fits[label].frequency for label in sector])
        dynamic = hessian * mass_metric[:, None] * mass_metric

        def zero_point_difference(block):
            levels = np.linalg.eigvalsh(block)
            if np.any(levels <= 0):
                raise ValueError(f"harmonic diagnostic is unstable in {irrep}")
            harmonic = np.sqrt(levels)
            unmixed = np.sqrt(np.diagonal(block, axis1=-2, axis2=-1))
            return .5 * np.sum(harmonic - unmixed, axis=1)

        if irrep != "E2g":
            contribution[irrep] = fits[sector[0]].multiplicity * zero_point_difference(dynamic)
            continue
        # All odd E2g components are spectators. The even ν6/ν8 partners
        # are active, leaving only the fully omitted E2g modes in this block.
        inactive = [i for i, label in enumerate(sector) if fits[label].multiplicity == 2]
        active = [sector.index("nu6_partner"), sector.index("nu8_partner")]
        contribution[irrep] = (zero_point_difference(dynamic)
                               + zero_point_difference(dynamic[:, inactive][:, :, inactive]))
        outside = hessian[:, inactive][:, :, inactive]
        cross = hessian[:, active][:, :, inactive]
        active_softening = -cross @ np.linalg.solve(outside, cross.transpose(0, 2, 1))
    return contribution, active_softening


def partner_invariants(surfaces, contractions, nodes):
    """First-order conditional expectation of the known E2g partner terms."""
    mode_labels = ("nu6_partner", "nu8_partner")
    mean_square = {}
    quartic = {}
    for item in contractions:
        label = item.fit.label
        if label not in mode_labels:
            continue
        _, frame, _, _ = item.resolve(nodes, details=True)
        ground = frame[:, :, 0]
        q2 = item.oscillator.moments[item.fit.powers.index(2)]
        mean_square[label] = np.einsum("qi,ij,qj->q", ground, q2, ground)
        quartic[label] = item.fit.curves(nodes, .5)[0][:, item.fit.powers.index(4)]
    excited = surfaces[1]
    effective = defaultdict(lambda: np.zeros(len(nodes)))
    # Cubic Re(z6**a z8**b) terms follow from the known even-axis cut.
    # The odd components are in independent, even local ground functions.
    for (radial, a, b), value in zip(excited.terms, excited.values, strict=True):
        if a + b != 3:
            continue
        if a >= 2:
            effective[a-2, b] -= value * (a*(a-1)/2) * nodes**radial * mean_square[mode_labels[0]]
        if b >= 2:
            effective[a, b-2] -= value * (b*(b-1)/2) * nodes**radial * mean_square[mode_labels[1]]
    # The single-mode quartic invariant is proportional to (x²+y²)².
    effective[2, 0] += 2 * quartic[mode_labels[0]] * mean_square[mode_labels[0]]
    effective[0, 2] += 2 * quartic[mode_labels[1]] * mean_square[mode_labels[1]]
    terms, coefficients = [], []
    for (a, b), values in effective.items():
        for power, coefficient in enumerate(np.polynomial.polynomial.polyfit(nodes, values, deg=2)):
            terms.append((power, a, b))
            coefficients.append((0., coefficient))
    return with_transverse_correction(surfaces, np.asarray(terms), np.asarray(coefficients))


def angular_cuts(path):
    measurements = defaultdict(list)
    for point in json.loads(path.read_text(encoding="utf-8")):
        group, q1, amplitude = point["group"], point["q1"], point["amplitude"]
        odd = point["even_axis_odd_in_q_cm1"]
        measurements[group].append((q1, amplitude, odd))
    fits, validation = {}, {}
    for group, rows in measurements.items():
        q1, amplitude, odd = np.asarray(rows).T
        train = np.isclose(amplitude, 1.7)
        fits[group] = np.polynomial.Polynomial.fit(q1[train], odd[train] / amplitude[train]**3, 2).convert()
        validation[group] = (fits[group](q1[~train]) * amplitude[~train]**3 - odd[~train]).tolist()
    return fits, validation


def angular_energies(nodes, fit, cubic, size):
    coefficients, _ = fit.curves(nodes, .5)
    quadratic, quartic = coefficients[:, fit.powers.index(2)], coefficients[:, fit.powers.index(4)]
    strength = cubic(np.clip(nodes, 0., 3.))
    if np.any(quadratic <= 0) or np.any(quartic <= 0):
        raise ValueError(f"the angular sensitivity needs confining positive quadratic and quartic terms for {fit.label}")
    omega = np.sqrt(2 * fit.frequency * np.median(quadratic))
    length = np.sqrt(fit.frequency / omega)
    upper = length * np.sqrt(np.arange(1, size+4) / 2)
    x = np.diag(upper, 1) + np.diag(upper, -1)
    powers = [np.linalg.matrix_power(x, n)[:size, :size] for n in range(1, 5)]
    x, x2, x3, x4 = powers
    identity = np.eye(size)
    kinetic = np.diag(omega * (np.arange(size) + .5)) - fit.frequency / (2 * length**4) * x2
    total_kinetic = np.kron(kinetic, identity) + np.kron(identity, kinetic)
    r2 = np.kron(x2, identity) + np.kron(identity, x2)
    r4 = np.kron(x4, identity) + 2*np.kron(x2, x2) + np.kron(identity, x4)
    angular = np.kron(x3, identity) - 3*np.kron(x, x2)
    shifts = []
    for c2, c4, c3 in zip(quadratic, quartic, strength, strict=True):
        isotropic = total_kinetic + c2*r2 + c4*r4
        energy = eigh(isotropic, eigvals_only=True, subset_by_index=(0, 0))[0]
        full = eigh(isotropic + c3*angular, eigvals_only=True, subset_by_index=(0, 0))[0]
        shifts.append(full - energy)
    return np.asarray(shifts)


@dataclass(frozen=True)
class MixedRadial:
    baseline: object
    fits: dict
    irreps: dict
    pair_fits: dict
    bounds: tuple

    @property
    def limits(self):
        return self.baseline.limits

    def points(self, nodes):
        nodes = np.asarray(nodes)
        contribution, _ = mixing_profile(np.r_[0., nodes], self.fits, self.irreps, self.pair_fits, self.bounds)
        offset = np.sum(list(contribution.values()), axis=0)
        return self.baseline.points(nodes) + offset[1:] - offset[0]

    def harmonic_reference(self, surface):
        return self.baseline.harmonic_reference(surface)


@dataclass(frozen=True)
class AngularRadial:
    baseline: MixedRadial
    angular_fits: dict
    size: int = 10

    @property
    def limits(self):
        return self.baseline.limits

    def points(self, nodes):
        q1 = np.r_[0., nodes]
        shifts = sum(angular_energies(q1, self.baseline.fits[group], fit, self.size)
                     for group, fit in self.angular_fits.items())
        return self.baseline.points(nodes) + shifts[1:] - shifts[0]

    def harmonic_reference(self, surface):
        return self.baseline.harmonic_reference(surface)


def analyze(scan, baseline, mode_file, cut_path, pair_path, degree, angular_path=None):
    pair_fits, checks = fit_mixed_cuts(pair_path)
    spectators = read_spectators(mode_file, cut_path, (.75, 2.25))
    fits = {item.label: fit_cut(item, degree=degree, max_power=4) for item in spectators}
    with np.load(mode_file, allow_pickle=False) as metadata:
        irreps = dict(zip(metadata["labels"].tolist(), metadata["irreps"].tolist(), strict=True))
    surfaces, radial, gap = replay_baseline(scan, baseline)
    frequency = scan["frequencies_cm1"]
    contractions = [confining_contraction(fit, 20, .25, .5, isotropic=True, tail="matched")
                    for fit in fits.values()]
    diagonal = ContractedRadial(radial[1], contractions, frequency[0], False)
    bounds = (0., 3.)
    mixed = MixedRadial(diagonal, fits, irreps, pair_fits, bounds)
    sample_nodes = np.array([0., .75, 1.5, 2.25, 3.])
    components, schur = mixing_profile(sample_nodes, fits, irreps, pair_fits, bounds)
    strength = np.stack((schur[:, 0, 0]/2, schur[:, 0, 1], schur[:, 1, 1]/2), axis=1)
    coefficients = np.polynomial.polynomial.polyfit(sample_nodes, strength, deg=2)
    exponents = np.array([(n, a, b) for n in range(3) for a, b in ((2, 0), (1, 1), (0, 2))])
    correction_surface = with_transverse_correction(
        surfaces, exponents, np.column_stack((np.zeros(9), coefficients.ravel())),
    )
    combined = partner_invariants(correction_surface, contractions, sample_nodes)

    def calculate(potentials, excited):
        result = calculation(potentials, frequency, gap, 12, 42, "capped", "anharmonic",
                             radial_corrections=(radial[0], excited))
        result["absorption_progression"] = absorption_progression(result)
        return result

    report = {
        "model_scope": "Harmonic spectator-block mixing sensitivity after direct anharmonic one-mode contraction. "
                       "The separate Schur-complement model is an adiabatic estimate for active–spectator E2g mixing.",
        "sample_q1": sample_nodes.tolist(),
        "mixed_zero_point_corrections_by_irrep_cm_inverse": {k: v.tolist() for k, v in components.items()},
        "schur_correction_to_active_curvature_cm_inverse": schur.tolist(),
        "mixed_cut_fits": checks,
        "models": {
            "diagonal": calculate(surfaces, diagonal),
            "spectator_mixing": calculate(surfaces, mixed),
            "spectator_and_active_mixing": calculate(correction_surface, mixed),
            "including_e2g_partner_invariants": calculate(combined, mixed),
        },
    }
    if angular_path is not None:
        angular_fits, validation = angular_cuts(angular_path)
        report["angular_cubic_cut_holdout_residuals_cm_inverse"] = validation
        report["angular_cubic_ground_shift_by_group_cm_inverse"] = {
            group: angular_energies(sample_nodes, fits[group], fit, 12).tolist()
            for group, fit in angular_fits.items()
        }
        report["models"]["including_e2g_angular_invariants"] = calculate(
            combined, AngularRadial(mixed, angular_fits),
        )
        report["models"]["larger_e2g_angular_basis"] = calculate(
            combined, AngularRadial(mixed, angular_fits, 12),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--angular", type=Path)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = analyze(read_scan(args.scan), checkpoint, args.modes, args.points, args.mixed, args.degree, args.angular)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
