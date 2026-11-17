"""Upper bounds on photon moments omitted from harmonic stick bases."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicPhotonMoment:
    """Thermal dipole strength times photon energy in Hartree to the given power."""

    retained: float
    omitted_initial_upper_bound: float
    omitted_final_upper_bound: float

    @property
    def coverage_lower_bound(self) -> float:
        omitted = self.omitted_initial_upper_bound + self.omitted_final_upper_bound
        if omitted == 0:
            return 1.0
        return 1.0 / (1.0 + omitted / self.retained) if self.retained else 0.0


def _moment_orders(process):
    return (1,) if process == "absorption" else (3, 4)


def photon_moment_report(moments, process, target_coverage):
    report = {}
    for order in _moment_orders(process):
        moment = None if moments is None else moments.get(order)
        if moment is None:
            report[f"M{order}"] = {"status": "unassessed"}
            continue
        lower = moment.coverage_lower_bound
        status = "unassessed"
        if target_coverage is not None:
            status = "complete" if lower >= target_coverage else "incomplete"
        report[f"M{order}"] = {**asdict(moment), "coverage_lower_bound": lower, "status": status}
    return report


def _roundoff(total, captured):
    return 128 * np.finfo(float).eps * (np.abs(total) + np.abs(captured))


def _upper_difference(total, captured):
    return np.maximum(0.0, total - captured + _roundoff(total, captured))


def _energy_moment(offset, cumulants, order):
    mean = offset + cumulants[..., 0]
    if order == 1:
        return mean
    variance = cumulants[..., 1]
    third = cumulants[..., 2]
    if order == 3:
        return mean**3 + 3 * mean * variance + third
    return (mean**4 + 6 * mean**2 * variance + 4 * mean * third
            + 3 * variance**2 + cumulants[..., 3])


def _thermal_energy_strength_moment(order, offset, omega, mean_occupation, vacuum_strength, ht_strength):
    """All-state E[(E_i + offset)^order * <i|mu† mu|i>] for offset >= 0."""
    n = mean_occupation
    variance = n * (1 + n)
    mode_cumulants = np.column_stack((
        omega * n, omega**2 * variance, omega**3 * variance * (1 + 2 * n),
        omega**4 * variance * (1 + 6 * variance),
    ))
    total_cumulants = mode_cumulants.sum(axis=0)
    moment = vacuum_strength * _energy_moment(offset, total_cumulants, order)
    # Multiplying a geometric distribution's mass by n gives mean(n) times
    # the distribution of 1 + two independent copies of that occupation.
    moment += (2 * ht_strength * n) @ _energy_moment(
        offset + omega, total_cumulants + mode_cumulants, order,
    )
    return moment


def _signed_absorption_moment(
    omega_i, omega_f, rotation, shift, gap, dipole, derivatives,
    variances, fourth_cumulants, closure,
):
    """Close the signed first moment over all final states for each initial distribution."""
    # The two Hamiltonians share the kinetic operator. Contract the quadratic
    # potential difference with |mu(Q)|² in the initial coordinate basis.
    curvature = rotation.T @ (omega_f[:, None]**2 * rotation) - np.diag(omega_i**2)
    slope = rotation.T @ (omega_f**2 * shift)
    shift_energy = 0.5 * np.dot(omega_f**2, shift**2)
    offset = gap + 0.5 * (omega_i.sum() - omega_f.sum()) + shift_energy
    moment = closure * (offset + 0.5 * (variances @ np.diag(curvature)))
    if derivatives is not None:
        correlations = (derivatives.conj() @ derivatives.T).real
        moment += np.sum((variances @ (curvature * correlations)) * variances, axis=-1)
        moment += 2 * (variances @ (slope * (derivatives @ dipole.conj()).real))
        moment += 0.5 * (fourth_cumulants @ (np.diag(curvature) * np.diag(correlations)))
        # The initial kinetic commutator contributes 1/2 sum |dmu/dQ|².
        moment += 0.5 * np.trace(correlations)
    return moment


def harmonic_photon_moments(
    omega_i, omega_f, rotation, shift, gap, dipole, derivatives, process,
    initial_states, initial_energies, populations, final_cutoff, strengths, closure, photons,
    mean_occupation,
):
    """Bound missing positive moments using the complete initial-state thermal distribution."""
    if derivatives is None:
        ht_strength = np.zeros_like(omega_i)
    else:
        ht_strength = np.sum(np.abs(derivatives)**2, axis=1) / (2 * omega_i)
    vacuum_strength = np.vdot(dipole, dipole).real + ht_strength.sum()
    row_omitted_strength = _upper_difference(closure, strengths.sum(axis=1))
    row_ceiling = np.maximum(0, -gap + initial_energies)
    min_unrepresented_final_energy = (final_cutoff + 1) * omega_f.min() if omega_f.size else np.inf
    final_tail_ceiling = np.maximum(0, row_ceiling - min_unrepresented_final_energy)
    # Ignoring max(gap, 0) gives a polynomial upper bound on each missing
    # hot state's emission energy (or negative absorption energy).
    offset = max(0.0, -gap)
    thermal_ceiling = offset + initial_energies
    has_hot_states = np.any(mean_occupation > 0)

    signed_initial_tail = signed_final_tail = 0.0
    initial_roundoff = final_roundoff = 0.0
    if process == "absorption":
        variances = (initial_states + 0.5) / omega_i
        fourth_cumulants = -1.5 * initial_states.astype(float) * (initial_states + 1) / omega_i**2
        row_signed = _signed_absorption_moment(
            omega_i, omega_f, rotation, shift, gap, dipole, derivatives,
            variances, fourth_cumulants, closure,
        )
        captured_signed = float(populations @ np.sum(strengths * photons, axis=1))
        initial_signed = float(populations @ row_signed)
        signed_final_tail = initial_signed - captured_signed
        final_roundoff = _roundoff(initial_signed, captured_signed)
        if has_hot_states:
            total_signed = _signed_absorption_moment(
                omega_i, omega_f, rotation, shift, gap, dipole, derivatives,
                (mean_occupation + 0.5) / omega_i, np.zeros_like(omega_i),
                vacuum_strength + 2 * (mean_occupation @ ht_strength),
            )
            signed_initial_tail = total_signed - initial_signed
            initial_roundoff = _roundoff(total_signed, initial_signed)

    positive_energies = np.maximum(0, photons)
    thermal_strengths = populations[:, None] * strengths
    moments = {}
    for order in _moment_orders(process):
        missing_initial = 0.0
        if has_hot_states:
            whole_thermal_ceiling = _thermal_energy_strength_moment(
                order, offset, omega_i, mean_occupation, vacuum_strength, ht_strength,
            )
            retained_thermal_ceiling = populations @ (thermal_ceiling**order * closure)
            missing_initial = _upper_difference(whole_thermal_ceiling, retained_thermal_ceiling)
        missing_final = (populations * row_omitted_strength) @ final_tail_ceiling**order
        # For absorption, positive = signed + |negative|. The energy-ceiling
        # bounds apply to the negative part. Emission uses them directly.
        initial_bound = np.maximum(0.0, signed_initial_tail + missing_initial + initial_roundoff)
        final_bound = np.maximum(0.0, signed_final_tail + missing_final + final_roundoff)
        retained = np.sum(thermal_strengths * positive_energies**order)
        if not np.all(np.isfinite([retained, initial_bound, final_bound])):
            continue
        moments[order] = HarmonicPhotonMoment(float(retained), float(initial_bound), float(final_bound))
    return moments
