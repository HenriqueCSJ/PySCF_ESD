"""Interpret the electronic and harmonic inputs to a vibronic spectrum."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from .pyscf_adapter import AdiabaticHessianParameters, VerticalGradientParameters
from .spectral_properties import HARTREE_TO_WAVENUMBER


def condon_ht_line_strengths(spectrum) -> dict[str, np.ndarray]:
    """Signed line-strength terms in the harmonic linear-dipole expansion."""
    condon = getattr(spectrum, "condon_transition_dipole_au", None)
    if condon is None:
        return {}
    # Expand |mu_C + mu_HT|² at the amplitude level to preserve interference.
    ht = spectrum.transition_dipole_au - condon
    return {
        "condon_strength_au2": np.sum(condon**2, axis=1),
        "ht_strength_au2": np.sum(ht**2, axis=1),
        "condon_ht_interference_au2": 2 * np.sum(condon * ht, axis=1),
    }


def condon_ht_properties(spectrum, process: str) -> dict | None:
    terms = condon_ht_line_strengths(spectrum)
    if not terms:
        return None
    weight = spectrum.population * spectrum.photon_energy_hartree**(1 if process == "absorption" else 3)
    sums = {name: float(weight @ values) for name, values in terms.items()}
    total = float(weight @ spectrum.dipole_strength_au2)
    return {
        "expansion_origin": "s0_minimum" if process == "absorption" else "modeled_excited_origin",
        "interpretation": (
            "The constant and linear terms depend on the expansion origin. "
            "The interference term is signed."
        ),
        "photon_weighted_terms_relative_units": sums,
        "fractions_of_total_signal": {name: value / total for name, value in sums.items()} if total else None,
    }


def electronic_relaxation_properties(
    *,
    ground_at_ground_hartree: float,
    ground_at_excited_hartree: float,
    excited_at_ground_hartree: float,
    excited_at_excited_hartree: float,
    transition_dipole_at_ground_au: ArrayLike | None = None,
    transition_dipole_at_excited_au: ArrayLike | None = None,
) -> dict:
    """Report vertical gaps and relaxation energies on one electronic reference.

    All four energies are total electronic energies at the two geometries.
    Vibrational zero-point energies enter the harmonic 0–0 gap separately.
    """
    e00, e01, e10, e11 = map(float, (
        ground_at_ground_hartree, ground_at_excited_hartree,
        excited_at_ground_hartree, excited_at_excited_hartree,
    ))
    if not all(isfinite(energy) for energy in (e00, e01, e10, e11)):
        raise ValueError("electronic energies must be finite")
    differences = {
        "vertical_absorption": e10 - e00,
        "vertical_emission": e11 - e01,
        "adiabatic_electronic_gap": e11 - e00,
        "ground_deformation_at_excited": e01 - e00,
        "excited_relaxation_from_ground": e10 - e11,
        "vertical_absorption_minus_emission": (e10 - e00) - (e11 - e01),
    }
    result = {
        "electronic_energies_hartree": {
            "ground_at_ground": e00, "ground_at_excited": e01, "excited_at_ground": e10, "excited_at_excited": e11,
        },
        "energy_differences_hartree": differences,
        "energy_differences_cm^-1": {
            name: energy * HARTREE_TO_WAVENUMBER for name, energy in differences.items()
        },
    }
    vertical_strength = {}
    for direction, dipole in (
        ("absorption", transition_dipole_at_ground_au), ("emission", transition_dipole_at_excited_au),
    ):
        if dipole is not None:
            vector = np.asarray(dipole, dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("the vertical transition dipole must be a finite Cartesian vector")
            vertical_strength[direction] = float(vector @ vector)
    if vertical_strength:
        result["vertical_electronic_dipole_strength_au2"] = vertical_strength
    return result


def _mode_properties(frequencies, displacement, derivatives=None):
    factors = 0.5 * frequencies * displacement**2
    reorganization_cm = frequencies * factors * HARTREE_TO_WAVENUMBER
    rows = [
        {
            "mode": index + 1,
            "frequency_cm^-1": float(frequency * HARTREE_TO_WAVENUMBER),
            "displacement_sqrt_electron_mass_bohr": float(shift),
            "huang_rhys_displacement_factor": float(factor),
            "quadratic_reorganization_cm^-1": float(reorganization),
        }
        for index, (frequency, shift, factor, reorganization) in enumerate(
            zip(frequencies, displacement, factors, reorganization_cm, strict=True)
        )
    ]
    if derivatives is not None:
        scales = np.linalg.norm(derivatives, axis=1) / np.sqrt(2 * frequencies)
        for row, scale in zip(rows, scales, strict=True):
            row["linear_ht_one_quantum_dipole_scale_au"] = float(scale)
    return {"total_quadratic_reorganization_cm^-1": float(reorganization_cm.sum()), "modes": rows}


def harmonic_model_properties(
    model: AdiabaticHessianParameters | VerticalGradientParameters,
    *,
    dipole_derivatives_au: ArrayLike | None = None,
    transition_dipole_at_ground_au: ArrayLike | None = None,
    direct_transition_dipole_at_excited_au: ArrayLike | None = None,
) -> dict:
    """Return local mode descriptors for the harmonic reference Hamiltonian.

    Displacements point from each minimum to the other geometry in that
    surface's normal coordinates. The individual Huang–Rhys factors are
    displacement descriptors. With Duschinsky mixing they do not partition
    the calculated spectral intensity.
    """
    if isinstance(model, VerticalGradientParameters):
        ground = excited = model.frequencies_hartree
        ground_displacement = -model.shift_au
        excited_displacement = model.shift_au
        rotation = None
    else:
        ground, excited = model.ground_frequencies_hartree, model.excited_frequencies_hartree
        rotation = model.rotation
        ground_displacement = -rotation.T @ model.shift_au
        excited_displacement = model.shift_au
    ground_derivatives = excited_derivatives = None
    if dipole_derivatives_au is not None:
        ground_derivatives = np.asarray(dipole_derivatives_au, dtype=float)
        if ground_derivatives.shape != (len(ground), 3) or not np.isfinite(ground_derivatives).all():
            raise ValueError("provide one finite Cartesian dipole derivative per ground mode")
        excited_derivatives = ground_derivatives if rotation is None else rotation @ ground_derivatives
    result = {
        "harmonic_00_cm^-1": float(model.gap_00_hartree * HARTREE_TO_WAVENUMBER),
        "harmonic_zpe_shift_cm^-1": float(0.5 * (excited.sum() - ground.sum()) * HARTREE_TO_WAVENUMBER),
        "ground_at_excited": _mode_properties(ground, ground_displacement, ground_derivatives),
        "excited_at_ground": _mode_properties(excited, excited_displacement, excited_derivatives),
        "mode_interpretation": (
            "Mode displacements and quadratic relaxation terms use each surface's harmonic basis. "
            "The decomposition within degenerate mode subspaces is basis-dependent."
        ),
    }
    if rotation is not None:
        result["duschinsky_rows_excited_columns_ground"] = rotation.tolist()
    vector_fields = (
        (("ground_at_excited", "cartesian_mode_derivatives_au"), ("excited_at_ground", "cartesian_mode_derivatives_au"))
        if rotation is None else
        (("ground_at_excited", "ground_cartesian_mode_derivatives_au"),
         ("excited_at_ground", "excited_cartesian_mode_derivatives_au"))
    )
    for surface, attribute in vector_fields:
        vectors = getattr(model, attribute, None)
        if vectors is not None:
            result[surface]["cartesian_dR_dQ_inverse_sqrt_electron_mass"] = vectors.tolist()
    if (ground_derivatives is not None and transition_dipole_at_ground_au is not None
            and direct_transition_dipole_at_excited_au is not None):
        center_dipole = np.asarray(transition_dipole_at_ground_au, dtype=float)
        direct_dipole = np.asarray(direct_transition_dipole_at_excited_au, dtype=float)
        if any(vector.shape != (3,) or not np.isfinite(vector).all() for vector in (center_dipole, direct_dipole)):
            raise ValueError("provide finite Cartesian transition dipoles in a common frame and phase")
        extrapolated = center_dipole + ground_displacement @ ground_derivatives
        result["linear_ht_at_excited_minimum"] = {
            "extrapolated_dipole_au": extrapolated.tolist(),
            "direct_dipole_au": direct_dipole.tolist(),
            "vector_error_norm_au": float(np.linalg.norm(extrapolated - direct_dipole)),
            "extrapolated_strength_au2": float(extrapolated @ extrapolated),
            "direct_strength_au2": float(direct_dipole @ direct_dipole),
        }
    return result


def harmonic_deformation_check(electronic: dict, harmonic: dict) -> dict:
    """Compare local quadratic predictions at the other surface's minimum."""
    direct = electronic["energy_differences_cm^-1"]
    differences = (
        ("ground_deformation_at_excited", "ground_at_excited"),
        ("excited_relaxation_from_ground", "excited_at_ground"),
    )
    report = {}
    for energy_key, surface_key in differences:
        direct_energy = direct[energy_key]
        quadratic = harmonic[surface_key]["total_quadratic_reorganization_cm^-1"]
        report[energy_key] = {
            "direct_electronic_cm^-1": direct_energy,
            "local_quadratic_cm^-1": quadratic,
            "quadratic_minus_direct_cm^-1": quadratic - direct_energy,
            "quadratic_over_direct_ratio": quadratic / direct_energy if direct_energy > 0 else None,
        }
    return report
