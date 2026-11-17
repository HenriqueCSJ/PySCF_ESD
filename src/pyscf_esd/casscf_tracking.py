"""Signed overlaps between real restricted complete-active-space states."""

from math import comb
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from .state_tracking import StateTrackingError, _real_array


def casscf_state_overlap(
    reference_ci: ArrayLike,
    candidate_ci: ArrayLike,
    reference_mo_coeff: ArrayLike,
    candidate_mo_coeff: ArrayLike,
    ao_cross_overlap: ArrayLike,
    *,
    ncore: int,
    ncas: int,
    nelecas: tuple[int, int],
) -> float:
    """Return the many-electron overlap in the states' input phases.

    Use one CAS specification with the same active alpha and beta electron
    counts and a restricted doubly occupied core. Supply normalized real CI
    arrays in PySCF alpha/beta determinant order. MO columns are orthonormal
    in their native AO metrics and ordered as core, active, then any virtuals.
    The cross-AO overlap has reference rows and candidate columns. Both
    geometries must use the same Cartesian frame.

    The doubly occupied core contributes its cross-overlap determinant
    squared. The Schur complement then accounts for core–active mixing in
    the occupied determinant overlap of each spin block. A singular or poorly
    conditioned core precludes this reduction and raises StateTrackingError.
    """
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in (ncore, ncas)) or ncore < 0 or ncas < 1:
        raise ValueError("ncore and ncas must be integers with ncore >= 0 and ncas >= 1")
    if len(nelecas) != 2 or any(
        isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or not 0 <= count <= ncas
        for count in nelecas
    ):
        raise ValueError("nelecas must give the alpha and beta active electron counts")

    orbitals = ncore + ncas
    reference = _real_array(reference_mo_coeff, 2, "reference MO coefficients")
    candidate = _real_array(candidate_mo_coeff, 2, "candidate MO coefficients")
    cross = _real_array(ao_cross_overlap, 2, "AO cross-overlap")
    if reference.shape[1] < orbitals or candidate.shape[1] < orbitals:
        raise ValueError("MO coefficients must include all core and active orbitals")
    if cross.shape != (reference.shape[0], candidate.shape[0]):
        raise ValueError("AO cross-overlap must match the reference and candidate AO dimensions")

    shape = tuple(comb(ncas, count) for count in nelecas)
    bra = _real_array(reference_ci, 2, "reference CI coefficients")
    ket = _real_array(candidate_ci, 2, "candidate CI coefficients")
    for coefficients in (bra, ket):
        if coefficients.shape != shape:
            raise ValueError("CI coefficients must match the CAS alpha and beta determinant counts")
        if not np.isclose(np.linalg.norm(coefficients), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("CAS CI coefficients must have unit norm")

    mo_cross = reference[:, :orbitals].T @ cross @ candidate[:, :orbitals]
    active_cross = mo_cross[ncore:, ncore:]
    core_weight = 1.0
    if ncore:
        core_cross = mo_cross[:ncore, :ncore]
        singular = np.linalg.svd(core_cross, compute_uv=False)
        # Even a uniformly small core is unsafe to invert on the scale of
        # the full orbital overlap, despite its small condition number.
        threshold = np.sqrt(np.finfo(float).eps) * max(1.0, singular[0])
        if singular[-1] <= threshold:
            raise StateTrackingError(
                f"common core overlap is singular or ill-conditioned (smallest singular value {singular[-1]:.3g})"
            )
        active_cross = active_cross - mo_cross[ncore:, :ncore] @ np.linalg.solve(
            core_cross, mo_cross[:ncore, ncore:],
        )
        core_weight = float(np.linalg.det(core_cross)) ** 2

    from pyscf.fci.addons import overlap

    result = core_weight * float(overlap(bra, ket, ncas, nelecas, s=active_cross))
    if not np.isfinite(result) or abs(result) > 1.0 + 1e-8:
        raise StateTrackingError("CAS orbital overlaps produced an invalid normalized state overlap")
    return float(np.clip(result, -1.0, 1.0))
