"""Signed overlaps between real expansions of collinear Slater determinants."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .state_tracking import _real_array


def determinant_expansion_overlap(
    bra_coefficients: ArrayLike,
    ket_coefficients: ArrayLike,
    bra_alpha: ArrayLike,
    bra_beta: ArrayLike,
    ket_alpha: ArrayLike,
    ket_beta: ArrayLike,
    mo_overlap_alpha: ArrayLike,
    *,
    mo_overlap_beta: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Return overlaps in the electronic states' input phases.

    On each side, the alpha and beta occupation arrays have one row per
    determinant. Orbitals are zero-based and ordered in creation-operator
    order within each row, with the alpha string followed by the beta string.
    Changing this order changes the determinant phase. An empty spin string
    has shape ``(ndeterminants, 0)``. Electron counts per spin must match.

    Each MO overlap has bra rows and ket columns and uses the native orbital
    order and phases. Orbitals are orthonormal within each side. If the beta
    matrix is omitted, both spins use the alpha matrix. Coefficients are used
    as supplied, including phases and norms. Repeated occupation pairs add
    their coefficients with the signs implied by their occupation order.

    Supply coefficient vectors of shape ``(ndeterminants,)`` or batches of
    shape ``(nstates, ndeterminants)``. The result has a bra and a ket state
    axis, with an axis omitted for each coefficient vector input.
    """
    alpha = _real_array(mo_overlap_alpha, 2, "alpha MO overlap")
    beta = alpha if mo_overlap_beta is None else _real_array(
        mo_overlap_beta, 2, "beta MO overlap"
    )
    bra, bra_a, bra_b, bra_scalar = _expansion(
        bra_coefficients, bra_alpha, bra_beta, (alpha.shape[0], beta.shape[0]), "bra"
    )
    ket, ket_a, ket_b, ket_scalar = _expansion(
        ket_coefficients, ket_alpha, ket_beta, (alpha.shape[1], beta.shape[1]), "ket"
    )
    if bra_a.shape[1] != ket_a.shape[1] or bra_b.shape[1] != ket_b.shape[1]:
        raise ValueError("bra and ket must have the same electron count in each spin")

    bra_a, bra_a_index = np.unique(bra_a, axis=0, return_inverse=True)
    bra_b, bra_b_index = np.unique(bra_b, axis=0, return_inverse=True)
    ket_a, ket_a_index = np.unique(ket_a, axis=0, return_inverse=True)
    ket_b, ket_b_index = np.unique(ket_b, axis=0, return_inverse=True)
    # NumPy releases have differed in the shape of return_inverse with axis.
    bra_a_index, bra_b_index = bra_a_index.ravel(), bra_b_index.ravel()
    ket_a_index, ket_b_index = ket_a_index.ravel(), ket_b_index.ravel()
    alpha_minors = _spin_minors(alpha, bra_a, ket_a)
    beta_minors = _spin_minors(beta, bra_b, ket_b)
    result = _contract(
        bra, ket, bra_a_index, bra_b_index, ket_a_index, ket_b_index, alpha_minors, beta_minors
    )
    if bra_scalar:
        result = result[0]
    if ket_scalar:
        result = result[..., 0]
    return float(result) if bra_scalar and ket_scalar else result


def _expansion(coefficients, alpha, beta, orbital_counts, label):
    dimensions = np.ndim(coefficients)
    if dimensions not in (1, 2):
        raise ValueError(f"{label} coefficients must be a vector or a matrix of state rows")
    coefficients = np.atleast_2d(_real_array(coefficients, dimensions, f"{label} coefficients"))
    states, determinants = coefficients.shape
    if states == 0 or determinants == 0:
        raise ValueError(f"{label} must contain at least one state and determinant")
    alpha, alpha_sign = _canonical_occupations(alpha, determinants, orbital_counts[0], f"{label} alpha")
    beta, beta_sign = _canonical_occupations(beta, determinants, orbital_counts[1], f"{label} beta")
    return coefficients * (alpha_sign * beta_sign), alpha, beta, dimensions == 1


def _canonical_occupations(data, determinants, orbital_count, label):
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] != determinants:
        raise ValueError(f"{label} occupations must have one row per determinant")
    if data.size and (
        not np.issubdtype(data.dtype, np.integer)
        or np.any(data < 0)
        or np.any(data >= orbital_count)
    ):
        raise ValueError(f"{label} occupations must contain integer orbital indices in range")
    data = data.astype(np.intp, copy=False)
    ordered = np.sort(data, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(f"{label} occupations contain a repeated orbital in one string")
    signs = np.ones(determinants)
    for index in np.flatnonzero(np.any(ordered != data, axis=1)):
        row = data[index]
        inversions = np.count_nonzero(np.triu(row[:, None] > row[None, :], k=1))
        signs[index] = (-1) ** inversions
    return ordered, signs


def _determinant_and_cofactors(matrix):
    # Cofactors from the SVD use products with one singular value omitted.
    # They remain defined for a singular block and never divide by its determinant.
    left, values, right = np.linalg.svd(matrix)
    orientation = np.linalg.slogdet(left)[0] * np.linalg.slogdet(right)[0]
    products = np.ones_like(values)
    products[1:] *= np.cumprod(values[:-1])
    products[:-1] *= np.cumprod(values[:0:-1])[::-1]
    return orientation * np.prod(values), orientation * (left * products) @ right


def _removed_orbital(occupations, orbitals):
    positions = np.searchsorted(orbitals, occupations)
    return orbitals.size * (orbitals.size - 1) // 2 - np.sum(positions, axis=1)


def _core_and_particle(occupations, orbitals, counts):
    occupied_count = occupations.shape[1]
    if len(occupations) == 1:
        return occupations[0, :-1], occupations[:, -1], np.ones(1)
    core = orbitals[counts == len(occupations)]
    if core.size != occupied_count - 1:
        return None
    particle_rows, particle_positions = np.nonzero(~np.isin(occupations, core))
    signs = 1 - 2 * ((occupied_count - 1 - particle_positions) % 2)
    return core, occupations[particle_rows, particle_positions], signs


def _spin_minors(metric, bra, ket):
    size = bra.shape[1]
    shape = (len(bra), len(ket))
    if size == 0:
        return np.ones(shape)
    if size == 1:
        return metric[np.ix_(bra[:, 0], ket[:, 0])]
    if shape == (1, 1):
        return np.linalg.det(metric[np.ix_(bra[0], ket[0])]).reshape(shape)

    bra_union, bra_counts = np.unique(bra, return_counts=True)
    ket_union, ket_counts = np.unique(ket, return_counts=True)
    if bra_union.size == ket_union.size == size + 1:
        # One hole per string. The sorted occupations select unsigned minors.
        bra_hole = _removed_orbital(bra, bra_union)
        ket_hole = _removed_orbital(ket, ket_union)
        _, cofactors = _determinant_and_cofactors(metric[np.ix_(bra_union, ket_union)])
        signs = 1 - 2 * ((bra_hole[:, None] + ket_hole[None, :]) % 2)
        return signs * cofactors[np.ix_(bra_hole, ket_hole)]

    bra_parts = _core_and_particle(bra, bra_union, bra_counts)
    ket_parts = _core_and_particle(ket, ket_union, ket_counts)
    if bra_parts is not None and ket_parts is not None:
        bra_core, bra_particle, bra_sign = bra_parts
        ket_core, ket_particle, ket_sign = ket_parts
        core_det, cofactors = _determinant_and_cofactors(metric[np.ix_(bra_core, ket_core)])
        # For a bordered block [C B; D E], det = det(C) * E - D @ adj(C) @ B.
        rows = metric[np.ix_(bra_particle, ket_core)]
        cols = metric[np.ix_(bra_core, ket_particle)]
        values = core_det * metric[np.ix_(bra_particle, ket_particle)] - rows @ cofactors.T @ cols
        return bra_sign[:, None] * values * ket_sign[None, :]

    # Bound the temporary stack of occupied blocks for general occupations.
    values = np.empty(shape)
    flat = values.ravel()
    batch_size = max(1, (1 << 18) // (size * size))
    for start in range(0, flat.size, batch_size):
        stop = min(flat.size, start + batch_size)
        bra_rows, ket_rows = np.divmod(np.arange(start, stop), len(ket))
        blocks = metric[bra[bra_rows, :, None], ket[ket_rows, None, :]]
        flat[start:stop] = np.linalg.det(blocks)
    return values


def _contract(bra, ket, bra_a, bra_b, ket_a, ket_b, alpha_minors, beta_minors):
    # Spin flips usually fill the alpha-hole by beta-particle grid. Contract
    # it with two matrix products instead of building all determinant pairs.
    bra_grid = alpha_minors.shape[0] * beta_minors.shape[0]
    ket_shape = (alpha_minors.shape[1], beta_minors.shape[1])
    if bra_grid <= 2 * bra.shape[1] and np.prod(ket_shape) <= 2 * ket.shape[1]:
        result = np.empty((len(bra), len(ket)))
        for root, coefficients in enumerate(ket):
            grid = np.zeros(ket_shape)
            np.add.at(grid, (ket_a, ket_b), coefficients)
            transformed = alpha_minors @ grid @ beta_minors.T
            result[:, root] = bra @ transformed[bra_a, bra_b]
        return result

    result = np.zeros((len(bra), len(ket)))
    batch_size = max(1, (1 << 18) // ket.shape[1])
    for start in range(0, bra.shape[1], batch_size):
        block = slice(start, start + batch_size)
        values = alpha_minors[np.ix_(bra_a[block], ket_a)] * beta_minors[np.ix_(bra_b[block], ket_b)]
        result += (bra[:, block] @ values) @ ket.T
    return result
