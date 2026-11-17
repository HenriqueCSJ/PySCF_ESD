"""Evaluate OpenQP AO metrics and import open-shell orbitals with PySCF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .openqp_mrsf_prepared import OPENQP_ENGINE_COMMIT
from .state_tracking import _real_array

if TYPE_CHECKING:
    from pyscf.gto import Mole


_SHELL_LABELS = "spdfg"
_METRIC_ATOL = 1e-8


@dataclass(frozen=True)
class _OpenQPAOBasis:
    atomic_numbers: NDArray[np.int64]
    shells: list[tuple[int, int, list[list[float]]]]
    spherical: bool
    nao: int
    orders: dict[int, NDArray[np.intp]]
    factors: NDArray[np.float64]
    contraction_scales: NDArray[np.float64]

    def build(
        self, coordinates, native_metric, label, *, charge: int, spin: int,
    ) -> tuple[Mole, NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
        from pyscf import gto
        from pyscf.data.elements import ELEMENTS

        # Unique labels preserve independent per-center bases for identical
        # elements. PySCF sorts shells by center and angular momentum.
        labels = [
            f"{ELEMENTS[number]}{index}"
            for index, number in enumerate(self.atomic_numbers, start=1)
        ]
        grouped_basis = {atom: [] for atom in labels}
        for center, angular, primitives in self.shells:
            grouped_basis[labels[center]].append([angular, *primitives])
        mol = gto.M(
            atom=list(zip(labels, coordinates)), unit="Bohr", basis=grouped_basis,
            cart=not self.spherical, charge=charge, spin=spin, verbose=0,
            parse_arg=False, dump_input=False,
        )
        indices = _native_ao_indices(mol, self.shells, self.orders)
        factors = self.factors
        if gto.mole.NORMALIZE_GTO:
            factors = factors * self.contraction_scales
        pyscf_metric = mol.intor_symmetric("int1e_ovlp")
        calculated = np.outer(factors, factors) * pyscf_metric[np.ix_(indices, indices)]
        if not np.allclose(calculated, native_metric, rtol=0, atol=_METRIC_ATOL):
            raise ValueError(f"{label} PySCF AO self overlap disagrees with the native OpenQP metric")
        return mol, indices, factors, pyscf_metric


def exact_openqp_ao_overlap(
    atoms: ArrayLike,
    bra_coordinates_bohr: ArrayLike,
    ket_coordinates_bohr: ArrayLike,
    basis_set: Mapping[str, object],
    *,
    engine_commit: str,
    bra_native_ao_overlap: ArrayLike,
    ket_native_ao_overlap: ArrayLike,
    native_components: Mapping[int, tuple[ArrayLike, ArrayLike]] | None = None,
) -> NDArray[np.float64]:
    """Return the signed AO overlap with native OpenQP bra rows and ket columns.

    Both geometries must use the same Cartesian frame, atom order, and
    portable basis. Supply the pinned OpenQP ``basis_set`` record shared
    by the checkpoints. The shell data, rather than the basis name or
    library, define the contractions. The portable record labels its
    exported MO rows as Molden-ordered. This function returns the native
    order used by the raw OpenQP ``OQP::SM`` and ``OQP::VEC_MO_A`` tags.

    Both native self metrics are required, as full AO matrices or packed
    lower triangles. The primitive coefficients supply the raw contraction
    norms that PySCF normally removes when building its basis. The default
    convention supports spherical shells through f and Cartesian s and p
    shells. To specify an independently verified angular convention, pass
    ``native_components[l] = (indices, factors)``. Native component ``i``
    uses PySCF angular component ``indices[i]`` with factor ``factors[i]``.
    """
    basis = _read_portable_basis(atoms, basis_set, engine_commit, native_components)
    bra = _real_array(bra_coordinates_bohr, 2, "bra coordinates")
    ket = _real_array(ket_coordinates_bohr, 2, "ket coordinates")
    if bra.shape != (len(basis.atomic_numbers), 3) or ket.shape != bra.shape:
        raise ValueError("both Bohr coordinate arrays must have one Cartesian row per atom")
    bra_metric = _native_metric(bra_native_ao_overlap, basis.nao, "bra")
    ket_metric = _native_metric(ket_native_ao_overlap, basis.nao, "ket")
    spin = int(basis.atomic_numbers.sum() % 2)
    bra_mol, bra_indices, bra_factors, _ = basis.build(bra, bra_metric, "bra", charge=0, spin=spin)
    ket_mol, ket_indices, ket_factors, _ = basis.build(ket, ket_metric, "ket", charge=0, spin=spin)

    from pyscf import gto

    cross = gto.intor_cross("int1e_ovlp", bra_mol, ket_mol)
    return np.outer(bra_factors, ket_factors) * cross[np.ix_(bra_indices, ket_indices)]


def import_openqp_orbitals(
    atoms: ArrayLike,
    coordinates_bohr: ArrayLike,
    basis_set: Mapping[str, object],
    *,
    engine_commit: str,
    native_ao_overlap: ArrayLike,
    native_mo_coefficients: ArrayLike,
    charge: int = 0,
    spin: int | None = None,
    native_components: Mapping[int, tuple[ArrayLike, ArrayLike]] | None = None,
) -> tuple[Mole, NDArray[np.float64]]:
    """Return a PySCF molecule and the same signed MOs in its AO basis.

    Supply the pinned portable ``basis_set``, the full or packed native AO
    overlap, and the real raw MO coefficient matrix in native AO-by-MO order.
    The AO convention and both MO orthonormality conditions are checked.
    All MO columns and their phases are retained, including for an orbital
    subset. The portable Molden-ordered MO rows require conversion to native
    order before calling this function.

    ``charge`` sets the molecular charge and ``spin`` sets the PySCF
    alpha-minus-beta electron count. Omit ``spin`` to use the minimum
    compatible with electron parity. The molecule is built without running
    SCF or assigning MO occupations.
    """
    basis = _read_portable_basis(atoms, basis_set, engine_commit, native_components)
    coordinates = _real_array(coordinates_bohr, 2, "coordinates")
    if coordinates.shape != (len(basis.atomic_numbers), 3):
        raise ValueError("Bohr coordinates must have one Cartesian row per atom")
    charge, spin = _charge_and_spin(charge, spin, int(basis.atomic_numbers.sum()))
    metric = _native_metric(native_ao_overlap, basis.nao, "input")
    coefficients = _real_array(native_mo_coefficients, 2, "native MO coefficients")
    if coefficients.shape[0] != basis.nao or not 1 <= coefficients.shape[1] <= basis.nao:
        raise ValueError("native MO coefficients must have one row per AO and at least one MO column")
    identity = np.eye(coefficients.shape[1])
    if not np.allclose(coefficients.T @ metric @ coefficients, identity, rtol=0, atol=_METRIC_ATOL):
        raise ValueError("native MO coefficients are not orthonormal in the OpenQP AO metric")
    mol, indices, factors, pyscf_metric = basis.build(coordinates, metric, "input", charge=charge, spin=spin)
    imported = np.empty_like(coefficients)
    imported[indices] = factors[:, None] * coefficients
    if not np.allclose(imported.T @ pyscf_metric @ imported, identity, rtol=0, atol=_METRIC_ATOL):
        raise ValueError("imported MO coefficients are not orthonormal in the PySCF AO metric")
    return mol, imported


def _charge_and_spin(charge, spin, nuclear_charge):
    if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, Integral):
        raise ValueError("charge must be an integer")
    charge = int(charge)
    electrons = nuclear_charge - charge
    if electrons < 0:
        raise ValueError("charge leaves a negative electron count")
    if spin is None:
        return charge, electrons % 2
    if isinstance(spin, (bool, np.bool_)) or not isinstance(spin, Integral):
        raise ValueError("spin must be an integer alpha-minus-beta electron count")
    spin = int(spin)
    if abs(spin) > electrons or (electrons - spin) % 2:
        raise ValueError("spin is incompatible with the electron count")
    return charge, spin


def _read_portable_basis(atoms, basis_set, engine_commit, native_components):
    if engine_commit != OPENQP_ENGINE_COMMIT:
        raise ValueError("unsupported OpenQP basis engine commit")
    atomic_numbers = np.asarray(atoms)
    if (
        atomic_numbers.ndim != 1
        or not atomic_numbers.size
        or not np.issubdtype(atomic_numbers.dtype, np.integer)
        or np.any((atomic_numbers < 1) | (atomic_numbers > 118))
    ):
        raise ValueError("atoms must contain OpenQP atomic numbers from 1 to 118")
    shells, spherical, nao = _portable_shells(basis_set, len(atomic_numbers))
    orders, factors = _native_component_maps(shells, spherical, native_components)
    return _OpenQPAOBasis(
        atomic_numbers.astype(np.int64, copy=False), shells, spherical, nao, orders, factors,
        _contraction_scales(shells, orders),
    )


def _portable_shells(basis_set, natoms):
    if not isinstance(basis_set, Mapping) or (
        basis_set.get("format") != "OpenQP portable basis v1"
        or basis_set.get("coefficient_convention") != "molden"
        or basis_set.get("basis_function_order") != "molden"
        or type(basis_set.get("spherical_harmonics")) is not bool
        or type(basis_set.get("nbf")) is not int
    ):
        raise ValueError("requires the pinned OpenQP portable basis v1 metadata")
    portable_shells = basis_set.get("shells")
    if not isinstance(portable_shells, Sequence) or not portable_shells:
        raise ValueError("OpenQP portable basis must contain a sequence of shells")
    spherical = basis_set["spherical_harmonics"]
    nao = basis_set["nbf"]
    shells = []
    for shell in portable_shells:
        if not isinstance(shell, Mapping):
            raise ValueError("invalid OpenQP portable shell")
        center, angular = shell.get("atom_index"), shell.get("angular_momentum")
        if (
            type(center) is not int
            or not 0 <= center < natoms
            or type(angular) is not int
            or not 0 <= angular < len(_SHELL_LABELS)
            or shell.get("shell") != _SHELL_LABELS[angular]
        ):
            raise ValueError("invalid OpenQP portable shell center or angular momentum")
        exponents = _real_array(shell.get("exponents"), 1, "primitive exponents")
        coefficients = _real_array(shell.get("coefficients"), 1, "primitive coefficients")
        if (
            exponents.size == 0
            or exponents.shape != coefficients.shape
            or np.any(exponents <= 0)
            or not np.any(coefficients)
        ):
            raise ValueError("invalid OpenQP portable primitive exponents or coefficients")
        shells.append((center, angular, np.column_stack((exponents, coefficients)).tolist()))
    if {shell[0] for shell in shells} != set(range(natoms)):
        raise ValueError("OpenQP portable basis must contain a shell for every atom")
    if nao != sum(_component_count(angular, spherical) for _, angular, _ in shells):
        raise ValueError("OpenQP portable nbf disagrees with the shell representation")
    return shells, spherical, nao


def _component_count(angular, spherical):
    return 2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2


def _contraction_scales(shells, orders):
    scales = []
    for _, angular, primitives in shells:
        exponents, coefficients = np.asarray(primitives).T
        # In this convention the primitives are normalized independently.
        # Their same-center overlap fixes the contraction norm without fitting
        # to the native overlap. The ratio form avoids large products of exponents.
        ratio = np.minimum.outer(exponents, exponents) / np.maximum.outer(exponents, exponents)
        overlap = (2 * np.sqrt(ratio) / (1 + ratio)) ** (angular + 1.5)
        squared_norm = coefficients @ overlap @ coefficients
        if not np.isfinite(squared_norm) or squared_norm <= 0:
            raise ValueError("OpenQP portable shell has an invalid contraction norm")
        scales.append(np.full(orders[angular].size, np.sqrt(squared_norm)))
    return np.concatenate(scales)


def _native_component_maps(shells, spherical, native_components):
    if native_components is None:
        native_components = {}
    if not isinstance(native_components, Mapping):
        raise ValueError("native component maps must be keyed by shell angular momentum")
    orders, factors = {}, {}
    for angular in {shell[1] for shell in shells}:
        size = _component_count(angular, spherical)
        if angular in native_components:
            indices, weights = native_components[angular]
            indices = np.asarray(indices)
            weights = _real_array(weights, 1, "native AO component factors")
            if (
                indices.shape != (size,)
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(np.sort(indices), np.arange(size))
                or weights.shape != (size,)
                or np.any(weights == 0)
            ):
                raise ValueError("invalid native AO component permutation or factors")
            orders[angular], factors[angular] = indices, weights
        elif angular <= 1 or (spherical and angular <= 3):
            orders[angular], factors[angular] = np.arange(size), np.ones(size)
        else:
            raise ValueError(f"an explicit native AO component map is required for {_SHELL_LABELS[angular]} shells")
    return orders, np.concatenate([factors[angular] for _, angular, _ in shells])


def _native_metric(values, nao, label):
    ndim = np.ndim(values)
    if ndim not in (1, 2):
        raise ValueError(f"{label} native AO overlap must be packed or square")
    array = _real_array(values, ndim, f"{label} native AO overlap")
    if array.shape == (nao, nao):
        return array
    if array.shape != (nao * (nao + 1) // 2,):
        raise ValueError(f"{label} native AO overlap size must agree with the basis")
    rows, cols = np.tril_indices(nao)
    matrix = np.empty((nao, nao))
    matrix[rows, cols] = matrix[cols, rows] = array
    return matrix


def _native_ao_indices(mol, shells, orders):
    shell_order = sorted(range(len(shells)), key=lambda i: shells[i][:2])
    starts = mol.ao_loc_nr()
    indices = [None] * len(shells)
    if mol.nbas != len(shells):
        raise ValueError("PySCF changed the number of OpenQP shells")
    for pyscf_shell, native_shell in enumerate(shell_order):
        center, angular, _ = shells[native_shell]
        order = orders[angular]
        start, stop = starts[pyscf_shell:pyscf_shell + 2]
        if (
            mol.bas_atom(pyscf_shell) != center
            or mol.bas_angular(pyscf_shell) != angular
            or stop - start != order.size
        ):
            raise ValueError("PySCF AO shell layout disagrees with the OpenQP shell data")
        indices[native_shell] = start + order
    return np.concatenate(indices)
