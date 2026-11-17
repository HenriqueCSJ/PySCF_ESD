"""Read raw OpenQP MRSF energy points and evaluate the cross-geometry MO metric."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mrsf_overlap import MRSFSingletExpansion, expand_restricted_mrsf_singlets
from .roks_stability import ROKSReferenceLink, assess_roks_reference_link
from .state_tracking import _real_array


OPENQP_ENGINE_COMMIT = "1db79089b285406338bd28a255b967e4011f8df7"
_ENERGY_ATOL = 1e-9
_MATRIX_ATOL = 1e-8


@dataclass(frozen=True)
class OpenQPMRSFPoint:
    """Physical response roots and alpha orbitals in one checkpoint's raw gauge."""

    atoms: tuple[int, ...]
    coordinates_bohr: NDArray[np.float64]
    basis: tuple[str, str, bool]
    mo_coefficients: NDArray[np.float64]
    root_energies_hartree: NDArray[np.float64]
    _transition_dipoles_au: NDArray[np.float64]
    singlet_expansion: MRSFSingletExpansion | None = None
    _auxiliary_energy_hartree: float | None = None
    _ao_metric: NDArray[np.float64] | None = None
    _portable_basis_set: Mapping[str, object] | None = None

    def transition_dipole(self, first_root: int, second_root: int) -> NDArray[np.float64]:
        """Return the raw signed Cartesian dipole for two distinct zero-based roots."""
        nroots = self.root_energies_hartree.size
        roots = (first_root, second_root)
        if first_root == second_root or any(
            isinstance(root, (bool, np.bool_))
            or not isinstance(root, Integral)
            or not 0 <= root < nroots for root in roots
        ):
            raise ValueError("select two distinct zero-based MRSF response roots")
        return self._transition_dipoles_au[:, min(roots), max(roots)].copy()

    def cross_ao_overlap(self, current: "OpenQPMRSFPoint") -> NDArray[np.float64]:
        """Evaluate raw cross-AO integrals from the two complete checkpoint records."""
        from .openqp_ao_overlap import exact_openqp_ao_overlap

        if self.atoms != current.atoms or self.basis != current.basis:
            raise ValueError("OpenQP points must use the same atom ordering and basis")
        if (self._portable_basis_set is None or current._portable_basis_set is None
                or self._ao_metric is None or current._ao_metric is None):
            raise ValueError("both checkpoints need portable basis data and a native AO metric")
        if self._portable_basis_set != current._portable_basis_set:
            raise ValueError("the OpenQP portable basis definitions differ between points")
        return exact_openqp_ao_overlap(
            self.atoms, self.coordinates_bohr, current.coordinates_bohr, self._portable_basis_set,
            engine_commit=OPENQP_ENGINE_COMMIT, bra_native_ao_overlap=self._ao_metric,
            ket_native_ao_overlap=current._ao_metric,
        )

    def signed_state_overlap(
        self, current: "OpenQPMRSFPoint", ao_cross_overlap: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate the signed overlap of the decoded restricted MRSF roots."""
        if self.singlet_expansion is None or current.singlet_expansion is None:
            raise ValueError("read both checkpoints with the verified n_core for state overlap")
        if ao_cross_overlap is None:
            ao_cross_overlap = self.cross_ao_overlap(current)
        metric = exact_openqp_mo_overlap(self, current, ao_cross_overlap)
        return self.singlet_expansion.overlap(current.singlet_expansion, metric)

    def auxiliary_reference_link(
        self, current: "OpenQPMRSFPoint", ao_cross_overlap: ArrayLike | None = None, *,
        minimum_singular_value: float,
    ) -> ROKSReferenceLink:
        """Screen core and SOMO retention before transporting the singlet roots."""
        n_core = _shared_core_count(self, current)
        if ao_cross_overlap is None:
            ao_cross_overlap = self.cross_ao_overlap(current)
        metric = exact_openqp_mo_overlap(self, current, ao_cross_overlap)
        return assess_roks_reference_link(
            metric, n_core=n_core, minimum_singular_value=minimum_singular_value,
        )


def _shared_core_count(previous: OpenQPMRSFPoint, current: OpenQPMRSFPoint) -> int:
    left, right = previous.singlet_expansion, current.singlet_expansion
    if left is None or right is None:
        raise ValueError("read both checkpoints with the verified n_core for reference overlap")
    if left.n_core != right.n_core:
        raise ValueError("the two references must have the same core count")
    return left.n_core


@dataclass(frozen=True)
class _OpenQPMRSFEnergyPoint:
    atoms: tuple[int, ...]
    coordinates_bohr: NDArray[np.float64]
    basis: tuple[str, str, bool]
    mo_coefficients: NDArray[np.float64]
    ao_metric: NDArray[np.float64]
    auxiliary_energy_hartree: float
    root_energies_hartree: NDArray[np.float64]
    response_offsets_hartree: NDArray[np.float64]


def _unpack_ao_matrix(packed: NDArray[np.float64], nao: int) -> NDArray[np.float64]:
    rows, cols = np.tril_indices(nao)
    matrix = np.empty((nao, nao))
    matrix[rows, cols] = packed
    matrix[cols, rows] = packed
    return matrix


def _array(checkpoint: Mapping[str, object], key: str, ndim: int) -> NDArray[np.float64]:
    try:
        return _real_array(checkpoint[key], ndim, key)
    except KeyError as exc:
        raise ValueError(f"missing {key} in full OpenQP MRSF energy checkpoint") from exc


def _raw_tag(
    checkpoint: Mapping[str, object], key: str, shape: tuple[int, ...],
) -> NDArray[np.float64]:
    try:
        values = np.asarray(checkpoint[key])
    except KeyError as exc:
        raise ValueError(f"missing {key} in full OpenQP MRSF energy checkpoint") from exc
    flat = _real_array(values.ravel(order="C"), 1, key)
    if flat.size != np.prod(shape):
        raise ValueError(f"{key} has an inconsistent OpenQP tag size")
    return flat.reshape(shape, order="F")


def _check_portable_occupations(checkpoint: Mapping[str, object], n_core: int, nao: int) -> None:
    if "molecular_orbitals" not in checkpoint:
        return
    orbitals = checkpoint["molecular_orbitals"]
    for spin, occupied in (("alpha", n_core + 2), ("beta", n_core)):
        entry = orbitals.get(spin) if isinstance(orbitals, Mapping) else None
        if not isinstance(entry, Mapping) or not np.array_equal(
            entry.get("occupancies"), np.arange(nao) < occupied,
        ):
            raise ValueError("OpenQP portable occupations disagree with the stated two-SOMO reference")


def _read_energy_point(
    checkpoint: Mapping[str, object], engine_commit: str,
) -> _OpenQPMRSFEnergyPoint:
    if engine_commit != OPENQP_ENGINE_COMMIT:
        raise ValueError("unsupported OpenQP checkpoint engine commit")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("OpenQP checkpoint must be a decoded JSON object")
    if any(
        key in ("state_tracking", "OQP::td_states_overlap")
        or (isinstance(key, str) and key.startswith(("OQP::state_tracking_", "OQP::mo_tracking_")))
        for key in checkpoint
    ):
        raise ValueError("requires an untransported OpenQP MRSF energy checkpoint")

    config = checkpoint.get("json")
    if not isinstance(config, Mapping) or config.get("scf_type") != "rohf":
        raise ValueError("requires an OpenQP ROHF checkpoint")
    nao = config.get("nbf")
    basis_name = config.get("basis")
    basis_library = config.get("library")
    spherical = config.get("spherical_harmonics")
    if (
        type(nao) is not int
        or nao < 1
        or not isinstance(basis_name, str)
        or not basis_name
        or not isinstance(basis_library, str)
        or type(spherical) is not bool
    ):
        raise ValueError("missing or invalid OpenQP checkpoint AO basis metadata")
    basis = (basis_name, basis_library, spherical)

    atoms = _array(checkpoint, "atoms", 1)
    coordinates = _array(checkpoint, "coord", 1)
    if (
        not atoms.size
        or not np.all((atoms >= 0) & (atoms <= 118))
        or not np.all(atoms == np.round(atoms))
        or coordinates.size != 3 * atoms.size
    ):
        raise ValueError("invalid OpenQP checkpoint atom numbers or Cartesian coordinates")
    coordinates = coordinates.reshape(-1, 3).copy()

    offsets = _array(checkpoint, "OQP::td_energies", 1)
    totals = _array(checkpoint, "energies", 1)
    reference = _array(checkpoint, "energy", 0).item()
    public_offsets = _array(checkpoint, "td_energies", 1)
    nroots = offsets.size
    if (
        nroots < 2
        or totals.shape != (nroots + 1,)
        or public_offsets.shape != offsets.shape
        or abs(reference - totals[0]) > _ENERGY_ATOL
        or not np.allclose(public_offsets, offsets, rtol=0, atol=_ENERGY_ATOL)
        or not np.allclose(totals[1:] - totals[0], offsets, rtol=0, atol=_ENERGY_ATOL)
    ):
        raise ValueError("OpenQP total Hartree energies must match the checkpoint's response-root order")
    energies = totals[1:].copy()
    coefficients = _raw_tag(checkpoint, "OQP::VEC_MO_A", (nao, nao)).copy()
    ntri = nao * (nao + 1) // 2
    ao_metric = _unpack_ao_matrix(_raw_tag(checkpoint, "OQP::SM", (ntri,)), nao)
    if not np.allclose(coefficients.T @ ao_metric @ coefficients, np.eye(nao), rtol=0, atol=_MATRIX_ATOL):
        raise ValueError("OpenQP checkpoint alpha MOs are not orthonormal in its AO metric")
    offsets = offsets.copy()
    for values in (coordinates, energies, coefficients, ao_metric, offsets):
        values.setflags(write=False)
    return _OpenQPMRSFEnergyPoint(
        tuple(atoms.astype(int).tolist()), coordinates, basis, coefficients, ao_metric,
        reference, energies, offsets,
    )


def read_openqp_mrsf_point(
    checkpoint: Mapping[str, object], *, engine_commit: str, n_core: int | None = None,
) -> OpenQPMRSFPoint:
    """Read a completed OpenQP singlet MRSF-TDDFT energy run.

    The caller verifies the pinned engine, the singlet MRSF input recipe,
    and the electronic model. The saved JSON lacks the full input recipe.
    Root energies retain the original Hartree zero. Dipoles are checked
    against the signed state-interaction density.

    Supply the verified core count to enable state overlap and auxiliary
    reference diagnostics. The pinned JSON serializer provides excitation
    rows and root columns. With n_core supplied, the vectors must reconstruct
    the stored one-body densities in that root order.
    """
    energy_point = _read_energy_point(checkpoint, engine_commit)
    coefficients = energy_point.mo_coefficients
    nao = coefficients.shape[0]
    nroots = energy_point.root_energies_hartree.size
    ntri = nao * (nao + 1) // 2

    expansion = None
    if n_core is not None:
        vectors = _array(checkpoint, "OQP::td_bvec_mo", 2)
        if vectors.shape[1] != nroots:
            raise ValueError("OpenQP response-vector columns must match the checkpoint root energies")
        expansion = expand_restricted_mrsf_singlets(vectors, n_core=n_core, n_orbitals=nao)
        _check_portable_occupations(checkpoint, expansion.n_core, nao)

    shape = (3, nroots, nroots)
    raw_dipoles = _raw_tag(checkpoint, "OQP::td_trans_dipole", shape)
    dipoles = _array(checkpoint, "td_trans_dipole", 3)
    if dipoles.shape != shape or not np.allclose(dipoles, raw_dipoles, rtol=0, atol=_MATRIX_ATOL):
        raise ValueError("OpenQP public transition dipoles disagree with the signed raw tag")
    density_mo = _raw_tag(checkpoint, "OQP::td_trans_density_mo", (nao, nao, nroots, nroots))
    if expansion is not None:
        for bra, ket in zip(*np.triu_indices(nroots)):
            if not np.allclose(
                density_mo[:, :, bra, ket], expansion._response_density_mo(bra, ket),
                rtol=0, atol=_MATRIX_ATOL,
            ):
                raise ValueError("OpenQP response vectors disagree with the signed density in root order")
    dipole_ao = _raw_tag(checkpoint, "OQP::td_dip_ao", (ntri, 3))
    first, second = np.triu_indices(nroots, k=1)
    for axis in range(3):
        operator = _unpack_ao_matrix(dipole_ao[:, axis], nao)
        operator_mo = coefficients.T @ operator @ coefficients
        reconstructed = -np.einsum("ab,abij->ij", operator_mo, density_mo)
        if not np.allclose(
            dipoles[axis, first, second], reconstructed[first, second], rtol=0, atol=_MATRIX_ATOL
        ):
            raise ValueError("OpenQP signed transition dipole disagrees with its state-interaction density")

    dipoles = dipoles.copy()
    dipoles.setflags(write=False)
    basis_set = checkpoint.get("basis_set")
    portable_basis = deepcopy(dict(basis_set)) if isinstance(basis_set, Mapping) else None
    return OpenQPMRSFPoint(
        energy_point.atoms, energy_point.coordinates_bohr, energy_point.basis, coefficients,
        energy_point.root_energies_hartree, dipoles, expansion, energy_point.auxiliary_energy_hartree,
        energy_point.ao_metric, portable_basis,
    )


def paired_openqp_triplet_gap_hartree(
    singlet: OpenQPMRSFPoint, triplet_checkpoint: Mapping[str, object], *, engine_commit: str,
) -> float:
    """Return the lowest two triplet response roots' separation at a singlet point.

    The caller verifies that the companion used the same input recipe except
    for the requested response spin and root count, and that it converged
    both lowest triplet roots. A full companion checkpoint must agree with
    the singlet's geometry, AO convention and auxiliary determinant.
    The gap is a branch-risk diagnostic, independent of the SCF stability.
    """
    if singlet.singlet_expansion is None or singlet._ao_metric is None:
        raise ValueError("read the singlet checkpoint with the verified n_core for the triplet check")
    triplet = _read_energy_point(triplet_checkpoint, engine_commit)
    if (triplet.atoms != singlet.atoms or triplet.basis != singlet.basis
            or triplet.ao_metric.shape != singlet._ao_metric.shape
            or not np.allclose(triplet.coordinates_bohr, singlet.coordinates_bohr, atol=1e-10, rtol=0)
            or not np.allclose(triplet.ao_metric, singlet._ao_metric, atol=_MATRIX_ATOL, rtol=0)):
        raise ValueError("the triplet checkpoint must use the same geometry and AO convention")
    metric = singlet.mo_coefficients.T @ singlet._ao_metric @ triplet.mo_coefficients
    link = assess_roks_reference_link(
        metric, n_core=singlet.singlet_expansion.n_core, minimum_singular_value=1.0 - _MATRIX_ATOL,
    )
    if (not link.passes_overlap_screen or singlet._auxiliary_energy_hartree is None
            or abs(triplet.auxiliary_energy_hartree - singlet._auxiliary_energy_hartree) > _ENERGY_ATOL):
        raise ValueError("the triplet checkpoint belongs to a different auxiliary determinant")
    _check_portable_occupations(triplet_checkpoint, singlet.singlet_expansion.n_core, metric.shape[1])
    offsets = triplet.response_offsets_hartree
    if np.any(np.diff(offsets) < 0):
        raise ValueError("OpenQP triplet responses must be in ascending raw root order")
    return float(offsets[1] - offsets[0])


def exact_openqp_mo_overlap(
    previous: OpenQPMRSFPoint,
    current: OpenQPMRSFPoint,
    ao_cross_overlap: ArrayLike,
    *,
    native_current_by_previous_mo_overlap: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Contract an independently evaluated raw AO overlap into the checkpoint MO gauges.

    AO rows belong to the previous point and columns to the current point.
    Supply AO overlaps in the exact OpenQP AO order, phase, and normalization
    at each point, in a common Cartesian frame. The optional native value is
    the OpenQP Python tag view with current-MO rows and previous-MO columns.
    Compare its transpose elementwise to the independent contraction; any
    detectable column normalization, phase change, or reordering fails.
    The returned MO metric is an array oracle for downstream state-overlap
    evaluation. Agreement alone does not qualify the native state kernel.
    """
    if previous.atoms != current.atoms or previous.basis != current.basis:
        raise ValueError("OpenQP points must use the same atom ordering and basis")
    left, right = previous.mo_coefficients, current.mo_coefficients
    cross = _real_array(ao_cross_overlap, 2, "ao_cross_overlap")
    if cross.shape != (left.shape[0], right.shape[0]):
        raise ValueError("AO cross overlap dimensions must match the OpenQP checkpoint AO dimensions")
    overlap = left.T @ cross @ right
    if not np.all(np.isfinite(overlap)) or np.linalg.norm(overlap, ord=2) > 1.0 + _MATRIX_ATOL:
        raise ValueError("MO cross overlap is incompatible with orthonormal orbitals")
    if native_current_by_previous_mo_overlap is not None:
        native = _real_array(native_current_by_previous_mo_overlap, 2, "native OpenQP MO overlap")
        if native.T.shape != overlap.shape or not np.allclose(native.T, overlap, rtol=0, atol=_MATRIX_ATOL):
            raise ValueError("native OpenQP MO overlap disagrees with the signed raw AO contraction")
    return overlap
