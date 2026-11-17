"""Audit and assess a saved OpenQP high-spin DFT determinant without SCF."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .openqp_ao_overlap import import_openqp_orbitals
from .openqp_mrsf_prepared import (
    OPENQP_ENGINE_COMMIT,
    _array,
    _check_portable_occupations,
    _raw_tag,
    _unpack_ao_matrix,
)
from .roks_stability import (
    ROKSInternalStabilityReport,
    _roks_orbital_gradient,
    assess_fixed_roks_internal_stability,
)
from .state_tracking import _real_array


@dataclass(frozen=True)
class OpenQPSavedROKSReference:
    """One snapshot's determinant and gradient from its recorded spin Focks."""

    atoms: tuple[int, ...]
    coordinates_bohr: NDArray[np.float64]
    native_ao_overlap: NDArray[np.float64]
    native_mo_coefficients: NDArray[np.float64]
    occupations: NDArray[np.intp]
    spin_fock_mo_hartree: NDArray[np.float64]
    total_energy_hartree: float
    saved_fock_residual_hartree: float


def read_openqp_saved_roks_reference(
    checkpoint: Mapping[str, object], *, engine_commit: str, n_core: int,
) -> OpenQPSavedROKSReference:
    """Read raw spin-density and Fock records in a shared two-SOMO AO basis.

    Verify the generating DFT input and reference multiplicity externally.
    The densities must reproduce the two occupied spin projectors before
    the saved spin Focks are used. The returned residual evaluates those
    records and does not certify a fresh, final-density Fock.
    """
    if engine_commit != OPENQP_ENGINE_COMMIT:
        raise ValueError("unsupported OpenQP checkpoint engine commit")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("OpenQP checkpoint must be a decoded JSON object")
    config = checkpoint.get("json")
    if not isinstance(config, Mapping) or config.get("scf_type") != "rohf":
        raise ValueError("requires a saved OpenQP restricted open-shell reference")
    nao = config.get("nbf")
    if (type(nao) is not int or isinstance(n_core, (bool, np.bool_)) or not isinstance(n_core, Integral)
            or not 0 <= n_core <= nao - 2):
        raise ValueError("supply the core count for a saved two-SOMO orbital space")
    _check_portable_occupations(checkpoint, n_core, nao)

    atoms = _array(checkpoint, "atoms", 1)
    coordinates = _array(checkpoint, "coord", 1)
    if (not atoms.size or np.any((atoms < 1) | (atoms > 118)) or np.any(atoms != np.round(atoms))
            or coordinates.size != 3 * atoms.size):
        raise ValueError("invalid OpenQP atom numbers or Bohr coordinates")
    coordinates = coordinates.reshape(-1, 3).copy()
    coefficients = _raw_tag(checkpoint, "OQP::VEC_MO_A", (nao, nao)).copy()
    ntri = nao * (nao + 1) // 2

    def unpack(tag):
        return _unpack_ao_matrix(_raw_tag(checkpoint, "OQP::" + tag, (ntri,)), nao)

    overlap = unpack("SM")
    dual = overlap @ coefficients
    if not np.allclose(coefficients.T @ dual, np.eye(nao), atol=1e-8, rtol=0):
        raise ValueError("OpenQP shared MOs are not orthonormal in the saved AO metric")
    indices = np.arange(nao)
    spin_occupation = np.array([indices < n_core + 2, indices < n_core], dtype=np.intp)
    fock_mo = np.empty((2, nao, nao))
    for spin_index, spin in enumerate(("A", "B")):
        density_mo = dual.T @ unpack("DM_" + spin) @ dual
        if not np.allclose(density_mo, np.diag(spin_occupation[spin_index]), atol=1e-8, rtol=0):
            raise ValueError(f"OpenQP saved spin-{spin} density disagrees with its occupied orbitals")
        fock_mo[spin_index] = coefficients.T @ unpack("FOCK_" + spin) @ coefficients
    occupations = spin_occupation.sum(axis=0)
    residual = float(np.linalg.norm(_roks_orbital_gradient(fock_mo, spin_occupation)))
    for array in (coordinates, overlap, coefficients, occupations, fock_mo):
        array.setflags(write=False)
    return OpenQPSavedROKSReference(
        tuple(atoms.astype(int)), coordinates, overlap, coefficients, occupations, fock_mo,
        _array(checkpoint, "energy", 0).item(), residual,
    )


def assess_openqp_roks_internal_stability(
    checkpoint: Mapping[str, object], *, engine_commit: str, n_core: int, charge: int,
    xc: str, grid_coordinates_bohr: ArrayLike, grid_weights: ArrayLike,
    energy_tolerance_hartree: float = 1e-8, fock_tolerance_hartree: float = 1e-7,
    scf_residual_tolerance: float = 1e-7, minimum_curvature: float = 1e-6,
) -> ROKSInternalStabilityReport:
    """Check the imported OpenQP determinant with its specified DFT quadrature.

    Pass the effective reference XC, in PySCF notation, and the actual
    nuclear-frame quadrature arrays. This gas-phase all-electron adapter
    uses exact PySCF Coulomb and exchange. The caller checks that this
    treatment and the declared functional correspond to the saved model.
    This runs no SCF and checks fresh energy and spin Focks against the
    stored reference before evaluating the analytic orbital Hessian.
    """
    from pyscf import dft

    reference = read_openqp_saved_roks_reference(checkpoint, engine_commit=engine_commit, n_core=n_core)
    molecule, coefficients = import_openqp_orbitals(
        reference.atoms, reference.coordinates_bohr, checkpoint.get("basis_set"),
        engine_commit=engine_commit, native_ao_overlap=reference.native_ao_overlap,
        native_mo_coefficients=reference.native_mo_coefficients, charge=charge, spin=2,
    )
    if molecule.nelec != (n_core + 2, n_core):
        raise ValueError("the declared charge disagrees with the saved two-SOMO reference")
    mean_field = dft.ROKS(molecule, xc=xc)
    mean_field.mo_coeff = coefficients
    mean_field.mo_occ = reference.occupations
    mean_field.grids.coords = _real_array(grid_coordinates_bohr, 2, "OpenQP XC grid coordinates").copy()
    mean_field.grids.weights = _real_array(grid_weights, 1, "OpenQP XC grid weights").copy()
    mean_field.small_rho_cutoff = 0
    return assess_fixed_roks_internal_stability(
        mean_field, reference_energy_hartree=reference.total_energy_hartree,
        energy_tolerance_hartree=energy_tolerance_hartree,
        reference_fock_mo=reference.spin_fock_mo_hartree,
        fock_tolerance_hartree=fock_tolerance_hartree, scf_residual_tolerance=scf_residual_tolerance,
        minimum_curvature=minimum_curvature,
    )
