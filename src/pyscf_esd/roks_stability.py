"""Internal real-orbital stability of a high-spin CPU ROKS determinant."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import re
from types import SimpleNamespace

import numpy as np
from numpy.typing import ArrayLike

from .cpu import (
    ElectronicConvergenceError,
    OrbitalStabilityChannel,
    ReferenceStabilityUnresolvedError,
    _STABILITY_EIGEN_TOL,
    _checked_array,
    _stability_curvature,
)
from .state_tracking import _real_array


@dataclass(frozen=True)
class ROKSInternalStabilityReport:
    """Curvature evidence for shared real rotations at fixed ROKS occupations."""

    scf_residual_hartree: float
    scf_residual_tolerance_hartree: float
    minimum_curvature: float
    rotation_dimension: int
    internal: OrbitalStabilityChannel
    fixed_energy_difference_hartree: float | None = None
    fixed_fock_max_difference_hartree: float | None = None

    @property
    def internal_qualified(self) -> bool:
        """Whether the internal real ROKS curvature met the requested margin."""
        return self.internal.status == "qualified"


@dataclass(frozen=True)
class ROKSReferenceLink:
    """Gauge-invariant occupation-subspace retention between two ROKS points."""

    core_singular_values: tuple[float, ...]
    somo_singular_values: tuple[float, ...]
    alpha_occupied_singular_values: tuple[float, ...]
    core_full_space_retention: tuple[float, float] | None
    somo_full_space_retention: tuple[float, float]
    minimum_singular_value: float

    @property
    def passes_overlap_screen(self) -> bool:
        """Whether the core and SOMO spaces independently meet the retention cutoff."""
        return all(
            value >= self.minimum_singular_value
            for value in (*self.core_singular_values, *self.somo_singular_values,
                          *self.alpha_occupied_singular_values)
        )


def assess_roks_reference_link(
    mo_overlap: ArrayLike, *, n_core: int, minimum_singular_value: float,
) -> ROKSReferenceLink:
    """Screen a link between two shared-orbital references with two SOMOs.

    Supply the raw bra-by-ket MO metric in a common Cartesian frame, with
    orbitals ordered core, SOMOs, virtuals on each side. Singular values are
    invariant to rotations within those occupation classes. The full alpha
    space is also reported because it can remain unchanged during a core/SOMO
    exchange. The full-space retention of each class is the smallest
    singular value of its projection onto the other point's complete MO
    space, in forward and reverse order. This exposes loss of basis support
    without renormalizing the link. Stationarity, orbital Hessian stability
    and the identity of the lowest triplet require separate checks.
    """
    metric = _real_array(mo_overlap, 2, "raw ROKS cross-MO overlap")
    if (isinstance(n_core, (bool, np.bool_)) or not isinstance(n_core, Integral)
            or n_core < 0 or n_core + 2 > min(metric.shape)):
        raise ValueError("the two MO spaces must have the stated core and two SOMOs")
    if (isinstance(minimum_singular_value, (bool, np.bool_))
            or not isinstance(minimum_singular_value, Real)
            or not 0 < minimum_singular_value <= 1):
        raise ValueError("minimum_singular_value must be in (0, 1]")
    occupied = n_core + 2
    if np.linalg.norm(metric, ord=2) > 1.0 + 1e-8:
        raise ValueError("the cross-MO overlap is incompatible with orthonormal orbitals")

    def singular_values(block):
        return tuple(np.minimum(np.linalg.svd(block, compute_uv=False), 1.0).tolist())

    def full_space_retention(orbitals):
        return singular_values(metric[orbitals, :])[-1], singular_values(metric[:, orbitals])[-1]

    return ROKSReferenceLink(
        singular_values(metric[:n_core, :n_core]),
        singular_values(metric[n_core:occupied, n_core:occupied]),
        singular_values(metric[:occupied, :occupied]),
        full_space_retention(slice(0, n_core)) if n_core else None,
        full_space_retention(slice(n_core, occupied)),
        minimum_singular_value,
    )


def assess_roks_internal_stability(
    mean_field, *, scf_residual_tolerance: float = 1e-7, minimum_curvature: float = 1e-6,
) -> ROKSInternalStabilityReport:
    """Assess a converged molecular CPU ROKS reference with positive spin.

    Use PySCF's analytic shared-orbital Hessian at the SCF quadrature, including
    every real rotation between different occupation classes. The residual is
    the norm of PySCF's orbital gradient (half the energy derivative in these
    coordinates). Curvature and its margin use the native stability scaling.
    A failure curvature can be a verified Rayleigh quotient. This assessment
    is scoped to the real, shared-orbital ROKS manifold.
    """
    from pyscf.dft.roks import ROKS

    if not isinstance(mean_field, ROKS):
        raise TypeError("internal ROKS stability requires a molecular CPU ROKS reference")
    if not mean_field.converged:
        raise ElectronicConvergenceError("Converge the ROKS determinant before assessing orbital stability.")
    return _assess_roks_internal(
        mean_field, scf_residual_tolerance=scf_residual_tolerance,
        minimum_curvature=minimum_curvature,
    )


def assess_fixed_roks_internal_stability(
    mean_field, *, reference_energy_hartree: float,
    energy_tolerance_hartree: float = 1e-8, reference_fock_mo: ArrayLike | None = None,
    fock_tolerance_hartree: float = 1e-7, scf_residual_tolerance: float = 1e-7,
    minimum_curvature: float = 1e-6,
) -> ROKSInternalStabilityReport:
    """Check a fixed external ROKS determinant with a prescribed numerical model.

    Use a configured molecular CPU ROKS object with imported orbitals,
    occupations, and the actual reference quadrature in the grid coordinates
    (bohr) and weights. The caller checks the XC and Coulomb/exchange recipe.
    The supplied energy is the total for the auxiliary ROKS determinant.
    An optional native Fock comparison uses the fresh alpha and beta
    operators in the unchanged input MO gauge.

    The routine evaluates the energy, Fock and residual at those orbitals
    and uses them for the analytic internal Hessian. It never invokes SCF
    or changes the convergence flag. Failure of the energy, Fock or
    stationarity check precludes a curvature result.
    """
    from pyscf.dft.roks import ROKS
    from pyscf.lib import temporary_env

    if not isinstance(mean_field, ROKS):
        raise TypeError("internal ROKS stability requires a molecular CPU ROKS reference")
    if mean_field.grids.coords is None or mean_field.grids.weights is None:
        raise ReferenceStabilityUnresolvedError(
            "internal", "fixed ROKS reference", "Supply the actual reference XC grid coordinates and weights.",
        )
    coordinates = _real_array(mean_field.grids.coords, 2, "fixed ROKS quadrature coordinates")
    weights = _real_array(mean_field.grids.weights, 1, "fixed ROKS quadrature weights")
    if coordinates.shape != (weights.size, 3) or not np.any(weights):
        raise ValueError("fixed ROKS quadrature must have usable weights and one Cartesian row per point")
    if isinstance(reference_energy_hartree, (bool, np.bool_)) or (
        not isinstance(reference_energy_hartree, Real) or not isfinite(reference_energy_hartree)
    ):
        raise ValueError("reference_energy_hartree must be finite")
    for name, value in (
        ("energy_tolerance_hartree", energy_tolerance_hartree),
        ("fock_tolerance_hartree", fock_tolerance_hartree),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    # A grid object's old screening tables may describe earlier coordinates.
    # Evaluate both the Fock and response from the supplied points themselves.
    with temporary_env(mean_field.grids, non0tab=None, screen_index=None):
        return _assess_roks_internal(
            mean_field, scf_residual_tolerance=scf_residual_tolerance, minimum_curvature=minimum_curvature,
            reference_energy_hartree=reference_energy_hartree, energy_tolerance_hartree=energy_tolerance_hartree,
            reference_fock_mo=reference_fock_mo, fock_tolerance_hartree=fock_tolerance_hartree,
        )


def _assess_roks_internal(
    mean_field, *, scf_residual_tolerance, minimum_curvature,
    reference_energy_hartree=None, energy_tolerance_hartree=None,
    reference_fock_mo=None, fock_tolerance_hartree=None,
):
    from pyscf.scf.rohf import make_rdm1
    from pyscf.soscf.newton_ah import gen_g_hop_rohf

    for name, value in (
        ("scf_residual_tolerance", scf_residual_tolerance), ("minimum_curvature", minimum_curvature),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    mol = mean_field.mol
    nalpha, nbeta = mol.nelec
    if nalpha <= nbeta:
        raise ValueError("internal ROKS stability requires positive-spin high-spin occupations")
    _require_matching_response(mean_field)
    nao = mol.nao_nr()
    if not isinstance(mean_field.mo_coeff, np.ndarray):
        raise TypeError("internal ROKS stability requires real NumPy molecular orbitals")
    coefficients = _checked_array(mean_field.mo_coeff, (nao, nao), "complete ROKS orbitals")
    occupations = _checked_array(mean_field.mo_occ, (nao,), "ROKS occupations")
    spin_occupation = np.stack((occupations > 0, occupations == 2)).astype(int)
    if not np.isin(occupations, (0, 1, 2)).all() or tuple(spin_occupation.sum(axis=1)) != (nalpha, nbeta):
        raise ElectronicConvergenceError("ROKS occupations do not match the molecule's high-spin electron counts.")
    overlap = _checked_array(mean_field.get_ovlp(), (nao, nao), "ROKS AO overlap")
    if not np.allclose(coefficients.T @ overlap @ coefficients, np.eye(nao), atol=1e-7, rtol=0):
        raise ElectronicConvergenceError("ROKS orbitals are not orthonormal in the AO overlap.")

    density = make_rdm1(coefficients, occupations)
    hcore = mean_field.get_hcore()
    potential = mean_field.get_veff(dm=density)
    fock = _checked_array(hcore + potential, (2, nao, nao), "fresh ROKS spin Fock matrices")
    if not np.allclose(fock, fock.transpose(0, 2, 1), atol=1e-9, rtol=0):
        raise ElectronicConvergenceError("Fresh ROKS spin Fock matrices are not symmetric.")

    fock_mo = coefficients.T @ fock @ coefficients
    energy_difference = fock_difference = None
    if reference_energy_hartree is not None:
        total = mean_field.energy_tot(dm=density, h1e=hcore, vhf=potential)
        energy_difference = float(total - reference_energy_hartree)
        if not isfinite(energy_difference) or abs(energy_difference) > energy_tolerance_hartree:
            raise ReferenceStabilityUnresolvedError(
                "internal", "fixed ROKS reference",
                f"Fresh total energy differs by {energy_difference:.3g} Hartree. "
                "Check the imported determinant, XC, exchange, and quadrature.",
            )
    if reference_fock_mo is not None:
        expected = _checked_array(reference_fock_mo, fock_mo.shape, "reference spin Fock matrices in MO order")
        fock_difference = float(np.max(np.abs(fock_mo - expected)))
        if fock_difference > fock_tolerance_hartree:
            raise ReferenceStabilityUnresolvedError(
                "internal", "fixed ROKS reference",
                f"Fresh spin Fock matrices differ by up to {fock_difference:.3g} Hartree in the imported MO basis.",
            )
    gradient = _roks_orbital_gradient(fock_mo, spin_occupation)
    rotation_dimension = gradient.size
    residual = float(np.linalg.norm(gradient))
    if not isfinite(residual) or residual > scf_residual_tolerance:
        raise ElectronicConvergenceError(
            f"ROKS residual {residual:.3g} Hartree exceeds the stability input tolerance "
            f"{scf_residual_tolerance:.3g}."
        )

    channel = "internal"
    try:
        if not rotation_dimension:
            raise ElectronicConvergenceError("The nonredundant ROKS rotation space is empty.")
        native_gradient, hop, diagonal = gen_g_hop_rohf(
            mean_field, coefficients, occupations,
            fock_ao=SimpleNamespace(focka=fock[0], fockb=fock[1]), with_symmetry=False,
        )
        shape = (rotation_dimension,)
        native_gradient = _checked_array(native_gradient, shape, "native ROKS orbital gradient")
        if not np.allclose(native_gradient, gradient, atol=1e-10, rtol=0):
            raise ElectronicConvergenceError("The native ROKS gradient disagrees with the fresh spin Fock derivative.")
        diagonal = _checked_array(diagonal, shape, "ROKS orbital Hessian preconditioner")

        def operator(vector):
            return 2 * _checked_array(hop(vector), shape, "ROKS orbital Hessian product")

        evidence = _stability_curvature(operator, 2 * diagonal, np.nextafter(minimum_curvature, np.inf))
    except (ElectronicConvergenceError, ImportError, NotImplementedError) as error:
        internal = OrbitalStabilityChannel(channel, "unresolved", reason=str(error))
    else:
        if evidence.below_threshold_direction is None:
            status = "qualified"
        elif evidence.curvature < -_STABILITY_EIGEN_TOL:
            status = "negative_direction"
        else:
            status = "below_margin"
        internal = OrbitalStabilityChannel(channel, status, evidence.curvature, evidence.residual)
    return ROKSInternalStabilityReport(
        residual, float(scf_residual_tolerance), float(minimum_curvature), rotation_dimension, internal,
        energy_difference, fock_difference,
    )


def _roks_orbital_gradient(spin_fock_mo, spin_occupation):
    # A closed–open pair contributes beta, open–virtual alpha, and
    # closed–virtual both. The convention matches PySCF's half-gradient.
    occupations = spin_occupation.sum(axis=0)
    rows, columns = np.nonzero(occupations[:, None] < occupations[None, :])
    weights = spin_occupation[:, columns] - spin_occupation[:, rows]
    return np.sum(weights * spin_fock_mo[:, rows, columns], axis=0)


def _require_matching_response(mean_field) -> None:
    """Check that PySCF's internal response differentiates this SCF Fock."""
    from pyscf import __version__

    channel, stage = "internal", "ROKS reference"
    version = re.match(r"^(\d+)\.(\d+)", __version__)
    if version is None or tuple(map(int, version.groups())) < (2, 15):
        raise ReferenceStabilityUnresolvedError(
            channel, stage, "The analytic shared-orbital ROKS Hessian requires PySCF 2.15 or newer.",
        )
    if mean_field.do_nlc():
        raise ReferenceStabilityUnresolvedError(
            channel, stage, "PySCF's internal orbital Hessian omits the nonlocal correlation response.",
        )
    if mean_field._numint._xc_type(mean_field.xc) not in ("LDA", "GGA"):
        raise ValueError("internal ROKS stability requires an LDA or GGA DFT functional, including hybrid GGAs")
    if getattr(mean_field, "with_solvent", None) is not None:
        raise ReferenceStabilityUnresolvedError(
            channel, stage, "The checked internal ROKS response requires a gas-phase reference.",
        )
    second_grids = getattr(mean_field, "second_grids", None)
    if second_grids is not None and second_grids is not mean_field.grids:
        raise ReferenceStabilityUnresolvedError(
            channel, stage, "PySCF's internal orbital Hessian would use a different quadrature from the SCF Fock.",
        )
    scf = getattr(mean_field, "_scf", None)
    if scf is not None and scf.mol is not mean_field.mol:
        raise ReferenceStabilityUnresolvedError(
            channel, stage, "Internal ROKS stability requires the SCF and response AO bases to match.",
        )
