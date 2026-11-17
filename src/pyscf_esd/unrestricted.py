"""Analytic nuclear derivatives of a prepared high-spin UKS determinant."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .cpu import ElectronicConvergenceError, _checked_array, _readonly


@dataclass(frozen=True)
class UKSTripletEndpoint:
    """Spin-free UKS total energy and native analytic nuclear derivatives.

    Axes of the Hessian are (atom A, atom B, Cartesian A, Cartesian B).
    Derivatives are read-only host arrays in the input molecule's atom order.
    The reported spin is a diagnostic of the unrestricted KS determinant.
    """

    energy_hartree: float
    gradient_hartree_per_bohr: NDArray[np.float64]
    hessian_hartree_per_bohr2: NDArray[np.float64] | None
    spin_squared: float
    effective_multiplicity: float
    electron_counts: tuple[int, int]
    scf_residual_hartree: float
    backend: Literal["pyscf", "gpu4pyscf"]
    density_fitting: bool
    gradient_grid_response: bool
    hessian_grid_response: bool | None

    @property
    def spin_contamination(self) -> float:
        """Return the determinant's signed excess over the triplet S(S+1) = 2."""
        return self.spin_squared - 2.0


def analytic_uks_triplet_endpoint(
    mean_field,
    *,
    include_hessian: bool = False,
    gradient_grid_response: bool = True,
    scf_residual_tolerance: float = 1e-7,
) -> UKSTripletEndpoint:
    """Evaluate an already converged molecular UKS reference with M_S = +1.

    Use the native direct or fully density-fitted PySCF or GPU4PySCF kernels.
    The model is all-electron, nonrelativistic, gas-phase LDA or GGA, including
    hybrid GGAs, with atom-centered XC quadrature and without point-group
    constraints. The gradient includes
    quadrature response by default. Use ``gradient_grid_response=False`` to
    hold the quadrature fixed in both derivatives. The Hessian uses fixed
    XC quadrature and the input reference's response settings. Density-fitted
    derivatives include auxiliary-basis response.

    The orbital gradient at the supplied density must satisfy the residual
    tolerance (Hartree, Euclidean norm). Retain the converged canonical
    orbitals, energies, geometry and electronic settings. The caller is responsible
    for orbital stability and tracking the intended triplet character. The
    endpoint can be at a nonstationary nuclear geometry. No SCF iteration,
    geometry change, numerical differentiation or backend conversion is used.
    """
    if not isinstance(include_hessian, (bool, np.bool_)):
        raise ValueError("include_hessian must be a boolean")
    if not isinstance(gradient_grid_response, (bool, np.bool_)):
        raise ValueError("gradient_grid_response must be a boolean")
    if (isinstance(scf_residual_tolerance, (bool, np.bool_))
            or not isinstance(scf_residual_tolerance, Real)
            or not isfinite(scf_residual_tolerance)
            or scf_residual_tolerance <= 0.0):
        raise ValueError("scf_residual_tolerance must be positive and finite")

    backend = _uks_backend(mean_field)
    mol = mean_field.mol
    if mol.spin != 2:
        raise ValueError("the high-spin UKS endpoint requires mol.spin = 2 (M_S = +1)")
    if not mean_field.converged:
        raise ElectronicConvergenceError("the UKS reference must already be converged")
    density_fitting = getattr(mean_field, "with_df", None) is not None
    _check_model(mean_field, density_fitting)

    coefficients = mean_field.mo_coeff
    if backend == "gpu4pyscf":
        import cupy

        if not isinstance(coefficients, cupy.ndarray):
            raise ElectronicConvergenceError("GPU4PySCF UKS orbitals must be CUDA arrays")
        array_module = cupy
        device = coefficients.device
    else:
        array_module = np
        device = nullcontext()

    with device:
        electron_counts, nmo = _check_orbitals(mean_field, array_module)
        energy = float(_checked_array(_to_host(mean_field.e_tot), (), "UKS total energy"))
        # The native get_grad builds a fresh physical Fock matrix when none is given.
        orbital_gradient = mean_field.get_grad(coefficients, mean_field.mo_occ)
        residual_shape = (sum(nocc * (nmo - nocc) for nocc in electron_counts),)
        residual = float(np.linalg.norm(_checked_array(
            _to_host(orbital_gradient), residual_shape, "UKS orbital gradient",
        )))
        if residual > scf_residual_tolerance:
            raise ElectronicConvergenceError(
                f"UKS orbital residual {residual:.3g} Hartree exceeds tolerance"
            )
        # GPU4PySCF and PySCF both return the two determinant spin diagnostics as scalars.
        spin_squared, multiplicity = _checked_array(
            mean_field.spin_square(), (2,), "UKS determinant spin diagnostics",
        )
        if spin_squared < 2.0 - 1e-7:
            raise ElectronicConvergenceError("UKS spin diagnostics are inconsistent with M_S = +1")

        gradient = _native_derivative(
            mean_field.nuc_grad_method(), backend, (mol.natm, 3), "gradient",
            grid_response=bool(gradient_grid_response),
            auxbasis_response=True if density_fitting else None,
        )
        hessian = None
        if include_hessian:
            hess_shape = (mol.natm, mol.natm, 3, 3)
            hessian = _native_derivative(
                mean_field.Hessian(), backend, hess_shape,
                "Hessian", grid_response=False, auxbasis_response=2 if density_fitting else None,
            )

    return UKSTripletEndpoint(
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        spin_squared=float(spin_squared),
        effective_multiplicity=float(multiplicity),
        electron_counts=electron_counts,
        scf_residual_hartree=residual,
        backend=backend,
        density_fitting=density_fitting,
        gradient_grid_response=bool(gradient_grid_response),
        hessian_grid_response=False if include_hessian else None,
    )


def _uks_backend(mean_field) -> Literal["pyscf", "gpu4pyscf"]:
    # DF and solver decorators subclass the native UKS type.
    for cls in type(mean_field).__mro__:
        if cls.__name__ == "UKS":
            if cls.__module__ == "pyscf.dft.uks":
                return "pyscf"
            if cls.__module__ == "gpu4pyscf.dft.uks":
                return "gpu4pyscf"
    raise TypeError("mean_field must be a molecular PySCF or GPU4PySCF UKS reference")


def _check_model(mean_field, density_fitting: bool) -> None:
    mol = mean_field.mol
    if getattr(mol, "symmetry", False):
        raise NotImplementedError("the UKS endpoint requires a reference without point-group symmetry")
    if mol.ecp or mol.pseudo or getattr(mean_field, "with_x2c", None) is not None:
        raise NotImplementedError("the UKS endpoint requires an all-electron, nonrelativistic model")
    if (getattr(mean_field, "with_solvent", None) is not None
            or getattr(mean_field, "mm_mol", None) is not None):
        raise NotImplementedError("the UKS endpoint requires a gas-phase model")
    if mean_field.do_nlc() or mean_field.do_disp():
        raise NotImplementedError("the UKS endpoint does not cover NLC or empirical dispersion")
    do_gcp = getattr(mean_field, "do_gcp", None)
    if do_gcp is not None and do_gcp():
        raise NotImplementedError("the UKS endpoint does not cover geometrical counterpoise corrections")
    if mean_field._numint._xc_type(mean_field.xc) not in ("LDA", "GGA"):
        raise NotImplementedError("the UKS endpoint requires LDA or GGA XC, including hybrid GGAs")
    if density_fitting and getattr(mean_field, "only_dfj", False):
        raise NotImplementedError("the UKS endpoint requires full J/K density fitting when DF is enabled")


def _check_orbitals(mean_field, array_module) -> tuple[tuple[int, int], int]:
    coefficients = array_module.asarray(mean_field.mo_coeff)
    energies = array_module.asarray(mean_field.mo_energy)
    if (coefficients.ndim != 3 or coefficients.shape[:2] != (2, mean_field.mol.nao_nr())
            or coefficients.shape[2] == 0):
        raise ElectronicConvergenceError("UKS requires both sets of molecular orbitals")
    nmo = coefficients.shape[2]
    if energies.shape != (2, nmo):
        raise ElectronicConvergenceError("UKS requires both sets of orbital energies")
    for values in (coefficients, energies):
        if values.dtype.kind != "f" or not bool(array_module.all(array_module.isfinite(values))):
            raise ElectronicConvergenceError("UKS orbital data must be real and finite")

    occupations = _checked_array(_to_host(mean_field.mo_occ), (2, nmo), "UKS occupations")
    if not np.all((occupations == 0) | (occupations == 1)):
        raise ElectronicConvergenceError("UKS triplet occupations must be 0 or 1 in each spin channel")
    electron_counts = (int(occupations[0].sum()), int(occupations[1].sum()))
    n_alpha, n_beta = electron_counts
    if n_alpha - n_beta != 2 or n_alpha + n_beta != mean_field.mol.nelectron:
        raise ElectronicConvergenceError("UKS occupations must match the molecular charge and M_S = +1")
    return electron_counts, nmo


def _native_derivative(
    derivative, backend: str, shape: tuple[int, ...], name: str, *,
    grid_response: bool, auxbasis_response: bool | int | None,
) -> NDArray[np.float64]:
    family = "grad" if name == "gradient" else "hessian"
    package = backend if auxbasis_response is None else f"{backend}.df"
    if type(derivative).__module__ != f"{package}.{family}.uks" or not hasattr(derivative, "grid_response"):
        raise NotImplementedError(f"native {backend} UKS {name} is unavailable for this model")
    derivative.grid_response = grid_response
    if auxbasis_response is not None:
        if not hasattr(derivative, "auxbasis_response"):
            raise NotImplementedError(f"native {backend} UKS {name} lacks auxiliary-basis response")
        derivative.auxbasis_response = auxbasis_response
    return _readonly(_checked_array(_to_host(derivative.kernel()), shape, f"UKS nuclear {name}"))


def _to_host(value):
    if hasattr(value, "__cuda_array_interface__"):
        return value.get()
    return value
