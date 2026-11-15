"""Direct GPU4PySCF RKS/PBE0 and singlet TDA for the analytic VG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cpu import (
    ElectronicConvergenceError,
    ElectronicPoint,
    OrbitalSeed,
    ScfAttempt,
    Settings,
    _checked_array,
    _MAX_DENSE_TDA_DIMENSION,
    _readonly,
    _TDA_RESPONSE_BLOCK_SIZE,
    build_molecule,
)

if TYPE_CHECKING:
    from .cpu_workflow import CpuVibronicResult


@dataclass(frozen=True)
class GpuElectronicPoint(ElectronicPoint):
    """Device-resident RKS/TDA state with read-only NumPy electronic data."""

    hessian_grid_response: bool = False

    def ground_gradient(self) -> NDArray[np.float64]:
        """Return the analytic S0 gradient with grid response in Hartree/bohr."""
        with self._mean_field.mo_coeff.device:
            gradient = self._mean_field.nuc_grad_method()
            gradient.grid_response = True
            return _fp64_host(gradient.kernel(), (self.mol.natm, 3), "ground-state gradient")

    def excited_gradient(self, root_index: int) -> NDArray[np.float64]:
        """Return the analytic total gradient of the zero-based singlet root."""
        if isinstance(root_index, bool) or not isinstance(root_index, Integral):
            raise ValueError("root_index must be a zero-based integer")
        if not 0 <= root_index < len(self.excitation_energies_hartree):
            raise ValueError("root_index is outside the solved TDA roots")
        with self._mean_field.mo_coeff.device:
            gradient = self._tda.Gradients()
            gradient.cphf_max_cycle = self.settings.cphf_max_cycle
            gradient.cphf_conv_tol = self.settings.cphf_conv_tol
            return _fp64_host(
                gradient.kernel(state=int(root_index) + 1),
                (self.mol.natm, 3), "excited-state gradient",
            )

    def ground_hessian(self) -> NDArray[np.float64]:
        """Return the analytic S0 Hessian in PySCF order and Hartree/bohr²."""
        with self._mean_field.mo_coeff.device:
            hessian = self._mean_field.Hessian()
            hessian.grid_response = self.hessian_grid_response
            hessian.max_cycle = self.settings.cphf_max_cycle
            natm = self.mol.natm
            hessian.solve_mo1 = partial(_checked_hessian_response, hessian, hessian.solve_mo1)
            try:
                return _fp64_host(hessian.kernel(), (natm, natm, 3, 3), "ground-state Hessian")
            finally:
                del hessian.solve_mo1

    def reference_stability(self) -> tuple[bool, bool]:
        """Return GPU4PySCF's internal and external RKS stability flags."""
        with self._mean_field.mo_coeff.device:
            return super().reference_stability()

    def direct_jk_evidence(self) -> dict:
        """Probe the direct J/K builder on the final device density for validation."""
        import cupy

        mf = self._mean_field
        with mf.mo_coeff.device:
            coulomb, exchange = mf.get_jk(self.mol, mf.make_rdm1())
            matrices = {}
            nao = self.mol.nao_nr()
            for name, matrix in (("coulomb", coulomb), ("exchange", exchange)):
                _require_device_fp64(matrix, cupy, name)
                _fp64_host(matrix, (nao, nao), name)
                matrices[name] = {
                    "array_type": f"{type(matrix).__module__}.{type(matrix).__name__}",
                    "dtype": matrix.dtype.name,
                    "device_id": matrix.device.id,
                    "shape": list(matrix.shape),
                }
        return {
            "mean_field": f"{type(mf).__module__}.{type(mf).__name__}",
            "get_j": mf.get_j.__module__,
            "get_k": mf.get_k.__module__,
            "get_jk": mf.get_jk.__module__,
            "direct_scf": bool(mf.direct_scf),
            "density_fitting": getattr(mf, "with_df", None) is not None,
            "cphf_uses_primary_grid": mf.cphf_grids is mf.grids,
            "hessian_grid_response": self.hessian_grid_response,
            **matrices,
        }


def solve_point(
    symbols: Sequence[str],
    coordinates_bohr: ArrayLike,
    settings: Settings,
    *,
    previous: OrbitalSeed | ElectronicPoint | None = None,
    hessian_grid_response: bool = False,
) -> GpuElectronicPoint:
    """Solve one all-electron closed-shell PBE0/TDA point with native GPU operators.

    The analytic VG route uses the primary unpruned grid for all response
    steps. The default fixed-grid ground Hessian matches the effective CPU
    route. ``hessian_grid_response=True`` requests GPU4PySCF's grid response.
    GPU4PySCF's analytic TD-RKS gradient has no grid-response switch.
    """
    if previous is not None:
        raise ValueError("GPU solve_point does not support an orbital seed")
    if settings.xc.strip().upper() != "PBE0":
        raise ValueError("the direct GPU backend requires PBE0")
    if not isinstance(hessian_grid_response, (bool, np.bool_)):
        raise ValueError("hessian_grid_response must be a boolean")

    mol = build_molecule(symbols, coordinates_bohr, settings)
    nocc = mol.nelectron // 2
    if settings.nstates > nocc * (mol.nao_nr() - nocc):
        raise ValueError("nstates exceeds the number of occupied–virtual pairs")
    cupy, mean_field = _make_mean_field(mol, settings)
    mean_field.kernel()
    residual, coefficients = _scf_residual(mean_field, cupy)
    attempt = ScfAttempt(
        "gpu_minao_diis", bool(mean_field.converged), residual,
        bool(mean_field.converged and residual <= settings.scf_residual_tol),
    )
    if not attempt.accepted:
        raise ElectronicConvergenceError(f"GPU RKS did not meet convergence checks: {attempt}")

    nvir = coefficients.shape[1] - nocc
    if settings.nstates > nocc * nvir:
        raise ValueError("nstates exceeds the retained occupied–virtual dimension")
    td = mean_field.TDA()
    td.singlet = True
    td.wfnsym = None
    td.nstates = int(settings.nstates)
    td.conv_tol = settings.td_residual_tol * 0.2
    td.max_cycle = int(settings.max_cycle)
    td.chkfile = None
    td.positive_eig_threshold = 0.0
    td.exclude_nlc = True
    td.kernel()
    tda_route = "gpu_davidson"
    try:
        roots, amplitudes, ratios = _validated_tda(td, cupy, nocc, nvir, settings.td_residual_tol)
    except ElectronicConvergenceError:
        _dense_tda_retry(td, cupy, nocc, nvir, settings.td_residual_tol)
        roots, amplitudes, ratios = _validated_tda(td, cupy, nocc, nvir, settings.td_residual_tol)
        tda_route = "gpu_dense_same_operator"
    dipoles = _fp64_host(td.transition_dipole(), (td.nstates, 3), "transition dipoles")
    return GpuElectronicPoint(
        mol=mol, settings=settings, mo_coeff=_readonly(coefficients), nocc=nocc,
        amplitudes=_readonly(amplitudes), excitation_energies_hartree=_readonly(roots),
        transition_dipoles_au=_readonly(dipoles), ground_energy_hartree=float(mean_field.e_tot),
        scf_residual=residual, tda_residuals=_readonly(ratios), attempts=(attempt,),
        _mean_field=mean_field, _tda=td, tda_route=tda_route,
        hessian_grid_response=bool(hessian_grid_response),
    )


def run_gpu_vibronic(
    symbols: Sequence[str], coordinates_bohr: ArrayLike, settings: Settings, *,
    target_index: int, stationary_tolerance: float = 1e-5,
    minimum_gap_hartree: float = 1e-4, ground_hessian_symmetry_tolerance: float = 1e-5,
    hessian_grid_response: bool = False,
) -> CpuVibronicResult:
    """Run the analytic VG model at a single stationary S0 geometry on GPU."""
    from .cpu_workflow import _prepare_vg_center, _validate_center_options

    center_options = dict(
        stationary_tolerance=stationary_tolerance, minimum_gap_hartree=minimum_gap_hartree,
        ground_hessian_symmetry_tolerance=ground_hessian_symmetry_tolerance,
    )
    _validate_center_options(settings, target_index, **center_options)
    center = solve_point(
        symbols, coordinates_bohr, settings, hessian_grid_response=hessian_grid_response,
    )
    return _prepare_vg_center(center, target_index, **center_options)


def _make_mean_field(mol, settings):
    try:
        import cupy
        from gpu4pyscf.dft.rks import RKS
        from gpu4pyscf.scf import diis, j_engine, jk
    except (ImportError, OSError) as exc:
        raise RuntimeError("the GPU backend requires a working CuPy and GPU4PySCF installation") from exc
    if cupy.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("the GPU backend requires a visible CUDA device")
    if mol.has_ecp() or getattr(mol, "pseudo", None):
        raise ValueError("the direct GPU backend requires an all-electron basis")
    if max(mol.bas_angular(shell) for shell in range(mol.nbas)) > min(j_engine.LMAX, jk.LMAX):
        raise ValueError("basis angular momentum exceeds the native GPU direct J/K limit")

    mean_field = RKS(mol, xc="pbe0")
    mean_field.grids.level = int(settings.grid_level)
    mean_field.grids.prune = None
    mean_field.cphf_grids = mean_field.grids
    mean_field.small_rho_cutoff = 0.0
    mean_field.conv_tol = settings.scf_conv_tol
    mean_field.conv_tol_grad = settings.scf_residual_tol * 0.2
    mean_field.conv_tol_cpscf = settings.cphf_conv_tol
    mean_field.max_cycle = int(settings.max_cycle)
    mean_field.conv_check = True
    mean_field.init_guess = "minao"
    mean_field.direct_scf = True
    mean_field.direct_scf_tol = settings.direct_scf_tol
    mean_field.diis = True
    mean_field.DIIS = diis.SCF_DIIS
    mean_field.diis_space = 8
    mean_field.diis_start_cycle = 1
    mean_field.diis_damp = 0.0
    mean_field.level_shift = 0.0
    mean_field.damp = 0.0
    mean_field.chkfile = None
    return cupy, mean_field


def _scf_residual(mean_field, cupy):
    mo = mean_field.mo_coeff
    _require_device_fp64(mo, cupy, "RKS orbitals")
    nao = mean_field.mol.nao_nr()
    nocc = mean_field.mol.nelectron // 2
    if mo.ndim != 2 or mo.shape[0] != nao or mo.shape[1] <= nocc:
        raise ElectronicConvergenceError("RKS did not provide the occupied and virtual orbitals")
    nmo = mo.shape[1]
    coefficients = _fp64_host(mo, (nao, nmo), "RKS orbitals")
    occupations = _fp64_host(mean_field.mo_occ, (nmo,), "RKS occupations")
    _fp64_host(mean_field.mo_energy, (nmo,), "RKS orbital energies")
    _fp64_host(mean_field.e_tot, (), "RKS energy")
    if not np.all(occupations[:nocc] == 2) or np.any(occupations[nocc:] != 0):
        raise ElectronicConvergenceError("RKS determinant does not have closed-shell occupations")
    overlap = mean_field.get_ovlp()
    _require_device_fp64(overlap, cupy, "RKS AO overlap")
    overlap_mo = _fp64_host(mo.T @ overlap @ mo, (nmo, nmo), "RKS MO overlap")
    if not np.allclose(overlap_mo, np.eye(nmo), atol=1e-7, rtol=0.0):
        raise ElectronicConvergenceError("RKS orbitals are not orthonormal in the AO overlap")

    # Rebuild the Fock matrix at the final determinant, bypassing DIIS history
    # and the incremental potential from the SCF iterations.
    density = mean_field.make_rdm1()
    _require_device_fp64(density, cupy, "RKS density")
    fock = mean_field.get_hcore() + mean_field.get_veff(dm=density)
    _require_device_fp64(fock, cupy, "fresh RKS Fock matrix")
    gradient = mean_field.get_grad(mo, mean_field.mo_occ, fock)
    _require_device_fp64(gradient, cupy, "RKS orbital gradient")
    return float(_fp64_host(cupy.linalg.norm(gradient), (), "RKS residual")), coefficients


def _checked_hessian_response(
    hessian, native_solve, mo_energy, mo_coeff, mo_occ, h1mo,
    fx=None, atmlst=None, max_memory=4000, verbose=None,
):
    """Check the native CPHF solution and correct its residual with the same GPU operator."""
    import cupy
    from gpu4pyscf.hessian.rhf import ALIGNED
    from gpu4pyscf.lib.cupy_helper import get_avail_mem
    from gpu4pyscf.pbc.gto.int1e import int1e_ipovlp
    from pyscf import lib

    if fx is None:
        fx = hessian.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = native_solve(mo_energy, mo_coeff, mo_occ, h1mo, fx, atmlst, max_memory, verbose)
    occupied_device = mo_occ > 0
    occupied = occupied_device.get()
    energy = mo_energy.get()
    occupied_coeff = mo_coeff[:, occupied_device]
    nao, nmo = mo_coeff.shape
    nocc = occupied_coeff.shape[1]
    overlap_derivative = -int1e_ipovlp(hessian.mol)

    # Match the native solver's workspace estimate using the current free memory.
    bytes_per_atom = 8 * 3 * max(nao * nocc * 3 + nao * nao, nao * nao * 3)
    alignment = ALIGNED**2
    available_atoms = int(get_avail_mem() * 0.5 / bytes_per_atom)
    if available_atoms < 1:
        raise ElectronicConvergenceError("GPU memory is insufficient to verify the Hessian CPHF response")
    block_atoms = available_atoms // alignment * alignment or available_atoms

    def apply_response(orbitals):
        response = fx(cupy.asarray(orbitals))
        _require_device_fp64(response, cupy, "Hessian CPHF response")
        return _fp64_host(response.reshape(orbitals.shape), orbitals.shape, "Hessian CPHF response")

    aoslices = hessian.mol.aoslice_by_atom()
    for start, stop in lib.prange(0, hessian.mol.natm, block_atoms):
        orbital_shape = (3 * (stop - start), nmo, nocc)
        first_order_overlap = cupy.empty((stop - start, 3, nmo, nocc))
        for atom, (left, right) in enumerate(aoslices[start:stop, 2:]):
            overlap_ao = cupy.zeros_like(overlap_derivative)
            overlap_ao[:, left:right] += overlap_derivative[:, left:right]
            overlap_ao[:, :, left:right] += overlap_derivative[:, left:right].transpose(0, 2, 1)
            first_order_overlap[atom] = mo_coeff.T @ overlap_ao @ occupied_coeff
        overlap = _fp64_host(first_order_overlap.reshape(orbital_shape), orbital_shape,
                             "first-order MO overlap")
        fock = _fp64_host(h1mo[start:stop].reshape(orbital_shape), orbital_shape,
                          "first-order MO Fock matrix")
        tolerance = hessian.base.conv_tol_cpscf * (stop - start)
        block_response, block_energy = _refine_hessian_response_block(
            mo1[start:stop].reshape(orbital_shape), apply_response, fock, overlap,
            energy, occupied, tolerance, hessian.max_cycle, max_memory, verbose,
        )
        mo1[start:stop] = block_response.reshape(mo1[start:stop].shape)
        mo_e1[start:stop] = block_energy.reshape(mo_e1[start:stop].shape)
    return mo1, mo_e1


def _refine_hessian_response_block(
    orbitals, apply_response, fock, overlap, energy, occupied, tolerance,
    max_cycle, max_memory, verbose,
):
    from pyscf import lib

    virtual = ~occupied
    occupied_energy = energy[occupied]
    gaps = energy[virtual, None] - occupied_energy
    orbitals = orbitals.copy()
    orbitals[:, occupied] = -0.5 * overlap[:, occupied]
    orbital_rhs = fock - overlap * occupied_energy

    def residual(potential):
        return orbitals[:, virtual] + (orbital_rhs[:, virtual] + potential[:, virtual]) / gaps

    potential = apply_response(orbitals)
    error = residual(potential)
    scale = np.linalg.norm(error.reshape(len(error), -1), axis=1).max()
    if not np.isfinite(scale):
        raise ElectronicConvergenceError("GPU Hessian CPHF residual is non-finite")
    if scale > tolerance:
        # Solve the residual equation with zero occupied correction. Scaling keeps
        # small residual directions above the Krylov linear-dependence cutoff.
        def induced_correction(trials):
            trial_blocks = trials.reshape(-1, *gaps.shape)
            trial_orbitals = np.zeros((len(trial_blocks), *orbitals.shape[1:]))
            trial_orbitals[:, virtual] = trial_blocks
            induced = apply_response(trial_orbitals)[:, virtual] / gaps
            return induced.reshape(trials.shape)

        scaled_tolerance = tolerance / scale
        correction = lib.krylov(
            induced_correction, -error.reshape(len(error), -1) / scale,
            tol=scaled_tolerance, lindep=max(scaled_tolerance**2, np.finfo(orbitals.dtype).eps),
            max_cycle=max_cycle, max_memory=max_memory,
            verbose=lib.logger.WARN if verbose is None else verbose,
        )
        orbitals[:, virtual] += scale * correction.reshape(error.shape)
        potential = apply_response(orbitals)
        error = residual(potential)
        worst = np.linalg.norm(error.reshape(len(error), -1), axis=1).max()
        if not np.isfinite(worst) or worst > tolerance:
            raise ElectronicConvergenceError(
                f"GPU Hessian CPHF residual {worst:.3g} exceeds tolerance {tolerance:.3g}"
            )

    first_order_energy = orbital_rhs[:, occupied] + potential[:, occupied]
    first_order_energy += orbitals[:, occupied] * (occupied_energy[:, None] - occupied_energy)
    return orbitals, first_order_energy


def _validated_tda(td, cupy, nocc, nvir, tolerance):
    converged = td.converged.get() if isinstance(td.converged, cupy.ndarray) else np.asarray(td.converged)
    if converged.shape != (td.nstates,) or not np.all(converged):
        raise ElectronicConvergenceError("one or more requested GPU TDA roots did not converge")
    roots = _fp64_host(td.e, (td.nstates,), "TDA excitation energies")
    if np.any(roots <= 0.0) or np.any(np.diff(roots) < 0.0):
        raise ElectronicConvergenceError("TDA roots must be positive and ordered by energy")
    if td.xy is None or len(td.xy) != td.nstates:
        raise ElectronicConvergenceError("one or more GPU TDA amplitudes are missing")
    for x, y in td.xy:
        _require_device_fp64(x, cupy, "TDA amplitude")
        if x.shape != (nocc, nvir) or not np.isscalar(y) or y != 0:
            raise ElectronicConvergenceError("invalid GPU singlet TDA amplitudes")
    amplitudes = cupy.stack([x for x, _ in td.xy])
    vectors = amplitudes.reshape(td.nstates, -1)
    gram = _fp64_host(2 * vectors @ vectors.T, (td.nstates, td.nstates), "TDA amplitude overlap")
    if not np.allclose(gram, np.eye(td.nstates), atol=1e-7, rtol=0.0):
        raise ElectronicConvergenceError("TDA amplitudes fail restricted orthonormality")

    apply_response, _ = td.gen_vind()
    applied = apply_response(vectors)
    _require_device_fp64(applied, cupy, "TDA response")
    if applied.shape != vectors.shape:
        raise ElectronicConvergenceError("invalid GPU TDA response dimensions")
    difference = applied - cupy.asarray(roots)[:, None] * vectors
    ratios = cupy.linalg.norm(difference, axis=1) / cupy.linalg.norm(vectors, axis=1)
    residuals = _fp64_host(ratios, (td.nstates,), "TDA residuals")
    if np.any(residuals > tolerance):
        raise ElectronicConvergenceError(f"GPU TDA residual {max(residuals):.3g} Hartree exceeds tolerance")
    return roots, _fp64_host(amplitudes, (td.nstates, nocc, nvir), "TDA amplitudes"), residuals


def _dense_tda_retry(td, cupy, nocc, nvir, tolerance):
    dimension = nocc * nvir
    if dimension > _MAX_DENSE_TDA_DIMENSION:
        raise ElectronicConvergenceError(
            f"occupied–virtual dimension {dimension} exceeds dense GPU TDA limit {_MAX_DENSE_TDA_DIMENSION}",
        )
    apply_response, _ = td.gen_vind()
    operator = cupy.empty((dimension, dimension), dtype=cupy.float64)
    for start in range(0, dimension, _TDA_RESPONSE_BLOCK_SIZE):
        stop = min(start + _TDA_RESPONSE_BLOCK_SIZE, dimension)
        identity_rows = cupy.zeros((stop - start, dimension), dtype=cupy.float64)
        identity_rows[cupy.arange(stop - start), cupy.arange(start, stop)] = 1.0
        block = apply_response(identity_rows)
        _require_device_fp64(block, cupy, "TDA response")
        if block.shape != identity_rows.shape:
            raise ElectronicConvergenceError("invalid GPU TDA response dimensions")
        operator[start:stop] = block
    asymmetry = cupy.linalg.norm(operator - operator.T, ord=cupy.inf)
    if float(_fp64_host(asymmetry, (), "TDA response asymmetry")) > tolerance:
        raise ElectronicConvergenceError("GPU TDA response is not Hermitian within tolerance")
    roots, eigenvectors = cupy.linalg.eigh(0.5 * (operator + operator.T))
    if bool(cupy.any(roots <= 0.0)):
        raise ElectronicConvergenceError("TDA roots must be positive")
    td.e = roots[:td.nstates].get()
    td.xy = [
        (vector.reshape(nocc, nvir) / np.sqrt(2), 0)
        for vector in eigenvectors[:, :td.nstates].T
    ]
    td.converged = cupy.ones(td.nstates, dtype=bool)


def _require_device_fp64(value, cupy, name):
    if not isinstance(value, cupy.ndarray) or value.dtype != np.dtype(np.float64):
        raise ElectronicConvergenceError(f"{name} must be a CUDA FP64 array")


def _fp64_host(value, shape, name):
    if hasattr(value, "__cuda_array_interface__"):
        value = value.get()
    value = np.asarray(value)
    if value.dtype != np.dtype(np.float64):
        raise ElectronicConvergenceError(f"{name} must have FP64 precision")
    return _checked_array(value, shape, name)
