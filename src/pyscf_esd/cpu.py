"""Direct CPU RKS/singlet TDA electronic inputs for vibronic spectroscopy."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from types import FunctionType, SimpleNamespace
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from pyscf.dft.rks import RKS
    from pyscf.gto import Mole
    from pyscf.tdscf.rks import TDA


_MAX_DENSE_TDA_DIMENSION = 256
_TDA_RESPONSE_BLOCK_SIZE = 16


class ElectronicConvergenceError(RuntimeError):
    """The electronic solver or an independent numerical check failed."""


@dataclass(frozen=True)
class Settings:
    """Electronic model and numerical tolerances for every geometry in a run.

    `basis` is a PySCF basis name and `xc` uses PySCF's LibXC syntax. The
    backend uses unpruned, atom-centered grids and supports HF, LDA and GGA
    exchange-correlation types, including hybrid GGAs. Nonlocal correlation,
    empirical dispersion, ECPs, symmetry constraints, and solvation are outside
    this backend. SCF and TDA residual tolerances are in Hartree.
    """

    basis: str
    xc: str
    nstates: int = 5
    charge: int = 0
    grid_level: int = 4
    scf_conv_tol: float = 1e-11
    scf_residual_tol: float = 1e-7
    td_residual_tol: float = 1e-8
    max_cycle: int = 100
    cphf_conv_tol: float = 1e-10
    cphf_max_cycle: int = 100
    direct_scf_tol: float = 1e-13

    def __post_init__(self) -> None:
        for name in ("basis", "xc"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty PySCF specification")
        for name, minimum in (("nstates", 1), ("grid_level", 0), ("max_cycle", 1),
                              ("cphf_max_cycle", 1), ("charge", None)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if minimum is not None and value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        # PySCF's default atomic radial and angular grids are tabulated for 0..9.
        if self.grid_level > 9:
            raise ValueError("grid_level must be <= 9")
        for name in ("scf_conv_tol", "scf_residual_tol", "td_residual_tol",
                     "cphf_conv_tol", "direct_scf_tol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class OrbitalSeed:
    """Converged occupied orbitals from a nearby geometry using the same recipe."""

    mol: Mole
    mo_coeff: NDArray[np.float64]
    nocc: int
    settings: Settings


@dataclass(frozen=True)
class ScfAttempt:
    route: str
    converged: bool
    residual_hartree: float | None
    accepted: bool


@dataclass(frozen=True)
class ElectronicPoint:
    """Converged electronic data and derivative entry points at one geometry.

    Orbitals are ordered with occupied orbitals first. TDA amplitudes have
    shape (nstates, nocc, nvir) and PySCF's restricted normalization
    2 * sum(X**2) = 1. Dipoles are real length-gauge vectors in e * bohr and
    use the same state phases as X. Treat `mol` as read-only.
    """

    mol: Mole
    settings: Settings
    mo_coeff: NDArray[np.float64]
    nocc: int
    amplitudes: NDArray[np.float64]
    excitation_energies_hartree: NDArray[np.float64]
    transition_dipoles_au: NDArray[np.float64]
    ground_energy_hartree: float
    scf_residual: float
    tda_residuals: NDArray[np.float64]
    attempts: tuple[ScfAttempt, ...]
    _mean_field: RKS = field(repr=False)
    _tda: TDA = field(repr=False)
    tda_route: str = "davidson"

    def excited_gradient(self, root_index: int) -> NDArray[np.float64]:
        """Return the chosen root's total nuclear gradient in Hartree/bohr.

        Supply the zero-based root index obtained by tracking this particular
        geometry. The upstream TD-RKS gradient lacks a quadrature-grid response
        switch, so check grid and finite-difference step convergence separately.
        """
        if isinstance(root_index, bool) or not isinstance(root_index, Integral):
            raise ValueError("root_index must be a zero-based integer")
        if not 0 <= root_index < len(self.excitation_energies_hartree):
            raise ValueError("root_index is outside the solved TDA roots")
        gradient = self._tda.Gradients()
        gradient.cphf_max_cycle = self.settings.cphf_max_cycle
        gradient.cphf_conv_tol = self.settings.cphf_conv_tol
        return _checked_array(
            _checked_excited_gradient(gradient, int(root_index) + 1),
            (self.mol.natm, 3), "excited-state gradient",
        )

    def ground_gradient(self) -> NDArray[np.float64]:
        """Return the analytic ground-state gradient in Hartree/bohr."""
        gradient = self._mean_field.nuc_grad_method()
        gradient.grid_response = True
        return _checked_array(gradient.kernel(), (self.mol.natm, 3), "ground-state gradient")

    def ground_hessian(self) -> NDArray[np.float64]:
        """Return the analytic ground-state Hessian in PySCF order and Hartree/bohr²."""
        hessian = self._mean_field.Hessian()
        # The CPU semilocal RKS Hessian does not include quadrature-grid response.
        hessian.grid_response = False
        hessian.max_cycle = self.settings.cphf_max_cycle
        natm = self.mol.natm
        return _checked_array(hessian.kernel(), (natm, natm, 3, 3), "ground-state Hessian")

    def reference_stability(self) -> tuple[bool, bool]:
        """Check internal and external RKS stability in PySCF's stability model."""
        _, _, internal, external = self._mean_field.stability(
            internal=True, external=True, return_status=True,
        )
        return bool(internal), bool(external)


def _checked_excited_gradient(gradient, state: int) -> NDArray[np.float64]:
    from pyscf.grad import tdrks

    if type(gradient) is not tdrks.Gradients:
        raise ElectronicConvergenceError("cannot verify CPHF for this CPU gradient implementation")

    response_checked = False

    def solve(fvind, mo_energy, mo_occ, rhs, *, max_cycle, tol):
        nonlocal response_checked
        result = _solve_excited_response(
            fvind, mo_energy, mo_occ, rhs, max_cycle=max_cycle, tol=tol,
        )
        response_checked = True
        return result

    # PySCF's electronic gradient calls the module-level CPHF solver directly.
    # Bind that function to a private namespace for this gradient instance so
    # that the derivative expressions and other PySCF callers are untouched.
    native_grad_elec = tdrks.grad_elec
    checked_grad_elec = FunctionType(
        native_grad_elec.__code__,
        {**native_grad_elec.__globals__, "cphf": SimpleNamespace(solve=solve)},
        argdefs=native_grad_elec.__defaults__,
        closure=native_grad_elec.__closure__,
    )

    def grad_elec(xy, singlet, atmlst=None):
        return checked_grad_elec(gradient, xy, singlet, atmlst, gradient.max_memory, gradient.verbose)

    gradient.grad_elec = grad_elec
    try:
        result = gradient.kernel(state=state)
    finally:
        del gradient.grad_elec
    if not response_checked:
        raise ElectronicConvergenceError("CPU excited-state gradient did not expose its CPHF response")
    return result


def _solve_excited_response(fvind, mo_energy, mo_occ, rhs, *, max_cycle, tol):
    """Verify and, if needed, refine PySCF's virtual–occupied Z-vector."""
    from pyscf import lib
    from pyscf.scf import cphf

    gaps = mo_energy[mo_occ == 0, None] - mo_energy[mo_occ > 0]
    if not np.all(np.isfinite(gaps)) or np.any(gaps == 0):
        raise ElectronicConvergenceError("CPU excited-state CPHF has invalid orbital energy gaps")
    inverse_gaps = 1 / gaps
    preconditioned_rhs = (-rhs * inverse_gaps).ravel()

    def induced_response(trial):
        induced = fvind(trial.reshape(gaps.shape)).reshape(gaps.shape)
        return (induced * inverse_gaps).reshape(trial.shape)

    # This attempt can report subspace convergence prematurely. Use a local
    # logger and judge the returned response by a fresh operator application.
    solution, _ = cphf.solve(
        fvind, mo_energy, mo_occ, rhs, max_cycle=max_cycle, tol=tol, verbose=lib.logger.QUIET,
    )
    solution = _checked_array(solution, rhs.shape, "CPU excited-state CPHF response").ravel()
    residual = preconditioned_rhs - solution - induced_response(solution)
    error = np.linalg.norm(residual)
    if np.isfinite(error) and error > tol:
        # Normalizing the correction RHS keeps the trial vector above the
        # Krylov linear-dependence cutoff without relaxing the residual target.
        scaled_tol = tol / error
        try:
            correction = lib.krylov(
                induced_response, residual / error, tol=scaled_tol,
                lindep=max(scaled_tol**2, np.finfo(solution.dtype).eps), max_cycle=max_cycle,
            )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            raise ElectronicConvergenceError(
                f"CPU excited-state CPHF correction failed at residual {error:.3g} (tolerance {tol:.3g})"
            ) from exc
        solution = solution + error * correction
        error = np.linalg.norm(preconditioned_rhs - solution - induced_response(solution))
    if not np.isfinite(error) or error > tol:
        raise ElectronicConvergenceError(
            f"CPU excited-state CPHF residual {error:.3g} exceeds tolerance {tol:.3g}"
        )
    return solution.reshape(rhs.shape), None


def build_molecule(symbols: Sequence[str], coordinates_bohr: ArrayLike, settings: Settings) -> Mole:
    """Build a symmetry-unconstrained all-electron PySCF Mole from finite bohr coordinates."""
    from pyscf import gto
    from pyscf.data.elements import ELEMENTS
    from pyscf.lib.exceptions import BasisNotFoundError

    if isinstance(symbols, str):
        raise ValueError("symbols must contain one element symbol per atom")
    elements = tuple(symbols)
    if not elements or any(not isinstance(symbol, str) or symbol.capitalize() not in ELEMENTS[1:]
                           for symbol in elements):
        raise ValueError("symbols must contain standard chemical elements")
    for symbol in sorted({item.capitalize() for item in elements}):
        try:
            ecp = gto.basis.load_ecp(settings.basis, symbol)
        except BasisNotFoundError:
            ecp = None
        if ecp:
            raise ElectronicConvergenceError(
                f"{settings.basis} requires an effective core potential for {symbol}. "
                "Select an all-electron basis for this backend"
            )
    try:
        coordinates = _checked_array(coordinates_bohr, (len(elements), 3), "geometry in bohr")
    except ElectronicConvergenceError as exc:
        raise ValueError(str(exc)) from exc
    nelectrons = sum(gto.charge(symbol) for symbol in elements) - settings.charge
    if nelectrons <= 0 or nelectrons % 2:
        raise ValueError("closed-shell RKS requires a positive even electron count")
    mol = gto.M(
        atom=list(zip(elements, coordinates)), unit="Bohr", basis=settings.basis,
        charge=int(settings.charge), spin=0, symmetry=False, verbose=0,
    )
    if mol.nao_nr() <= mol.nelectron // 2:
        raise ValueError("the basis must provide at least one virtual orbital")
    return mol


def solve_point(
    symbols: Sequence[str],
    coordinates_bohr: ArrayLike,
    settings: Settings,
    *,
    previous: OrbitalSeed | ElectronicPoint | None = None,
) -> ElectronicPoint:
    """Run RKS and singlet TDA at one point. Derivatives are explicitly requested later."""
    mol = build_molecule(symbols, coordinates_bohr, settings)
    mean_field, residual, attempts = _solve_reference(mol, settings, previous)
    nocc = mol.nelectron // 2
    nmo = mean_field.mo_coeff.shape[1]
    nvir = nmo - nocc
    if settings.nstates > nocc * nvir:
        raise ValueError("nstates exceeds the number of occupied–virtual pairs")
    td = mean_field.TDA()
    td.singlet = True
    td.wfnsym = None
    td.nstates = int(settings.nstates)
    td.conv_tol = settings.td_residual_tol * 0.2
    # Davidson filters the squared norm of a correction vector. Its library
    # default can remove useful directions before a tight residual is reached.
    td.lindep = min(1e-17, 0.2 * td.conv_tol**2)
    td.max_cycle = int(settings.max_cycle)
    td.chkfile = None
    td.positive_eig_threshold = 0.0
    td.exclude_nlc = True
    td.kernel()
    tda_route = "davidson"
    try:
        roots, amplitudes, ratios = _validated_tda(td, nocc, nvir, settings.td_residual_tol)
    except ElectronicConvergenceError as first_error:
        try:
            if nocc * nvir <= _MAX_DENSE_TDA_DIMENSION:
                _dense_tda_retry(td, nocc, nvir, settings.td_residual_tol)
                tda_route = "dense_same_operator"
            else:
                # Reuse finite Ritz vectors as guesses for the unchanged
                # response operator, including partially converged roots.
                previous_roots = getattr(td, "xy", None)
                guesses = np.array([np.ravel(x) for x, _ in previous_roots]) if previous_roots is not None else None
                if (guesses is None or guesses.shape != (td.nstates, nocc*nvir)
                        or not np.isfinite(guesses).all()):
                    guesses = None
                td.max_cycle *= 2
                td.lindep *= .01
                td.kernel(x0=guesses)
                tda_route = "davidson_restart_same_operator"
            roots, amplitudes, ratios = _validated_tda(td, nocc, nvir, settings.td_residual_tol)
        except ElectronicConvergenceError as retry_error:
            raise ElectronicConvergenceError(
                f"TDA initial solve failed ({first_error}). Same-operator recovery failed ({retry_error})"
            ) from retry_error
    dipoles = _checked_array(td.transition_dipole(), (td.nstates, 3), "transition dipoles")
    return ElectronicPoint(
        mol=mol, settings=settings, mo_coeff=_readonly(mean_field.mo_coeff),
        nocc=nocc, amplitudes=_readonly(amplitudes), excitation_energies_hartree=_readonly(roots),
        transition_dipoles_au=_readonly(dipoles), ground_energy_hartree=float(mean_field.e_tot),
        scf_residual=residual, tda_residuals=_readonly(ratios), attempts=attempts,
        _mean_field=mean_field, _tda=td, tda_route=tda_route,
    )


def _make_mean_field(mol: Mole, settings: Settings) -> RKS:
    from pyscf import dft, scf
    from pyscf.scf.dispersion import parse_dft

    class DirectRKS(dft.rks.RKS):
        # RKS inherits RHF's auto-incore J/K choice. The base SCF builder
        # evaluates screened integrals on demand and avoids that choice.
        get_jk = scf.hf.SCF.get_jk

    xc, _, dispersion = parse_dft(settings.xc)
    if dispersion is not None or dft.libxc.is_nlc(xc):
        raise ValueError("CPU RKS/TDA does not support nonlocal correlation or dispersion")
    if dft.libxc.xc_type(xc) not in ("HF", "LDA", "GGA"):
        raise ValueError("CPU RKS/TDA derivatives require an HF, LDA or GGA-type functional")
    dft.libxc.test_deriv_order(xc, 3, raise_error=True)
    mean_field = DirectRKS(mol, xc=xc)
    mean_field.grids.level = int(settings.grid_level)
    mean_field.grids.prune = None
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
    mean_field.DIIS = scf.diis.SCF_DIIS
    mean_field.diis_space = 8
    mean_field.diis_start_cycle = 1
    mean_field.diis_damp = 0.0
    mean_field.level_shift = 0.0
    mean_field.damp = 0.0
    mean_field.chkfile = None
    return mean_field


def _solve_reference(
    mol: Mole, settings: Settings, previous: OrbitalSeed | ElectronicPoint | None,
) -> tuple[RKS, float, tuple[ScfAttempt, ...]]:
    attempts = []
    routes = []
    if previous is not None:
        try:
            density = _project_density(previous, mol, settings)
        except np.linalg.LinAlgError:
            attempts.append(ScfAttempt("projected_occupied_diis", False, None, False))
        else:
            routes.append(("projected_occupied_diis", density))
    routes.extend((("minao_diis", None), ("minao_soscf", None)))
    for route, density in routes:
        mean_field = _make_mean_field(mol, settings)
        if route == "minao_soscf":
            mean_field = mean_field.newton()
        mean_field.kernel(dm0=density)
        try:
            residual = _scf_residual(mean_field)
        except ElectronicConvergenceError:
            attempts.append(ScfAttempt(route, bool(mean_field.converged), None, False))
            continue
        accepted = bool(mean_field.converged and residual <= settings.scf_residual_tol)
        attempts.append(ScfAttempt(route, bool(mean_field.converged), residual, accepted))
        if accepted:
            return mean_field, residual, tuple(attempts)
    raise ElectronicConvergenceError(f"RKS did not meet convergence checks after {attempts}")


def _project_density(
    previous: OrbitalSeed | ElectronicPoint, mol: Mole, settings: Settings,
) -> NDArray[np.float64]:
    from pyscf.scf.addons import project_mo_nr2nr

    if previous.settings != settings or previous.mol.nelectron != mol.nelectron or not np.array_equal(
        previous.mol.atom_charges(), mol.atom_charges(),
    ):
        raise ValueError("previous orbitals must use the same electronic settings and atom map")
    nocc = mol.nelectron // 2
    coefficients = np.asarray(previous.mo_coeff)
    if previous.nocc != nocc or coefficients.ndim != 2 or coefficients.shape[1] < nocc:
        raise ValueError("previous orbitals must contain the occupied closed-shell space")
    occupied = _checked_array(
        coefficients[:, :nocc], (previous.mol.nao_nr(), nocc), "previous occupied orbitals",
    )
    projected = project_mo_nr2nr(previous.mol, occupied, mol)
    overlap = mol.intor_symmetric("int1e_ovlp")
    return _orthogonalized_density(projected, overlap)


def _orthogonalized_density(
    orbitals: NDArray[np.float64], overlap: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not np.all(np.isfinite(orbitals)):
        raise np.linalg.LinAlgError("projected occupied orbitals are nonfinite")
    eigenvalues, eigenvectors = np.linalg.eigh(orbitals.T @ overlap @ orbitals)
    if np.any(eigenvalues < 1e-8):
        raise np.linalg.LinAlgError("projected occupied space lost rank")
    metric_inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    orthogonalized = orbitals @ metric_inverse_root
    return 2 * orthogonalized @ orthogonalized.T


def _scf_residual(mean_field: RKS) -> float:
    coefficients = np.asarray(mean_field.mo_coeff)
    nao = mean_field.mol.nao_nr()
    nocc = mean_field.mol.nelectron // 2
    if coefficients.ndim != 2 or coefficients.shape[0] != nao or coefficients.shape[1] <= nocc:
        raise ElectronicConvergenceError("RKS did not provide the occupied and virtual orbitals")
    nmo = coefficients.shape[1]
    coefficients = _checked_array(coefficients, (nao, nmo), "RKS orbitals")
    occupations = _checked_array(mean_field.mo_occ, (nmo,), "RKS occupations")
    _checked_array(mean_field.mo_energy, (nmo,), "RKS orbital energies")
    _checked_array(mean_field.e_tot, (), "RKS energy")
    if not np.all(occupations[:nocc] == 2) or np.any(occupations[nocc:] != 0):
        raise ElectronicConvergenceError("RKS determinant does not have closed-shell occupations")
    overlap = mean_field.get_ovlp()
    if not np.allclose(coefficients.T @ overlap @ coefficients, np.eye(nmo), atol=1e-7, rtol=0.0):
        raise ElectronicConvergenceError("RKS orbitals are not orthonormal in the AO overlap")
    density = mean_field.make_rdm1()
    fock = _checked_array(
        mean_field.get_hcore() + mean_field.get_veff(dm=density),
        (nao, nao), "fresh RKS Fock matrix",
    )
    orbital_gradient = mean_field.get_grad(coefficients, occupations, fock)
    return float(np.linalg.norm(orbital_gradient))


def _validated_tda(
    td: TDA, nocc: int, nvir: int, tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if np.asarray(td.converged).shape != (td.nstates,) or not np.all(td.converged):
        raise ElectronicConvergenceError("one or more requested TDA roots did not converge")
    roots = _checked_array(td.e, (td.nstates,), "TDA excitation energies")
    if np.any(roots <= 0.0) or np.any(np.diff(roots) < 0.0):
        raise ElectronicConvergenceError("TDA roots must be positive and ordered by energy")
    amplitudes = _checked_array(
        [x for x, _ in td.xy], (td.nstates, nocc, nvir), "TDA amplitudes",
    )
    residuals = _tda_residuals(td, roots, amplitudes)
    if np.any(residuals > tolerance):
        raise ElectronicConvergenceError(f"TDA residual {max(residuals):.3g} Hartree exceeds tolerance")
    return roots, amplitudes, residuals


def _dense_tda_retry(td: TDA, nocc: int, nvir: int, tolerance: float) -> None:
    dimension = nocc * nvir
    if dimension > _MAX_DENSE_TDA_DIMENSION:
        raise ElectronicConvergenceError(
            f"occupied–virtual dimension {dimension} exceeds dense TDA limit {_MAX_DENSE_TDA_DIMENSION}",
        )

    apply_response, _ = td.gen_vind()
    operator = np.empty((dimension, dimension))
    for start in range(0, dimension, _TDA_RESPONSE_BLOCK_SIZE):
        stop = min(start + _TDA_RESPONSE_BLOCK_SIZE, dimension)
        identity_rows = np.zeros((stop - start, dimension))
        identity_rows[np.arange(stop - start), np.arange(start, stop)] = 1.0
        operator[start:stop] = _checked_array(
            apply_response(identity_rows), identity_rows.shape, "TDA response",
        )

    asymmetry = np.linalg.norm(operator - operator.T, ord=np.inf)
    if asymmetry > tolerance:
        raise ElectronicConvergenceError(
            f"TDA response is not Hermitian within tolerance (asymmetry {asymmetry:.3g} Hartree)",
        )
    try:
        roots, eigenvectors = np.linalg.eigh(0.5 * (operator + operator.T))
    except np.linalg.LinAlgError as exc:
        raise ElectronicConvergenceError("dense TDA diagonalization did not converge") from exc
    if np.any(roots <= 0.0):
        raise ElectronicConvergenceError("TDA roots must be positive")

    td.e = roots[:td.nstates].copy()
    td.xy = [
        (vector.reshape(nocc, nvir) / np.sqrt(2), 0)
        for vector in eigenvectors[:, :td.nstates].T
    ]
    td.converged = np.ones(td.nstates, dtype=bool)


def _tda_residuals(
    td: TDA, roots: NDArray[np.float64], amplitudes: NDArray[np.float64],
) -> NDArray[np.float64]:
    vectors = amplitudes.reshape(len(roots), -1)
    gram = 2 * vectors @ vectors.T
    if not np.allclose(gram, np.eye(len(roots)), atol=1e-7, rtol=0.0):
        raise ElectronicConvergenceError("TDA amplitudes fail restricted orthonormality")
    apply_response, _ = td.gen_vind()
    applied = _checked_array(apply_response(vectors), vectors.shape, "TDA response")
    difference = applied - roots[:, None] * vectors
    return np.linalg.norm(difference, axis=1) / np.linalg.norm(vectors, axis=1)


def _checked_array(
    value: ArrayLike, shape: tuple[int, ...], name: str,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.shape != shape or np.iscomplexobj(array) or array.dtype.kind not in "fiu":
        raise ElectronicConvergenceError(f"{name} must be real with shape {shape}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ElectronicConvergenceError(f"{name} contains nonfinite data")
    return array


def _readonly(value: NDArray[np.float64]) -> NDArray[np.float64]:
    result = value.copy()
    result.setflags(write=False)
    return result
