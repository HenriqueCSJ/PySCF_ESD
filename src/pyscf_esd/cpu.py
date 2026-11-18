"""Direct CPU RKS/singlet TDA electronic inputs for vibronic spectroscopy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral, Real
from types import FunctionType, SimpleNamespace
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from pyscf.dft.rks import RKS
    from pyscf.gto import Mole
    from pyscf.tdscf.rks import TDA


_MAX_DENSE_TDA_DIMENSION = 256
_TDA_RESPONSE_BLOCK_SIZE = 16
_MAX_DENSE_STABILITY_DIMENSION = 24
_RKS_SPIN_PROBE_MIN_DIMENSION = 256
_STABILITY_EIGEN_TOL = 1e-7
_STABILITY_CUTOFF = -1e-5  # PySCF's threshold in its native orbital-Hessian scaling.


class ElectronicConvergenceError(RuntimeError):
    """The electronic solver or an independent numerical check failed."""


class UnstableReferenceError(ElectronicConvergenceError):
    """A converged restricted reference has an unstable orbital direction."""

    def __init__(self, channel: str, stage: str, detail: str = ""):
        self.channel = channel
        self.stage = stage
        name = {"internal": "internal", "restricted_to_unrestricted": "restricted-to-unrestricted"}[channel]
        suffix = f" {detail}" if detail else ""
        super().__init__(f"{stage} has a {name} orbital instability.{suffix}")


class ReferenceStabilityUnresolvedError(ElectronicConvergenceError):
    """The orbital-Hessian solve could not qualify a restricted reference."""

    def __init__(self, channel: str, stage: str, detail: str = ""):
        self.channel = channel
        self.stage = stage
        name = channel.replace("_", "-")
        suffix = f" {detail}" if detail else ""
        super().__init__(f"{stage} has unresolved {name} orbital stability.{suffix}")


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
    stability_checks: str = "minima"

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
        if self.stability_checks not in ("minima", "all"):
            raise ValueError("stability_checks must be minima or all")
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

    def reference_stability(self, stage: str = "Electronic point reference") -> tuple[bool, bool]:
        """Check internal and restricted-to-unrestricted orbital stability."""
        _, _, internal, external = _restricted_stability(self._mean_field, stage, internal=True, external=True)
        return bool(internal), bool(external)


def _stability_options(mean_field) -> dict:
    occupied = int((mean_field.mo_occ > 0).sum())
    pairs = (len(mean_field.mo_occ) - occupied) * occupied
    return {"nroots": min(2, int(pairs)), "tol": 1e-7}


def _restricted_stability(mean_field, stage: str, *, internal: bool, external: bool):
    xc = getattr(mean_field, "xc", None)
    # Keep GPU and pure-HF stability on their existing native routes.
    if isinstance(xc, str) and isinstance(getattr(mean_field, "mo_coeff", None), np.ndarray):
        from pyscf.dft import libxc

        if libxc.xc_type(xc) != "HF":
            return _dft_stability(mean_field, stage, internal=internal, external=external)
    return mean_field.stability(
        internal=internal, external=external, return_status=True, **_stability_options(mean_field),
    )


def _dft_stability(mean_field, stage: str, *, internal: bool, external: bool):
    rotated = internal_stable = external_stable = None
    if internal:
        channel = "internal"
        try:
            from pyscf.scf.stability import _rotate_mo

            hop, diagonal = _rks_internal_stability_operator(mean_field)
            mode = _negative_stability_mode(hop, diagonal)
        except (ElectronicConvergenceError, ImportError) as error:
            raise ReferenceStabilityUnresolvedError(channel, stage, str(error)) from error
        internal_stable = mode is None
        rotated = mean_field.mo_coeff if internal_stable else _rotate_mo(
            mean_field.mo_coeff, mean_field.mo_occ, mode,
        )
    if external:
        channel = "restricted_to_unrestricted"
        try:
            from pyscf.scf.stability import _gen_hop_rhf_external

            # PySCF's external status is the spin-breaking channel. Reuse that
            # operator and its scaling without constructing a UKS determinant.
            _, _, hop, diagonal = _gen_hop_rhf_external(mean_field)
            block = None
            if len(diagonal) >= _RKS_SPIN_PROBE_MIN_DIMENSION and not mean_field.mol.symmetry:
                block = _optional_rks_stability_block(mean_field, channel)
            external_stable = _negative_stability_mode(hop, diagonal, block_operator=block) is None
        except (ElectronicConvergenceError, ImportError) as error:
            raise ReferenceStabilityUnresolvedError(channel, stage, str(error)) from error
    return rotated, None, internal_stable, external_stable


def _rks_internal_stability_operator(mean_field, *, with_symmetry: bool = True):
    from pyscf.soscf.newton_ah import gen_g_hop_rhf

    _, hop, diagonal = gen_g_hop_rhf(
        mean_field, mean_field.mo_coeff, mean_field.mo_occ, with_symmetry=with_symmetry,
    )
    # PySCF scales the internal operator by two in its stability wrapper.
    return lambda vector: 2 * hop(vector).real, 2 * diagonal


def _has_native_rks_response(mean_field) -> bool:
    from pyscf.scf._response_functions import _gen_rhf_response

    underlying = getattr(mean_field, "_scf", mean_field)
    return (getattr(mean_field.gen_response, "__func__", None) is _gen_rhf_response
            and getattr(underlying, "mol", mean_field.mol) is mean_field.mol)


def _rks_fock_blocks(mean_field):
    occupied = mean_field.mo_coeff[:, mean_field.mo_occ == 2]
    virtual = mean_field.mo_coeff[:, mean_field.mo_occ == 0]
    density = mean_field.make_rdm1(mean_field.mo_coeff, mean_field.mo_occ)
    fock = mean_field.get_hcore() + mean_field.get_veff(mean_field.mol, density)
    return occupied, virtual, occupied.T @ fock @ occupied, virtual.T @ fock @ virtual


def _semilocal_complex_curvature(mean_field, operator, threshold: float) -> _StabilityCurvature | None:
    """Find the exact minimum of the native semilocal real-to-complex operator."""
    if not _has_native_rks_response(mean_field) or mean_field._numint.libxc.is_hybrid_xc(mean_field.xc):
        return None
    occupied, virtual, foo, fvv = _rks_fock_blocks(mean_field)
    if occupied.shape[1] == 0 or virtual.shape[1] == 0:
        return None

    # An antisymmetric density perturbation has zero Coulomb and semilocal
    # XC response. The native Hessian acts as Fvv X - X Foo on the full space.
    occupied_energies, occupied_modes = np.linalg.eigh(foo)
    virtual_energies, virtual_modes = np.linalg.eigh(fvv)
    mode = np.outer(virtual_modes[:, 0], occupied_modes[:, -1]).ravel()
    product = _checked_array(operator(mode), mode.shape, "real-to-complex orbital Hessian product")
    curvature = float(mode @ product)
    residual = float(np.linalg.norm(product - curvature * mode))
    expected = virtual_energies[0] - occupied_energies[-1]
    if abs(expected - curvature) > _STABILITY_EIGEN_TOL or residual > _STABILITY_EIGEN_TOL:
        return None
    if curvature < threshold:
        return _StabilityCurvature(curvature, residual, mode)
    if curvature - residual >= threshold:
        return _StabilityCurvature(curvature, residual, None)
    return None


def _rks_stability_block_operator(mean_field, channel):
    """Batch restricted internal or spin-breaking HVPs through one native response call."""
    if not _has_native_rks_response(mean_field):
        return None
    if channel == "internal":
        scale = 4
        singlet = None
    elif channel == "restricted_to_unrestricted":
        scale = 1
        singlet = False
    else:
        return None

    occupied, virtual, foo, fvv = _rks_fock_blocks(mean_field)
    response = mean_field.gen_response(singlet=singlet, hermi=1, with_nlc=False)

    def apply(rows):
        amplitudes = rows.reshape(len(rows), virtual.shape[1], occupied.shape[1])
        density_tr = virtual @ (2 * amplitudes) @ occupied.T
        density = density_tr + density_tr.swapaxes(-2, -1)
        product = fvv @ amplitudes - amplitudes @ foo + virtual.T @ response(density) @ occupied
        return (scale * product).reshape(rows.shape)

    return apply


def _optional_rks_stability_block(mean_field, channel):
    try:
        return _rks_stability_block_operator(mean_field, channel)
    except (ElectronicConvergenceError, TypeError, ValueError, NotImplementedError):
        return None


def _negative_stability_mode(
    operator, diagonal: NDArray[np.float64], *, block_operator=None,
) -> NDArray[np.float64] | None:
    """Return a negative direction or qualify the lowest curvature of this operator."""
    evidence = _stability_curvature(
        operator, diagonal, _STABILITY_CUTOFF, block_operator=block_operator,
        probe_low_gap=block_operator is not None,
    )
    return evidence.below_threshold_direction


@dataclass(frozen=True)
class _StabilityCurvature:
    curvature: float
    residual: float
    below_threshold_direction: NDArray[np.float64] | None


def _stability_curvature(
    operator, diagonal: NDArray[np.float64], threshold: float, *, block_operator=None, probe_low_gap: bool = False,
) -> _StabilityCurvature:
    """Verify a lowest-mode estimate or a direction below the required curvature."""
    dimension = len(diagonal)
    diagonal = _checked_array(diagonal, (dimension,), "orbital Hessian diagonal")
    if not dimension:
        raise ElectronicConvergenceError("The occupied–virtual rotation space is empty.")

    def apply(vector):
        return _checked_array(operator(vector), (dimension,), "orbital Hessian product")

    block_verified = False

    def apply_many(rows):
        nonlocal block_operator, block_verified
        if block_operator is None or len(rows) < 2:
            return [apply(row) for row in rows]
        rows = np.asarray(rows)
        try:
            # Limit the intermediate AO density storage independently of the
            # Davidson subspace size.
            products = np.concatenate([block_operator(rows[i:i + 2]) for i in range(0, len(rows), 2)])
            products = _checked_array(products, rows.shape, "batched orbital Hessian products")
            if not block_verified:
                for row, product in zip(rows[:2], products[:2]):
                    scalar = apply(row)
                    if np.linalg.norm(product - scalar) > 1e-10 * max(1., np.linalg.norm(scalar)):
                        raise ElectronicConvergenceError("Batched and native orbital Hessian actions disagree.")
                block_verified = True
            return products
        except (ElectronicConvergenceError, TypeError, ValueError, NotImplementedError):
            block_operator = None
            return [apply(row) for row in rows]

    if dimension <= _MAX_DENSE_STABILITY_DIMENSION:
        matrix = np.column_stack([apply(axis) for axis in np.eye(dimension)])
        if np.linalg.norm(matrix - matrix.T, ord=np.inf) > _STABILITY_EIGEN_TOL:
            raise ElectronicConvergenceError("The orbital Hessian is not symmetric to the eigensolver tolerance.")
        try:
            _, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
        except np.linalg.LinAlgError as error:
            raise ElectronicConvergenceError("Dense orbital-Hessian diagonalization failed.") from error
        mode = vectors[:, 0]
        product = matrix @ mode
        curvature = float(mode @ product)
        residual = float(np.linalg.norm(product - curvature * mode))
        if curvature < threshold:
            return _StabilityCurvature(curvature, residual, mode)
        if residual <= _STABILITY_EIGEN_TOL and curvature - residual >= threshold:
            return _StabilityCurvature(curvature, residual, None)
        raise ElectronicConvergenceError("The lowest orbital curvature could not be resolved at the cutoff.")

    from pyscf import lib
    from pyscf.lib.linalg_helper import LinearDependenceError

    guesses = np.zeros((2, dimension))
    lowest_pairs = np.argsort(diagonal, kind="stable")
    guesses[np.arange(2), lowest_pairs[:2]] = 1.0
    if probe_low_gap:
        # A small projection can expose a downhill direction quickly. Only
        # the full native HVP decides whether that direction fails the cutoff.
        indices = lowest_pairs[:_MAX_DENSE_STABILITY_DIMENSION]
        basis = np.zeros((len(indices), dimension))
        basis[np.arange(len(indices)), indices] = 1.0
        projected = np.asarray(apply_many(basis))[:, indices]
        if np.linalg.norm(projected - projected.T, ord=np.inf) <= _STABILITY_EIGEN_TOL:
            _, vectors = np.linalg.eigh((projected + projected.T) * 0.5)
            guesses[0, indices] = vectors[:, 0]
            product = apply(guesses[0])
            curvature = float(guesses[0] @ product)
            if curvature < threshold:
                residual = float(np.linalg.norm(product - curvature * guesses[0]))
                return _StabilityCurvature(curvature, residual, guesses[0].copy())
    random = np.random.default_rng(0)
    lowest_verified = np.inf
    confirmations = 0
    for max_cycle, max_space, use_guesses in ((50, 16, True), (100, 32, False), (100, 32, True)):
        # Mix coordinate or Ritz guesses into full-support probes. Providing
        # exact low-gap or prior Ritz vectors as separate seeds can stop the
        # solver before it explores another Hessian block.
        seeds = random.standard_normal((2, dimension))
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
        if use_guesses and len(guesses):
            seeds[:len(guesses)] += guesses
        try:
            converged, eigenvalues, vectors = lib.davidson1(
                apply_many, list(seeds), diagonal, nroots=2,
                tol=_STABILITY_EIGEN_TOL, tol_residual=_STABILITY_EIGEN_TOL,
                max_cycle=max_cycle, max_space=max_space,
            )
        except (LinearDependenceError, np.linalg.LinAlgError):
            continue
        verified = []
        guesses = []
        for converged_root, eigenvalue, vector in zip(converged, eigenvalues, vectors):
            try:
                vector = _checked_array(vector, (dimension,), "orbital-Hessian Ritz vector")
            except ElectronicConvergenceError:
                continue
            length = np.linalg.norm(vector)
            if not np.isfinite(length) or length == 0:
                continue
            mode = vector / length
            guesses.append(mode)
            product = apply(mode)
            curvature = float(mode @ product)
            residual = float(np.linalg.norm(product - curvature * mode))
            # A Rayleigh quotient below the threshold directly disproves the
            # required curvature bound, even for an unconverged eigenpair.
            if curvature < threshold:
                return _StabilityCurvature(curvature, residual, mode)
            if (
                converged_root and abs(length - 1.0) <= 1e-6 and residual <= _STABILITY_EIGEN_TOL
                and abs(eigenvalue - curvature) <= _STABILITY_EIGEN_TOL
            ):
                verified.append((curvature, residual, mode))
        if len(verified) != 2:
            continue
        if abs(float(verified[0][2] @ verified[1][2])) > 1e-6:
            continue
        curvature, residual, _ = min(verified, key=lambda root: root[0])
        if curvature < lowest_verified - 2 * _STABILITY_EIGEN_TOL:
            lowest_verified = curvature
            confirmations = 0
        if (
            abs(curvature - lowest_verified) <= 2 * _STABILITY_EIGEN_TOL
            and curvature - residual >= threshold
        ):
            confirmations += 1
            if confirmations == 2:
                return _StabilityCurvature(curvature, residual, None)
    raise ElectronicConvergenceError("Independent starts did not confirm the lowest orbital-Hessian curvature.")


@dataclass(frozen=True)
class OrbitalStabilityChannel:
    """An orbital-Hessian result in the native PySCF scaling for this channel."""

    channel: str
    status: Literal["qualified", "negative_direction", "below_margin", "unresolved"]
    curvature: float | None = None
    eigenpair_residual: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RKSStabilityReport:
    """Evidence for an independent three-channel assessment of one CPU RKS determinant."""

    scf_residual_hartree: float
    scf_residual_tolerance_hartree: float
    minimum_curvature: float
    channels: tuple[OrbitalStabilityChannel, ...]

    @property
    def qualified(self) -> bool:
        return all(channel.status == "qualified" for channel in self.channels)


def assess_rks_stability(
    mean_field, *, scf_residual_tolerance: float = 1e-7, minimum_curvature: float = 1e-6,
) -> RKSStabilityReport:
    """Check internal, real-to-complex and spin-breaking stability in one CPU RKS reference.

    Requires native convergence and an independently recomputed SCF residual.
    Positive qualification needs the verified lowest mode of every channel above
    the requested margin. A reported failure curvature can be a Rayleigh
    witness. Its eigenpair residual indicates how far it is from an eigenmode.
    """
    from pyscf.dft import libxc
    from pyscf.dft.rks import RKS

    for name, value in (("scf_residual_tolerance", scf_residual_tolerance), ("minimum_curvature", minimum_curvature)):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(mean_field, RKS) or libxc.xc_type(mean_field.xc) == "HF":
        raise TypeError("three-channel stability requires a CPU restricted Kohn–Sham DFT reference")
    if mean_field.do_nlc():
        raise ReferenceStabilityUnresolvedError(
            "internal", "RKS reference", "PySCF's internal orbital Hessian omits the nonlocal correlation response.",
        )
    if not mean_field.converged:
        raise ElectronicConvergenceError("Converge the RKS determinant before assessing orbital stability.")
    if not isinstance(mean_field.mo_coeff, np.ndarray) or not np.isrealobj(mean_field.mo_coeff):
        raise TypeError("three-channel stability requires real NumPy molecular orbitals")
    residual = _scf_residual(mean_field)
    if residual > scf_residual_tolerance:
        raise ElectronicConvergenceError(
            f"RKS residual {residual:.3g} Hartree exceeds the stability input tolerance {scf_residual_tolerance:.3g}."
        )

    reports = []
    threshold = np.nextafter(minimum_curvature, np.inf)

    def assess(name, operator, diagonal):
        try:
            evidence = None
            if name == "real_to_complex":
                try:
                    evidence = _semilocal_complex_curvature(mean_field, operator, threshold)
                except (ElectronicConvergenceError, np.linalg.LinAlgError):
                    pass
            if evidence is None:
                block = None
                if len(diagonal) > _MAX_DENSE_STABILITY_DIMENSION:
                    block = _optional_rks_stability_block(mean_field, name)
                evidence = _stability_curvature(
                    operator, diagonal, threshold, block_operator=block,
                    probe_low_gap=(name == "restricted_to_unrestricted" and block is not None
                                   and len(diagonal) >= _RKS_SPIN_PROBE_MIN_DIMENSION),
                )
        except (ElectronicConvergenceError, ImportError, NotImplementedError) as error:
            reports.append(OrbitalStabilityChannel(name, "unresolved", reason=str(error)))
        else:
            if evidence.below_threshold_direction is None:
                status = "qualified"
            elif evidence.curvature < -_STABILITY_EIGEN_TOL:
                status = "negative_direction"
            else:
                status = "below_margin"
            reports.append(OrbitalStabilityChannel(name, status, evidence.curvature, evidence.residual))

    try:
        internal, internal_diagonal = _rks_internal_stability_operator(mean_field, with_symmetry=False)
    except (ImportError, NotImplementedError) as error:
        reports.append(OrbitalStabilityChannel("internal", "unresolved", reason=str(error)))
    else:
        assess("internal", internal, internal_diagonal)

    try:
        from pyscf.scf.stability import _gen_hop_rhf_external

        complex_rotation, complex_diagonal, unrestricted, unrestricted_diagonal = _gen_hop_rhf_external(
            mean_field, with_symmetry=False,
        )
    except (ImportError, NotImplementedError) as error:
        for name in ("real_to_complex", "restricted_to_unrestricted"):
            reports.append(OrbitalStabilityChannel(name, "unresolved", reason=str(error)))
    else:
        assess("real_to_complex", complex_rotation, complex_diagonal)
        assess("restricted_to_unrestricted", unrestricted, unrestricted_diagonal)
    return RKSStabilityReport(residual, float(scf_residual_tolerance), float(minimum_curvature), tuple(reports))


def require_stable_reference(internal: bool, unrestricted: bool, stage: str) -> None:
    """Report which orbital-stability channel failed at the given geometry."""
    if not internal:
        raise UnstableReferenceError("internal", stage)
    if not unrestricted:
        raise UnstableReferenceError("restricted_to_unrestricted", stage)


def check_restricted_reference(mean_field, stage: str) -> None:
    """Check a stationary endpoint and report a verified instability when available."""
    try:
        _, _, internal, _ = _restricted_stability(mean_field, stage, internal=True, external=False)
    except ReferenceStabilityUnresolvedError as internal_failure:
        # An inconclusive internal solve can coexist with a confirmed direction
        # towards spin breaking. Preserve the internal failure if the external
        # check cannot establish an instability either.
        try:
            _, _, _, unrestricted = _restricted_stability(mean_field, stage, internal=False, external=True)
        except ReferenceStabilityUnresolvedError:
            raise internal_failure from None
        if unrestricted is not None and not unrestricted:
            raise UnstableReferenceError("restricted_to_unrestricted", stage) from internal_failure
        raise
    else:
        require_stable_reference(internal, True, stage)
        _, _, _, unrestricted = _restricted_stability(mean_field, stage, internal=False, external=True)
        require_stable_reference(True, unrestricted, stage)


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
    _vg_target_index: int | None = None,
) -> ElectronicPoint:
    """Run RKS and singlet TDA at one point. Derivatives are explicitly requested later."""
    mol = build_molecule(symbols, coordinates_bohr, settings)
    mean_field, residual, attempts = _solve_reference(mol, settings, previous)
    nocc = mol.nelectron // 2
    nmo = mean_field.mo_coeff.shape[1]
    nvir = nmo - nocc
    if _vg_target_index is not None:
        count = _vg_root_count(settings.nstates, _vg_target_index, nocc * nvir)
        if count != settings.nstates:
            settings = replace(settings, nstates=count)
    elif settings.nstates > nocc * nvir:
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


def _vg_root_count(requested: int, target_index: int, dimension: int) -> int:
    if not 0 <= target_index < dimension:
        raise ValueError(f"target index {target_index} is outside the {dimension} singlet TDA pairs")
    count = min(requested, dimension)
    if target_index >= count or (target_index == count - 1 and count < dimension):
        raise ValueError("request the target and at least one higher reference root, or the full TDA space")
    return count


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
    """Try projected and fresh restricted guesses and verify stationarity."""
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
            if settings.stability_checks == "all":
                mean_field, residual = _stabilize_reference(mean_field, mol, settings, residual, attempts)
            return mean_field, residual, tuple(attempts)
    raise ElectronicConvergenceError(f"RKS did not meet convergence checks after {attempts}")


def _stabilize_reference(mean_field, mol, settings: Settings, residual: float, attempts: list[ScfAttempt]):
    """Repair an internal saddle using fresh restricted SCF solves at the same geometry."""
    stage = "SCF reference at a sampled geometry"
    for attempt in range(5):
        rotated, _, stable, _ = _restricted_stability(mean_field, stage, internal=True, external=False)
        if stable:
            # A spin-breaking direction requires a different electronic
            # reference. The restricted rotation search only treats internal saddles.
            _, _, _, unrestricted = _restricted_stability(mean_field, stage, internal=False, external=True)
            require_stable_reference(True, unrestricted, stage)
            return mean_field, residual
        if attempt == 4:
            break
        # The proposed orbitals seed a new SCF. An energy decrease beyond
        # the SCF tolerance distinguishes a new solution from the same saddle.
        density = mean_field.make_rdm1(mo_coeff=rotated, mo_occ=mean_field.mo_occ)
        for route in ("stability_rotation_diis", "stability_rotation_soscf"):
            candidate = _make_mean_field(mol, settings)
            if route.endswith("soscf"):
                candidate = candidate.newton()
            candidate.kernel(dm0=density)
            try:
                candidate_residual = _scf_residual(candidate)
            except ElectronicConvergenceError:
                candidate_residual = None
            accepted = bool(
                candidate.converged and candidate_residual is not None
                and candidate_residual <= settings.scf_residual_tol
                and candidate.e_tot < mean_field.e_tot - settings.scf_conv_tol
            )
            attempts.append(ScfAttempt(route, bool(candidate.converged), candidate_residual, accepted))
            if accepted:
                mean_field, residual = candidate, candidate_residual
                break
        else:
            raise UnstableReferenceError(
                "internal", stage,
                "The rotated orbitals did not converge to a lower-energy restricted solution.",
            )
    raise UnstableReferenceError("internal", stage, "Restricted orbital-rotation recovery exhausted its budget.")


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
    # Restore S-orthonormal occupied orbitals so Tr(D S) equals the closed-shell electron count.
    metric_inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    orthogonalized = orbitals @ metric_inverse_root
    return 2 * orthogonalized @ orthogonalized.T


def _scf_residual(mean_field: RKS) -> float:
    """Measure the occupied–virtual orbital gradient of the final density."""
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
    # Rebuild the physical Fock matrix from the converged density. The
    # iterative solver may have used DIIS or second-order intermediates.
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

    # Match PySCF's restricted TDA convention X.T @ X = 1/2 after flattening X.
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
