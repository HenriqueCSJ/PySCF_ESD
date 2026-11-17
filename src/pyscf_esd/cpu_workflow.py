"""Analytic vertical-gradient spectroscopy and optional state-tracked differences on CPU."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cpu import (
    ElectronicConvergenceError, ElectronicPoint, Settings, build_molecule, require_stable_reference, solve_point,
)
from .harmonic import HarmonicSpectrum, harmonic_spectrum
from .pyscf_adapter import VerticalGradientParameters, vertical_gradient
from .state_tracking import StateMatch, StateTrackingError, orbital_overlap_blocks, track_state


@dataclass(frozen=True)
class Displacement:
    axis: int
    sign: int
    match: StateMatch
    cached: bool
    scf_residual_hartree: float
    tda_max_residual_hartree: float
    tda_route: str


@dataclass(frozen=True)
class CpuVibronicResult:
    center: ElectronicPoint
    target_index: int
    model: VerticalGradientParameters
    ground_gradient_hartree_per_bohr: NDArray[np.float64]
    excited_gradient_hartree_per_bohr: NDArray[np.float64]
    energy_difference_gradient_hartree_per_bohr: NDArray[np.float64] | None
    ground_hessian_asymmetry_hartree_per_bohr2: float
    excited_hessian_hartree_per_bohr2: NDArray[np.float64] | None
    hessian_asymmetry_hartree_per_bohr2: float | None
    dipole_jacobian_au_per_bohr: NDArray[np.float64] | None
    normal_dipole_derivatives_au: NDArray[np.float64] | None
    displacements: tuple[Displacement, ...]
    ground_hessian_hartree_per_bohr2: NDArray[np.float64] | None = None

    def spectrum(
        self, *, initial_max_quanta: int = 0, final_max_quanta: int = 5,
        temperature_kelvin: float = 0.0, process: str = "absorption",
        include_ht: bool | None = None,
        max_overlaps: int = 250_000,
    ) -> HarmonicSpectrum:
        """Calculate VG sticks with an optional linear transition dipole.

        Include HT by default when the result has dipole derivatives.
        With HT, the VG emission dipole is extrapolated linearly to the
        modeled excited minimum. VG uses the ground curvature for both surfaces.
        """
        if include_ht is None:
            include_ht = self.normal_dipole_derivatives_au is not None
        elif not isinstance(include_ht, (bool, np.bool_)):
            raise ValueError("include_ht must be a boolean or None")
        if include_ht and self.normal_dipole_derivatives_au is None:
            raise ValueError("HT sticks require dipole derivatives from run_cpu_vibronic(include_ht=True)")

        model = self.model
        gap = model.gap_00_hartree
        shift = model.shift_au
        dipole = model.transition_dipole_au
        derivative = self.normal_dipole_derivatives_au if include_ht else None
        if process == "emission":
            if derivative is not None:
                dipole = dipole - shift @ derivative
            gap = -gap
            shift = -shift
        return harmonic_spectrum(
            model.frequencies_hartree, model.frequencies_hartree,
            gap_00_hartree=gap, transition_dipole_au=dipole, shift_au=shift,
            dipole_derivatives_au=derivative, initial_max_quanta=initial_max_quanta,
            final_max_quanta=final_max_quanta, temperature_kelvin=temperature_kelvin,
            process=process, max_overlaps=max_overlaps,
        )


def _write_checkpoint(path: Path, **fields) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **fields)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _match_point(center, other_mol, data, target_index, thresholds):
    from pyscf import gto

    cross_overlap = gto.intor_cross("int1e_ovlp", center.mol, other_mol)
    occ, vir = orbital_overlap_blocks(
        center.mo_coeff, data["mo_coeff"], cross_overlap,
        2 * (np.arange(center.mo_coeff.shape[1]) < center.nocc),
        2 * (np.arange(data["mo_coeff"].shape[1]) < int(data["nocc"])),
    )
    match = track_state(
        center.amplitudes[target_index], data["amplitudes"], occ, vir,
        candidate_energies_hartree=data["excitation_energies_hartree"], **thresholds,
    )
    if match.index == len(data["excitation_energies_hartree"]) - 1:
        raise StateTrackingError("the match is the highest solved root; request additional roots")
    return match


def _validate_center_options(
    settings: Settings, target_index: int, *, stationary_tolerance: float,
    minimum_gap_hartree: float, ground_hessian_symmetry_tolerance: float,
) -> None:
    if isinstance(target_index, (bool, np.bool_)) or not isinstance(target_index, (int, np.integer)):
        raise ValueError("target_index must be an integer")
    if not 0 <= target_index < settings.nstates - 1:
        raise ValueError("request the target and at least one higher reference root")
    if not np.isfinite(stationary_tolerance) or stationary_tolerance <= 0:
        raise ValueError("stationary_tolerance must be positive and finite")
    if not np.isfinite(ground_hessian_symmetry_tolerance) or ground_hessian_symmetry_tolerance <= 0:
        raise ValueError("ground_hessian_symmetry_tolerance must be positive and finite")
    if not np.isfinite(minimum_gap_hartree) or minimum_gap_hartree < 0:
        raise ValueError("minimum_gap_hartree is out of range")


def _prepare_vg_center(
    center: ElectronicPoint, target_index: int, *, stationary_tolerance: float,
    minimum_gap_hartree: float, ground_hessian_symmetry_tolerance: float,
    ground_hessian: NDArray[np.float64] | None = None,
) -> CpuVibronicResult:
    """Apply the same central-state checks and construct VG on either device."""
    require_stable_reference(*center.reference_stability(), "Vertical-gradient center reference")
    target_index = int(target_index)
    differences = np.abs(center.excitation_energies_hartree - center.excitation_energies_hartree[target_index])
    differences[target_index] = np.inf
    if differences.min() <= minimum_gap_hartree:
        raise StateTrackingError("the reference target is too close to another root")
    ground_gradient = center.ground_gradient()
    if np.max(np.abs(ground_gradient)) > stationary_tolerance:
        raise ValueError("the supplied ground-state geometry is not stationary at the requested tolerance")
    excited_gradient = center.excited_gradient(target_index)
    hessian = center.ground_hessian() if ground_hessian is None else ground_hessian
    hessian_transpose = hessian.transpose(1, 0, 3, 2)
    asymmetry = float(np.max(np.abs(hessian - hessian_transpose)))
    if asymmetry > ground_hessian_symmetry_tolerance:
        raise ValueError(f"ground Hessian asymmetry {asymmetry:.3g} Hartree/bohr² exceeds tolerance")
    symmetric_hessian = (hessian + hessian_transpose) * 0.5
    model = vertical_gradient(
        center.mol, symmetric_hessian, excited_gradient,
        vertical_excitation_hartree=center.excitation_energies_hartree[target_index],
        transition_dipole_au=center.transition_dipoles_au[target_index],
    )
    return CpuVibronicResult(
        center=center, target_index=target_index, model=model,
        ground_gradient_hartree_per_bohr=ground_gradient,
        excited_gradient_hartree_per_bohr=excited_gradient,
        energy_difference_gradient_hartree_per_bohr=None,
        ground_hessian_asymmetry_hartree_per_bohr2=asymmetry,
        excited_hessian_hartree_per_bohr2=None,
        hessian_asymmetry_hartree_per_bohr2=None,
        dipole_jacobian_au_per_bohr=None,
        normal_dipole_derivatives_au=None,
        displacements=(),
        ground_hessian_hartree_per_bohr2=symmetric_hessian,
    )


def run_cpu_vibronic(
    symbols: tuple[str, ...], coordinates_bohr: ArrayLike, settings: Settings, *,
    target_index: int, checkpoint_dir: str | Path | None = None,
    step_bohr: float = 0.005, stationary_tolerance: float = 1e-5,
    minimum_overlap: float = 0.65, minimum_margin: float = 0.10,
    minimum_gap_hartree: float = 1e-4, warm_start: bool = True,
    ground_hessian_symmetry_tolerance: float = 1e-5,
    excited_hessian_symmetry_tolerance: float = 2e-4,
    include_ht: bool = False,
    compute_excited_hessian: bool = False,
) -> CpuVibronicResult:
    """Run a CPU RKS/TDA vertical-gradient model at a stationary ground-state geometry.

    ``target_index`` is zero-based at the reference geometry. The default model
    uses the analytic ground Hessian and excited gradient at that geometry.
    Request ``include_ht`` for transition dipole differences or
    ``compute_excited_hessian`` for differences of the excited gradients.
    Both use central differences with state tracking in the supplied lab frame.
    Checkpoints retain raw dipoles and amplitudes for consistent phase matching.
    Each Hessian is symmetrized within its own tolerance (Hartree/bohr²).
    """
    for name, value in (("include_ht", include_ht), ("compute_excited_hessian", compute_excited_hessian)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    _validate_center_options(
        settings, target_index, stationary_tolerance=stationary_tolerance,
        minimum_gap_hartree=minimum_gap_hartree,
        ground_hessian_symmetry_tolerance=ground_hessian_symmetry_tolerance,
    )
    if not np.isfinite(step_bohr) or step_bohr <= 0:
        raise ValueError("step_bohr must be positive and finite")
    if not np.isfinite(excited_hessian_symmetry_tolerance) or excited_hessian_symmetry_tolerance <= 0:
        raise ValueError("excited_hessian_symmetry_tolerance must be positive and finite")
    for name, value, limit in (
        ("minimum_overlap", minimum_overlap, 1), ("minimum_margin", minimum_margin, 1),
    ):
        if not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"{name} is out of range")

    if isinstance(symbols, str):
        raise ValueError("symbols must contain one element symbol per atom")
    symbols = tuple(symbols)
    coordinates = np.asarray(coordinates_bohr)
    center = solve_point(symbols, coordinates, settings)
    target_index = int(target_index)
    result = _prepare_vg_center(
        center, target_index, stationary_tolerance=stationary_tolerance,
        minimum_gap_hartree=minimum_gap_hartree,
        ground_hessian_symmetry_tolerance=ground_hessian_symmetry_tolerance,
    )
    model = result.model
    if not (include_ht or compute_excited_hessian):
        return result

    import pyscf
    from pyscf import dft

    config = json.dumps({
        "schema_version": 1, "pyscf": pyscf.__version__, "numpy": np.__version__,
        "libxc": dft.libxc.__version__,
        "symbols": list(symbols), "coordinates_bohr": coordinates.tolist(),
        "settings": asdict(settings), "target_index": target_index, "step_bohr": step_bohr,
        "warm_start": bool(warm_start),
    }, sort_keys=True)
    folder = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    thresholds = dict(minimum_overlap=minimum_overlap, minimum_margin=minimum_margin,
                      minimum_gap_hartree=minimum_gap_hartree)
    size = coordinates.size
    raw_hessian = np.empty((size, size)) if compute_excited_hessian else None
    dipole_jacobian = np.empty((size, 3)) if include_ht else None
    energy_gradient = np.empty(size)
    displacements = []
    for axis in range(size):
        gradients, dipoles, energies = [], [], []
        for sign in (-1, 1):
            displaced = np.array(coordinates, dtype=float, copy=True)
            displaced.flat[axis] += sign * step_bohr
            if displaced.flat[axis] == coordinates.flat[axis]:
                raise ValueError("step_bohr is below coordinate precision")
            path = None if folder is None else folder / f"axis_{axis:03d}_{'minus' if sign < 0 else 'plus'}.npz"
            try:
                data = {}
                if path is not None and path.is_file():
                    with np.load(path, allow_pickle=False) as record:
                        data = dict(record)
                    if data["config"].item() != config or not np.array_equal(data["geometry_bohr"], displaced):
                        raise ValueError(f"checkpoint uses a different calculation: {path}")
                cached = bool(data) and (not compute_excited_hessian or "gradient" in data)
                if cached:
                    other_mol = build_molecule(symbols, displaced, settings)
                    match = _match_point(center, other_mol, data, target_index, thresholds)
                    if compute_excited_hessian and int(data["root_index"]) != match.index:
                        raise StateTrackingError("the checkpoint gradient belongs to a different root")
                else:
                    point = solve_point(symbols, displaced, settings, previous=center if warm_start else None)
                    data = {
                        name: getattr(point, name) for name in (
                            "mo_coeff", "nocc", "amplitudes", "excitation_energies_hartree",
                            "transition_dipoles_au", "ground_energy_hartree",
                        )
                    }
                    match = _match_point(center, point.mol, data, target_index, thresholds)
                    if compute_excited_hessian:
                        data["gradient"] = point.excited_gradient(match.index)
                    data["scf_residual"] = point.scf_residual
                    data["tda_residuals"] = point.tda_residuals
                    data["tda_route"] = point.tda_route
                    data["attempts_json"] = json.dumps([asdict(attempt) for attempt in point.attempts])
            except StateTrackingError as error:
                raise StateTrackingError(f"axis {axis}, displacement {sign:+d}: {error}") from error
            except ElectronicConvergenceError as error:
                raise ElectronicConvergenceError(f"axis {axis}, displacement {sign:+d}: {error}") from error

            dipoles_at_point = np.asarray(data["transition_dipoles_au"])
            ground_energy = np.asarray(data["ground_energy_hartree"])
            if (dipoles_at_point.shape != (settings.nstates, 3) or ground_energy.shape != ()
                or dipoles_at_point.dtype.kind != "f" or ground_energy.dtype.kind != "f"
                or not np.isfinite(dipoles_at_point).all() or not np.isfinite(ground_energy)):
                raise ValueError(f"invalid electronic data for axis {axis}, displacement {sign:+d}")
            energy = float(ground_energy) + data["excitation_energies_hartree"][match.index]
            if compute_excited_hessian:
                gradient = data["gradient"]
                if (gradient.shape != coordinates.shape or gradient.dtype.kind != "f"
                    or not np.isfinite(gradient).all()):
                    raise ValueError(f"invalid gradient for axis {axis}, displacement {sign:+d}")
                gradients.append(gradient.ravel())
            if include_ht:
                dipoles.append(match.phase * dipoles_at_point[match.index])
            scf_residual = float(data["scf_residual"])
            tda_residuals = np.asarray(data["tda_residuals"])
            if (not 0 <= scf_residual <= settings.scf_residual_tol
                or tda_residuals.shape != (settings.nstates,)
                or tda_residuals.dtype.kind != "f" or not np.isfinite(tda_residuals).all()
                or np.any(tda_residuals < 0) or np.max(tda_residuals) > settings.td_residual_tol):
                raise ValueError(f"invalid residuals for axis {axis}, displacement {sign:+d}")
            if path is not None and not cached:
                _write_checkpoint(path, config=config, geometry_bohr=displaced,
                                  root_index=match.index, **data)
            energies.append(energy)
            displacements.append(Displacement(
                axis, sign, match, cached, scf_residual, float(np.max(tda_residuals)),
                str(np.asarray(data.get("tda_route", "davidson")).item()),
            ))
        scale = 1 / (2 * step_bohr)
        if raw_hessian is not None:
            raw_hessian[:, axis] = (gradients[1] - gradients[0]) * scale
        if dipole_jacobian is not None:
            dipole_jacobian[axis] = (dipoles[1] - dipoles[0]) * scale
        energy_gradient[axis] = (energies[1] - energies[0]) * scale

    asymmetry = symmetric = None
    if raw_hessian is not None:
        asymmetry = float(np.max(np.abs(raw_hessian - raw_hessian.T)))
        if asymmetry > excited_hessian_symmetry_tolerance:
            raise ValueError(f"excited Hessian asymmetry {asymmetry:.3g} Hartree/bohr² exceeds tolerance")
        symmetric = (raw_hessian + raw_hessian.T) * 0.5
    normal_derivatives = None
    if dipole_jacobian is not None:
        normal_derivatives = model.cartesian_mode_derivatives_au.reshape(-1, size) @ dipole_jacobian
    return replace(
        result,
        energy_difference_gradient_hartree_per_bohr=energy_gradient.reshape(coordinates.shape),
        excited_hessian_hartree_per_bohr2=symmetric,
        hessian_asymmetry_hartree_per_bohr2=asymmetry,
        dipole_jacobian_au_per_bohr=dipole_jacobian,
        normal_dipole_derivatives_au=normal_derivatives,
        displacements=tuple(displacements),
    )
