"""Prepare harmonic spectroscopy parameters from completed PySCF calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .harmonic import HarmonicSpectrum, harmonic_spectrum

if TYPE_CHECKING:
    from pyscf.gto import Mole


@dataclass(frozen=True)
class VerticalGradientParameters:
    """Equal-curvature harmonic surfaces in the ground-state normal modes.

    ``frequencies_hartree`` stores vibrational quanta hbar*omega. The
    mass-weighted coordinates obey ``Q_excited = Q_ground + shift_au``,
    with units sqrt(electron mass) * bohr. ``gap_00_hartree`` is the 0-0
    gap, equal to the minima gap because the curvatures are identical.
    ``transition_dipole_au`` is a length-gauge vector in e * bohr.
    ``cartesian_mode_derivatives_au[k, a, x]`` gives dR[a,x]/dQ[k]
    in 1/sqrt(electron mass), with the same signs as ``shift_au``.
    """

    frequencies_hartree: NDArray[np.float64]
    shift_au: NDArray[np.float64]
    gap_00_hartree: float
    transition_dipole_au: NDArray[np.float64]
    reorganization_energy_hartree: float
    cartesian_mode_derivatives_au: NDArray[np.float64]


@dataclass(frozen=True)
class AdiabaticHessianParameters:
    """Harmonic model of two minima in the ground Mole's Cartesian frame.

    With normal coordinates in sqrt(electron mass) * bohr, the fitted
    relation is ``Q1 = rotation @ Q0 + shift_au``. The raw overlap between
    vibrational subspaces has the reported ``subspace_singular_values``.
    ``rotation_correction_norm`` is the spectral norm of the difference
    between the fitted orthogonal rotation and that overlap. The reported
    displacement norm and maximum residual after projection onto either
    vibrational subspace use the same mass-weighted coordinate units.

    Both sets of ``cartesian_mode_derivatives_au[k, a, x]`` are signed
    dR[a,x]/dQ[k] in the common frame, in 1/sqrt(electron mass). Modes
    follow frequency order with eigensolver-dependent signs and bases in
    degenerate spaces. The excited coordinates align as row vectors via
    ``R1 @ excited_alignment_rotation + excited_alignment_translation_bohr``.
    The alignment rotation has determinant +1, while the signed Duschinsky
    matrix can have either determinant.
    """

    ground_frequencies_hartree: NDArray[np.float64]
    excited_frequencies_hartree: NDArray[np.float64]
    gap_00_hartree: float
    rotation: NDArray[np.float64]
    shift_au: NDArray[np.float64]
    ground_cartesian_mode_derivatives_au: NDArray[np.float64]
    excited_cartesian_mode_derivatives_au: NDArray[np.float64]
    excited_alignment_rotation: NDArray[np.float64]
    excited_alignment_translation_bohr: NDArray[np.float64]
    subspace_singular_values: NDArray[np.float64]
    rotation_correction_norm: float
    displacement_norm_au: float
    displacement_projection_residual_au: float

    def spectrum(
        self,
        *,
        transition_dipole_au: ArrayLike,
        dipole_derivatives_au: ArrayLike | None = None,
        initial_max_quanta: int = 0,
        final_max_quanta: int = 5,
        temperature_kelvin: float = 0.0,
        process: Literal["absorption", "emission"] = "absorption",
        max_overlaps: int = 250_000,
    ) -> HarmonicSpectrum:
        """Calculate sticks with a real dipole expanded at the S0 minimum.

        The length-gauge dipole (e * bohr) and the vector components of
        ``dipole_derivatives_au`` share the ground Mole's Cartesian frame
        and one electronic phase. Use the returned S0 mode signs and order
        for the derivative rows dμ/dQ0. The same input expansion serves
        both processes. Emission uses the inverse affine relation and
        evaluates that expansion at the modeled S1 origin.
        """
        dipole = _as_real_array(transition_dipole_au, (3,), "transition dipole")
        derivatives = None
        if dipole_derivatives_au is not None:
            derivatives = _as_real_array(
                dipole_derivatives_au, (self.ground_frequencies_hartree.size, 3),
                "ground normal-mode dipole derivatives",
            )

        initial = self.ground_frequencies_hartree
        final = self.excited_frequencies_hartree
        gap = self.gap_00_hartree
        rotation = self.rotation
        shift = self.shift_au
        if process == "emission":
            initial, final = final, initial
            gap = -gap
            rotation = self.rotation.T
            shift = -(rotation @ self.shift_au)
            if derivatives is not None:
                # Evaluating Q0 = J.T @ Q1 - J.T @ K at Q1=0 shifts the
                # constant term. The derivative rows transform with J.
                dipole = dipole + shift @ derivatives
                derivatives = self.rotation @ derivatives

        return harmonic_spectrum(
            initial, final, gap_00_hartree=gap, rotation=rotation, shift_au=shift,
            transition_dipole_au=dipole, dipole_derivatives_au=derivatives,
            initial_max_quanta=initial_max_quanta, final_max_quanta=final_max_quanta,
            temperature_kelvin=temperature_kelvin, process=process, max_overlaps=max_overlaps,
        )


def adiabatic_hessian(
    ground_mol: Mole,
    excited_mol: Mole,
    ground_hessian: ArrayLike,
    excited_hessian: ArrayLike,
    *,
    ground_energy_hartree: float,
    excited_energy_hartree: float,
    ground_gradient: ArrayLike,
    excited_gradient: ArrayLike,
    masses_amu: ArrayLike | None = None,
    stationary_tolerance: float = 1e-5,
    subspace_tolerance: float = 1e-3,
) -> AdiabaticHessianParameters:
    """Build an adiabatic-Hessian model from completed, separate minima.

    The caller establishes atom correspondence, state continuity, and
    consistency of the electronic method and energy reference across the
    minima. Supply each total energy (Hartree), total gradient (natm, 3;
    Hartree/bohr), and total Hessian (natm, natm, 3, 3; Hartree/bohr**2)
    at its own built PySCF Mole, in that Mole's native frame and atom order.
    The same atom-ordered masses are used on both surfaces. With no explicit
    masses, the PySCF isotope-averaged lists including overrides must agree.

    ``stationary_tolerance`` bounds the largest per-atom gradient norm in
    Hartree/bohr. After proper mass-weighted alignment, each Hessian is
    projected against the translations and rotations of its own geometry.
    Rotational generators with singular values at or below 1e-10 of the
    largest are treated as redundant, giving five rigid motions for linear
    molecules and six for nonlinear molecules. The coordinate overlap is
    fitted by its orthogonal polar factor only for full-rank overlaps and
    when max(abs(1 - singular_values)) is within ``subspace_tolerance``.
    The tolerance must be in [0, 1). The default 0.001 permits a maximum
    principal angle of about 2.56 degrees. Increasing it explicitly
    accepts larger changes to the projected coordinate relation.
    """
    from pyscf.data.nist import AMU2AU

    natm = ground_mol.natm
    if natm < 2 or excited_mol.natm != natm:
        raise ValueError("adiabatic-Hessian analysis requires matching molecules with at least two atoms")
    if (ground_mol.charge != excited_mol.charge or any(
        ground_mol.atom_pure_symbol(atom) != excited_mol.atom_pure_symbol(atom)
        for atom in range(natm)
    )):
        raise ValueError("the two Mole inputs must have matching ordered elements and molecular charge")

    stationary_tolerance = _positive_tolerance(stationary_tolerance, "stationary_tolerance")
    subspace_tolerance = float(_as_real_array(subspace_tolerance, (), "subspace_tolerance"))
    if not 0 <= subspace_tolerance < 1:
        raise ValueError("subspace_tolerance must be in [0, 1)")

    if masses_amu is None:
        masses_amu = ground_mol.atom_mass_list(isotope_avg=True)
        if not np.array_equal(masses_amu, excited_mol.atom_mass_list(isotope_avg=True)):
            raise ValueError("the two Mole inputs must have matching atom-ordered masses")
    mass = _as_real_array(masses_amu, (natm,), "masses in amu")
    if np.any(mass <= 0):
        raise ValueError("masses in amu must be positive")

    energy0 = float(_as_real_array(ground_energy_hartree, (), "ground total energy"))
    energy1 = float(_as_real_array(excited_energy_hartree, (), "excited total energy"))
    for name, gradient in (("ground", ground_gradient), ("excited", excited_gradient)):
        gradient = _as_real_array(gradient, (natm, 3), f"{name} gradient")
        maximum = np.max(np.linalg.norm(gradient, axis=1))
        if maximum > stationary_tolerance:
            raise ValueError(
                f"{name} geometry fails the stationarity tolerance: maximum per-atom "
                f"gradient norm {maximum:.3g} Hartree/bohr exceeds {stationary_tolerance:.3g}"
            )

    hessian0 = _symmetric_hessian(ground_hessian, natm, "ground Hessian")
    hessian1 = _symmetric_hessian(excited_hessian, natm, "excited Hessian")
    coords0 = _as_real_array(ground_mol.atom_coords(), (natm, 3), "ground coordinates")
    coords1 = _as_real_array(excited_mol.atom_coords(), (natm, 3), "excited coordinates")
    center0 = np.average(coords0, weights=mass, axis=0)
    center1 = np.average(coords1, weights=mass, axis=0)
    centered0 = coords0 - center0
    centered1 = coords1 - center1
    left, _, right = np.linalg.svd(centered1.T @ (mass[:, None] * centered0))
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    alignment = left @ right
    translation = center0 - center1 @ alignment

    sqrt_mass = np.sqrt(mass) * np.sqrt(AMU2AU)
    omega0, modes0 = _stationary_modes(centered0, hessian0, sqrt_mass, "ground")
    omega1, modes1 = _stationary_modes(centered1, hessian1, sqrt_mass, "excited")
    if omega0.size != omega1.size:
        raise ValueError("the two geometries must have the same vibrational dimension")
    aligned_modes1 = (modes1.T.reshape(-1, natm, 3) @ alignment).reshape(omega1.size, -1).T
    overlap = aligned_modes1.T @ modes0
    left, singular, right = np.linalg.svd(overlap)
    correction = float(np.max(np.abs(1 - singular)))
    rank_tolerance = np.finfo(float).eps * omega0.size * singular[0]
    if singular[-1] <= rank_tolerance or correction > max(subspace_tolerance, rank_tolerance):
        raise ValueError(
            f"vibrational subspace loss exceeds subspace_tolerance: "
            f"min singular value {singular.min():.8g}, correction norm {correction:.3g}, "
            f"tolerance {subspace_tolerance:.3g}"
        )
    rotation = left @ right
    displacement = ((centered0 - centered1 @ alignment) * sqrt_mass[:, None]).ravel()
    shift = aligned_modes1.T @ displacement
    residual = max(
        np.linalg.norm(displacement - modes0 @ (modes0.T @ displacement)),
        np.linalg.norm(displacement - aligned_modes1 @ shift),
    )
    gap = float(energy1 - energy0 + 0.5 * (omega1.sum() - omega0.sum()))
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("adiabatic-Hessian model requires a finite, positive 0-0 gap")

    return AdiabaticHessianParameters(
        ground_frequencies_hartree=omega0,
        excited_frequencies_hartree=omega1,
        gap_00_hartree=gap,
        rotation=rotation,
        shift_au=shift,
        ground_cartesian_mode_derivatives_au=modes0.T.reshape(-1, natm, 3) / sqrt_mass[:, None],
        excited_cartesian_mode_derivatives_au=aligned_modes1.T.reshape(-1, natm, 3) / sqrt_mass[:, None],
        excited_alignment_rotation=alignment,
        excited_alignment_translation_bohr=translation,
        subspace_singular_values=singular,
        rotation_correction_norm=correction,
        displacement_norm_au=float(np.linalg.norm(displacement)),
        displacement_projection_residual_au=float(residual),
    )


def vertical_gradient(
    mol: Mole,
    ground_hessian: ArrayLike,
    excited_gradient: ArrayLike,
    *,
    vertical_excitation_hartree: float,
    transition_dipole_au: ArrayLike,
    masses_amu: ArrayLike | None = None,
) -> VerticalGradientParameters:
    """Prepare a vertical-gradient model, using PySCF's normal-mode analysis.

    Supply a built PySCF Mole at the stationary ground-state minimum. All
    inputs refer to that geometry and use the Mole's Cartesian frame and
    atom order. The ground-state Hessian has PySCF's (natm, natm, 3, 3)
    shape and units Hartree/bohr**2. The total excited-state
    gradient has shape (natm, 3) and units Hartree/bohr. The vertical
    excitation is E_excited(R_ground) - E_ground(R_ground) in Hartree.
    The real length-gauge transition dipole has shape (3,) in e * bohr.

    With ``masses_amu=None``, this uses PySCF's isotope-averaged atomic
    masses including any mass overrides in the Mole's nuclear properties.
    All masses must be positive. The model requires at least one
    vibrational mode, positive vibrational curvatures and a positive 0-0 gap.
    """
    from pyscf.data.nist import AMU2AU
    from pyscf.hessian.thermo import harmonic_analysis

    natm = mol.natm
    if natm < 2:
        raise ValueError("vertical-gradient analysis requires at least two atoms")

    hessian = _symmetric_hessian(ground_hessian, natm, "ground Hessian")

    gradient = _as_real_array(excited_gradient, (natm, 3), "excited gradient")
    dipole = _as_real_array(transition_dipole_au, (3,), "transition dipole").copy()
    vertical_excitation = float(
        _as_real_array(vertical_excitation_hartree, (), "vertical excitation")
    )
    if vertical_excitation <= 0:
        raise ValueError("vertical excitation must be positive")

    if masses_amu is None:
        masses_amu = mol.atom_mass_list(isotope_avg=True)
    mass = _as_real_array(masses_amu, (natm,), "masses in amu")
    if np.any(mass <= 0):
        raise ValueError("masses in amu must be positive")

    modes = harmonic_analysis(mol, hessian, imaginary_freq=False, mass=mass)
    mass_factor = np.sqrt(AMU2AU)
    frequencies = modes["freq_au"] / mass_factor
    if frequencies.size == 0:
        raise ValueError("the geometry has no retained vibrational modes")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("ground Hessian yields nonfinite vibrational frequencies")
    if np.any(frequencies <= 0):
        raise ValueError("ground Hessian has imaginary or zero vibrational frequencies")

    # PySCF's norm_mode includes the Cartesian amu**(-1/2) factors.
    cartesian_mode_derivatives = modes["norm_mode"] / mass_factor
    projected_gradient = np.einsum("kax,ax->k", cartesian_mode_derivatives, gradient)
    shift = projected_gradient / np.square(frequencies)
    relaxation = float(0.5 * np.dot(projected_gradient, shift))
    gap_00 = vertical_excitation - relaxation
    if not np.isfinite(gap_00) or gap_00 <= 0:
        raise ValueError("vertical-gradient model predicts an invalid or nonpositive 0-0 gap")

    return VerticalGradientParameters(
        frequencies, shift, gap_00, dipole, relaxation, cartesian_mode_derivatives,
    )


def _stationary_modes(
    centered: NDArray[np.float64],
    hessian: NDArray[np.float64],
    sqrt_mass: NDArray[np.float64],
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    natm = len(sqrt_mass)
    weights = sqrt_mass[:, None, None]
    translations = np.broadcast_to(np.eye(3), (natm, 3, 3)) * weights
    rotations = np.cross(np.eye(3), centered[:, None, :]).transpose(0, 2, 1) * weights
    axes, singular, _ = np.linalg.svd(rotations.reshape(-1, 3), full_matrices=False)
    rank = np.count_nonzero(singular > 1e-10 * singular[0])
    if rank < 2:
        raise ValueError(f"{name} geometry has undefined rotational degrees of freedom")

    rigid = np.column_stack((translations.reshape(-1, 3), axes[:, :rank]))
    frame, _ = np.linalg.qr(rigid, mode="complete")
    vibrational = frame[:, rigid.shape[1]:]
    if not vibrational.shape[1]:
        raise ValueError(f"{name} geometry has no retained vibrational modes")
    cartesian = hessian.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    mass_inv = np.repeat(1 / sqrt_mass, 3)
    weighted = mass_inv[:, None] * cartesian * mass_inv
    force_constants, eigenvectors = np.linalg.eigh(vibrational.T @ weighted @ vibrational)
    if not np.all(np.isfinite(force_constants)):
        raise ValueError(f"{name} Hessian yields nonfinite vibrational frequencies")
    zero = 64 * np.finfo(float).eps * np.linalg.norm(weighted, ord=np.inf)
    if np.any(force_constants <= zero):
        raise ValueError(f"{name} Hessian has imaginary or zero vibrational frequencies")
    return np.sqrt(force_constants), vibrational @ eigenvectors


def _symmetric_hessian(values: ArrayLike, natm: int, name: str) -> NDArray[np.float64]:
    hessian = _as_real_array(values, (natm, natm, 3, 3), name)
    transpose = hessian.transpose(1, 0, 3, 2)
    if not np.allclose(hessian, transpose, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} must be symmetric in its two Cartesian indices")
    return (hessian + transpose) * 0.5


def _positive_tolerance(value: float, name: str) -> float:
    tolerance = float(_as_real_array(value, (), name))
    if tolerance <= 0:
        raise ValueError(f"{name} must be positive")
    return tolerance


def _as_real_array(values: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.shape != shape or np.iscomplexobj(array):
        raise ValueError(f"{name} must be real with shape {shape}")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
