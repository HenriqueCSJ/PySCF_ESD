"""Diagonalization and coherent redistribution of vibrational line strengths."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class VibrationalMixingResult:
    """Vibrational term energies and transition dipoles in the mixed basis.

    Energies are sorted in ascending order. Each column of ``eigenvectors``
    gives the coefficients of the corresponding mixed state in the input
    basis. Transition dipoles have shape ``(..., n_states, 3)`` with Cartesian
    components in x, y, z order. Dipole strengths are their squared norms.
    Eigenvector phases are arbitrary, and eigenvectors within a degenerate
    subspace may undergo any orthogonal rotation.
    """

    energies_hartree: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    transition_dipole_au: NDArray[np.float64]
    dipole_strength_au2: NDArray[np.float64]


def _finite_real_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite in double precision")
    return array


def mix_vibrational_states(
    hamiltonian_hartree: ArrayLike,
    basis_transition_dipole_au: ArrayLike,
) -> VibrationalMixingResult:
    """Mix a real vibrational Hamiltonian block and its transition dipoles.

    Supply a nonempty, exactly symmetric Hamiltonian in an orthonormal
    vibrational basis. Its diagonal holds term energies in Hartree with a
    common arbitrary reference, retained in the output. Its off-diagonal
    entries are couplings in Hartree.

    Supply signed Cartesian transition dipoles in atomic units, with shape
    ``(..., n_basis, 3)`` and the same order and basis phases as the Hamiltonian.
    Each leading index can describe a separate untransformed partner state.
    The amplitudes are rotated along the basis axis before taking squared
    norms. Strengths do not include populations or photon-energy factors.
    """
    hamiltonian = _finite_real_array(hamiltonian_hartree, "hamiltonian_hartree")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1] or not hamiltonian.size:
        raise ValueError("hamiltonian_hartree must be a nonempty square matrix")
    if not np.array_equal(hamiltonian, hamiltonian.T):
        raise ValueError("hamiltonian_hartree must be symmetric")
    dipoles = _finite_real_array(basis_transition_dipole_au, "basis_transition_dipole_au")
    n_basis = hamiltonian.shape[0]
    if dipoles.shape[-2:] != (n_basis, 3):
        raise ValueError("basis_transition_dipole_au must have shape (..., n_basis, 3)")

    # Remove a common offset without subtracting opposite-signed large energies.
    diagonal = np.diag(hamiltonian)
    reference = np.clip(0.0, diagonal.min(), diagonal.max())
    shifted = hamiltonian.copy()
    shifted[np.diag_indices(n_basis)] -= reference
    energies, eigenvectors = np.linalg.eigh(shifted)
    with np.errstate(over="ignore", invalid="ignore"):
        energies += reference
        amplitudes = eigenvectors.T @ dipoles
        strengths = np.sum(np.square(amplitudes), axis=-1)
    for array in (energies, eigenvectors, amplitudes, strengths):
        if not np.all(np.isfinite(array)):
            raise ValueError("mixed energies, eigenvectors, or dipoles exceed the numerical range")
        array.flags.writeable = False
    return VibrationalMixingResult(energies, eigenvectors, amplitudes, strengths)
