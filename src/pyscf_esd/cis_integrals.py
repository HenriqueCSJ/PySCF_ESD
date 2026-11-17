"""Dense AO integral derivatives for nonrelativistic, all-electron RHF/CIS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from pyscf.gto import Mole


@dataclass(frozen=True)
class IntegralValues:
    """AO overlap, core Hamiltonian, chemists' ERIs, and nuclear repulsion."""

    overlap: NDArray[np.float64]
    hcore: NDArray[np.float64]
    eri: NDArray[np.float64]
    enuc: float


@dataclass(frozen=True)
class OneElectronValues:
    overlap: NDArray[np.float64]
    hcore: NDArray[np.float64]
    enuc: float


def _all_eri_centers(contribution):
    """Sum a canonical first-center term over all four ERI centers."""
    bra = contribution + contribution.transpose(1, 0, 2, 3)
    return bra + bra.transpose(2, 3, 0, 1)


class IntegralDerivativeEngine:
    """Evaluate first and mixed second derivatives for one fixed geometry.

    A direction is a real array shaped ``(natm, 3)``. For two directions
    ``left`` and ``right``, the derivatives are taken at ``s = t = 0`` for
    ``R(s, t) = R + s * left + t * right``, with coordinates in Bohr.
    The AO order is PySCF's spherical AO order. Full derivative kernels
    are cached on demand, so the ERI storage scales as the fourth power
    of the number of AOs.
    """

    def __init__(self, mol: Mole) -> None:
        if mol.cart or mol.has_ecp() or mol._pseudo:
            raise ValueError("Integral derivatives require all-electron spherical AOs")

        # Hold the geometry fixed and isolate changes of the rinv origin.
        self._mol = mol.copy()
        self._nao = self._mol.nao_nr()
        self._charges = self._mol.atom_charges()
        ao_bounds = self._mol.aoslice_by_atom()[:, 2:]
        self._ao_atoms = np.repeat(np.arange(self._mol.natm), ao_bounds[:, 1] - ao_bounds[:, 0])

        charged = np.flatnonzero(self._charges)
        pair_a, pair_b = np.tril_indices(charged.size, k=-1)
        self._pair_a, self._pair_b = charged[pair_a], charged[pair_b]
        coordinates = self._mol.atom_coords(unit="Bohr")
        self._pair_vectors = coordinates[self._pair_a] - coordinates[self._pair_b]
        distances = np.linalg.norm(self._pair_vectors, axis=1)
        if np.any(distances == 0):
            raise ValueError("Nuclear repulsion is undefined for coincident charged nuclei")
        products = self._charges[self._pair_a] * self._charges[self._pair_b]
        self._pair_inverse_cubed = products / distances**3
        self._pair_inverse_fifth = products / distances**5

    @cached_property
    def one_electron(self) -> OneElectronValues:
        """Reference one-electron integrals and nuclear repulsion."""
        return OneElectronValues(
            self._mol.intor("int1e_ovlp"),
            self._mol.intor("int1e_kin") + self._mol.intor("int1e_nuc"),
            float(self._mol.energy_nuc()),
        )

    @cached_property
    def base(self) -> IntegralValues:
        """Undifferentiated integral values at the reference geometry."""
        one = self.one_electron
        return IntegralValues(one.overlap, one.hcore, self._mol.intor("int2e", aosym="s1"), one.enuc)

    @cached_property
    def _first_one_electron(self):
        mol = self._mol
        nuclear = {}
        with mol.with_rinv_zeta(0):
            for atom in np.flatnonzero(self._charges):
                with mol.with_rinv_at_nucleus(atom):
                    nuclear[atom] = mol.intor("int1e_iprinv")
        return mol.intor("int1e_ipovlp"), mol.intor("int1e_ipkin"), nuclear

    @cached_property
    def _second_one_electron(self):
        mol = self._mol

        def second_kernel(name):
            return mol.intor(name).reshape(3, 3, self._nao, self._nao)

        nuclear = {}
        with mol.with_rinv_zeta(0):
            for atom in np.flatnonzero(self._charges):
                with mol.with_rinv_at_nucleus(atom):
                    nuclear[atom] = second_kernel("int1e_ipiprinv"), second_kernel("int1e_iprinvip")
        return second_kernel("int1e_ipipovlp"), second_kernel("int1e_ipipkin"), nuclear

    @cached_property
    def _first_eri(self):
        return self._mol.intor("int2e_ip1", aosym="s1")

    @cached_property
    def _second_eri(self):
        return tuple(
            self._mol.intor(name, aosym="s1").reshape(3, 3, *(self._nao,) * 4)
            for name in ("int2e_ipip1", "int2e_ipvip1", "int2e_ip1ip2")
        )

    def _direction(self, direction: ArrayLike):
        displacement = np.asarray(direction)
        if displacement.shape != (self._mol.natm, 3) or np.iscomplexobj(displacement):
            raise ValueError("A nuclear displacement must be real and have shape (natm, 3)")
        displacement = displacement.astype(float, copy=False)
        if not np.isfinite(displacement).all():
            raise ValueError("Nuclear displacements must be finite")
        ao = displacement[self._ao_atoms].T
        pairs = displacement[self._pair_a] - displacement[self._pair_b]
        return displacement, ao, pairs

    def first_one_electron(self, direction: ArrayLike) -> OneElectronValues:
        """Return analytic one-electron derivatives without building any ERIs."""
        nuclear, ao, pairs = self._direction(direction)
        overlap_kernel, kinetic_kernel, nuclear_kernels = self._first_one_electron

        # libcint differentiates the basis with respect to the electron
        # coordinates. A derivative with respect to its center reverses the sign.
        relative = ao[:, :, None] - ao[:, None, :]
        overlap = -np.einsum("xpq,xpq->pq", relative, overlap_kernel)
        hcore = -np.einsum("xpq,xpq->pq", relative, kinetic_kernel)

        # A translated nuclear attraction integral is invariant when both
        # AOs and the nucleus move together. Use AO velocities relative to
        # each nucleus to include its motion in the same contractions.
        for atom, kernel in nuclear_kernels.items():
            contribution = np.einsum("xp,xpq->pq", ao - nuclear[atom, :, None], kernel)
            hcore += self._charges[atom] * (contribution + contribution.T)

        enuc = -np.einsum("a,ax,ax->", self._pair_inverse_cubed, self._pair_vectors, pairs)
        return OneElectronValues(overlap, hcore, float(enuc))

    def first(self, direction: ArrayLike) -> IntegralValues:
        """Return the derivative along one nuclear displacement."""
        one = self.first_one_electron(direction)
        _, ao, _ = self._direction(direction)
        eri_first = -np.einsum("xp,xpqrs->pqrs", ao, self._first_eri)
        return IntegralValues(one.overlap, one.hcore, _all_eri_centers(eri_first), one.enuc)

    def second_one_electron(self, left: ArrayLike, right: ArrayLike) -> OneElectronValues:
        """Return mixed one-electron derivatives without building any ERIs."""
        left_nuclear, left_ao, left_pairs = self._direction(left)
        right_nuclear, right_ao, right_pairs = self._direction(right)
        overlap_kernel, kinetic_kernel, nuclear_kernels = self._second_one_electron

        left_difference = left_ao[:, :, None] - left_ao[:, None, :]
        right_difference = right_ao[:, :, None] - right_ao[:, None, :]
        overlap = np.einsum("xpq,ypq,xypq->pq", left_difference, right_difference, overlap_kernel)
        hcore = np.einsum("xpq,ypq,xypq->pq", left_difference, right_difference, kinetic_kernel)

        for atom, (same, different) in nuclear_kernels.items():
            relative_left = left_ao - left_nuclear[atom, :, None]
            relative_right = right_ao - right_nuclear[atom, :, None]
            contribution = np.einsum("xp,yp,xypq->pq", relative_left, relative_right, same)
            contribution += np.einsum("xp,yq,xypq->pq", relative_left, relative_right, different)
            hcore -= self._charges[atom] * (contribution + contribution.T)

        enuc = 3 * np.einsum(
            "a,ax,ax,ay,ay->", self._pair_inverse_fifth,
            self._pair_vectors, left_pairs, self._pair_vectors, right_pairs,
        ) - np.einsum("a,ax,ax->", self._pair_inverse_cubed, left_pairs, right_pairs)
        return OneElectronValues(overlap, hcore, float(enuc))

    def second(self, left: ArrayLike, right: ArrayLike) -> IntegralValues:
        """Return the mixed derivative along two nuclear displacements."""
        one = self.second_one_electron(left, right)
        _, left_ao, _ = self._direction(left)
        _, right_ao, _ = self._direction(right)
        same, same_pair, opposite_pair = self._second_eri
        diagonal = np.einsum("xp,yp,xypqrs->pqrs", left_ao, right_ao, same)
        within = np.einsum("xp,yq,xypqrs->pqrs", left_ao, right_ao, same_pair)
        across = np.einsum("xp,yr,xypqrs->pqrs", left_ao, right_ao, opposite_pair)

        # The derivative centers can be on the same AO, on distinct AOs in
        # the same pair, or on opposite pairs. Integral symmetry gives the
        # other placements, including the two possible derivative orders.
        eri = _all_eri_centers(diagonal)
        eri += _all_eri_centers(within)
        across = across + across.transpose(2, 3, 0, 1)
        across = across + across.transpose(1, 0, 2, 3)
        eri += across + across.transpose(0, 1, 3, 2)

        return IntegralValues(one.overlap, one.hcore, eri, one.enuc)


def integral_derivatives(
    mol: Mole, left: ArrayLike, right: ArrayLike,
) -> tuple[IntegralValues, IntegralValues, IntegralValues, IntegralValues]:
    """Return base values, two first derivatives, and the mixed second derivative."""
    engine = IntegralDerivativeEngine(mol)
    return engine.base, engine.first(left), engine.first(right), engine.second(left, right)
