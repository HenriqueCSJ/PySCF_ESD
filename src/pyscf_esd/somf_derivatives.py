"""Analytic Cartesian nuclear derivatives of molecular Breit–Pauli SOMF.

The operator, spin normalization, and spin-summed reference density are those
of :mod:`pyscf_esd.pyscf_soc`. For related SOC derivative factorization see
Hagai, Fujimoto, and Yanai, J. Chem. Theory Comput. 2025, 21, 11604–11619,
doi:10.1021/acs.jctc.5c01442.
"""

import numpy as np
from numpy.typing import NDArray

from .pyscf_soc import _breit_pauli_somf_ao


def _transpose(matrix):
    return matrix.swapaxes(-1, -2)


def _spin_vector(integrals):
    # The libcint sigma-dot-p pair emits (cross_x, cross_y, cross_z, dot)
    # for each component of the leading electronic-coordinate derivative.
    return integrals.reshape(3, 4, *integrals.shape[1:])[:, :3]


def _integral_derivatives(mol, density):
    # The electronic gradient of the bra has the opposite sign to its
    # basis-center derivative. The derivative of a specific nuclear
    # potential is minus the sum of both basis-center derivatives of the
    # corresponding integral, by translation invariance.
    bra_gradient = _spin_vector(mol.intor("int1e_ipspnucsp", comp=12))
    nuclear_mol = mol.copy()
    nuclear_mol.set_rinv_zeta(0.)
    for atom, (shell_start, shell_end, ao_start, ao_end) in enumerate(mol.aoslice_by_atom()):
        orbitals = slice(ao_start, ao_end)
        bra = np.zeros_like(bra_gradient)
        bra[..., orbitals, :] = -bra_gradient[..., orbitals, :]
        charge = mol.atom_charge(atom)
        if charge:
            with nuclear_mol.with_rinv_at_nucleus(atom):
                bra -= charge * _spin_vector(nuclear_mol.intor("int1e_ipsprinvsp", comp=12))
        derivative = bra - _transpose(bra)
        if shell_start != shell_end:
            derivative += _two_electron_integral_derivative(
                mol, density, (shell_start, shell_end), orbitals,
            )
        yield derivative


def _two_electron_integral_derivative(mol, density, shells, orbitals):
    from pyscf.scf import jk

    # G[i,j,k,l] = <i(1) k(2) | p1 × (1/r12) p1 | j(1) l(2)> is
    # antisymmetric in i,j and symmetric in k,l.
    # The two native gradient integrals put the electronic gradient in
    # their first slot, once before and once after exchanging electron pairs.
    whole = (0, mol.nbas)
    atom_columns = density[:, orbitals]
    direct_i, exchange_i, exchange_j = jk.get_jk(
        mol, (density, density, atom_columns),
        ("ijkl,lk->ij", "ijkl,jk->il", "ijkl,li->kj"),
        intor="int2e_ipspsp1", aosym="s2kl", comp=12, shls_slice=shells + whole * 3,
    )
    direct_k, exchange_k, exchange_l = jk.get_jk(
        mol, (atom_columns, atom_columns, density),
        ("ijkl,ji->kl", "ijkl,li->kj", "ijkl,lj->ki"),
        intor="int2e_ip1spsp2", comp=12, shls_slice=shells + whole * 3,
    )
    direct = -2 * _spin_vector(direct_k)
    direct[..., orbitals, :] -= _spin_vector(direct_i)
    direct[..., :, orbitals] += _transpose(_spin_vector(direct_i))

    exchange = _transpose(_spin_vector(exchange_j)) - _spin_vector(exchange_k)
    exchange[..., orbitals, :] -= _spin_vector(exchange_i)
    exchange[..., :, orbitals] -= _spin_vector(exchange_l)
    return direct - 1.5 * (exchange - _transpose(exchange))


def _two_electron_density_derivative(mol, derivatives):
    from pyscf.scf import jk

    matrices = tuple(derivatives)
    count = len(matrices)
    contractions = jk.get_jk(
        mol, matrices * 2, ("ijkl,lk->ij",) * count + ("ijkl,jk->il",) * count,
        intor="int2e_p1vxp1", aosym="s2kl", comp=3,
    )
    direct = np.stack(contractions[:count])
    exchange = np.stack(contractions[count:])
    return direct - 1.5 * (exchange - _transpose(exchange))


def _real_array(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return np.asarray(array, dtype=float)


def breit_pauli_somf_ao_derivatives(
    mol, density, density_derivatives=None,
) -> NDArray[np.complex128]:
    """Return SOMF slopes in Hartree/bohr as (atom, Cartesian, spin, AO, AO).

    The spin index selects the three spatial coefficients multiplying the
    electronic spin operators. Supply the real symmetric spin-summed AO
    density and, for a relaxed density, its coefficient derivatives with
    shape (atom, Cartesian, AO, AO). If these derivatives are omitted, the
    AO density coefficients are held fixed while every AO center and each
    nuclear Coulomb potential are differentiated. Stock libcint gradient
    integrals are contracted directly in AO shells.
    """
    from pyscf.data.nist import LIGHT_SPEED

    if mol.has_ecp() or mol.pseudo:
        raise ValueError("Breit–Pauli SOMF derivatives require an all-electron molecule")
    nao = mol.nao_nr()
    density = _real_array(density, (nao, nao), "density")
    if not np.allclose(density, _transpose(density), rtol=0, atol=1e-12):
        raise ValueError("density must be symmetric")
    if density_derivatives is not None:
        density_derivatives = _real_array(
            density_derivatives, (mol.natm, 3, nao, nao), "density_derivatives",
        )
        if not np.allclose(density_derivatives, _transpose(density_derivatives), rtol=0, atol=1e-12):
            raise ValueError("density_derivatives must be symmetric")

    derivative = np.empty((mol.natm, 3, 3, nao, nao))
    for atom, integral_derivative in enumerate(_integral_derivatives(mol, density)):
        if density_derivatives is not None:
            integral_derivative += _two_electron_density_derivative(mol, density_derivatives[atom])
        derivative[atom] = integral_derivative
    return 1j * derivative / (2 * LIGHT_SPEED**2)


def breit_pauli_somf_mo_derivatives(
    mol, coefficients, occupations, coefficient_derivatives, *, directions=None,
) -> NDArray[np.complex128]:
    """Return SOMF slopes in Hartree/bohr as (atom, Cartesian, spin, MO, MO).

    Supply real molecular orbitals, their fixed 0 or 2 occupations, and
    their first derivatives with shape (atom, Cartesian, AO, MO). Optionally
    pass ``directions`` as (direction, atom, Cartesian) in bohr per input
    coordinate, with coefficient derivatives shaped (direction, AO, MO).
    The returned shape is then (direction, spin, MO, MO), in Hartree per
    input coordinate. The result includes the AO integral, spin-summed
    density and MO transformation responses. The supplied orbital
    derivatives define the MO gauge.
    """
    from pyscf.data.nist import LIGHT_SPEED

    if mol.has_ecp() or mol.pseudo:
        raise ValueError("Breit–Pauli SOMF derivatives require an all-electron molecule")
    nao = mol.nao_nr()
    occupations = np.asarray(occupations)
    if (occupations.ndim != 1 or not len(occupations) or occupations.dtype.kind not in "fiu"
            or not np.all((occupations == 0) | (occupations == 2))
            or occupations.sum() != mol.nelectron):
        raise ValueError("occupations must be a vector of 0 or 2 for the molecular electron count")
    nmo = len(occupations)
    coefficients = _real_array(coefficients, (nao, nmo), "coefficients")
    coordinate_shape = (mol.natm, 3)
    if directions is not None:
        directions = np.asarray(directions)
        if directions.ndim != 3 or not directions.shape[0]:
            raise ValueError("directions must have shape (direction, atom, Cartesian)")
        coordinate_shape = (len(directions),)
        directions = _real_array(directions, (*coordinate_shape, mol.natm, 3), "directions")
    dc = _real_array(coefficient_derivatives, (*coordinate_shape, nao, nmo), "coefficient_derivatives")
    occupied = coefficients[:, occupations == 2]
    density = 2 * occupied @ occupied.T
    slope = np.zeros((*coordinate_shape, 3, nmo, nmo))
    for atom, integral_derivative in enumerate(_integral_derivatives(mol, density)):
        mo_derivative = coefficients.T @ integral_derivative @ coefficients
        if directions is None:
            slope[atom] = mo_derivative
        else:
            slope += np.einsum("du,uxij->dxij", directions[:, atom], mo_derivative)

    # Batch density perturbations for the AO shell traversal while bounding
    # the size of the temporary AO matrices.
    prefactor = 1 / (2 * LIGHT_SPEED**2)
    ao_times_coefficients = (_breit_pauli_somf_ao(mol, density).imag / prefactor) @ coefficients
    flat_dc = dc.reshape(-1, nao, nmo)
    flat_slope = slope.reshape(-1, 3, nmo, nmo)
    batch_size = max(1, (64 * 1024**2) // (16 * density.nbytes))
    for start in range(0, len(flat_dc), batch_size):
        batch = slice(start, start + batch_size)
        density_half_derivative = 2 * flat_dc[batch][..., occupations == 2] @ occupied.T
        density_derivatives = density_half_derivative + _transpose(density_half_derivative)
        density_response = _two_electron_density_derivative(mol, density_derivatives)
        flat_slope[batch] += coefficients.T @ density_response @ coefficients
        orbital_response = _transpose(flat_dc[batch])[:, None, :, :] @ ao_times_coefficients
        flat_slope[batch] += orbital_response - _transpose(orbital_response)
    return (1j * prefactor) * slope
