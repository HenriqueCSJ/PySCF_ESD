"""Molecular Breit–Pauli SOMF state interaction from restricted PySCF DFT/TDA."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .phosphorescence import PhosphorescenceResult, spin_orbit_phosphorescence
from .tda_state_interaction import assemble_tda_state_interaction


@dataclass(frozen=True)
class PySCFTDASOC:
    """Spin-free energies and operators in the same restricted TDA state basis."""

    spin_free_energies_hartree: NDArray[np.float64]
    spin_orbit_hartree: NDArray[np.complex128]
    dipole_operator_au: NDArray[np.float64]
    singlet_roots: tuple[int, ...]
    triplet_roots: tuple[int, ...]

    def phosphorescence(
        self, triplet_root: int = 0, *, temperature_kelvin: float | None = None,
    ) -> PhosphorescenceResult:
        """Return electronic phosphorescence for a retained zero-based TDA root."""
        if (isinstance(triplet_root, (bool, np.bool_)) or not isinstance(triplet_root, Integral)
                or triplet_root not in self.triplet_roots):
            raise ValueError("triplet_root must select a retained native triplet TDA root")
        start = 1 + len(self.singlet_roots) + 3 * self.triplet_roots.index(triplet_root)
        return spin_orbit_phosphorescence(
            self.spin_free_energies_hartree, self.spin_orbit_hartree, self.dipole_operator_au,
            ground_index=0, triplet_indices=(start, start + 1, start + 2),
            temperature_kelvin=temperature_kelvin,
        )


def _validate_reference(mf):
    from pyscf.dft.rks import RKS
    from pyscf.dft.rks_symm import RKS as SymmetryRKS

    if not isinstance(mf, (RKS, SymmetryRKS)) or not mf.converged:
        raise ValueError("supply a converged CPU PySCF molecular RKS reference")
    mol = mf.mol
    if (mol.spin or mol.has_ecp() or mol.pseudo or getattr(mf, "with_x2c", None) is not None
            or getattr(mf, "with_solvent", None) is not None or getattr(mf, "mm_mol", None) is not None
            or "get_hcore" in vars(mf) or mf._numint._xc_type(mf.xc) == "HF"):
        raise ValueError("SOMF requires nonrelativistic, gas-phase, all-electron closed-shell DFT")
    coeff, occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    if (coeff.ndim != 2 or coeff.shape[0] != mol.nao_nr() or np.iscomplexobj(coeff)
            or not np.isfinite(coeff).all() or occ.shape != (coeff.shape[1],)
            or not np.all((occ == 0) | (occ == 2)) or occ.sum() != mol.nelectron
            or not np.any(occ == 2) or not np.any(occ == 0)):
        raise ValueError("SOMF needs finite real molecular orbitals with integer RKS occupations")
    return coeff[:, occ == 2], coeff[:, occ == 0]


def _read_roots(td, roots, shape, *, singlet):
    from pyscf.tdscf.rks import TDA, dTDA

    if (not isinstance(td, TDA) or isinstance(td, dTDA) or td.singlet != singlet
            or getattr(td, "frozen", None) not in (None, 0)
            or getattr(td, "with_solvent", None) is not None):
        raise ValueError("supply standard singlet and triplet PySCF RKS/TDA without frozen orbitals or solvent")
    if td.xy is None or td.e is None or td.converged is None or not len(td.xy):
        raise ValueError("solve both singlet and triplet TDA before assembling state matrices")
    nroots = len(td.xy)
    indices = tuple(range(nroots)) if roots is None else tuple(roots)
    if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                           or not 0 <= i < nroots for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("roots must select distinct zero-based indices of solved TDA roots")
    indices = tuple(map(int, indices))
    energies, converged = np.asarray(td.e), np.asarray(td.converged)
    if energies.shape != (nroots,) or converged.shape != (nroots,) or converged.dtype.kind != "b":
        raise ValueError("TDA energies and convergence flags must match the stored roots")
    energies = energies[list(indices)]
    if (np.iscomplexobj(energies) or not np.isfinite(energies).all() or np.any(energies <= 0)
            or not np.all(converged[list(indices)])):
        raise ValueError("all selected TDA roots must converge with positive excitation energies")
    amplitudes = []
    for index in indices:
        x, y = td.xy[index]
        x, y = np.asarray(x), np.asarray(y)
        if (x.shape != shape or np.iscomplexobj(x) or not np.isfinite(x).all()
                or not np.all(y == 0)):
            raise ValueError("TDA roots need finite real X amplitudes of occupied–virtual shape and zero Y")
        amplitudes.append(np.sqrt(2.) * x)
    return indices, energies, np.stack(amplitudes)


def _breit_pauli_somf_ao(mol, density):
    from pyscf.data.nist import LIGHT_SPEED
    from pyscf.scf import jk

    nuclear = mol.intor("int1e_pnucxp", comp=3)
    # G_ijkl has momenta on i and j and is symmetric in k,l. Contract
    # directly in AO shells with the spin-summed density. Exchange from
    # the other electron equals -K.T for a real symmetric density.
    direct, exchange = jk.get_jk(
        mol, (density, density), ("ijkl,lk->ij", "ijkl,jk->il"),
        intor="int2e_p1vxp1", aosym="s2kl", comp=3,
    )
    spatial = nuclear + direct - 1.5 * (exchange - exchange.swapaxes(-1, -2))
    # This coefficient multiplies electronic S = sigma/2 in the spin algebra.
    return 1j * spatial / (2 * LIGHT_SPEED**2)


def assemble_pyscf_tda_soc(singlet_tda, triplet_tda, *, singlet_roots=None, triplet_roots=None) -> PySCFTDASOC:
    """Assemble analytic SOC and electric-dipole matrices at one geometry.

    Both converged spin manifolds must use the same CPU PySCF RKS object.
    The operator is all-electron molecular Breit–Pauli SOMF with the RKS
    density. Interstate dipoles are unrelaxed matrix elements of normalized
    spin-adapted TDA singles, including the common nuclear dipole.
    This approximation uses no geometry displacements and does not evaluate
    nuclear derivatives of the SOC or interstate matrices.
    """
    mf = getattr(singlet_tda, "_scf", None)
    if mf is not getattr(triplet_tda, "_scf", None):
        raise ValueError("singlet and triplet TDA must share the same RKS reference object")
    occupied, virtual = _validate_reference(mf)
    shape = occupied.shape[1], virtual.shape[1]
    si, se, sx = _read_roots(singlet_tda, singlet_roots, shape, singlet=True)
    ti, te, tx = _read_roots(triplet_tda, triplet_roots, shape, singlet=False)
    mol = mf.mol
    coeff = np.column_stack((occupied, virtual))
    density = 2 * (occupied @ occupied.T)
    soc_ao = _breit_pauli_somf_ao(mol, density)
    with mol.with_common_origin(np.zeros(3)):
        dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
    nuclear = mol.atom_charges() @ mol.atom_coords()
    ground = nuclear + np.einsum("xij,ji->x", dipole_ao, density)
    operators = assemble_tda_state_interaction(
        sx, tx, coeff.T @ dipole_ao @ coeff, coeff.T @ soc_ao @ coeff, dipole_reference=ground,
    )
    return PySCFTDASOC(
        np.r_[0., se, np.repeat(te, 3)], operators.soc, np.moveaxis(operators.dipole, 0, -1), si, ti,
    )


def run_pyscf_tda_soc(
    mf, *, n_singlets: int = 12, n_triplets: int = 12, response_tolerance: float = 1e-7,
) -> PySCFTDASOC:
    """Solve both restricted DFT/TDA manifolds and assemble their state matrices.

    The reference must already be converged at the desired geometry.
    Increase the manifold sizes to check convergence of the mixed rate.
    The response tolerance bounds the TDA residual norm in Hartree.
    """
    occupied, virtual = _validate_reference(mf)
    max_roots = occupied.shape[1] * virtual.shape[1]
    if any(isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1
           for count in (n_singlets, n_triplets)):
        raise ValueError("n_singlets and n_triplets must be positive integers")
    if (isinstance(response_tolerance, (bool, np.bool_)) or not isinstance(response_tolerance, Real)
            or not isfinite(response_tolerance) or response_tolerance <= 0.):
        raise ValueError("response_tolerance must be finite and positive")
    solvers = []
    for singlet, count in ((True, n_singlets), (False, n_triplets)):
        td = mf.TDA()
        td.singlet = singlet
        td.positive_eig_threshold = 0.
        td.conv_tol = float(response_tolerance)
        # Davidson uses lindep as a squared correction-vector cutoff. Keep
        # its norm scale below the residual target so useful updates survive.
        td.lindep = min(td.lindep, (td.conv_tol / 10)**2)
        count = min(count, max_roots)
        td.kernel(nstates=count)
        if td.xy is None or len(td.xy) != count:
            raise ValueError("PySCF returned fewer TDA roots than requested")
        solvers.append(td)
    return assemble_pyscf_tda_soc(*solvers)
