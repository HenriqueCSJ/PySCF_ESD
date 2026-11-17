"""PySCF namespace extension for vibronic absorption and fluorescence."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

from pyscf.gto import Mole

from pyscf_esd import *  # Re-export the same public objects under the PySCF namespace.
from pyscf_esd import __all__ as _esd_exports
from pyscf_esd.cpu import Settings
from pyscf_esd.cpu_workflow import run_cpu_vibronic
from pyscf_esd.gpu import run_gpu_vibronic
from pyscf_esd.gpu_spin_flip import solve_spin_flip_tda, spin_flip_tda_gradient
from pyscf_esd.pyscf_adapter import adiabatic_hessian
from pyscf_esd.workflow import DEFAULT_SETTINGS, VibronicResult, run_vibronic

__all__ = [
    *_esd_exports, "Settings", "adiabatic_hessian", "run", "run_cpu_vibronic",
    "run_gpu_vibronic", "solve_spin_flip_tda", "spin_flip_tda_gradient",
]


def run(
    mol: Mole, *, xc: str = DEFAULT_SETTINGS.xc, target_index: int = 0, **options,
) -> VibronicResult:
    """Run ESD from a built, closed-shell PySCF molecule.

    Use one named all-electron spherical orbital basis and standard point
    nuclei. The driver uses the molecule's coordinates, basis name and charge,
    and follows the zero-based singlet root at the relaxed ground geometry.
    Other keywords are passed to :func:`run_vibronic`.
    """
    if not isinstance(mol, Mole) or not mol._built:
        raise ValueError("esd.run requires a built molecular pyscf.gto.Mole")
    if mol.spin != 0:
        raise ValueError("esd.run requires a closed-shell molecule (spin=0)")
    if not isinstance(mol.basis, str):
        raise ValueError("esd.run requires one named basis set on the PySCF Mole")
    if mol.cart or mol.symmetry or mol.has_ecp() or mol.pseudo or mol.nucmod:
        raise ValueError("esd.run requires spherical all-electron orbitals, point nuclei and symmetry=False")
    if any(charge == 0 for charge in mol.atom_charges()):
        raise ValueError("esd.run requires real atoms without ghost centers")
    if isinstance(target_index, bool) or not isinstance(target_index, Integral) or target_index < 0:
        raise ValueError("target_index must be a zero-based integer")

    settings = replace(
        DEFAULT_SETTINGS, basis=mol.basis, charge=mol.charge, xc=xc,
        nstates=max(DEFAULT_SETTINGS.nstates, target_index + 2),
    )
    symbols = tuple(mol.atom_pure_symbol(index) for index in range(mol.natm))
    return run_vibronic(symbols, mol.atom_coords(unit="Bohr"), settings, target_index=target_index, **options)
