"""Vertical optical lines from prepared spin-conserving molecular UKS/TDA roots."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .cpu import ElectronicConvergenceError, _checked_array, _readonly
from .spectral_properties import spectral_line_properties
from .unrestricted import _to_host


_Backend = Literal["pyscf", "gpu4pyscf"]
_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UnrestrictedTDAVerticalSpectrum:
    """Electronic sticks sharing the prepared reference geometry and spin projection.

    Arrays are read-only, in input root order. The signed Cartesian dipoles
    follow the native root phases. Each oscillator strength is for excitation
    from the reference and each Einstein A is for the reverse vacuum transition.
    Reference spin squared describes the UKS determinant. A root spin diagnostic
    is unavailable in the supported native spin-conserving TDA implementations.
    """

    root_indices: tuple[int, ...]
    photon_energy_hartree: _FloatArray
    transition_dipole_au: _FloatArray
    dipole_strength_au2: _FloatArray
    oscillator_strength: _FloatArray
    einstein_a_s_inverse: _FloatArray
    reference_energy_hartree: float
    reference_ms: float
    reference_spin_squared: float
    backend: _Backend
    root_spin_squared: _FloatArray | None = None


@dataclass(frozen=True)
class _VerticalLines:
    photon_energy_hartree: _FloatArray
    dipole_strength_au2: _FloatArray
    population: _FloatArray
    initial_population_coverage: float = 1.0
    final_dipole_strength_coverage: float | None = None


def unrestricted_tda_vertical_spectrum(
    td, *, roots: Iterable[int] | None = None,
) -> UnrestrictedTDAVerticalSpectrum:
    """Read native molecular UKS/TDA lines from a converged calculation.

    Supply the still-prepared PySCF or GPU4PySCF TDA solver at the geometry of
    interest. ``roots`` selects distinct zero-based native root indices and
    defaults to all stored roots. Each selected root must have converged and
    have a positive excitation energy. The unrestricted response conserves
    M_S and can mix states of different total spin. The rates cover only the
    electric-dipole decay to the reference at the same geometry.
    """
    backend = _native_backend(td)
    mf = td._scf
    if not mf.converged:
        raise ElectronicConvergenceError("the UKS reference must already be converged")
    if mf._numint._xc_type(mf.xc) == "HF":
        raise ValueError("the UKS/TDA reference requires a density functional")
    if td.xy is None or len(td.xy) == 0 or td.e is None or td.converged is None:
        raise ElectronicConvergenceError("the UKS/TDA response must already contain solved roots")

    nroots = len(td.xy)
    indices = _select_roots(roots, nroots)
    selected = list(indices)
    device = mf.mo_coeff[0].device if backend == "gpu4pyscf" else nullcontext()
    with device:
        energies = np.asarray(_to_host(td.e))
        converged = np.asarray(_to_host(td.converged))
        if energies.shape != (nroots,) or converged.shape != (nroots,) or converged.dtype.kind != "b":
            raise ElectronicConvergenceError("UKS/TDA energies and convergence flags must match the roots")
        if not np.all(converged[selected]):
            raise ElectronicConvergenceError("each selected UKS/TDA root must be converged")
        energy = _checked_array(energies[selected], (len(indices),), "UKS/TDA photon energies")
        if np.any(energy <= 0):
            raise ElectronicConvergenceError("each selected UKS/TDA root must have positive excitation energy")

        alpha, beta = (_to_host(occupation) for occupation in mf.mo_occ)
        occupations = np.asarray((alpha, beta))
        if occupations.ndim != 2 or not np.all((occupations == 0) | (occupations == 1)):
            raise ValueError("UKS/TDA requires integer alpha and beta occupations")
        n_alpha, n_beta = occupations.sum(axis=1)
        if (n_alpha, n_beta) != tuple(mf.nelec):
            raise ElectronicConvergenceError("UKS occupations disagree with the reference electron counts")
        ms = float(n_alpha - n_beta) / 2
        reference_energy = float(_checked_array(_to_host(mf.e_tot), (), "UKS reference energy"))
        spin_squared, _ = mf.spin_square()
        spin_squared = float(_checked_array(_to_host(spin_squared), (), "UKS reference spin squared"))
        dipoles = _checked_array(
            _to_host(td.transition_dipole(xy=[td.xy[index] for index in indices])),
            (len(indices), 3), "UKS/TDA length transition dipoles",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        strengths = np.einsum("ij,ij->i", dipoles, dipoles)
    # Unit weights recover the intrinsic properties of each independent line.
    lines = _VerticalLines(energy, strengths, np.ones(len(indices)))
    absorption = spectral_line_properties(lines, "absorption")
    emission = spectral_line_properties(lines, "emission")
    return UnrestrictedTDAVerticalSpectrum(
        root_indices=indices,
        photon_energy_hartree=_readonly(energy),
        transition_dipole_au=_readonly(dipoles),
        dipole_strength_au2=_readonly(strengths),
        oscillator_strength=_readonly(absorption.oscillator_strength),
        einstein_a_s_inverse=_readonly(emission.einstein_a_s_inverse),
        reference_energy_hartree=reference_energy,
        reference_ms=ms,
        reference_spin_squared=spin_squared,
        backend=backend,
    )


def _select_roots(roots: Iterable[int] | None, nroots: int) -> tuple[int, ...]:
    if roots is None:
        return tuple(range(nroots))
    indices = tuple(roots)
    if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                           or not 0 <= i < nroots for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("roots must select distinct zero-based UKS/TDA root indices")
    return tuple(int(index) for index in indices)


def _native_backend(td) -> _Backend:
    response_types = {(cls.__module__, cls.__name__) for cls in type(td).__mro__}
    mean_field = getattr(td, "_scf", None)
    reference_types = {(cls.__module__, cls.__name__) for cls in type(mean_field).__mro__}
    # Inspect ancestry to allow native wrappers without importing an optional backend.
    cpu_reference = {
        ("pyscf.dft.uks", "UKS"), ("pyscf.dft.uks_symm", "SymAdaptedUKS"),
    }
    cpu_response = "pyscf.tdscf.uks"
    if ((cpu_response, "TDA") in response_types and cpu_reference & reference_types
            and not {(cpu_response, "TDDFT"), (cpu_response, "dTDA")} & response_types):
        return "pyscf"
    if (("gpu4pyscf.tdscf.uks", "TDA") in response_types
            and ("gpu4pyscf.dft.uks", "UKS") in reference_types):
        return "gpu4pyscf"
    raise TypeError("supply a native spin-conserving molecular PySCF or GPU4PySCF UKS/TDA solver")
