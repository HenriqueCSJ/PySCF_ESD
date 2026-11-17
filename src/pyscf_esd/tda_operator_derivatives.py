"""Analytic fixed-density contribution of the C-PCM transition kernel."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CPCMTransitionKernelDerivatives:
    """Reaction-field bilinear value and directional nuclear derivatives."""

    value: float
    first_x: float
    first_y: float | None
    mixed_xy: float | None


@dataclass(frozen=True)
class CPCMTransitionKernelAction:
    """Reaction-field AO potential and directional nuclear derivatives."""

    value: NDArray[np.float64]
    first_x: NDArray[np.float64]
    first_y: NDArray[np.float64] | None
    mixed_xy: NDArray[np.float64] | None


def _real_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return value.astype(float, copy=False)


def _surface_diagonal(surface, owners, x, y):
    # The diagonal Coulomb element is proportional to the reciprocal of
    # the product of SWIG switching factors. Differentiate its logarithm.
    relative = surface["grid_coords"][:, None, :] - surface["atom_coords"][None, :, :]
    radius = np.linalg.norm(relative, axis=-1)
    width = surface["R_sw_J"]
    t = (radius - surface["R_in_J"]) / width
    active = (t > 0) & (t < 1)
    active[np.arange(owners.size), owners] = False
    t = np.where(active, t, 1.0)
    switch = t**3 * (10 - 15*t + 6*t*t)
    switch_log_deriv = 30*t*t * (1-t)**2 / switch
    unit = np.divide(relative, radius[..., None], out=np.zeros_like(relative), where=active[..., None])

    relative_x = x[owners, None, :] - x[None, :, :]
    radius_x = np.einsum("gac,gac->ga", unit, relative_x)
    log_x = np.sum(switch_log_deriv * radius_x / width, axis=1)
    if y is None:
        return -log_x, None, None

    relative_y = y[owners, None, :] - y[None, :, :]
    radius_y = np.einsum("gac,gac->ga", unit, relative_y)
    log_y = np.sum(switch_log_deriv * radius_y / width, axis=1)
    cross = np.einsum("gac,gac->ga", relative_x, relative_y) - radius_x * radius_y
    radius_xy = np.divide(cross, radius, out=np.zeros_like(radius), where=active)
    switch_log_curvature = 60*t*(1-t)*(1-2*t) / switch - switch_log_deriv**2
    log_xy = np.sum(
        switch_log_curvature * radius_x * radius_y / width**2 + switch_log_deriv * radius_xy / width,
        axis=1,
    )
    return -log_x, -log_y, log_x * log_y - log_xy


def _surface_actions(surface, owners, cavity, x, y, charges):
    """Apply first and mixed Coulomb-matrix derivatives in surface row blocks."""
    from scipy.special import erf, hyp1f1

    diagonal = _surface_diagonal(surface, owners, x, y)
    coords = surface["grid_coords"]
    exponent = surface["charge_exp"]
    grid_x = x[owners]
    grid_y = None if y is None else y[owners]
    actions = [np.empty_like(charges) for _ in range(1 if y is None else 3)]
    ngrid = owners.size
    for first in range(0, ngrid, 64):
        rows = slice(first, first + 64)
        i = np.arange(ngrid)[rows]
        j = np.arange(i.size)
        delta = coords[rows, None, :] - coords[None, :, :]
        dx = grid_x[rows, None, :] - grid_x[None, :, :]
        r2 = np.einsum("ijc,ijc->ij", delta, delta)
        xi = exponent[rows, None] * exponent[None, :] / np.hypot(exponent[rows, None], exponent[None, :])
        z2 = xi**2 * r2
        # The hypergeometric limits remove cancellation for overlapping
        # Gaussian centers. The erf form is cheaper elsewhere.
        small = z2 < 0.01
        safe_r2 = np.where(small, 1.0, r2)
        prefactor = 2*xi / np.sqrt(np.pi)
        gaussian = prefactor * np.exp(-z2)
        radial_first = (gaussian - erf(np.sqrt(z2)) / np.sqrt(safe_r2)) / safe_r2
        radial_first[small] = (-2/3 * prefactor[small] * xi[small]**2
                               * hyp1f1(1.5, 2.5, -z2[small]))
        projection_x = np.einsum("ijc,ijc->ij", delta, dx)
        ds_x = radial_first * projection_x
        ds_x[j, i] = cavity[i, i] * diagonal[0][rows]
        actions[0][rows] = ds_x @ charges
        if y is not None:
            dy = grid_y[rows, None, :] - grid_y[None, :, :]
            projection_y = np.einsum("ijc,ijc->ij", delta, dy)
            ds_y = radial_first * projection_y
            ds_y[j, i] = cavity[i, i] * diagonal[1][rows]
            actions[1][rows] = ds_y @ charges
            radial_mixed = (-2*xi**2 * gaussian - 3*radial_first) / safe_r2
            radial_mixed[small] = (4/5 * prefactor[small] * xi[small]**4
                                   * hyp1f1(2.5, 3.5, -z2[small]))
            ds_xy = (radial_first * np.einsum("ijc,ijc->ij", dx, dy)
                     + radial_mixed * projection_x * projection_y)
            ds_xy[j, i] = cavity[i, i] * diagonal[2][rows]
            actions[2][rows] = ds_xy @ charges
    return actions


def _surface_integrals(pcm, owners, x, y):
    from pyscf import df, gto, lib

    mol = pcm.mol
    ao_bounds = mol.aoslice_by_atom()[:, 2:]
    ao_owners = np.repeat(np.arange(mol.natm), ao_bounds[:, 1] - ao_bounds[:, 0])
    surface = pcm.surface
    # Relative displacements use translation invariance of the three-center
    # Coulomb integral to put each charge center at rest for its derivative.
    # A nuclear-translation derivative on the first AO is minus int3c2e_ip1.
    available = pcm.max_memory - lib.current_memory()[0]
    block_size = max(1, min(64, int(available*0.4e6 / (8*mol.nao**2*9))))
    derivatives = ["int3c2e", "int3c2e_ip1"]
    if y is not None:
        derivatives += ["int3c2e_ipip1", "int3c2e_ipvip1"]
    kernels = [mol._add_suffix(name) for name in derivatives]
    optimizers = [gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, name) for name in kernels]
    for first in range(0, owners.size, block_size):
        rows = slice(first, first + block_size)
        block_owners = owners[rows]
        fake = gto.fakemol_for_charges(surface["grid_coords"][rows], expnt=surface["charge_exp"][rows]**2)
        fake.cart = mol.cart

        def integral(index):
            return df.incore.aux_e2(mol, fake, intor=kernels[index], aosym="s1", cintopt=optimizers[index])

        dx = x[ao_owners, None, :] - x[None, block_owners, :]
        first_integral = integral(1)
        derivative_x = -np.einsum("cijg,igc->ijg", first_integral, dx)
        block = [integral(0), derivative_x + derivative_x.transpose(1, 0, 2)]
        if y is not None:
            dy = y[ao_owners, None, :] - y[None, block_owners, :]
            derivative_y = -np.einsum("cijg,igc->ijg", first_integral, dy)
            block.append(derivative_y + derivative_y.transpose(1, 0, 2))
        del first_integral
        if y is not None:
            same_ao = integral(2).reshape(3, 3, mol.nao, mol.nao, -1)
            derivative_xy = np.einsum("cdijg,igc,igd->ijg", same_ao, dx, dy)
            del same_ao
            different_aos = integral(3).reshape(3, 3, mol.nao, mol.nao, -1)
            derivative_xy += np.einsum("cdijg,igc,jgd->ijg", different_aos, dx, dy)
            del different_aos
            block.append(derivative_xy + derivative_xy.transpose(1, 0, 2))
        yield rows, block


def _electronic_surface_potentials(pcm, owners, densities, x, y):
    potentials = [np.empty((owners.size, densities.shape[0])) for _ in range(2 if y is None else 4)]
    for rows, integrals in _surface_integrals(pcm, owners, x, y):
        for potential, integral in zip(potentials, integrals):
            potential[rows] = np.einsum("ijg,mij->gm", integral, densities, optimize=True)
    return potentials


def _nuclear_surface_potentials(pcm, owners, x, y):
    """Point nuclei against moving Gaussian surface centers, in Bohr."""
    from scipy.special import erf, hyp1f1

    surface = pcm.surface
    delta = surface["grid_coords"][:, None, :] - surface["atom_coords"][None, :, :]
    r2 = np.einsum("gac,gac->ga", delta, delta)
    exponent = surface["charge_exp"][:, None]
    dx = x[owners, None, :] - x[None, :, :]
    projection_x = np.einsum("gac,gac->ga", delta, dx)
    safe_r2 = np.where(r2 == 0, 1, r2)
    gauss = 2 * exponent / np.sqrt(np.pi) * np.exp(-exponent**2 * r2)
    radial = erf(exponent * np.sqrt(r2)) / np.sqrt(safe_r2)
    first = (gauss - radial) / safe_r2
    mixed = (-2 * exponent**2 * gauss - 3 * first) / safe_r2
    small = exponent**2 * r2 < 0.01
    if np.any(small):
        # The integral representation avoids cancellation close to a nucleus.
        xi = np.broadcast_to(exponent, r2.shape)[small]
        z2 = -(xi**2) * r2[small]
        prefactor = 2 * xi / np.sqrt(np.pi)
        first[small] = -2/3 * prefactor * xi**2 * hyp1f1(1.5, 2.5, z2)
        mixed[small] = 4/5 * prefactor * xi**4 * hyp1f1(2.5, 3.5, z2)
    charge = pcm.mol.atom_charges()
    potentials = [pcm.v_grids_n[:, None], ((first * projection_x) @ charge)[:, None]]
    if y is not None:
        dy = y[owners, None, :] - y[None, :, :]
        projection_y = np.einsum("gac,gac->ga", delta, dy)
        relative = np.einsum("gac,gac->ga", dx, dy)
        potentials += [((first * projection_y) @ charge)[:, None],
                       ((first * relative + mixed * projection_x * projection_y) @ charge)[:, None]]
    return potentials


def _prepare_cavity(pcm):
    from pyscf.solvent.pcm import PCM
    from scipy.linalg import cho_factor

    if type(pcm) is not PCM or pcm.method.upper() not in {"C-PCM", "CPCM"}:
        raise TypeError("a native CPU PySCF C-PCM object is required")
    if pcm.frozen or pcm.surface_discretization_method.upper() != "SWIG":
        raise ValueError("an unfrozen SWIG cavity is required")
    if not pcm._intermediates:
        pcm.build()
    surface = pcm.surface
    if not np.array_equal(surface["atom_coords"], pcm.mol.atom_coords(unit="Bohr")):
        raise ValueError("C-PCM surface must belong to the supplied molecule geometry")
    owners = np.repeat(np.arange(pcm.mol.natm), np.diff(surface["gslice_by_atom"], axis=1).ravel())
    cavity = pcm._intermediates["K"]
    f = pcm._intermediates["f_epsilon"]
    if not np.isfinite(f) or not np.isclose(f, 1 - 1/pcm.eps, rtol=1e-14, atol=1e-14):
        raise ValueError("C-PCM cavity was built with a different dielectric constant")
    return owners, cho_factor(cavity, check_finite=True), f


def cpcm_transition_kernel_bilinear(pcm, left_density, right_density, direction_x, direction_y=None):
    """Differentiate ``Tr(D_left B[D_right])`` at fixed AO density matrices.

    ``pcm`` is the CPU PySCF C-PCM object containing the requested dielectric
    and SWIG cavity. It is built at its input geometry if needed. Directions
    have shape ``(natm, 3)`` and use Bohr. The calculation differentiates the
    retained surface points locally, with moving Gaussian centers and SWIG
    weights, and takes the first and mixed derivatives of the cavity solve.
    The PCM response is insensitive to antisymmetric parts of real densities.

    Orbital and density response, the C-PCM contribution to the SCF Fock
    operator, and gas-phase terms must be added when differentiating the full
    TDA operator. This bilinear does not include the TDA spin prefactor.
    """
    from scipy.linalg import cho_solve

    owners, factor, f = _prepare_cavity(pcm)
    mol = pcm.mol
    x = _real_array(direction_x, (mol.natm, 3), "direction_x")
    y = None if direction_y is None else _real_array(direction_y, x.shape, "direction_y")
    densities = np.stack([
        _real_array(left_density, (mol.nao, mol.nao), "left_density"),
        _real_array(right_density, (mol.nao, mol.nao), "right_density"),
    ])
    b, b_x, *other_potentials = _electronic_surface_potentials(pcm, owners, densities, x, y)
    charges = cho_solve(factor, b)
    p, q = charges.T
    value = -f * (b[:, 0] @ q)
    k_x, *other_actions = _surface_actions(pcm.surface, owners, pcm._intermediates["K"], x, y, charges)
    residual_x = b_x - k_x
    first_x = -f * (b_x[:, 0] @ q + p @ residual_x[:, 1])
    if y is None:
        return CPCMTransitionKernelDerivatives(float(value), float(first_x), None, None)

    b_y, b_xy = other_potentials
    k_y, k_xy = other_actions
    residual_y = b_y - k_y
    first_y = -f * (b_y[:, 0] @ q + p @ residual_y[:, 1])
    q_x = cho_solve(factor, residual_x[:, 1])
    q_y = cho_solve(factor, residual_y[:, 1])
    mixed_xy = -f * (b_xy[:, 0] @ q + p @ (b_xy[:, 1] - k_xy[:, 1])
                    + residual_x[:, 0] @ q_y + residual_y[:, 0] @ q_x)
    return CPCMTransitionKernelDerivatives(float(value), float(first_x), float(first_y), float(mixed_xy))


def cpcm_transition_kernel_action(pcm, density, direction_x, direction_y=None):
    """Apply the C-PCM transition response and its nuclear derivatives.

    Return AO matrices for ``B[D]``, ``B_x[D]`` and, when two directions are
    supplied, ``B_y[D]`` and ``B_xy[D]``, holding the real AO density ``D``
    fixed. The CPU PySCF C-PCM object, directions, and model scope follow
    :func:`cpcm_transition_kernel_bilinear`.
    """
    return _cpcm_action(pcm, density, direction_x, direction_y, include_nuclei=False)


def cpcm_fock_potential_derivatives(pcm, density, direction_x, direction_y=None):
    """Differentiate the C-PCM ground-state Fock potential at fixed AO density.

    Includes motion of the nuclear point charges, AO centers, Gaussian
    surface charges and switching weights in the equilibrium SWIG cavity.
    The density is the spin-summed SCF density. Add the transition-kernel
    action on each density derivative to obtain the relaxed Fock potential.
    """
    return _cpcm_action(pcm, density, direction_x, direction_y, include_nuclei=True)


def _cpcm_action(pcm, density, direction_x, direction_y, *, include_nuclei):
    from scipy.linalg import cho_solve

    owners, factor, f = _prepare_cavity(pcm)
    mol = pcm.mol
    x = _real_array(direction_x, (mol.natm, 3), "direction_x")
    y = None if direction_y is None else _real_array(direction_y, x.shape, "direction_y")
    density = _real_array(density, (mol.nao, mol.nao), "density")
    potentials = _electronic_surface_potentials(pcm, owners, density[None], x, y)
    if include_nuclei:
        potentials = [e - n for e, n in zip(potentials, _nuclear_surface_potentials(pcm, owners, x, y))]
    b, b_x, *other_potentials = potentials
    q = cho_solve(factor, b)
    surface, cavity = pcm.surface, pcm._intermediates["K"]
    k_x, *other_actions = _surface_actions(surface, owners, cavity, x, y, q)
    q_x = cho_solve(factor, b_x - k_x)
    value = np.zeros((mol.nao, mol.nao))
    first_x = np.zeros_like(value)
    first_y = mixed_xy = None
    if y is not None:
        b_y, b_xy = other_potentials
        k_y, k_xy = other_actions
        q_y = cho_solve(factor, b_y - k_y)
        k_x_q_y, k_y_q_x, _ = _surface_actions(surface, owners, cavity, x, y, np.hstack((q_y, q_x)))
        q_xy = cho_solve(factor, b_xy - k_xy - k_x_q_y[:, :1] - k_y_q_x[:, 1:])
        first_y = np.zeros_like(value)
        mixed_xy = np.zeros_like(value)

    for rows, integrals in _surface_integrals(pcm, owners, x, y):
        integral, integral_x, *other_integrals = integrals

        def project(kernel, charge):
            return np.einsum("ijg,g->ij", kernel, charge[rows, 0], optimize=True)

        value += project(integral, q)
        first_x += project(integral_x, q) + project(integral, q_x)
        if y is not None:
            integral_y, integral_xy = other_integrals
            first_y += project(integral_y, q) + project(integral, q_y)
            mixed_xy += (project(integral_xy, q) + project(integral_x, q_y)
                         + project(integral_y, q_x) + project(integral, q_xy))
    if y is not None:
        first_y *= -f
        mixed_xy *= -f
    return CPCMTransitionKernelAction(-f * value, -f * first_x, first_y, mixed_xy)
