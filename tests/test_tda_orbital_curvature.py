from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.cis_integrals import IntegralDerivativeEngine
from pyscf_esd.tda_orbital_curvature import (
    OrbitalResponse,
    fixed_amplitude_fock_density,
    tda_first_action,
    tda_mixed_contraction,
    tda_orbital_gradient,
    tda_transition_density,
    tda_transition_density_derivative,
)


def _multiply(a, b):
    """Product of exact coefficients in 1, x, y, xy."""
    return (
        a[0] @ b[0],
        a[1] @ b[0] + a[0] @ b[1],
        a[2] @ b[0] + a[0] @ b[2],
        a[3] @ b[0] + a[1] @ b[2] + a[2] @ b[1] + a[0] @ b[3],
    )


def _inverse(a):
    inverse = np.linalg.inv(a[0])
    x, y = -inverse @ a[1] @ inverse, -inverse @ a[2] @ inverse
    xy = -inverse @ (a[3] @ inverse + a[1] @ y + a[2] @ x)
    return inverse, x, y, xy


def _transpose(a):
    return tuple(value.T for value in a)


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-11)


def test_exact_stationary_orbital_and_tda_curvature():
    rng = np.random.default_rng(92)
    nocc, nvir = 2, 3
    nmo = nocc + nvir

    def random_matrix():
        return 0.13 * rng.normal(size=(nmo, nmo))

    def symmetric(array):
        return 0.5 * (array + array.T)

    def moving_frame(array):
        array[:nocc, :nocc] = symmetric(array[:nocc, :nocc])
        array[nocc:, nocc:] = symmetric(array[nocc:, nocc:])
        return array

    def block_diagonal(array):
        array[:nocc, nocc:] = 0
        array[nocc:, :nocc] = 0
        return symmetric(array)

    c0 = np.eye(nmo) + random_matrix()
    c = (c0, *(c0 @ moving_frame(random_matrix()) for _ in range(3)))
    c_inverse = _inverse(c)
    overlap = _multiply(_transpose(c_inverse), c_inverse)
    fock_mo = (np.diag([-0.7, -0.4, 0.1, 0.3, 0.5]),
               *(block_diagonal(random_matrix()) for _ in range(3)))
    fock = _multiply(_multiply(_transpose(c_inverse), fock_mo), c_inverse)
    occupied, virtual = tuple(m[:, :nocc] for m in c), tuple(m[:, nocc:] for m in c)
    density = tuple(2 * m for m in _multiply(occupied, _transpose(occupied)))

    # Define a density-dependent ground Fock operator that equals the target
    # Fock on the specified stationary orbitals. Its linear response along
    # that stationary path is K[D] = B D B. The quadratic off-path term adds
    # a third energy derivative, so the orbital adjoint exercises both kinds
    # of density dependence in the TDA Rayleigh quotient.
    b = tuple(symmetric(random_matrix()) for _ in range(4))
    u = symmetric(random_matrix())
    eta = 1.2

    def ground_response(matrix):
        return b[0] @ matrix @ b[0]

    response = OrbitalResponse(c0, nocc, fock[0], ground_response, atol=1e-12, rtol=1e-11)
    first_x = response.first(overlap[1], fock[1] - ground_response(density[1]))
    first_y = response.first(overlap[2], fock[2] - ground_response(density[2]))
    _assert_close(first_x.coefficients, c[1])
    _assert_close(first_y.coefficients, c[2])
    _assert_close(first_x.density, density[1])
    _assert_close(first_y.fock_mo, fock_mo[2])

    mixed_fock_source = fock[3] - ground_response(density[3])
    mixed = response.mixed(overlap[3], first_x, first_y, mixed_fock_source)
    mixed_rotation, residual = response.solve(-mixed.stationarity_source)
    assert residual < 1e-11
    rotation_full = np.zeros((nmo, nmo))
    rotation_full[nocc:, :nocc] = mixed_rotation
    rotation_full[:nocc, nocc:] = -mixed_rotation.T
    _assert_close(mixed.coefficients + c0 @ rotation_full, c[3])

    q = rng.normal(size=(nocc, nvir))
    q /= np.linalg.norm(q)
    transition = _multiply(tuple(m @ q.T for m in virtual), _transpose(occupied))
    potential = _multiply(_multiply(b, transition), b)
    projection = _multiply(_multiply(_transpose(virtual), potential), occupied)
    action = tuple(q @ fi[nocc:, nocc:] - fi[:nocc, :nocc] @ q + 2 * ki.T
                   for fi, ki in zip(fock_mo, projection))
    t, tx, ty, _ = transition
    _assert_close(tda_transition_density(c0, q), t)
    _assert_close(tda_transition_density_derivative(response, q, first_x), tx)
    _assert_close(tda_transition_density_derivative(response, q, first_y), ty)
    _assert_close(tda_first_action(response, q, first_x, potential[0], potential[1]), action[1])
    _assert_close(tda_first_action(response, q, first_y, potential[0], potential[2]), action[2])

    potential_x = potential[1] - ground_response(tx)
    potential_y = potential[2] - ground_response(ty)
    zero = np.zeros_like(t)
    fixed_t_potential = _multiply(_multiply(b, (t, zero, zero, zero)), b)[3]
    fixed_t_potential += eta * np.vdot(u, mixed.density - density[3]) * np.vdot(u, t) * u
    kernel_density_potential = 2 * eta * np.vdot(u, t)**2 * u
    source = tda_orbital_gradient(response, q, potential[0], kernel_density_potential)
    adjoint, _ = response.solve(source)

    def contraction(orbitals):
        return tda_mixed_contraction(
            response, q, orbitals, first_x, first_y, tx, ty, potential[0], potential[1], potential[2],
            potential_x, potential_y, np.vdot(t, fixed_t_potential),
        ) - np.vdot(adjoint, orbitals.stationarity_source)

    expected = np.vdot(q, action[3])
    _assert_close(contraction(mixed), expected)

    # Contract the entire mixed Fock derivative in AO space. This route
    # avoids computing an AO Fock matrix for each nuclear pair.
    geometry = response.mixed(overlap[3], first_x, first_y, include_density_response=False)
    relaxed = fixed_amplitude_fock_density(response, q, adjoint)
    contract_first = (np.vdot(relaxed, mixed_fock_source)
                      + np.vdot(ground_response(relaxed), geometry.density))
    _assert_close(contraction(geometry) + contract_first, expected)


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required")
def test_overlap_uses_analytic_nuclear_derivatives():
    from pyscf import gto

    positions = np.asarray([[0.1, -0.3, 0.4], [-0.4, 0.7, -1.1]])
    exponent = 0.71
    mol = gto.M(atom=[("H", point) for point in positions], unit="Bohr",
                basis={"H": [[0, [exponent, 1.0]]]}, verbose=0)
    x = np.asarray([[0.3, 1.1, 0.0], [0.5, -0.1, -0.3]])
    y = np.asarray([[1.2, 0.0, 0.6], [-0.3, 0.2, -1.0]])
    delta = positions[0] - positions[1]
    dx, dy = x[0] - x[1], y[0] - y[1]
    overlap = np.exp(-0.5 * exponent * np.dot(delta, delta))
    first = -exponent * np.dot(delta, dx) * overlap
    mixed = (exponent**2 * np.dot(delta, dx) * np.dot(delta, dy) - exponent * np.dot(dx, dy)) * overlap
    op = IntegralDerivativeEngine(mol)
    _assert_close(op.first_one_electron(x).overlap, [[0, first], [first, 0]])
    _assert_close(op.second_one_electron(x, y).overlap, [[0, mixed], [mixed, 0]])
    _assert_close(op.second_one_electron(y, x).overlap, op.second_one_electron(x, y).overlap)
    _assert_close(op.second_one_electron(np.broadcast_to(x[0], x.shape), y).overlap, np.zeros((2, 2)))


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required")
def test_native_df_rks_tda_action_and_occupied_cphf():
    from pyscf import dft, gto, solvent

    mol = gto.M(atom="O 0 0 0; H 0 .75 .55; H 0 -.75 .55", basis="sto-3g", verbose=0)
    mf = solvent.PCM(dft.RKS(mol, xc="PBE0").density_fit())
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.equilibrium_solvation = True
    mf.with_solvent.lebedev_order = 7
    mf.conv_tol = 1e-11
    mf.conv_tol_cpscf = 1e-12
    mf.kernel()
    assert mf.converged
    td = mf.TDA(equilibrium_solvation=True)
    native_action, _ = td.gen_vind()
    coeff = mf.mo_coeff
    nocc = int(np.count_nonzero(mf.mo_occ == 2))
    occupied, virtual = coeff[:, :nocc], coeff[:, nocc:]
    q = np.random.default_rng(84).normal(size=(occupied.shape[1], virtual.shape[1]))
    transition = tda_transition_density(coeff, q)
    potential = mf.gen_response(hermi=0)(transition)
    gap = mf.mo_energy[nocc:][None, :] - mf.mo_energy[:nocc][:, None]
    _assert_close(2 * occupied.T @ potential.T @ virtual + q * gap, native_action([q]).reshape(q.shape))

    hessian = mf.Hessian()
    hessian.auxbasis_response = 2
    h1 = hessian.make_h1(coeff, mf.mo_occ)
    mo1, _ = hessian.solve_mo1(mf.mo_energy, coeff, mf.mo_occ, h1)
    metric_coeff = mf.get_ovlp() @ coeff
    canonical_fock = (metric_coeff * mf.mo_energy) @ metric_coeff.T
    response = OrbitalResponse(coeff, nocc, canonical_fock, mf.gen_response(hermi=1), atol=1e-12)
    integrals = IntegralDerivativeEngine(mol)
    for atom, axis in ((0, 1), (1, 2)):
        direction = np.zeros((mol.natm, 3))
        direction[atom, axis] = 1
        first = response.first(integrals.first_one_electron(direction).overlap, h1[atom][axis])
        _assert_close(first.coefficients[:, :nocc], mo1[atom][axis])


def load_tests(loader, tests, pattern):
    for check in (test_exact_stationary_orbital_and_tda_curvature, test_overlap_uses_analytic_nuclear_derivatives,
                  test_native_df_rks_tda_action_and_occupied_cphf):
        tests.addTest(unittest.FunctionTestCase(check))
    return tests
