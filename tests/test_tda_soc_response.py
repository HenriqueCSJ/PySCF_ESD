"""Check full-space borrowing and its analytic derivative independently."""

import unittest

import numpy as np

from pyscf_esd.tda_soc_response import (
    _derivative_adjoint, _dipole_derivatives, _full_space_moments, _moment_derivative, _sources,
)
from pyscf_esd.tda_state_interaction import assemble_tda_state_interaction
from pyscf_esd import thermal_harmonic_spectrum


class TDASOCResponseTests(unittest.TestCase):
    def test_dipole_nuclear_translation_has_the_electron_charge_sign(self):
        try:
            from pyscf import gto
        except ImportError:
            self.skipTest("requires PySCF")
        mol = gto.M(atom="He 0 0 0; He .4 .7 1.2", basis="cc-pvdz", verbose=0)
        with mol.with_common_origin(np.zeros(3)):
            dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
        directions = np.broadcast_to(np.eye(3)[:, None, :], (3, mol.natm, 3))
        actual = _dipole_derivatives(mol, np.eye(mol.nao), np.zeros((3, mol.nao, mol.nao)), dipole_ao, directions)
        expected = -np.eye(3)[:, :, None, None] * mol.intor_symmetric("int1e_ovlp")
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)

    def test_borrowing_matches_explicit_states_and_derivative_matches_direct_response(self):
        rng = np.random.default_rng(72)
        symmetric = lambda h: (h + h.swapaxes(-1, -2)) / 2
        antisymmetric = lambda h: (h - h.swapaxes(-1, -2)) / 2
        s, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        t, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        es, et = np.array([.20, .31, .42, .55]), np.array([.10, .28, .40, .47])
        a_s, a_t = (s * es) @ s.T, (t * et) @ t.T
        q, energy = t[:, 0].reshape(2, 2), et[0]
        dipole = symmetric(rng.normal(size=(3, 4, 4)))
        somf = antisymmetric(rng.normal(size=(3, 4, 4)))
        operators = (lambda x: x @ a_s, a_s.diagonal(), lambda x: x @ a_t, a_t.diagonal())
        ex, ground, _, x, y = _full_space_moments(energy, q, dipole, somf, *operators, 1e-12, 20)

        matrices = assemble_tda_state_interaction(
            s.T.reshape(4, 2, 2), t.T.reshape(4, 2, 2), dipole, 1j * somf,
        )
        target = np.arange(5, 8)
        states = np.r_[0., es, np.repeat(et, 3)]
        ds = states[target] - states[:, None]
        dg = -states
        mixed_target = np.divide(matrices.soc[:, target], ds, where=ds != 0.,
                                 out=np.zeros_like(matrices.soc[:, target]))
        mixed_ground = np.divide(matrices.soc[0], dg, where=dg != 0., out=np.zeros_like(matrices.soc[0]))
        expected = mixed_target.T @ matrices.dipole[:, 0, :].T
        expected += np.einsum("i,kir->rk", mixed_ground, matrices.dipole[:, :, target])
        np.testing.assert_allclose(ex + ground, expected, atol=3e-13, rtol=1e-12)

        da_s, da_t = symmetric(rng.normal(size=(2, 4, 4))), symmetric(rng.normal(size=(2, 4, 4)))
        dmu = symmetric(rng.normal(size=(2, 3, 4, 4)))
        dh = antisymmetric(rng.normal(size=(2, 3, 4, 4)))
        w, v, u, _ = _derivative_adjoint(energy, q, dipole, somf, x, y, *operators, 1e-12, 20)
        actual = _moment_derivative(
            q, x, y, w, v, dmu, dh,
            w @ da_s @ x.T, y @ da_t @ v.T, np.einsum("ijv,dv->dij", u, da_t @ q.ravel()),
            q.ravel() @ da_t @ q.ravel(),
        )
        direct = []
        _, _, b, optical = _sources(q, dipole, somf)
        for dp, hp, asp, atp in zip(dmu, dh, da_s, da_t, strict=True):
            source = atp @ q.ravel()
            energy_prime = q.ravel() @ source
            qprime = -(t[:, 1:] @ ((t[:, 1:].T @ source) / (et[1:] - energy))).reshape(q.shape)
            dipole_prime, soc_ground_prime, b_fixed, optical_fixed = _sources(q, dp, hp)
            _, _, b_root, optical_root = _sources(qprime, dipole, somf)
            dx = np.linalg.solve(a_s - energy * np.eye(4), (dipole_prime - x @ asp + energy_prime * x).T).T
            dy = np.linalg.solve(a_t, (soc_ground_prime - y @ atp).T).T
            direct.append(-1j * ((b_fixed + b_root) @ x.T + b @ dx.T
                                + dy @ optical.T + y @ (optical_fixed + optical_root).T))
        np.testing.assert_allclose(actual, direct, atol=2e-12, rtol=3e-12)

    def test_complex_harmonic_ht_is_coherent_under_global_phase(self):
        options = dict(gap_00_hartree=-.09, temperature_kelvin=77., process="emission",
                       fwhm_cm_inverse=100., step_cm_inverse=8., shift_au=[8.])
        mu = np.array([.1 + .2j, 0., -.05j])
        slopes = np.array([[.02j, .01, -.015 + .001j]])
        call = lambda d, l: thermal_harmonic_spectrum(
            [.006], [.004], transition_dipole_au=d, dipole_derivatives_au=l, **options,
        )
        before = call(mu, slopes)
        phase = np.exp(.63j)
        after = call(phase * mu, phase * slopes)
        np.testing.assert_allclose(before.wavenumber_cm_inverse, after.wavenumber_cm_inverse)
        np.testing.assert_allclose(before.photon_weighted_density_per_cm_inverse,
                                   after.photon_weighted_density_per_cm_inverse, atol=1e-20, rtol=1e-10)
