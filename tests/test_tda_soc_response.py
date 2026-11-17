"""Check full-space borrowing and its analytic derivative independently."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyscf_esd.tda_soc_response import (
    _derivative_adjoint, _dipole_derivatives, _full_space_moments, _moment_derivative,
    _qualify_soc_response, _sources, tda_soc_response,
)
from pyscf_esd.tda_state_interaction import assemble_tda_state_interaction
from pyscf_esd import thermal_harmonic_spectrum


class TDASOCResponseTests(unittest.TestCase):
    def require_davidson(self):
        try:
            from pyscf import lib
        except ImportError:
            self.skipTest("matrix-free spectral qualification requires PySCF")

    @staticmethod
    def response_inputs(shifted_singlet, triplet):
        n = len(triplet)
        energy = .10
        q = np.eye(1, n)
        dipole = np.zeros((3, n + 1, n + 1))
        dipole[0, 0, 1] = dipole[0, 1, 0] = 1.
        somf = np.zeros_like(dipole)
        somf[0, 1, n] = 1.
        somf[0, n, 1] = -1.
        singlet = shifted_singlet + energy * np.eye(n)
        return (None, None, energy, q, dipole, somf,
                lambda vectors: vectors @ singlet, singlet.diagonal(),
                lambda vectors: vectors @ triplet, triplet.diagonal())

    def test_dark_singlet_singularity_is_rejected_even_with_zero_optical_residual(self):
        shifted_matrix = np.diag(np.linspace(.20, .82, 32))
        # The dark vector also lives outside the coordinate with the smallest
        # supplied diagonal, so a diagonal-only probe cannot find it.
        shifted_matrix[-2:, -2:] = .5
        for shifted in (np.diag([.20, 0.]), shifted_matrix):
            with self.subTest(dimension=len(shifted)):
                if shifted is shifted_matrix:
                    self.require_davidson()
                inputs = self.response_inputs(shifted, np.diag(np.linspace(.10, .40, len(shifted))))
                _, _, residuals, _, _ = _full_space_moments(*inputs[2:], 1e-8, 128)
                self.assertEqual(residuals, (0., 0.))
                with patch("pyscf_esd.tda_soc_response._response_inputs", return_value=inputs):
                    with self.assertRaisesRegex(ValueError, "lowest shifted singlet TDA eigenvalue"):
                        tda_soc_response(None)

    def test_triplet_eigenpair_is_verified_in_both_paths(self):
        inputs = self.response_inputs(np.diag([.20, .30]), np.diag([.12, .30]))
        target = SimpleNamespace(mol=SimpleNamespace(natm=1))
        with patch("pyscf_esd.tda_soc_response._response_inputs", return_value=inputs):
            for directions in (None, np.zeros((1, 1, 3))):
                with self.subTest(derivatives=directions is not None):
                    with self.assertRaisesRegex(ValueError, "triplet eigenpair residual"):
                        tda_soc_response(target, directions=directions)

    def test_triplet_ground_resolvent_and_analytic_root_gap_are_independent(self):
        shifted = np.diag([.20, .30])
        unstable = self.response_inputs(shifted, np.diag([.10, -.01]))
        target = SimpleNamespace(mol=SimpleNamespace(natm=1))
        with patch("pyscf_esd.tda_soc_response._response_inputs", return_value=unstable):
            with self.assertRaisesRegex(ValueError, "lowest triplet TDA eigenvalue"):
                tda_soc_response(target)
        for other_energy in (.08, .10):
            with self.subTest(other_energy=other_energy):
                unresolved = self.response_inputs(shifted, np.diag([.10, other_energy]))
                with patch("pyscf_esd.tda_soc_response._response_inputs", return_value=unresolved):
                    with self.assertRaisesRegex(ValueError, "lowest projected triplet TDA eigenvalue"):
                        tda_soc_response(target, directions=np.zeros((1, 1, 3)))

    def test_matrix_free_qualification_accepts_positive_separated_operators(self):
        self.require_davidson()
        shifted = np.diag(np.linspace(.20, .82, 32))
        shifted[-2:, -2:] = [[.5, .35], [.35, .5]]
        _, _, energy, q, _, _, sa, sd, ta, td = self.response_inputs(
            shifted, np.diag(np.linspace(.10, .40, len(shifted))),
        )
        for derivatives in (False, True):
            with self.subTest(derivatives=derivatives):
                _qualify_soc_response(energy, q, sa, sd, ta, td, derivatives=derivatives)

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
