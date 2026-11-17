"""Same-geometry checks of the analytic TDA transition-dipole algebra."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf_esd.tda_dipole import (
    _fixed_amplitude_dipole_derivative,
    _polarized_gradient,
    _rigid_motion_errors,
    _target_root,
)
from pyscf_esd.tda_hessian import prepare_tda_root_response


class TDADipoleTests(unittest.TestCase):
    def test_polarization_matches_explicit_eigenvector_derivative(self):
        rng = np.random.default_rng(714)
        eigenvectors, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        energies = np.array([.21, .58, .91, 1.17])
        q = eigenvectors[:, 1]
        complement = eigenvectors[:, [0, 2, 3]]
        gaps = energies[1] - energies[[0, 2, 3]]
        operator = (eigenvectors * energies) @ eigenvectors.T
        changes = rng.normal(size=(2, 3, 4, 4))
        changes = (changes + changes.swapaxes(-1, -2)) / 2
        offset = rng.normal(size=(2, 3))
        dipole = rng.normal(size=(3, 4))
        dipole[0] = q
        calls = []

        def native_gradient(x):
            calls.append(x)
            vector = np.sqrt(2) * x.ravel()
            return offset + np.einsum("u,kxuv,v->kx", vector, changes, vector)

        def calculate(phase):
            root = prepare_tda_root_response(operator, energies[1], phase * q.reshape(2, 2))
            adjoints = root.solve(dipole.reshape(3, 2, 2))
            return _polarized_gradient(native_gradient, root.amplitude, adjoints.vectors, 2)

        eigenvector_derivative = ((changes @ q @ complement) / gaps) @ complement.T
        expected = eigenvector_derivative @ dipole.T
        calculated = calculate(1)
        np.testing.assert_allclose(calculated, expected, atol=1e-13)
        np.testing.assert_allclose(calculate(-1), -calculated, atol=1e-13)
        np.testing.assert_allclose([np.vdot(x, x) for x in calls], 0.5, atol=1e-13)

    def test_fixed_root_derivative_matches_analytic_orbital_and_operator_contractions(self):
        rng = np.random.default_rng(511)
        nocc, nvir = 2, 3
        nmo = nocc + nvir
        slices = [(0, 3), (3, 5)]
        coeff = np.linalg.qr(rng.normal(size=(nmo, nmo)))[0] @ np.diag(np.arange(1, nmo + 1) ** 0.3)
        position = rng.normal(size=(3, nmo, nmo))
        position = (position + position.transpose(0, 2, 1)) / 2
        irp = rng.normal(size=(3, 3, nmo, nmo))
        ip_overlap = rng.normal(size=(3, nmo, nmo))
        q = rng.normal(size=(nocc, nvir))
        q /= np.linalg.norm(q)
        mo1 = np.zeros((len(slices), 3, nmo, nocc))
        expected = np.empty((len(slices), 3, 3))
        cocc, cvir = coeff[:, :nocc], coeff[:, nocc:]

        for atom, (left, right) in enumerate(slices):
            projector = np.diag((left <= np.arange(nmo)) & (np.arange(nmo) < right))
            for displacement in range(3):
                bra = projector @ ip_overlap[displacement]
                overlap1 = bra + bra.T
                angular = np.zeros((nmo, nmo))
                angular[nocc:, :nocc] = rng.normal(size=(nvir, nocc))
                angular -= angular.T.copy()
                orbitals1 = -0.5 * coeff.T @ overlap1 @ coeff + angular
                mo1[atom, displacement] = orbitals1[:, :nocc]
                cocc1, cvir1 = (coeff @ orbitals1[:, :nocc], coeff @ orbitals1[:, nocc:])
                for axis in range(3):
                    ket = -irp[axis, displacement] @ projector
                    matrix = (cocc1.T @ position[axis] @ cvir + cocc.T @ position[axis] @ cvir1
                              + cocc.T @ (ket + ket.T) @ cvir)
                    expected[atom, displacement, axis] = np.sqrt(2) * np.sum(matrix * q)

        calculated = _fixed_amplitude_dipole_derivative(
            coeff, q, coeff.T @ position @ coeff, irp, ip_overlap, slices, mo1,
        )
        np.testing.assert_allclose(calculated, expected, atol=1e-13)

    def test_derivative_rigid_motion_diagnostics(self):
        coords = np.array([[1.2, 0.03, 0.7], [0.3, 0.92, -0.2]])
        derivative = np.stack([np.eye(3), -np.eye(3)])
        dipole = coords[0] - coords[1]
        translation, rotation = _rigid_motion_errors(coords, dipole, derivative)
        self.assertEqual((translation, rotation), (0., 0.))
        _, rotation = _rigid_motion_errors(coords, -dipole, derivative)
        self.assertGreater(rotation, 1.)

    @unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required for the analytic AO integrals")
    def test_minimal_homonuclear_dipole_matches_contracted_gaussian_formula(self):
        from pyscf import gto

        centers = np.array([[0.3, -0.82, 0.27], [0.71, 0.61, 0.96]])
        mol = gto.M(atom=[("H", center) for center in centers], basis="sto-3g", unit="bohr", verbose=0)
        separation = centers[0] - centers[1]
        alpha = mol.bas_exp(0)
        contractions = mol.bas_ctr_coeff(0)[:, 0]
        exponent_sum = alpha[:, None] + alpha
        exponent_product = alpha[:, None] * alpha
        reduced_exponent = exponent_product / exponent_sum
        gaussian = (2 * np.sqrt(exponent_product) / exponent_sum) ** 1.5
        contraction = np.outer(contractions, contractions) * gaussian
        overlap_terms = contraction * np.exp(-reduced_exponent * np.vdot(separation, separation))
        overlap = overlap_terms.sum() / contraction.sum()
        overlap_gradient = -2 * separation * (overlap_terms * reduced_exponent).sum() / contraction.sum()
        denominators = np.sqrt(2 * (1 - overlap**2))
        expected_on_first = np.eye(3) / denominators
        expected_on_first += (np.outer(overlap_gradient, separation) * overlap
                              / ((1 - overlap**2) * denominators))

        # Spatial symmetry specifies the two normalized orbitals and precludes
        # occupied–virtual orbital relaxation in the symmetric overlap gauge.
        coeff = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2 * (1 + np.array([overlap, -overlap])))
        position = mol.intor_symmetric("int1e_r", comp=3)
        ip_position = mol.intor("int1e_irp", comp=9).reshape(3, 3, 2, 2)
        derivative = _fixed_amplitude_dipole_derivative(
            coeff, np.ones((1, 1)), coeff.T @ position @ coeff, ip_position,
            -mol.intor("int1e_ipovlp", comp=3), mol.aoslice_by_atom()[:, 2:], np.zeros((2, 3, 2, 1)),
        )
        np.testing.assert_allclose(coeff.T @ mol.intor_symmetric("int1e_ovlp") @ coeff, np.eye(2), atol=1e-13)
        np.testing.assert_allclose(derivative, np.stack([expected_on_first, -expected_on_first]), atol=1e-13)

    def test_isolation_requires_an_upper_neighbor_or_complete_spectrum(self):
        td = SimpleNamespace(
            xy=[(np.eye(3)[i].reshape(1, 3) / np.sqrt(2), 0) for i in range(2)],
            e=np.array([.4, .7]), converged=[True, True], positive_eig_threshold=1e-3,
        )
        _, energy, separation = _target_root(td, 0, (1, 3), 1e-5)
        self.assertAlmostEqual(energy, .4)
        self.assertAlmostEqual(separation, .3)
        with self.assertRaisesRegex(ValueError, "one TDA root above"):
            _target_root(td, 1, (1, 3), 1e-5)
        td.e[1] = td.e[0] + 1e-6
        with self.assertRaisesRegex(ValueError, "not electronically isolated"):
            _target_root(td, 0, (1, 3), 1e-5)


if __name__ == "__main__":
    unittest.main()
