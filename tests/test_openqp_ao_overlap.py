"""Test portable OpenQP AO metrics and MO import without an electronic calculation."""

from importlib.util import find_spec
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd.openqp_ao_overlap import exact_openqp_ao_overlap, import_openqp_orbitals
from pyscf_esd.openqp_mrsf_prepared import OPENQP_ENGINE_COMMIT


def _shell(atom, angular, exponents, coefficients):
    return {
        "atom_index": atom,
        "angular_momentum": angular,
        "shell": "spdfg"[angular],
        "exponents": exponents,
        "coefficients": coefficients,
    }


def _basis(shells, *, spherical=False):
    return {
        "format": "OpenQP portable basis v1",
        "coefficient_convention": "molden",
        "basis_function_order": "molden",
        "spherical_harmonics": spherical,
        "nbf": sum(
            2 * ang + 1 if spherical else (ang + 1) * (ang + 2) // 2
            for ang in (shell["angular_momentum"] for shell in shells)
        ),
        "shells": shells,
    }


def _overlap(atoms, bra, ket, basis, bra_metric, ket_metric, **options):
    return exact_openqp_ao_overlap(
        atoms, bra, ket, basis, engine_commit=OPENQP_ENGINE_COMMIT,
        bra_native_ao_overlap=bra_metric, ket_native_ao_overlap=ket_metric, **options,
    )


class OpenQPAOBasisValidationTests(unittest.TestCase):
    def test_rejects_unknown_layout_and_out_of_scope_higher_shells(self):
        basis = _basis([_shell(0, 2, [1.0], [1.0])])
        coords = [[0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(ValueError, "explicit native AO component map"):
            _overlap([1], coords, coords, basis, np.eye(6), np.eye(6))
        with self.assertRaisesRegex(ValueError, "nbf disagrees"):
            _overlap([1], coords, coords, {**basis, "nbf": 5}, np.eye(5), np.eye(5))
        with self.assertRaisesRegex(ValueError, "portable basis v1"):
            _overlap([1], coords, coords, {**basis, "basis_function_order": "unknown"}, [], [])
        with self.assertRaisesRegex(ValueError, "invalid native AO component permutation"):
            _overlap(
                [1], coords, coords, basis, np.eye(6), np.eye(6),
                native_components={2: ([0, 1, 2, 3, 4, 4], np.ones(6))},
            )

    def test_import_checks_charge_spin_and_mo_dimensions(self):
        basis = _basis([_shell(0, 0, [0.7], [1.0])])
        controls = {
            "engine_commit": OPENQP_ENGINE_COMMIT,
            "native_ao_overlap": [1.0],
            "native_mo_coefficients": [[1.0]],
        }
        for override, message in (
            ({"charge": True}, "charge must be an integer"),
            ({"charge": 2}, "negative electron count"),
            ({"spin": 0}, "spin is incompatible"),
            ({"spin": True}, "spin must be an integer"),
            ({"native_mo_coefficients": [[1.0, 0.0]]}, "one row per AO"),
            ({"native_mo_coefficients": [[2.0]]}, "not orthonormal in the OpenQP AO metric"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    import_openqp_orbitals(
                        [1], [[0.0, 0.0, 0.0]], basis, **{**controls, **override},
                    )


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for AO integrals")
class OpenQPAOOverlapTests(unittest.TestCase):
    def test_unrenormalized_contraction_matches_analytic_shifted_gaussians(self):
        exponent = np.array([1.3, 0.4])
        coeff = np.array([0.3, -0.6])
        basis = _basis([_shell(0, 0, exponent, coeff)])
        bra = [[0.0, 0.0, 0.0]]
        ket = [[0.3, -0.2, 0.8]]
        alpha, beta = np.meshgrid(exponent, exponent)
        overlap = (2 * np.sqrt(alpha * beta) / (alpha + beta)) ** 1.5
        native_self = [coeff @ overlap @ coeff]
        separation_squared = np.sum(np.square(ket))
        cross = overlap * np.exp(-alpha * beta / (alpha + beta) * separation_squared)
        expected = coeff @ cross @ coeff
        actual = _overlap([1], bra, ket, basis, native_self, [native_self])
        self.assertAlmostEqual(actual.item(), expected, places=14)
        with patch("pyscf.gto.mole.NORMALIZE_GTO", False):
            already_raw = _overlap([1], bra, ket, basis, native_self, [native_self])
        self.assertAlmostEqual(already_raw.item(), expected, places=14)
        with self.assertRaisesRegex(ValueError, "ket PySCF AO self overlap disagrees"):
            _overlap([1], bra, ket, basis, native_self, [[1.0]])

    def test_shell_order_and_signed_cross_for_pure_d_and_f(self):
        from pyscf import gto

        basis = _basis([
            _shell(1, 3, [0.85], [-1.0]),
            _shell(0, 2, [0.6], [1.0]),
            _shell(1, 0, [0.8], [1.0]),
            _shell(0, 1, [1.1], [1.0]),
        ], spherical=True)
        bra = [[0.0, 0.0, 0.0], [0.3, -0.2, 1.2]]
        ket = [[-0.1, 0.3, 0.2], [0.4, -0.1, 1.3]]
        grouped = {"H1": [[1, [1.1, 1]], [2, [0.6, 1]]], "H2": [[0, [0.8, 1]], [3, [0.85, -1]]]}
        bra_mol, ket_mol = [
            gto.M(atom=list(zip(grouped, coords)), unit="Bohr", basis=grouped, verbose=0)
            for coords in (bra, ket)
        ]
        order = np.r_[9:16, 3:8, 8, 0:3]
        lookup = np.ix_(order, order)
        native_metrics = [mol.intor_symmetric("int1e_ovlp")[lookup] for mol in (bra_mol, ket_mol)]
        expected = gto.intor_cross("int1e_ovlp", bra_mol, ket_mol)[lookup]
        self.assertGreater(np.max(np.abs(expected - expected.T)), 0.1)
        actual = _overlap([1, 1], bra, ket, basis, *native_metrics)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        reverse = _overlap([1, 1], ket, bra, basis, *reversed(native_metrics))
        np.testing.assert_allclose(reverse, expected.T, atol=1e-14, rtol=0)

    def test_applies_an_explicit_signed_component_map(self):
        from pyscf import gto

        basis = _basis([_shell(0, 2, [0.7], [0.8])])
        bra, ket = [[0.0, 0.0, 0.0]], [[0.4, -0.1, 0.3]]
        bra_mol, ket_mol = [
            gto.M(atom=[("H", xyz)], unit="Bohr", basis={"H": [[2, [0.7, 1.0]]]},
                  cart=True, spin=1, verbose=0)
            for (xyz,) in (bra, ket)
        ]
        order = [0, 3, 5, 1, 2, 4]
        signs = np.array([1, -1, 1, -1, 1, 1])
        lookup = np.ix_(order, order)
        weights = 0.8 * signs
        scale = np.outer(weights, weights)
        self_metric = scale * bra_mol.intor_symmetric("int1e_ovlp")[lookup]
        expected = scale * gto.intor_cross("int1e_ovlp", bra_mol, ket_mol)[lookup]
        actual = _overlap(
            [1], bra, ket, basis, self_metric, self_metric,
            native_components={2: (order, signs)},
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    def test_import_preserves_mo_values_and_phases_with_signed_ao_mapping(self):
        from pyscf import gto

        basis = _basis([
            _shell(0, 2, [0.7], [0.8]),
            _shell(0, 0, [1.1], [-0.4]),
        ])
        coords = [[0.0, 0.0, 0.0]]
        reference = gto.M(
            atom=[("H", coords[0])], unit="Bohr", basis={"H": [[0, [1.1, -1.0]], [2, [0.7, 1.0]]]},
            cart=True, spin=1, verbose=0,
        )
        primitive_order = np.array([0, 3, 5, 1, 2, 4])
        signs = np.array([1, -1, 1, -1, 1, 1])
        order = np.r_[primitive_order + 1, 0]
        factors = np.r_[0.8 * signs, 0.4]
        metric = reference.intor_symmetric("int1e_ovlp")
        native_metric = np.outer(factors, factors) * metric[np.ix_(order, order)]
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        canonical = eigenvectors[:, [2, 0]] / np.sqrt(eigenvalues[[2, 0]])
        canonical[:, 0] *= -1
        native = canonical[order] / factors[:, None]
        mol, imported = import_openqp_orbitals(
            [1], coords, basis, engine_commit=OPENQP_ENGINE_COMMIT,
            native_ao_overlap=native_metric[np.tril_indices(7)], native_mo_coefficients=native,
            charge=-1, spin=2, native_components={2: (primitive_order, signs)},
        )
        self.assertEqual((mol.charge, mol.spin, mol.nelec), (-1, 2, (2, 0)))
        np.testing.assert_allclose(imported, canonical, atol=1e-14, rtol=0)
        probes = np.array([[0.1, 0.3, -0.4], [-0.5, 0.2, 0.7]])
        native_ao = reference.eval_gto("GTOval_cart", probes)[:, order] * factors
        np.testing.assert_allclose(
            mol.eval_gto("GTOval_cart", probes) @ imported, native_ao @ native,
            atol=1e-14, rtol=0,
        )

    def test_import_checks_ao_metric_and_uses_the_imported_mo_metric(self):
        basis = _basis([_shell(0, 0, [0.7], [0.25])])
        coords = [[0.0, 0.0, 0.0]]
        exact = 0.25 ** 2
        for supplied, message in (
            (1.0, "AO self overlap disagrees"),
            (exact + 5e-9, "not orthonormal in the PySCF AO metric"),
        ):
            with self.subTest(metric=supplied):
                with self.assertRaisesRegex(ValueError, message):
                    import_openqp_orbitals(
                        [1], coords, basis, engine_commit=OPENQP_ENGINE_COMMIT,
                        native_ao_overlap=[supplied], native_mo_coefficients=[[1 / np.sqrt(supplied)]],
                    )


if __name__ == "__main__":
    unittest.main()
