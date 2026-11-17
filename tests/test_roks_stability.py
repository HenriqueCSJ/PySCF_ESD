"""Checks of the real shared-orbital ROKS stability channel."""

from importlib.util import find_spec
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_esd.cpu import ElectronicConvergenceError, ReferenceStabilityUnresolvedError, _stability_curvature
from pyscf_esd.openqp_mrsf_prepared import OPENQP_ENGINE_COMMIT
from pyscf_esd.openqp_roks_prepared import assess_openqp_roks_internal_stability, read_openqp_saved_roks_reference
from pyscf_esd.roks_stability import (
    assess_fixed_roks_internal_stability,
    assess_roks_internal_stability,
    assess_roks_reference_link,
)


class ROKSReferenceLinkTests(unittest.TestCase):
    def test_only_rotations_within_an_occupation_class_are_gauge(self):
        metric = np.eye(5)
        metric[1:3, 1:3] = [[0.6, -0.8], [-0.8, -0.6]]
        metric[4, 4] = 0.0  # A missing direction of purely virtual support.
        link = assess_roks_reference_link(metric, n_core=1, minimum_singular_value=0.9)
        self.assertTrue(link.passes_overlap_screen)
        np.testing.assert_allclose(link.somo_singular_values, [1, 1])

        for permutation, core, somo, occupied in (
            ([1, 0, 2, 3, 4], 0, 0, 1),
            ([0, 3, 2, 1, 4], 1, 0, 0),
        ):
            link = assess_roks_reference_link(np.eye(5)[:, permutation], n_core=1, minimum_singular_value=0.9)
            self.assertFalse(link.passes_overlap_screen)
            self.assertEqual(min(link.core_singular_values), core)
            self.assertEqual(min(link.somo_singular_values), somo)
            self.assertEqual(min(link.alpha_occupied_singular_values), occupied)
            self.assertEqual(link.somo_full_space_retention, (1.0, 1.0))

    def test_loss_of_full_space_support_is_reported_in_both_directions(self):
        metric = np.pad(np.diag([0.7, 0.95, 0.8, 1.0]), ((0, 1), (0, 0)))
        metric[0, 1], metric[4, 0] = 0.1, 0.2
        forward = assess_roks_reference_link(metric, n_core=1, minimum_singular_value=0.9)
        reverse = assess_roks_reference_link(metric.T, n_core=1, minimum_singular_value=0.9)
        self.assertFalse(forward.passes_overlap_screen)
        np.testing.assert_allclose(forward.somo_full_space_retention, [0.8, 0.8])
        np.testing.assert_allclose(forward.core_full_space_retention, [np.sqrt(0.50), np.sqrt(0.53)])
        np.testing.assert_allclose(reverse.core_full_space_retention, forward.core_full_space_retention[::-1])
        empty_core = assess_roks_reference_link(np.eye(2), n_core=0, minimum_singular_value=0.9)
        self.assertTrue(empty_core.passes_overlap_screen)
        self.assertIsNone(empty_core.core_full_space_retention)


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for ROKS stability")
class ROKSStabilityTests(unittest.TestCase):
    def setUp(self):
        from pyscf import dft, gto

        molecule = gto.M(
            atom="H 0 0 0; H 0.2 0 1.9; H 0 0.4 4.1; H 0 -0.2 6.2",
            unit="Bohr", basis="sto-3g", spin=2, verbose=0,
        )
        self.reference = dft.ROKS(molecule, xc="pbe0")
        self.reference.chkfile = None

    def _set_synthetic_orbitals(self):
        self.reference.converged = True
        self.reference.mo_coeff = np.eye(4)
        self.reference.mo_occ = np.array([1., 0., 2., 1.])
        self.reference.get_ovlp = lambda: np.eye(4)
        self.reference.get_hcore = lambda: np.zeros((4, 4))

    def test_all_occupation_blocks_enter_the_fresh_gradient(self):
        self._set_synthetic_orbitals()
        for spin, row, column in ((1, 0, 2), (0, 1, 3), (0, 1, 2), (1, 1, 2)):
            with self.subTest(spin=spin, row=row, column=column):
                fock = np.zeros((2, 4, 4))
                fock[spin, row, column] = fock[spin, column, row] = 0.003
                with (
                    patch.object(self.reference, "get_veff", return_value=fock) as veff,
                    patch("pyscf.soscf.newton_ah.gen_g_hop_rohf") as native,
                    self.assertRaisesRegex(ElectronicConvergenceError, "residual 0.003 Hartree"),
                ):
                    assess_roks_internal_stability(self.reference)
                density = veff.call_args.kwargs["dm"]
                np.testing.assert_array_equal(density, [
                    np.diag([1., 0., 1., 1.]), np.diag([0., 0., 1., 0.]),
                ])
                native.assert_not_called()

    def test_checked_response_classifies_only_the_internal_channel(self):
        self._set_synthetic_orbitals()
        fock = np.stack((np.diag([0., 1., -0.5, 0.]), np.diag([0., 1., -0.7, 0.])))
        expected = ((0.2, "qualified"), (-0.2, "negative_direction"), (0., "below_margin"))
        for curvature, status in expected:
            with self.subTest(status=status):
                diagonal = np.array([curvature, 1., 2., 3., 4.]) / 2
                response = (np.zeros(5), lambda vector: diagonal * vector, diagonal)
                with (
                    patch.object(self.reference, "get_veff", return_value=fock),
                    patch("pyscf.soscf.newton_ah.gen_g_hop_rohf", return_value=response) as native,
                ):
                    report = assess_roks_internal_stability(self.reference)
                kwargs = native.call_args.kwargs
                self.assertFalse(kwargs["with_symmetry"])
                np.testing.assert_array_equal(kwargs["fock_ao"].focka, fock[0])
                np.testing.assert_array_equal(kwargs["fock_ao"].fockb, fock[1])
                self.assertEqual(report.scf_residual_hartree, 0.)
                self.assertEqual(report.rotation_dimension, 5)
                self.assertEqual(report.internal.channel, "internal")
                self.assertEqual(report.internal.status, status)
                self.assertAlmostEqual(report.internal.curvature, curvature)
                self.assertEqual(report.internal_qualified, status == "qualified")

    def test_rejects_invalid_response_and_orbitals(self):
        self._set_synthetic_orbitals()
        for gradient, diagonal, reason in (
            (np.ones(5), np.ones(5), "disagrees"),
            (np.zeros(4), np.ones(5), "shape"),
            (np.zeros(5), np.full(5, np.nan), "nonfinite"),
        ):
            with (
                self.subTest(reason=reason),
                patch.object(self.reference, "get_veff", return_value=np.zeros((2, 4, 4))),
                patch("pyscf.soscf.newton_ah.gen_g_hop_rohf", return_value=(gradient, None, diagonal)),
            ):
                report = assess_roks_internal_stability(self.reference)
                self.assertFalse(report.internal_qualified)
                self.assertEqual(report.internal.status, "unresolved")
                self.assertIn(reason, report.internal.reason)
        self.reference.mo_occ = np.array([1., 0., 2., 0.9])
        with self.assertRaisesRegex(ElectronicConvergenceError, "electron counts"):
            assess_roks_internal_stability(self.reference)

    def test_rejects_response_of_different_energy(self):
        self._set_synthetic_orbitals()
        with (
            patch.object(self.reference, "do_nlc", return_value=True),
            self.assertRaisesRegex(ReferenceStabilityUnresolvedError, "nonlocal correlation"),
        ):
            assess_roks_internal_stability(self.reference)
        self.reference.second_grids = object()
        with self.assertRaisesRegex(ReferenceStabilityUnresolvedError, "different quadrature"):
            assess_roks_internal_stability(self.reference)

    def test_fixed_reference_rejects_missing_grid_energy_fock_and_residual(self):
        self._set_synthetic_orbitals()
        self.reference.converged = False
        fock = np.zeros((2, 4, 4))
        fock[1, 0, 2] = fock[1, 2, 0] = 0.001
        with (
            patch.object(self.reference, "kernel", side_effect=AssertionError("unexpected SCF")),
            patch.object(self.reference, "get_veff", return_value=fock) as potential,
            patch.object(self.reference, "energy_tot", return_value=-2.5),
            patch("pyscf.soscf.newton_ah.gen_g_hop_rohf") as response,
        ):
            with self.assertRaisesRegex(ReferenceStabilityUnresolvedError, "grid coordinates"):
                assess_fixed_roks_internal_stability(self.reference, reference_energy_hartree=-2.5)
            potential.assert_not_called()
            self.reference.grids.coords = np.zeros((1, 3))
            self.reference.grids.weights = np.ones(1)
            for energy, expected_fock, error in (
                (-2.51, fock, "total energy differs"),
                (-2.5, np.zeros_like(fock), "spin Fock matrices differ"),
                (-2.5, fock, "ROKS residual"),
            ):
                with self.subTest(error=error), self.assertRaisesRegex(ElectronicConvergenceError, error):
                    assess_fixed_roks_internal_stability(
                        self.reference, reference_energy_hartree=energy, reference_fock_mo=expected_fock,
                    )
            response.assert_not_called()
        self.assertFalse(self.reference.converged)

    def test_analytic_shared_rotation_hessian_agrees_with_spin_density_curvature(self):
        from pyscf import lib

        mean_field = self.reference
        mean_field.grids.level = 0
        mean_field.grids.prune = None
        mean_field.small_rho_cutoff = 0.
        mean_field.conv_tol = 1e-13
        mean_field.conv_tol_grad = 1e-9
        with lib.with_omp_threads(1):
            mean_field.kernel()
            with patch("pyscf_esd.roks_stability._stability_curvature", wraps=_stability_curvature) as solve:
                report = assess_roks_internal_stability(mean_field)
            self.assertTrue(report.internal_qualified)
            self.assertLess(report.scf_residual_hartree, 1e-7)
            self.assertEqual(report.rotation_dimension, 5)
            axes = np.eye(report.rotation_dimension)
            operator = solve.call_args.args[0]
            analytic = np.column_stack([operator(axis) for axis in axes])
            occupations = mean_field.mo_occ
            rotated = occupations[:, None] < occupations[None, :]
            coefficients = mean_field.mo_coeff
            spin_projectors = np.stack((np.diag(occupations > 0), np.diag(occupations == 2))).astype(float)
            density = coefficients @ spin_projectors @ coefficients.T
            fock = mean_field.get_hcore() + mean_field.get_veff(dm=density)
            fock_mo = coefficients.T @ fock @ coefficients
            # This response is evaluated from AO densities. PySCF's shared
            # rotation operator instead caches XC from occupied orbitals.
            response = mean_field.gen_response(dm0=density, hermi=1, with_nlc=False, grids=mean_field.grids)

            def directional_curvature(direction):
                generator = np.zeros((len(occupations), len(occupations)))
                generator[rotated] = direction
                generator -= generator.T.copy()
                # First and second exact variations of exp(K) D exp(-K).
                first_mo = generator @ spin_projectors - spin_projectors @ generator
                second_mo = generator @ first_mo - first_mo @ generator
                first_ao = coefficients @ first_mo @ coefficients.T
                delta_fock = response(first_ao)
                return (
                    np.einsum("sij,sji->", fock_mo, second_mo)
                    + np.einsum("sij,sji->", delta_fock, first_ao)
                )

            direct = np.diag([directional_curvature(axis) for axis in axes])
            for index, axis in enumerate(axes):
                for other in range(index):
                    mixed = (directional_curvature(axis + axes[other]) - direct[index, index] - direct[other, other]) / 2
                    direct[index, other] = direct[other, index] = mixed

            # Serialize the already converged test determinant in a scaled
            # AO convention and import it without another SCF call.
            trils = np.tril_indices(coefficients.shape[0])
            scales = np.array([0.8, 1.1, 0.7, 1.2])
            scale = np.outer(scales, scales)
            primitives = np.asarray(mean_field.mol._basis["H"][0][1:])
            exp, weights = primitives.T
            normalized_primitive_overlap = (2 * np.sqrt(np.outer(exp, exp)) / (exp[:, None] + exp)) ** 1.5
            weights = weights / np.sqrt(weights @ normalized_primitive_overlap @ weights)
            basis = {
                "format": "OpenQP portable basis v1", "coefficient_convention": "molden",
                "basis_function_order": "molden", "spherical_harmonics": False, "nbf": 4,
                "shells": [
                    {"atom_index": atom, "angular_momentum": 0, "shell": "s",
                     "exponents": primitives[:, 0].tolist(),
                     "coefficients": (norm * weights).tolist()}
                    for atom, norm in enumerate(scales)
                ],
            }
            checkpoint = {
                "json": {"scf_type": "rohf", "nbf": 4},
                "atoms": [1] * 4, "coord": mean_field.mol.atom_coords().ravel().tolist(),
                "basis_set": basis, "energy": mean_field.e_tot,
                "OQP::SM": (scale * mean_field.get_ovlp())[trils].tolist(),
                "OQP::VEC_MO_A": (coefficients / scales[:, None]).T.tolist(),
                "OQP::DM_A": (density[0] / scale)[trils].tolist(),
                "OQP::DM_B": (density[1] / scale)[trils].tolist(),
                "OQP::FOCK_A": (fock[0] * scale)[trils].tolist(),
                "OQP::FOCK_B": (fock[1] * scale)[trils].tolist(),
            }
            with patch("pyscf.dft.roks.ROKS.kernel", side_effect=AssertionError("unexpected SCF")):
                imported = assess_openqp_roks_internal_stability(
                    checkpoint, engine_commit=OPENQP_ENGINE_COMMIT, n_core=1, charge=0, xc="pbe0",
                    grid_coordinates_bohr=mean_field.grids.coords, grid_weights=mean_field.grids.weights,
                )
            corrupted = {**checkpoint, "OQP::DM_A": checkpoint["OQP::DM_B"]}
            with self.assertRaisesRegex(ValueError, "spin-A density disagrees"):
                read_openqp_saved_roks_reference(corrupted, engine_commit=OPENQP_ENGINE_COMMIT, n_core=1)
        np.testing.assert_allclose(analytic, direct, atol=1e-8, rtol=0)
        self.assertAlmostEqual(report.internal.curvature, np.linalg.eigvalsh(analytic)[0], places=11)
        self.assertTrue(imported.internal_qualified)
        self.assertAlmostEqual(imported.internal.curvature, report.internal.curvature, places=10)
        self.assertLess(abs(imported.fixed_energy_difference_hartree), 1e-10)
        self.assertLess(imported.fixed_fock_max_difference_hartree, 1e-10)
