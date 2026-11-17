"""CPU electronic backend tests without electronic-structure runs."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyscf_esd.cpu import (
    ElectronicConvergenceError,
    ElectronicPoint,
    ReferenceStabilityUnresolvedError,
    Settings,
    UnstableReferenceError,
    _checked_excited_gradient,
    _dense_tda_retry,
    _make_mean_field,
    _negative_stability_mode,
    _orthogonalized_density,
    _scf_residual,
    _semilocal_complex_curvature,
    _rks_internal_stability_operator,
    _rks_stability_block_operator,
    _stability_curvature,
    _solve_excited_response,
    _solve_reference,
    _tda_residuals,
    assess_rks_stability,
    build_molecule,
    check_restricted_reference,
    solve_point,
)


class CpuBackendTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required to inspect basis definitions")
    def test_def2_valence_basis_is_not_run_as_all_electron(self):
        with self.assertRaisesRegex(ElectronicConvergenceError, "effective core potential for Rb"):
            build_molecule(("Rb", "H"), ((0, 0, 0), (0, 0, 4)), Settings("def2-tzvpd", "pbe0"))

    def test_large_tda_restarts_the_same_operator_before_abstaining(self):
        energies = np.arange(1.0, 261.0)
        vectors = np.eye(len(energies))[:2]
        td = SimpleNamespace(
            e=energies[:2], converged=np.array([True, False]),
            xy=[(row[None, :] / np.sqrt(2.0), 0) for row in vectors],
            gen_vind=lambda: (lambda rows: rows * energies, None),
            transition_dipole=lambda: np.zeros((2, 3)),
        )

        def native_kernel(x0=None):
            if x0 is not None:
                td.converged[:] = True

        td.kernel = Mock(side_effect=native_kernel)
        mean_field = SimpleNamespace(mo_coeff=np.eye(len(energies) + 1), e_tot=-1.0, TDA=lambda: td)
        with (
            patch("pyscf_esd.cpu.build_molecule", return_value=SimpleNamespace(nelectron=2)),
            patch("pyscf_esd.cpu._solve_reference", return_value=(mean_field, 0.0, ())),
        ):
            point = solve_point(("H", "H"), ((0, 0, 0), (0, 0, 1)), Settings("sto-3g", "pbe0", nstates=2))
        self.assertEqual(point.tda_route, "davidson_restart_same_operator")
        self.assertEqual(td.kernel.call_count, 2)
        np.testing.assert_allclose(td.kernel.call_args.kwargs["x0"], vectors / np.sqrt(2))
        np.testing.assert_allclose(point.tda_residuals, 0.0)

    def test_rejects_invalid_settings(self):
        for kwargs in (
            {"nstates": True}, {"grid_level": 10}, {"charge": 1.5},
            {"td_residual_tol": float("nan")}, {"scf_conv_tol": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings("sto-3g", "pbe0", **kwargs)

    def test_projected_density_uses_ao_overlap_metric(self):
        overlap = np.diag([0.5, 2.0, 3.0])
        projected_orbitals = np.array([[1.0, 2.0], [1.0, -1.0], [-1.0, 0.5]])
        density = _orthogonalized_density(projected_orbitals, overlap)
        self.assertAlmostEqual(np.trace(density @ overlap), 4.0)
        np.testing.assert_allclose(density @ overlap @ density, 2 * density)
        with self.assertRaisesRegex(np.linalg.LinAlgError, "lost rank"):
            _orthogonalized_density(projected_orbitals[:, [0, 0]], overlap)

    def test_scf_residual_rebuilds_fock_at_final_density(self):
        density = np.diag([2.0, 0.0])
        mean_field = SimpleNamespace(
            mol=SimpleNamespace(nao_nr=lambda: 2, nelectron=2),
            mo_coeff=np.eye(2), mo_occ=np.array([2.0, 0.0]), mo_energy=np.array([-0.8, 0.4]),
            e_tot=-1.0, get_ovlp=lambda: np.eye(2), make_rdm1=lambda: density,
            get_hcore=lambda: np.diag([-1.0, 1.0]),
            get_veff=Mock(return_value=np.array([[0.5, 0.015], [0.015, 0.3]])),
            get_grad=lambda c, occ, fock: 2 * c[:, 1:].T @ fock @ c[:, :1],
        )
        self.assertAlmostEqual(_scf_residual(mean_field), 0.03)
        self.assertIs(mean_field.get_veff.call_args.kwargs["dm"], density)
        mean_field.mo_occ = np.array([1.0, 1.0])
        with self.assertRaisesRegex(ElectronicConvergenceError, "closed-shell occupations"):
            _scf_residual(mean_field)

    def test_recovery_refuses_flag_only_and_uses_fresh_objects(self):
        settings = Settings("sto-3g", "pbe0")
        cold = Mock(converged=True)
        retry_base = Mock()
        newton = Mock(converged=True)
        retry_base.newton.return_value = newton
        mol = object()
        with (
            patch("pyscf_esd.cpu._make_mean_field", side_effect=[cold, retry_base]) as factory,
            patch("pyscf_esd.cpu._scf_residual", side_effect=[1e-3, 1e-10]),
        ):
            reference, residual, attempts = _solve_reference(mol, settings, None)
        self.assertIs(reference, newton)
        self.assertEqual(residual, 1e-10)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual([attempt.route for attempt in attempts], ["minao_diis", "minao_soscf"])
        self.assertTrue(attempts[0].converged)
        self.assertFalse(attempts[0].accepted)
        self.assertTrue(attempts[1].accepted)
        cold.kernel.assert_called_once_with(dm0=None)
        newton.kernel.assert_called_once_with(dm0=None)

    def test_restricted_internal_recovery_uses_rotated_density_and_lower_energy(self):
        settings = Settings("sto-3g", "hf", stability_checks="all")
        orbitals = np.eye(2)[:, ::-1]
        occupations = np.array([2., 0.])
        first = Mock(converged=True, e_tot=-.4, mo_occ=occupations)
        first.stability.return_value = orbitals, None, False, None
        repaired = Mock(converged=True, e_tot=-.8, mo_occ=occupations)
        repaired.stability.side_effect = [(np.eye(2), None, True, None), (None, None, None, True)]
        with (
            patch("pyscf_esd.cpu._make_mean_field", side_effect=[first, repaired]),
            patch("pyscf_esd.cpu._scf_residual", side_effect=[1e-10, 2e-10]),
        ):
            result, residual, attempts = _solve_reference(object(), settings, None)
        self.assertIs(result, repaired)
        self.assertEqual(residual, 2e-10)
        first.make_rdm1.assert_called_once_with(mo_coeff=orbitals, mo_occ=occupations)
        repaired.kernel.assert_called_once_with(dm0=first.make_rdm1.return_value)
        self.assertEqual(attempts[-1].route, "stability_rotation_diis")
        self.assertEqual(repaired.stability.call_count, 2)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for molecular stability checks")
    def test_restricted_spin_instability_is_distinguished_from_scf_convergence(self):
        settings = Settings("sto-3g", "pbe0", nstates=1, grid_level=1, stability_checks="all")
        symbols = ("H", "H")
        equilibrium = build_molecule(symbols, [[0, 0, 0], [0, 0, 1.4]], settings)
        reference, residual, _ = _solve_reference(equilibrium, settings, None)
        self.assertTrue(reference.converged)
        self.assertLessEqual(residual, settings.scf_residual_tol)
        report = assess_rks_stability(reference)
        self.assertTrue(report.qualified)
        self.assertEqual([row.channel for row in report.channels],
                         ["internal", "real_to_complex", "restricted_to_unrestricted"])
        stretched = build_molecule(symbols, [[0, 0, 0], [0, 0, 4.5]], settings)
        with self.assertRaises(UnstableReferenceError) as failure:
            _solve_reference(stretched, settings, None)
        self.assertEqual(failure.exception.channel, "restricted_to_unrestricted")

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the RKS assessment interface")
    def test_three_channel_stability_distinguishes_margin_and_unresolved(self):
        settings = Settings("sto-3g", "pbe0")
        molecule = build_molecule(("H", "H"), [[0, 0, 0], [0, 0, 1.4]], settings)
        reference = _make_mean_field(molecule, settings)
        reference.converged = True
        reference.mo_coeff = np.eye(2)
        reference.mo_occ = np.array([2., 0.])
        with (
            patch("pyscf_esd.cpu._scf_residual", return_value=1e-9),
            patch("pyscf_esd.cpu._rks_internal_stability_operator", return_value=(lambda v: 0.5 * v, [0.5])) as internal,
            patch("pyscf.scf.stability._gen_hop_rhf_external", return_value=(
                lambda v: 1e-6 * v, [1e-6], lambda v: np.full_like(v, np.nan), [0.0],
            )) as external,
        ):
            report = assess_rks_stability(reference)
            with self.assertRaisesRegex(ElectronicConvergenceError, "exceeds"):
                assess_rks_stability(reference, scf_residual_tolerance=1e-10)
        self.assertFalse(report.qualified)
        self.assertEqual([row.status for row in report.channels], ["qualified", "below_margin", "unresolved"])
        self.assertEqual(report.channels[1].curvature, 1e-6)
        self.assertIn("nonfinite", report.channels[2].reason)
        internal.assert_called_once_with(reference, with_symmetry=False)
        external.assert_called_once_with(reference, with_symmetry=False)
        with (
            patch.object(reference, "do_nlc", return_value=True),
            self.assertRaisesRegex(ReferenceStabilityUnresolvedError, "nonlocal correlation"),
        ):
            assess_rks_stability(reference)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for exact semilocal orbital stability")
    def test_semilocal_complex_curvature_matches_full_hessian_with_rotated_orbitals(self):
        from pyscf.scf.stability import _gen_hop_rhf_external

        settings = Settings("sto-3g", "pbe", grid_level=1)
        mol = build_molecule(("O", "H", "H"), [[0, 0, 0], [0, 1.4, 1.1], [0, -1.4, 1.1]], settings)
        reference, _, _ = _solve_reference(mol, settings, None)
        occupied = reference.mo_occ == 2
        random = np.random.default_rng(12)
        for block in (occupied, ~occupied):
            nmo = np.count_nonzero(block)
            rotation, _ = np.linalg.qr(random.standard_normal((nmo, nmo)))
            reference.mo_coeff[:, block] = reference.mo_coeff[:, block] @ rotation
        operator, diagonal, spin, _ = _gen_hop_rhf_external(reference, with_symmetry=False)
        full = np.column_stack([operator(direction) for direction in np.eye(len(diagonal))])
        expected = np.linalg.eigvalsh(full)[0]
        result = _semilocal_complex_curvature(reference, operator, 1e-6)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.curvature, expected, places=12)
        self.assertLess(result.residual, 1e-12)
        self.assertIsNone(result.below_threshold_direction)
        vectors = random.standard_normal((2, len(diagonal)))
        internal, _ = _rks_internal_stability_operator(reference, with_symmetry=False)
        for name, native in (("internal", internal), ("restricted_to_unrestricted", spin)):
            batched = _rks_stability_block_operator(reference, name)
            np.testing.assert_allclose(batched(vectors), [native(vector) for vector in vectors], atol=1e-12, rtol=0)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for DFT stability checks")
    def test_restricted_dft_checks_stationary_orbitals_with_exactly_zero_gradient(self):
        mf = SimpleNamespace(xc="pbe0", mo_coeff=np.eye(2), mo_occ=np.array([2.0, 0.0]))
        internal = (np.zeros(1), lambda v: 0.5 * v, np.ones(1))
        nonfinite_spin_response = (None, None, lambda v: np.full_like(v, np.nan), np.zeros(1))
        with (
            patch("pyscf.soscf.newton_ah.gen_g_hop_rhf", return_value=internal),
            patch("pyscf.scf.stability._gen_hop_rhf_external", return_value=nonfinite_spin_response),
            self.assertRaises(ReferenceStabilityUnresolvedError) as failure,
        ):
            check_restricted_reference(mf, "Test stationary reference")
        self.assertEqual(failure.exception.stage, "Test stationary reference")
        self.assertEqual(failure.exception.channel, "restricted_to_unrestricted")
        self.assertIn("nonfinite", str(failure.exception))

    def test_endpoint_reports_verified_spin_instability_after_inconclusive_internal_solve(self):
        stage = "Reference at a tracked S1 stationary geometry"
        reference = object()
        internal = ReferenceStabilityUnresolvedError("internal", stage, "unconverged lowest mode")
        external = ReferenceStabilityUnresolvedError("restricted_to_unrestricted", stage)
        for external_result, expected in (
            ((None, None, None, False), UnstableReferenceError),
            ((None, None, None, True), ReferenceStabilityUnresolvedError),
            (external, ReferenceStabilityUnresolvedError),
        ):
            with (
                self.subTest(external_result=external_result),
                patch("pyscf_esd.cpu._restricted_stability", side_effect=[internal, external_result]) as check,
                self.assertRaises(expected) as failure,
            ):
                check_restricted_reference(reference, stage)
            check.assert_called_with(reference, stage, internal=False, external=True)
            if expected is UnstableReferenceError:
                self.assertEqual(failure.exception.channel, "restricted_to_unrestricted")
            else:
                self.assertIs(failure.exception, internal)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF Davidson is required")
    def test_stability_seeds_find_negative_mode_in_an_orthogonal_block(self):
        diagonal = np.linspace(0.1, 3.0, 32)
        diagonal[-2:] = 3.0
        hessian = np.diag(diagonal)
        hessian[-1, -2] = hessian[-2, -1] = 4.0
        from pyscf import lib

        native_root, _ = lib.davidson(lambda vector: hessian @ vector, 1 / diagonal, diagonal, nroots=2)
        self.assertGreater(native_root[0], 0.0)
        mode = _negative_stability_mode(lambda vector: hessian @ vector, diagonal)
        self.assertAlmostEqual(mode @ hessian @ mode, -1.0)
        for shift in (0.0, 0.01):
            block = Mock(side_effect=lambda vectors: vectors @ hessian + shift)
            evidence = _stability_curvature(lambda vector: hessian @ vector, diagonal, -1e-5, block_operator=block)
            self.assertAlmostEqual(evidence.curvature, -1.0)
            if shift:
                block.assert_called_once()
            else:
                self.assertGreater(block.call_count, 1)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the stability solver")
    def test_low_gap_projection_only_accepts_a_native_negative_witness(self):
        from pyscf import lib

        diagonal = np.linspace(0.1, 3, 32)
        stable = np.diag(diagonal)
        broken = stable.copy()
        broken[0, 1] = broken[1, 0] = -0.5
        for use_fake_block in (False, True):
            actual = stable if use_fake_block else broken
            with patch("pyscf.lib.davidson1", wraps=lib.davidson1) as iterative:
                result = _stability_curvature(
                    lambda vector: actual @ vector, diagonal, -1e-5,
                    block_operator=lambda rows: rows @ broken, probe_low_gap=True,
                )
            if use_fake_block:
                self.assertGreater(iterative.call_count, 0)
                self.assertIsNone(result.below_threshold_direction)
            else:
                iterative.assert_not_called()
                self.assertLess(result.curvature, 0)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF Davidson is required")
    def test_stability_verifies_negative_direction_even_with_unconverged_ritz_pair(self):
        diagonal = np.linspace(-0.5, 1.0, 32)
        trial = np.eye(len(diagonal))[0] + 0.3 * np.eye(len(diagonal))[-1]
        with patch("pyscf.lib.davidson1", return_value=([False], [0.0], [trial])) as davidson:
            mode = _negative_stability_mode(lambda vector: diagonal * vector, diagonal)
        davidson.assert_called_once()
        np.testing.assert_allclose(mode, trial / np.linalg.norm(trial))
        self.assertLess(mode @ (diagonal * mode), 0.0)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF Davidson is required")
    def test_stability_rejects_unverified_ritz_results_and_restarts_reproducibly(self):
        diagonal = np.linspace(0.1, 1.0, 32)
        trial = np.eye(len(diagonal))[:2]
        checked_seeds = []
        for flags, values in (([False, True], diagonal[:2]), ([True, True], [-0.1, diagonal[1]])):
            with (
                patch("pyscf.lib.davidson1", return_value=(flags, values, trial)) as davidson,
                self.assertRaisesRegex(ElectronicConvergenceError, "Independent starts"),
            ):
                _negative_stability_mode(lambda vector: diagonal * vector, diagonal)
            self.assertEqual(davidson.call_count, 3)
            checked_seeds.append(np.array([call.args[1][:2] for call in davidson.call_args_list]))
        np.testing.assert_array_equal(checked_seeds[0], checked_seeds[1])
        self.assertTrue(np.all(checked_seeds[0] != 0.0))
        self.assertFalse(np.array_equal(checked_seeds[0][0], checked_seeds[0][1]))

    def test_td_residual_and_amplitude_normalization(self):
        operator = np.diag([0.2, 0.4, 0.6])
        td = SimpleNamespace(gen_vind=lambda: (lambda x: x @ operator, None))
        amplitudes = (np.eye(3)[:2] / np.sqrt(2)).reshape(2, 1, 3)
        np.testing.assert_allclose(_tda_residuals(td, np.array([0.2, 0.4]), amplitudes), 0)
        np.testing.assert_allclose(
            _tda_residuals(td, np.array([0.1, 0.4]), amplitudes), [0.1, 0],
        )
        with self.assertRaisesRegex(ElectronicConvergenceError, "orthonormality"):
            _tda_residuals(td, np.array([0.2, 0.4]), amplitudes * 2)

    def test_tda_recovers_from_davidson_flag_or_residual_using_same_operator(self):
        operator = np.diag(np.arange(1.0, 18.0))
        operator[0, 1] = operator[1, 0] = 0.25
        known_roots, known_vectors = np.linalg.eigh(operator)
        mol = SimpleNamespace(nelectron=2)
        settings = Settings("sto-3g", "pbe0", nstates=2, td_residual_tol=1e-9)

        for converged, energies, expected_route in (
            ([True, False], known_roots[:2], "dense_same_operator"),
            ([True, True], known_roots[:2] + 0.01, "dense_same_operator"),
            ([True, True], known_roots[:2], "davidson"),
        ):
            with self.subTest(expected_route=expected_route, converged=converged):
                response = Mock(side_effect=lambda vectors: vectors @ operator)
                td = SimpleNamespace(
                    converged=converged, e=energies, kernel=Mock(),
                    xy=[(vector[None, :] / np.sqrt(2), 0) for vector in known_vectors[:, :2].T],
                    gen_vind=lambda: (response, None),
                )
                td.transition_dipole = lambda: np.array([x[0, :3] for x, _ in td.xy])
                mean_field = SimpleNamespace(mo_coeff=np.eye(18), e_tot=-1.0, TDA=lambda: td)
                with (
                    patch("pyscf_esd.cpu.build_molecule", return_value=mol),
                    patch("pyscf_esd.cpu._solve_reference", return_value=(mean_field, 0.0, ())),
                ):
                    point = solve_point(["He"], [[0, 0, 0]], settings)
                self.assertEqual(point.tda_route, expected_route)
                np.testing.assert_allclose(point.excitation_energies_hartree, known_roots[:2])
                np.testing.assert_allclose(point.tda_residuals, 0.0, atol=settings.td_residual_tol)
                np.testing.assert_allclose(point.transition_dipoles_au, point.amplitudes[:, 0, :3])
                self.assertTrue(np.all(td.converged))
                self.assertTrue(all(y == 0 for _, y in td.xy))
                self.assertEqual(td.nstates, settings.nstates)
                self.assertEqual(td.max_cycle, settings.max_cycle)
                self.assertEqual(td.conv_tol, settings.td_residual_tol * 0.2)
                td.kernel.assert_called_once_with()
                if expected_route == "dense_same_operator":
                    first_block, second_block = response.call_args_list[-3:-1]
                    np.testing.assert_array_equal(first_block.args[0], np.eye(17)[:16])
                    np.testing.assert_array_equal(second_block.args[0], np.eye(17)[16:])
                else:
                    self.assertEqual(response.call_count, 1)

    def test_dense_tda_refuses_nonhermitian_nonpositive_or_oversized_response(self):
        for operator, message in (
            (np.array([[1.0, 0.1], [0.0, 2.0]]), "not Hermitian"),
            (np.diag([-1.0, 2.0]), "must be positive"),
            (np.diag([0.0, 2.0]), "must be positive"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ElectronicConvergenceError, message):
                td = SimpleNamespace(nstates=1, gen_vind=lambda: (lambda x: x @ operator, None))
                _dense_tda_retry(td, 1, 2, 1e-9)

        td = SimpleNamespace(nstates=1, gen_vind=Mock())
        with self.assertRaisesRegex(ElectronicConvergenceError, "exceeds dense TDA limit"):
            _dense_tda_retry(td, 1, 257, 1e-9)
        td.gen_vind.assert_not_called()

    def test_excited_gradient_uses_selected_one_based_pyscf_root(self):
        gradient = Mock()
        gradient.kernel.return_value = np.zeros((2, 3))
        td = Mock()
        td.Gradients.return_value = gradient
        settings = Settings("sto-3g", "pbe0", nstates=2)
        point = ElectronicPoint(
            mol=SimpleNamespace(natm=2), settings=settings, mo_coeff=np.eye(3), nocc=1,
            amplitudes=np.zeros((2, 1, 2)), excitation_energies_hartree=np.array([0.1, 0.2]),
            transition_dipoles_au=np.zeros((2, 3)), ground_energy_hartree=-1.0, scf_residual=0.0,
            tda_residuals=np.zeros(2), attempts=(), _mean_field=Mock(), _tda=td,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            point.excited_gradient(-1)
        gradient.kernel.assert_not_called()
        with patch("pyscf_esd.cpu._checked_excited_gradient", return_value=np.zeros((2, 3))) as checked:
            point.excited_gradient(1)
        checked.assert_called_once_with(gradient, 2)
        self.assertEqual(gradient.cphf_conv_tol, settings.cphf_conv_tol)
        self.assertEqual(gradient.cphf_max_cycle, settings.cphf_max_cycle)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF required for the gradient adapter")
    def test_excited_gradient_cphf_override_is_local_and_restored_on_error(self):
        from pyscf.grad import tdrks
        from pyscf.scf import cphf as upstream_cphf

        def synthetic_grad_elec(gradient, xy, singlet, atmlst, max_memory, verbose):
            return globals()["cphf"].solve(
                *xy, max_cycle=gradient.cphf_max_cycle, tol=gradient.cphf_conv_tol,
            )[0]

        td = SimpleNamespace(
            verbose=0, stdout=None, mol=object(), chkfile=None, max_memory=1, _scf=object(),
        )
        gradient = tdrks.Gradients(td)
        other_gradient = tdrks.Gradients(td)
        native_solver = upstream_cphf.solve
        args = (object(), object(), object(), object())
        result = np.zeros((1, 3))
        failure = ElectronicConvergenceError("synthetic response failure")

        def kernel(*, state):
            self.assertEqual(state, 2)
            self.assertIs(upstream_cphf.solve, native_solver)
            self.assertIs(other_gradient.grad_elec.__func__, tdrks.Gradients.grad_elec)
            return gradient.grad_elec(args, True)

        with (
            patch.dict(globals(), {"cphf": upstream_cphf}),
            patch.object(tdrks, "grad_elec", synthetic_grad_elec),
            patch.object(gradient, "kernel", side_effect=kernel),
            patch("pyscf_esd.cpu._solve_excited_response", side_effect=[(result, None), failure]) as solve,
        ):
            self.assertIs(_checked_excited_gradient(gradient, 2), result)
            self.assertIs(gradient.grad_elec.__func__, tdrks.Gradients.grad_elec)
            with self.assertRaises(ElectronicConvergenceError) as raised:
                _checked_excited_gradient(gradient, 2)
            self.assertIs(raised.exception, failure)
        self.assertIs(gradient.grad_elec.__func__, tdrks.Gradients.grad_elec)
        self.assertIs(upstream_cphf.solve, native_solver)
        self.assertEqual(solve.call_count, 2)
        solve.assert_called_with(*args, max_cycle=gradient.cphf_max_cycle, tol=gradient.cphf_conv_tol)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF required for the Krylov solver")
    def test_excited_cphf_corrects_small_defect_and_checks_the_true_residual(self):
        from pyscf import lib
        from pyscf.scf import cphf

        energy = np.array([-1.0, -0.4, 0.3, 0.8])
        occupations = np.array([2, 2, 0, 0])
        gaps = energy[2:, None] - energy[:2]
        coupling = 0.08 * np.eye(4) + np.full((4, 4), 0.02)

        def response(trial):
            return coupling @ trial.reshape(4)

        exact = np.array([[0.15, -0.35], [0.45, 0.05]])
        rhs = -(gaps * exact + response(exact).reshape(gaps.shape))
        approximate = exact + np.array([[0.2, -0.1], [0.6, 0.05]]) * 1e-9
        tolerance = 1e-10
        args = (response, energy, occupations, rhs)
        options = dict(max_cycle=50, tol=tolerance)

        with patch.object(cphf, "solve", return_value=(approximate, None)) as native:
            corrected, first_order_energy = _solve_excited_response(*args, **options)
            native.assert_called_once_with(*args, **options, verbose=lib.logger.QUIET)
        defect = corrected + (rhs + response(corrected).reshape(gaps.shape)) / gaps
        self.assertLessEqual(np.linalg.norm(defect), tolerance)
        self.assertIsNone(first_order_energy)

        with (
            patch.object(cphf, "solve", return_value=(approximate, None)),
            patch.object(lib, "krylov", return_value=np.zeros(gaps.size)),
        ):
            with self.assertRaisesRegex(ElectronicConvergenceError, "CPHF residual .* exceeds tolerance"):
                _solve_excited_response(*args, **options)

        with (
            patch.object(cphf, "solve", return_value=(exact, None)),
            patch.object(lib, "krylov") as krylov,
        ):
            _solve_excited_response(*args, **options)
            krylov.assert_not_called()

    @unittest.skipUnless(find_spec("pyscf"), "PySCF required to inspect electronic setup")
    def test_real_pyscf_setup_is_direct_closed_shell_and_unpruned(self):
        from pyscf.scf.hf import SCF

        settings = Settings("sto-3g", "pbe0", nstates=1, grid_level=5)
        coordinates = [[0, 0, 0], [0, 0, 1.4]]
        mol = build_molecule(["H", "H"], coordinates, settings)
        np.testing.assert_allclose(mol.atom_coords(), coordinates)
        mf = _make_mean_field(mol, settings)
        self.assertTrue(mf.direct_scf)
        self.assertIs(mf.get_jk.__func__, SCF.get_jk)
        self.assertEqual(mf.grids.level, 5)
        self.assertIsNone(mf.grids.prune)
        self.assertEqual(mf.small_rho_cutoff, 0.0)
        self.assertIsNone(mf.chkfile)
        with self.assertRaisesRegex(ValueError, "even electron count"):
            build_molecule(["H"], [[0, 0, 0]], settings)
        with self.assertRaisesRegex(ValueError, "standard chemical elements"):
            build_molecule(["ghost-H", "H"], coordinates, settings)
        with self.assertRaisesRegex(ValueError, "nonlocal correlation or dispersion"):
            _make_mean_field(mol, Settings("sto-3g", "pbe0-d3bj"))


if __name__ == "__main__":
    unittest.main()
