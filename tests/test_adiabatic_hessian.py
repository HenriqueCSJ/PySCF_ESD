"""Analytic tests of the two-minimum PySCF adapter, without electronic runs."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.pyscf_adapter import adiabatic_hessian


def _molecule(coords, symbols=("O", "H", "H")):
    from pyscf import gto

    return gto.M(atom=list(zip(symbols, coords)), unit="Bohr", verbose=0)


def _proper_frame(seed):
    frame, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    frame[:, -1] *= np.sign(np.linalg.det(frame))
    return frame


def _vibrational_basis(coords, masses):
    centered = coords - np.average(coords, axis=0, weights=masses)
    displacements = [np.broadcast_to(axis, coords.shape) for axis in np.eye(3)]
    displacements.extend(np.cross(axis, centered) for axis in np.eye(3))
    constraints = np.array(displacements) * np.sqrt(masses)[None, :, None]
    _, _, right = np.linalg.svd(constraints.reshape(6, -1))
    return right[6:].T


def _native_hessian(freq, modes, sqrt_mass):
    cartesian = np.repeat(sqrt_mass, 3)[:, None] * modes
    matrix = (cartesian * freq**2) @ cartesian.T
    count = sqrt_mass.size
    return matrix.reshape(count, 3, count, 3).transpose(0, 2, 1, 3)


def _reframe(coords, hessian, frame, origin):
    return coords @ frame + origin, frame.T @ hessian @ frame


def _assignments(spectrum, *, reverse=False):
    result = {}
    for index, (initial, final) in enumerate(zip(spectrum.initial_quanta, spectrum.final_quanta)):
        pair = (tuple(initial), tuple(final))
        result[pair[::-1] if reverse else pair] = index
    return result


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required to test the adapter")
class AdiabaticHessianTests(unittest.TestCase):
    def setUp(self):
        from pyscf.data.nist import AMU2AU

        self.mass = np.array([15.995, 1.0078, 2.0141])
        self.sqrt_mass = np.sqrt(self.mass * AMU2AU)
        self.coords0 = np.array([[-0.05, 0.15, 0.12], [1.6, 0.1, 0.3], [-0.4, 1.4, -0.2]])
        self.omega0 = np.array([0.004, 0.006, 0.010])
        self.omega1 = np.array([0.005, 0.008, 0.011])
        self.modes0 = _vibrational_basis(self.coords0, self.mass) @ _proper_frame(4)
        self.hessian0 = _native_hessian(self.omega0, self.modes0, self.sqrt_mass)
        self.ground = _molecule(self.coords0)
        self.native_frame = _proper_frame(7)
        self.origin1 = np.array([1.2, -3.4, 5.6])
        self.coords1, self.modes1, hessian1 = self.excited_geometry(1.0)
        self.coords1_native, self.hessian1 = _reframe(
            self.coords1, hessian1, self.native_frame, self.origin1,
        )
        self.excited = _molecule(self.coords1_native)
        self.dipole = np.array([0.15, 0.25, -0.3])
        self.jacobian = np.random.default_rng(3).normal(scale=0.2, size=(3, 3, 3))

    def excited_geometry(self, distortion):
        # A displacement in the S0 vibrational space satisfies the Eckart
        # alignment condition in this common frame for both geometries.
        difference = (self.modes0 @ [2.0, -1.0, 0.5]).reshape(3, 3) / self.sqrt_mass[:, None]
        coords = self.coords0 + distortion * difference
        modes = _vibrational_basis(coords, self.mass) @ _proper_frame(5)
        hessian = _native_hessian(self.omega1, modes, self.sqrt_mass)
        return coords, modes, hessian

    def parameters(self, **kwargs):
        inputs = dict(
            ground_mol=self.ground, excited_mol=self.excited,
            ground_hessian=self.hessian0, excited_hessian=self.hessian1,
            ground_energy_hartree=-76.3, excited_energy_hartree=-76.1,
            ground_gradient=np.zeros((3, 3)), excited_gradient=np.zeros((3, 3)),
            masses_amu=self.mass,
        )
        inputs.update(kwargs)
        return adiabatic_hessian(**inputs)

    def test_distinct_minima_frequencies_alignment_and_signed_affine_relation(self):
        result = self.parameters()
        np.testing.assert_allclose(result.ground_frequencies_hartree, self.omega0)
        np.testing.assert_allclose(result.excited_frequencies_hartree, self.omega1)
        self.assertAlmostEqual(result.gap_00_hartree, 0.202)
        self.assertAlmostEqual(np.linalg.det(result.excited_alignment_rotation), 1.0)
        np.testing.assert_allclose(result.excited_alignment_rotation, self.native_frame.T)
        aligned = (self.coords1_native @ result.excited_alignment_rotation
                   + result.excited_alignment_translation_bohr)
        np.testing.assert_allclose(aligned, self.coords1, atol=2e-14)

        modes0 = (result.ground_cartesian_mode_derivatives_au * self.sqrt_mass[:, None]).reshape(3, -1).T
        modes1 = (result.excited_cartesian_mode_derivatives_au * self.sqrt_mass[:, None]).reshape(3, -1).T
        np.testing.assert_allclose(np.abs(self.modes0.T @ modes0), np.eye(3), atol=2e-14)
        np.testing.assert_allclose(np.abs(self.modes1.T @ modes1), np.eye(3), atol=2e-14)
        raw = modes1.T @ modes0
        np.testing.assert_allclose(result.subspace_singular_values, np.linalg.svd(raw, compute_uv=False))
        np.testing.assert_allclose(result.rotation.T @ result.rotation, np.eye(3), atol=2e-14)
        self.assertTrue(np.any(np.abs(result.rotation - np.diag(np.diag(result.rotation))) > 0.2))
        self.assertEqual(np.sign(np.linalg.det(result.rotation)), np.sign(np.linalg.det(raw)))
        self.assertAlmostEqual(result.rotation_correction_norm, np.linalg.norm(result.rotation - raw, ord=2))
        self.assertGreater(result.rotation_correction_norm, 1e-5)
        self.assertLess(result.rotation_correction_norm, 1e-3)
        displacement = ((self.coords0 - self.coords1) * self.sqrt_mass[:, None]).ravel()
        np.testing.assert_allclose(result.shift_au, modes1.T @ displacement, atol=2e-13)
        np.testing.assert_allclose(-result.rotation.T @ result.shift_au, -modes0.T @ displacement, atol=2e-13)
        self.assertAlmostEqual(result.displacement_norm_au, np.linalg.norm(displacement))
        self.assertLess(result.displacement_projection_residual_au, 2e-13)

    def test_reciprocal_signed_fc_ht_amplitudes(self):
        result = self.parameters()
        derivatives = np.einsum("kax,axc->kc", result.ground_cartesian_mode_derivatives_au, self.jacobian)
        for ht in (None, derivatives):
            with self.subTest(ht=ht is not None):
                inputs = dict(
                    transition_dipole_au=self.dipole, dipole_derivatives_au=ht,
                    initial_max_quanta=2, final_max_quanta=2, temperature_kelvin=1500.0,
                )
                absorption = result.spectrum(**inputs)
                emission = result.spectrum(process="emission", **inputs)
                forward = _assignments(absorption)
                backward = _assignments(emission, reverse=True)
                self.assertEqual(len(forward), 100)
                self.assertEqual(forward.keys(), backward.keys())
                for pair, i in forward.items():
                    j = backward[pair]
                    self.assertAlmostEqual(absorption.photon_energy_hartree[i], emission.photon_energy_hartree[j])
                    np.testing.assert_allclose(
                        absorption.transition_dipole_au[i], emission.transition_dipole_au[j], atol=2e-14,
                    )

        # Independent expectation from the Gaussian product for the 0-0 HT correction.
        precision = np.diag(self.omega0) + result.rotation.T @ (self.omega1[:, None] * result.rotation)
        mean = -np.linalg.solve(precision, result.rotation.T @ (self.omega1 * result.shift_au))
        ht_origin = result.spectrum(transition_dipole_au=self.dipole, dipole_derivatives_au=derivatives,
                                    final_max_quanta=0).transition_dipole_au[0]
        fc_origin = result.spectrum(transition_dipole_au=[1.0, 0.0, 0.0],
                                    final_max_quanta=0).transition_dipole_au[0, 0]
        np.testing.assert_allclose(ht_origin, fc_origin * (self.dipole + mean @ derivatives))

    def test_invariant_sticks_under_independent_proper_coordinate_frame_changes(self):
        result = self.parameters()
        frame0 = _proper_frame(8)
        coords0, hessian0 = _reframe(self.coords0, self.hessian0, frame0, [2.0, 3.0, -1.0])
        coords1, hessian1 = _reframe(self.coords1_native, self.hessian1, _proper_frame(9), [-1.0, 4.0, -2.0])
        other = self.parameters(
            ground_mol=_molecule(coords0), excited_mol=_molecule(coords1),
            ground_hessian=hessian0, excited_hessian=hessian1,
        )
        np.testing.assert_allclose(other.subspace_singular_values, result.subspace_singular_values, atol=2e-14)
        ground_phase = np.sign(np.einsum(
            "kax,kax,a->k", other.ground_cartesian_mode_derivatives_au,
            result.ground_cartesian_mode_derivatives_au @ frame0, self.mass,
        ))
        excited_phase = np.sign(np.einsum(
            "kax,kax,a->k", other.excited_cartesian_mode_derivatives_au,
            result.excited_cartesian_mode_derivatives_au @ frame0, self.mass,
        ))
        for direction in ("absorption", "emission"):
            with self.subTest(direction=direction):
                inputs = dict(process=direction, initial_max_quanta=1, final_max_quanta=2, temperature_kelvin=1500)
                reference = result.spectrum(
                    transition_dipole_au=self.dipole,
                    dipole_derivatives_au=np.einsum("kax,axc->kc", result.ground_cartesian_mode_derivatives_au,
                                                    self.jacobian),
                    **inputs,
                )
                changed = other.spectrum(
                    transition_dipole_au=self.dipole @ frame0,
                    dipole_derivatives_au=np.einsum("kax,axc->kc", other.ground_cartesian_mode_derivatives_au,
                                                    frame0.T @ self.jacobian @ frame0),
                    **inputs,
                )
                before, after = _assignments(reference), _assignments(changed)
                self.assertEqual(before.keys(), after.keys())
                phases = (ground_phase, excited_phase) if direction == "absorption" else (excited_phase, ground_phase)
                for pair, i in before.items():
                    j = after[pair]
                    self.assertAlmostEqual(reference.photon_energy_hartree[i], changed.photon_energy_hartree[j])
                    phase = np.prod(phases[0] ** pair[0]) * np.prod(phases[1] ** pair[1])
                    np.testing.assert_allclose(
                        changed.transition_dipole_au[j], phase * reference.transition_dipole_au[i] @ frame0,
                        atol=2e-14,
                    )

    def test_rejects_material_vibrational_subspace_loss_by_default(self):
        coords, _, hessian = self.excited_geometry(15.0)
        inputs = dict(excited_mol=_molecule(coords), excited_hessian=hessian)
        with self.assertRaisesRegex(ValueError, "vibrational subspace loss.*min singular value"):
            self.parameters(**inputs)
        result = self.parameters(**inputs, subspace_tolerance=0.5)
        self.assertGreater(result.rotation_correction_norm, 0.01)

    def test_rejects_invalid_surfaces(self):
        bad_gradient = np.zeros((3, 3))
        bad_gradient[0] = [7e-6, -7e-6, 7e-6]
        asymmetric = self.hessian1.copy()
        asymmetric[0, 1, 0, 0] += 0.01
        for replacement, message in (
            (dict(excited_gradient=bad_gradient), "excited geometry fails the stationarity tolerance"),
            (dict(ground_gradient=np.full((3, 3), np.nan)), "ground gradient must contain only finite"),
            (dict(excited_hessian=asymmetric), "excited Hessian must be symmetric"),
            (dict(excited_hessian=-self.hessian1), "excited Hessian has imaginary or zero"),
            (dict(ground_hessian=self.hessian0 * 0), "ground Hessian has imaginary or zero"),
            (dict(excited_energy_hartree=-76.5), "positive 0-0 gap"),
            (dict(masses_amu=[15.995, 0, 2.0141]), "masses in amu must be positive"),
            (dict(excited_mol=_molecule(self.coords1, ("H", "O", "H"))), "matching ordered elements"),
            (dict(subspace_tolerance=1.0), "subspace_tolerance must be in"),
            (dict(stationary_tolerance=0.0), "stationary_tolerance must be positive"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.parameters(**replacement)

    def test_default_mass_checks_molecule_isotope_overrides(self):
        shared = {index + 1: {"mass": mass} for index, mass in enumerate(self.mass)}
        ground = _molecule(self.coords0)
        excited = _molecule(self.coords1_native)
        ground.nucprop = shared
        excited.nucprop = shared
        implicit = self.parameters(ground_mol=ground, excited_mol=excited, masses_amu=None)
        explicit = self.parameters()
        np.testing.assert_array_equal(implicit.excited_frequencies_hartree, explicit.excited_frequencies_hartree)
        with self.assertRaisesRegex(ValueError, "matching atom-ordered masses"):
            self.parameters(ground_mol=ground, masses_amu=None)

    def test_linear_diatomic_rotational_rank_and_frequency(self):
        from pyscf.data.nist import AMU2AU

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
        direction = coords[1] / np.linalg.norm(coords[1])
        bond_derivative = np.array([-direction, direction])
        hessian = np.einsum("ia,jb->ijab", bond_derivative, bond_derivative)
        coords1, hessian1 = _reframe(coords * 1.1, hessian * 1.9, self.native_frame, self.origin1)
        masses = np.array([1.0, 2.0])
        result = self.parameters(
            ground_mol=_molecule(coords, ("H", "H")), excited_mol=_molecule(coords1, ("H", "H")),
            ground_hessian=hessian, excited_hessian=hessian1, masses_amu=masses,
            ground_gradient=np.zeros((2, 3)), excited_gradient=np.zeros((2, 3)),
        )
        reduced_mass = AMU2AU / np.sum(1 / masses)
        np.testing.assert_allclose(result.ground_frequencies_hartree, [np.sqrt(1 / reduced_mass)])
        np.testing.assert_allclose(result.excited_frequencies_hartree, [np.sqrt(1.9 / reduced_mass)])
        np.testing.assert_allclose(np.abs(result.shift_au), [0.1 * np.linalg.norm(coords[1]) * np.sqrt(reduced_mass)])

    def test_rejects_change_between_linear_and_nonlinear(self):
        linear_coords = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [-1.0, -2.0, -0.5]]
        identity = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
        with self.assertRaisesRegex(ValueError, "same vibrational dimension"):
            self.parameters(excited_mol=_molecule(linear_coords), excited_hessian=identity)

    def test_reflection_cannot_remove_the_minima_displacement(self):
        ground = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-0.3, -0.2, 0.002]])
        excited = ground * [1.0, 1.0, -1.0]
        hessian = np.eye(12).reshape(4, 3, 4, 3).transpose(0, 2, 1, 3)
        result = self.parameters(
            ground_mol=_molecule(ground, ("H",) * 4), excited_mol=_molecule(excited, ("H",) * 4),
            ground_hessian=hessian, excited_hessian=hessian, masses_amu=np.ones(4),
            ground_gradient=np.zeros((4, 3)), excited_gradient=np.zeros((4, 3)),
        )
        self.assertAlmostEqual(np.linalg.det(result.excited_alignment_rotation), 1.0)
        self.assertGreater(result.displacement_norm_au, 1e-2)

    def test_rank_deficient_overlap_is_rejected_with_relaxed_tolerance(self):
        ground = np.array([-3.0, 1.0, 2.0])[:, None] * [1, 0, 0]
        excited = np.array([1.0, -5.0, 4.0])[:, None] * [1, 0, 0]
        hessian = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
        with self.assertRaisesRegex(ValueError, "vibrational subspace loss"):
            self.parameters(
                ground_mol=_molecule(ground), excited_mol=_molecule(excited),
                ground_hessian=hessian, excited_hessian=hessian, masses_amu=np.ones(3),
                subspace_tolerance=np.nextafter(1.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
