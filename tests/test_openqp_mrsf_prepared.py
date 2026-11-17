"""Array fixtures for the prepared OpenQP MRSF checkpoint boundary."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.openqp_mrsf_prepared import (
    OPENQP_ENGINE_COMMIT,
    exact_openqp_mo_overlap,
    paired_openqp_triplet_gap_hartree,
    read_openqp_mrsf_point,
)


def make_checkpoint(coefficients, xyz, auxiliary_energy, root_gauge=(1, 1, 1), *, density=None):
    """Build the published and raw JSON views from independent input tensors."""
    nao = len(coefficients)
    inv = np.linalg.inv(coefficients)
    ao_metric = inv.T @ inv
    packed = np.tril_indices(nao)
    nao_pairs = len(packed[0])
    if density is None:
        density = np.zeros((nao, nao, 3, 3))
        density[:, :, 0, 1] = [[0.3, 0.5], [-0.4, -0.3]]
        density[:, :, 0, 2] = [[0.1, 0.2], [0.4, -0.1]]
        density[:, :, 1, 2] = [[0.2, -0.5], [0.0, -0.2]]
    density = density * np.outer(root_gauge, root_gauge)
    dipole_ao = np.array([[[0.5, 0.1], [0.1, 0.2]],
                          [[-0.3, 0.4], [0.4, 0.8]],
                          [[0.2, 0.2], [0.2, 0.3]]])
    dipole_ao = np.pad(dipole_ao, ((0, 0), (0, nao - 2), (0, nao - 2)))
    dipoles = np.zeros((3, 3, 3))
    for first, second in zip(*np.triu_indices(3, k=1)):
        density_ao = coefficients @ density[:, :, first, second] @ coefficients.T
        dipoles[:, first, second] = [-np.trace(density_ao @ op) for op in dipole_ao]

    response = np.array([-0.3, 0.05, 0.12])
    return {
        "json": {"scf_type": "rohf", "basis": "synthetic", "library": "", "nbf": nao,
                 "spherical_harmonics": False},
        "atoms": [1, 1],
        "coord": xyz,
        "energy": auxiliary_energy,
        "energies": [auxiliary_energy, *(auxiliary_energy + response)],
        "td_energies": response.tolist(),
        "td_trans_dipole": dipoles.tolist(),
        "OQP::td_energies": response.tolist(),
        "OQP::VEC_MO_A": coefficients.T.tolist(),
        "OQP::SM": ao_metric[packed].tolist(),
        "OQP::td_dip_ao": dipole_ao[:, packed[0], packed[1]].reshape(nao_pairs, 3).tolist(),
        "OQP::td_trans_density_mo": density.flatten(order="F").reshape(nao, nao, 9).tolist(),
        "OQP::td_trans_dipole": dipoles.flatten(order="F").reshape(3, 3, 3).tolist(),
    }


def two_somo_densities(n_orbitals, n_core):
    # Open-shell combination and both closed-shell determinants. These
    # elements follow from occupation numbers and single substitutions.
    density = np.zeros((n_orbitals, n_orbitals, 3, 3))
    p, q = n_core, n_core + 1
    density[p, p, 1, 1] = density[q, q, 2, 2] = 1
    density[q, q, 1, 1] = density[p, p, 2, 2] = -1
    density[p, q, 0, 1] = density[q, p, 1, 0] = -np.sqrt(2)
    density[q, p, 0, 2] = density[p, q, 2, 0] = np.sqrt(2)
    return density


class OpenQPMRSFPreparedTests(unittest.TestCase):
    def setUp(self):
        self.left = np.array([[1.0, 0.5], [0.0, 1.0]])
        self.right = np.array([[0.8, 0.2], [0.5, -1.0]])
        self.xyz_left = [0.0, 0.0, -0.7, 0.0, 0.0, 0.7]
        self.xyz_right = [0.0, 0.0, -0.8, 0.0, 0.0, 0.8]

    def test_checkpoint_uses_absolute_root_energies_and_signed_dipoles(self):
        raw = make_checkpoint(self.left, self.xyz_left, -75.0)
        point = read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT)
        changed = make_checkpoint(self.left, self.xyz_left, -75.1, (-1, 1, -1))
        changed_point = read_openqp_mrsf_point(changed, engine_commit=OPENQP_ENGINE_COMMIT)
        np.testing.assert_allclose(point.root_energies_hartree, [-75.3, -74.95, -74.88])
        np.testing.assert_allclose(changed_point.root_energies_hartree - point.root_energies_hartree, -0.1)
        np.testing.assert_allclose(point.mo_coefficients, self.left)
        np.testing.assert_allclose(point.coordinates_bohr, np.reshape(self.xyz_left, (-1, 3)))

        raw_dipole = point.transition_dipole(0, 1)
        np.testing.assert_allclose(raw_dipole, [-0.0575, 0.4025, 0.075])
        np.testing.assert_allclose(changed_point.transition_dipole(1, 0), -raw_dipole)
        np.testing.assert_allclose(changed_point.transition_dipole(0, 2), point.transition_dipole(0, 2))
        with self.assertRaisesRegex(ValueError, "distinct"):
            point.transition_dipole(0, 0)

    def test_checkpoint_rejects_inconsistent_root_order_and_unverified_phase(self):
        mutations = (
            ("energies", [-75.0, -74.95, -75.3, -74.88], "response-root order"),
            ("energy", 0.0, "response-root order"),
            ("OQP::SM", [1.0, 0.0, 1.0], "orthonormal"),
            ("state_tracking", {}, "untransported"),
            ("OQP::mo_tracking_phase", [1.0, -1.0], "untransported"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                raw = make_checkpoint(self.left, self.xyz_left, -75.0)
                raw[key] = value
                read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT)

        raw = make_checkpoint(self.left, self.xyz_left, -75.0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_openqp_mrsf_point(raw, engine_commit="v1.3.1")
        raw["td_trans_dipole"] = (-np.asarray(raw["td_trans_dipole"])).tolist()
        with self.assertRaisesRegex(ValueError, "signed raw tag"):
            read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT)
        raw["OQP::td_trans_dipole"] = (-np.asarray(raw["OQP::td_trans_dipole"])).tolist()
        with self.assertRaisesRegex(ValueError, "state-interaction density"):
            read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT)
        del raw["OQP::td_trans_density_mo"]
        with self.assertRaisesRegex(ValueError, "missing OQP::td_trans_density_mo"):
            read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT)

    def test_exact_mo_contraction_and_native_overlap_checks(self):
        left = read_openqp_mrsf_point(make_checkpoint(self.left, self.xyz_left, -75.0),
                                    engine_commit=OPENQP_ENGINE_COMMIT)
        right = read_openqp_mrsf_point(make_checkpoint(self.right, self.xyz_right, -75.1),
                                     engine_commit=OPENQP_ENGINE_COMMIT)
        expected = np.array([[0.54, -0.20], [0.16, -0.75]])
        cross = np.linalg.inv(self.left.T) @ expected @ np.linalg.inv(self.right)
        result = exact_openqp_mo_overlap(left, right, cross, native_current_by_previous_mo_overlap=expected.T)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(exact_openqp_mo_overlap(right, left, cross.T), expected.T)
        inv_left = np.linalg.inv(self.left)
        np.testing.assert_allclose(exact_openqp_mo_overlap(left, left, inv_left.T @ inv_left), np.eye(2))

        native = expected.T
        for invalid in (expected, native / np.linalg.norm(native, axis=0), -native):
            with self.assertRaisesRegex(ValueError, "signed raw AO contraction"):
                exact_openqp_mo_overlap(left, right, cross, native_current_by_previous_mo_overlap=invalid)

        with self.assertRaisesRegex(ValueError, "incompatible with orthonormal orbitals"):
            exact_openqp_mo_overlap(left, right, cross * 2)

    def test_signed_state_overlap_uses_checkpoint_root_axes(self):
        expected = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
        vectors = np.eye(4)[:, :3]
        density = two_somo_densities(2, 0)
        changed_density = np.einsum("ir,abij,js->abrs", expected, density, expected)
        left_raw = make_checkpoint(self.left, self.xyz_left, -75.0, density=density)
        right_raw = make_checkpoint(self.right, self.xyz_right, -75.1, density=changed_density)
        left_raw["OQP::td_bvec_mo"] = vectors.tolist()
        right_raw["OQP::td_bvec_mo"] = (vectors @ expected).tolist()
        left = read_openqp_mrsf_point(left_raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=0)
        right = read_openqp_mrsf_point(right_raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=0)
        cross = np.linalg.inv(self.left.T) @ np.linalg.inv(self.right)
        np.testing.assert_allclose(left.signed_state_overlap(right, cross), expected, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "read both checkpoints"):
            dipoles_only = read_openqp_mrsf_point(right_raw, engine_commit=OPENQP_ENGINE_COMMIT)
            left.signed_state_overlap(dipoles_only, cross)
        right_raw["OQP::td_bvec_mo"] = ((vectors @ expected) * [-1, 1, 1]).tolist()
        with self.assertRaisesRegex(ValueError, "disagree with the signed density"):
            read_openqp_mrsf_point(right_raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=0)
        right_raw["OQP::td_bvec_mo"] = (vectors @ expected).T.tolist()
        with self.assertRaisesRegex(ValueError, "columns"):
            read_openqp_mrsf_point(right_raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=0)

    def test_triplet_gap_requires_same_reference_and_preserves_response_order(self):
        orbitals = np.eye(4)
        density = two_somo_densities(4, 1)
        raw = make_checkpoint(orbitals, self.xyz_left, -75.0, density=density)
        raw["OQP::td_bvec_mo"] = np.eye(9)[:, (1, 2, 4)].tolist()
        singlet = read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=1)
        swapped_somos = orbitals[:, [0, 2, 1, 3]] * [-1, 1, 1, 1]
        triplet = make_checkpoint(swapped_somos, self.xyz_left, -75.0, density=density)
        triplet["molecular_orbitals"] = {
            "alpha": {"occupancies": [1, 1, 1, 0]},
            "beta": {"occupancies": [1, 0, 0, 0]},
        }
        # Use a small upper gap to check that zero-based roots 0 and 1
        # are used even when roots 1 and 2 are much closer.
        offsets = np.array([0.030565689422728776, 0.09187445559712014, 0.09694581973927008])
        triplet.update(energies=[-75, *(-75 + offsets)],
                       td_energies=offsets.tolist(), **{"OQP::td_energies": offsets.tolist()})
        for key in ("OQP::td_trans_density_mo", "OQP::td_trans_dipole", "td_trans_dipole"):
            del triplet[key]
        gap = paired_openqp_triplet_gap_hartree(singlet, triplet, engine_commit=OPENQP_ENGINE_COMMIT)
        self.assertAlmostEqual(gap, 0.061308766174391364, places=15)
        exchanged = orbitals[:, [1, 0, 2, 3]]
        wrong_reference = {**triplet, "OQP::VEC_MO_A": exchanged.T.tolist()}
        with self.assertRaisesRegex(ValueError, "different auxiliary determinant"):
            paired_openqp_triplet_gap_hartree(singlet, wrong_reference, engine_commit=OPENQP_ENGINE_COMMIT)
        with self.assertRaisesRegex(ValueError, "same geometry"):
            paired_openqp_triplet_gap_hartree(
                singlet, {**triplet, "coord": self.xyz_right}, engine_commit=OPENQP_ENGINE_COMMIT,
            )
        with self.assertRaisesRegex(ValueError, "portable occupations"):
            paired_openqp_triplet_gap_hartree(
                singlet, {**triplet, "molecular_orbitals": {}}, engine_commit=OPENQP_ENGINE_COMMIT,
            )
        order = [1, 0, 2]
        unsorted = {**triplet, "energies": [-75, *(-75 + offsets[order])],
                    "td_energies": offsets[order], "OQP::td_energies": offsets[order]}
        with self.assertRaisesRegex(ValueError, "ascending raw root order"):
            paired_openqp_triplet_gap_hartree(singlet, unsorted, engine_commit=OPENQP_ENGINE_COMMIT)

        current = read_openqp_mrsf_point(
            {**raw, "coord": self.xyz_right}, engine_commit=OPENQP_ENGINE_COMMIT, n_core=1,
        )
        link = singlet.auxiliary_reference_link(current, exchanged, minimum_singular_value=0.9)
        self.assertFalse(link.passes_overlap_screen)
        self.assertEqual(min(link.alpha_occupied_singular_values), 1.0)

    @unittest.skipUnless(find_spec("pyscf"), "PySCF is required for the AO integral adapter")
    def test_compute_cross_ao_and_state_overlap_from_saved_basis(self):
        exponents = np.array([1.1, 0.7])
        scales = np.array([0.8, 0.6])
        left = np.reshape(self.xyz_left, (2, 3))
        right = np.reshape(self.xyz_right, (2, 3)) + [0.12, 0.09, -0.03]

        def analytical_cross(bra, ket):
            a, b = exponents[:, None], exponents[None, :]
            distance_squared = np.sum((bra[:, None, :] - ket[None, :, :]) ** 2, axis=2)
            normalized = (2 * np.sqrt(a * b) / (a + b)) ** 1.5
            return np.outer(scales, scales) * normalized * np.exp(-(a * b / (a + b)) * distance_squared)

        basis = {
            "format": "OpenQP portable basis v1", "coefficient_convention": "molden",
            "basis_function_order": "molden", "spherical_harmonics": False, "nbf": 2,
            "shells": [
                {"atom_index": atom, "angular_momentum": 0, "shell": "s",
                 "exponents": [exponent], "coefficients": [scale]}
                for atom, (exponent, scale) in enumerate(zip(exponents, scales))
            ],
        }
        points = []
        for xyz in (left, right):
            eigenvalues, orbitals = np.linalg.eigh(analytical_cross(xyz, xyz))
            mo = (orbitals / np.sqrt(eigenvalues)) @ orbitals.T
            raw = make_checkpoint(mo, xyz.ravel(), -75, density=two_somo_densities(2, 0))
            raw["OQP::td_bvec_mo"] = np.eye(4)[:, :3].tolist()
            raw["basis_set"] = basis
            points.append(read_openqp_mrsf_point(raw, engine_commit=OPENQP_ENGINE_COMMIT, n_core=0))

        before, after = points
        expected = analytical_cross(left, right)
        cross = before.cross_ao_overlap(after)
        np.testing.assert_allclose(cross, expected, atol=2e-15, rtol=0)
        np.testing.assert_allclose(
            before.signed_state_overlap(after), before.signed_state_overlap(after, expected),
            atol=5e-15, rtol=0,
        )
        self.assertTrue(before.auxiliary_reference_link(after, minimum_singular_value=0.9).passes_overlap_screen)
