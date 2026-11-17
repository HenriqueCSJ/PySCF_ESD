import unittest

import numpy as np

from pyscf_esd import (
    bond_rotation, map_transported_spectators, project_spectator_modes, project_torsion,
    transport_bond_spectators,
)


def ethane_geometry():
    """Two tetrahedral methyl groups about a carbon–carbon axis, in bohr."""
    angles = np.arange(3) * (2 * np.pi / 3)
    radius = 1.94
    top = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(3, 2.04)))
    bottom = top * [1, 1, -1]
    carbons = [[0, 0, -1.43], [0, 0, 1.43]]
    masses = np.array([12, 12, 1, 1, 1, 1, 1, 1], dtype=float) * 1822.888
    return np.vstack((carbons, bottom, top)), masses, radius


class RotorProjectionTests(unittest.TestCase):
    def test_reduced_inertia_of_equivalent_methyl_tops(self):
        xyz, masses, radius = ethane_geometry()
        atoms = (5, 6, 7)
        rotated, tangent = bond_rotation(xyz, (0, 1), atoms, 0.3)
        projection = project_torsion(rotated, masses, tangent)
        self.assertAlmostEqual(projection.inertia_me_bohr2, 3 * masses[5] * radius**2 / 2)
        self.assertEqual(projection.spectator_axes.shape, (24, 17))
        np.testing.assert_allclose(projection.rigid_axes.T @ projection.unit_torsion, 0., atol=1e-13)

    def test_mode_frequencies_and_inertia_follow_atom_reindexing(self):
        xyz, masses, _ = ethane_geometry()
        _, tangent = bond_rotation(xyz, (0, 1), (5, 6, 7), 0.)
        projection = project_torsion(xyz, masses, tangent)
        bath, torsion = projection.spectator_axes, projection.unit_torsion
        frequencies = np.linspace(.0005, .014, bath.shape[1])
        weighted = (bath * frequencies**2) @ bath.T + .0001 * np.outer(torsion, torsion)
        coupling = 0.2 * frequencies[-1] * .01
        weighted += coupling * (np.outer(torsion, bath[:, -1]) + np.outer(bath[:, -1], torsion))
        root_mass = np.repeat(np.sqrt(masses), 3)
        hessian = weighted * np.outer(root_mass, root_mass)
        result = project_spectator_modes(hessian, masses, projection)
        self.assertAlmostEqual(result.conditional_torsion_curvature_hartree,
                               0.96 * result.frozen_torsion_curvature_hartree)

        order = np.array([6, 2, 0, 5, 4, 1, 7, 3])
        coordinate_order = (3 * order[:, None] + np.arange(3)).ravel()
        selected_hessian = hessian[np.ix_(coordinate_order, coordinate_order)]
        _, reordered_tangent = bond_rotation(xyz[order], (2, 5), (3, 0, 6), 0.)
        permuted = project_torsion(xyz[order], masses[order], reordered_tangent)
        reordered = project_spectator_modes(selected_hessian, masses[order], permuted)
        np.testing.assert_allclose(result.frequencies_hartree, frequencies)
        np.testing.assert_allclose(reordered.frequencies_hartree, frequencies)
        self.assertAlmostEqual(permuted.inertia_me_bohr2, projection.inertia_me_bohr2)

    def test_common_arc_removes_artificial_spectator_displacement(self):
        xyz, masses, _ = ethane_geometry()
        group = (5, 6, 7)
        bond = (0, 1)
        end, end_tangent = bond_rotation(xyz, bond, group, np.pi / 3)
        _, start_tangent = bond_rotation(xyz, bond, group, 0.)
        start_modes = project_torsion(xyz, masses, start_tangent).spectator_axes
        end_modes = project_torsion(end, masses, end_tangent).spectator_axes
        half_start, quality_start = transport_bond_spectators(xyz, masses, start_modes, bond, group, np.pi / 6)
        half_end, quality_end = transport_bond_spectators(end, masses, end_modes, bond, group, -np.pi / 6)
        first, tangent = bond_rotation(xyz, bond, group, np.pi / 6)
        last, _ = bond_rotation(end, bond, group, -np.pi / 6)
        common = project_torsion(first, masses, tangent)
        mapped = map_transported_spectators(half_start, half_end, common, first - last, masses)
        self.assertGreater(min(quality_start, quality_end), .999)
        np.testing.assert_allclose(mapped.shift_au, 0., atol=1e-12)
        np.testing.assert_allclose(mapped.rotation.T @ mapped.rotation, np.eye(17), atol=1e-12)

    def test_rotation_of_entire_geometry_is_not_an_internal_coordinate(self):
        xyz, masses, _ = ethane_geometry()
        with self.assertRaisesRegex(ValueError, "independent internal"):
            project_torsion(xyz, masses, np.cross([0, 0, 1], xyz))

    def test_separate_cut_uses_only_the_auxiliary_spectator_curvature(self):
        from examples.assemble_separable_rotor import Cut, _harmonic_cut

        xyz, masses, _ = ethane_geometry()
        angles = 2 * np.pi * np.arange(12) / 12
        scan = [bond_rotation(xyz, (0, 1), (5, 6, 7), angle) for angle in angles]
        samples = np.stack([geometry for geometry, _ in scan])
        tangents = np.stack([tangent for _, tangent in scan])
        projection = project_torsion(samples[0], masses, tangents[0])
        bath = projection.spectator_axes
        frequencies = np.linspace(.0005, .014, bath.shape[1])
        weighted = (bath * frequencies**2) @ bath.T
        weighted -= .0001 * np.outer(projection.unit_torsion, projection.unit_torsion)
        root_mass = np.repeat(np.sqrt(masses), 3)
        hessian = weighted * np.outer(root_mass, root_mass)
        cut = Cut(
            angles, samples, tangents, np.zeros_like(angles), (1,), np.ones(1),
            {"stationary": {
                "angle_deg": 0.0, "coordinates_bohr": xyz,
                "hessian_eh_bohr2": hessian, "hessian_method": "synthetic auxiliary curvature",
                "hessian_is_same_surface": False,
            }},
        )

        result = _harmonic_cut(cut, masses, None)

        self.assertFalse(result.is_same_surface)
        np.testing.assert_allclose(result.frequencies, frequencies, atol=1e-13, rtol=0)
