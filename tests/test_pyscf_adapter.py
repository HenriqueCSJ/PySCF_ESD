"""Integration checks with PySCF Mole and analytic model derivatives."""

from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd import harmonic_spectrum
from pyscf_esd.pyscf_adapter import vertical_gradient


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required to test the adapter")
class VerticalGradientTests(unittest.TestCase):
    def setUp(self):
        from pyscf import gto

        self.mol = gto.M(atom="H 0 0 0; H 1 2 3", unit="Bohr", verbose=0)
        self.direction = np.array([1, 2, 3]) / np.sqrt(14)
        bond_gradient = np.array([-self.direction, self.direction])
        self.spring_constant = 0.5
        self.excited_slope = 0.02
        self.hessian = self.spring_constant * np.einsum("ia,jb->ijab", bond_gradient, bond_gradient)
        self.gradient = self.excited_slope * bond_gradient
        self.dipole = np.array([0.1, 0.2, -0.3])

    def parameters(self, hessian=None, vertical_excitation=0.15, masses_amu=None):
        return vertical_gradient(
            self.mol,
            self.hessian if hessian is None else hessian,
            self.gradient,
            vertical_excitation_hartree=vertical_excitation,
            transition_dipole_au=self.dipole,
            masses_amu=masses_amu,
        )

    def test_linear_diatomic_frequency_and_relaxation(self):
        from pyscf.data.nist import AMU2AU

        masses = np.array([1.0, 2.0])
        reduced_mass = AMU2AU / np.sum(1 / masses)
        expected_frequency = np.sqrt(self.spring_constant / reduced_mass)
        expected_shift = self.excited_slope * np.sqrt(reduced_mass) / self.spring_constant
        expected_relaxation = self.excited_slope**2 / (2 * self.spring_constant)

        result = self.parameters(masses_amu=masses)
        np.testing.assert_allclose(result.frequencies_hartree, [expected_frequency])
        # The normal-mode eigenvector may point along either end of the bond.
        np.testing.assert_allclose(np.abs(result.shift_au), [expected_shift])
        # Transform -K back to Cartesian coordinates to check the shift sign.
        displacement = -result.shift_au[0] * result.cartesian_mode_derivatives_au[0]
        bond_shift = np.dot(displacement[1] - displacement[0], self.direction)
        self.assertAlmostEqual(bond_shift, -self.excited_slope / self.spring_constant)
        self.assertAlmostEqual(result.reorganization_energy_hartree, expected_relaxation)
        self.assertAlmostEqual(result.gap_00_hartree, 0.15 - expected_relaxation)
        np.testing.assert_array_equal(result.transition_dipole_au, self.dipole)

    def test_uses_pyscf_average_mass_by_default(self):
        default = self.parameters()
        explicit = self.parameters(masses_amu=self.mol.atom_mass_list(isotope_avg=True))
        np.testing.assert_array_equal(default.frequencies_hartree, explicit.frequencies_hartree)
        np.testing.assert_array_equal(default.shift_au, explicit.shift_au)

    def test_vertical_gradient_parameters_produce_assigned_absorption_sticks(self):
        result = self.parameters()
        spectrum = harmonic_spectrum(
            result.frequencies_hartree, result.frequencies_hartree,
            gap_00_hartree=result.gap_00_hartree, shift_au=result.shift_au,
            transition_dipole_au=result.transition_dipole_au,
            initial_max_quanta=0, final_max_quanta=3,
        )
        omega = result.frequencies_hartree[0]
        np.testing.assert_allclose(spectrum.photon_energy_hartree,
                                   result.gap_00_hartree + omega * np.arange(4))
        self.assertAlmostEqual(spectrum.dipole_strength_au2[1] / spectrum.dipole_strength_au2[0],
                               result.reorganization_energy_hartree / omega)

    def test_nonlinear_molecule_has_three_vibrational_modes(self):
        from pyscf import gto

        mol = gto.M(atom="O 0 0 0; H 1 0 0; H 0 1 0", unit="Bohr", verbose=0)
        hessian = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
        result = vertical_gradient(
            mol,
            hessian,
            np.zeros((mol.natm, 3)),
            vertical_excitation_hartree=0.15,
            transition_dipole_au=self.dipole,
        )
        self.assertEqual(result.frequencies_hartree.shape, (3,))
        np.testing.assert_array_equal(result.shift_au, [0.0, 0.0, 0.0])
        self.assertEqual(result.gap_00_hartree, 0.15)

    def test_rejects_unstable_stationary_point(self):
        for scale in (-1, 0):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "imaginary or zero"):
                self.parameters(hessian=scale * self.hessian)

    def test_rejects_inconsistent_inputs(self):
        with self.assertRaisesRegex(ValueError, "ground Hessian must be real with shape"):
            self.parameters(hessian=np.zeros((6, 6)))
        with self.assertRaisesRegex(ValueError, "masses in amu must be positive"):
            self.parameters(masses_amu=[1, 0])
        with self.assertRaisesRegex(ValueError, "nonpositive 0-0 gap"):
            self.parameters(vertical_excitation=1e-4)


if __name__ == "__main__":
    unittest.main()
