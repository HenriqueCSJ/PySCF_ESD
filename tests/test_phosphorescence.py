"""Synthetic controls for the spin–orbit phosphorescence state interaction."""

import unittest

import numpy as np

from pyscf_esd.phosphorescence import spin_orbit_phosphorescence
from pyscf_esd.spectral_properties import _EINSTEIN_A_FACTOR


def prepared_model():
    # S0, S1, the three T1 components, and three higher triplet components.
    energy = np.array([0., .30, .20, .20, .20, .40, .40, .40])
    spin_orbit = np.zeros((8, 8), dtype=complex)
    spin_orbit[0, 5] = spin_orbit[5, 0] = .01
    spin_orbit[1, 2] = spin_orbit[2, 1] = .006
    dipole = np.zeros((8, 8, 3), dtype=complex)
    dipole[0, 1, 0] = dipole[1, 0, 0] = .4
    for lower, higher in zip((2, 3, 4), (5, 6, 7), strict=True):
        dipole[higher, lower, 0] = dipole[lower, higher, 0] = -.7
    return energy, spin_orbit, dipole


def calculate(energy, spin_orbit, dipole, temperature=None):
    return spin_orbit_phosphorescence(
        energy, spin_orbit, dipole, ground_index=0, triplet_indices=(2, 3, 4),
        temperature_kelvin=temperature,
    )


class PhosphorescenceTests(unittest.TestCase):
    def test_without_soc_the_spin_forbidden_transition_is_dark(self):
        energy, spin_orbit, dipole = prepared_model()
        result = calculate(energy, np.zeros_like(spin_orbit), dipole, 298.)
        np.testing.assert_array_equal(result.sublevel_vacuum_rates_s_inverse, [0., 0., 0.])
        np.testing.assert_allclose(result.sublevel_populations, [1 / 3] * 3, rtol=0, atol=1e-15)
        self.assertEqual(result.vacuum_radiative_rate_s_inverse, 0.)
        self.assertIsNone(result.vacuum_radiative_lifetime_s)

    def test_exact_two_level_rotations_include_both_borrowing_amplitudes(self):
        energy, spin_orbit, dipole = prepared_model()
        result = calculate(energy, spin_orbit, dipole, 0.)
        g, t = .5 * np.arctan(2 * .01 / .4), .5 * np.arctan(2 * .006 / .1)
        amplitude = -np.cos(g) * np.sin(t) * .4 + np.sin(g) * np.cos(t) * .7
        ground = (.4 - np.hypot(.4, .02)) / 2
        emitting = (.5 - np.hypot(.1, .012)) / 2
        rate = _EINSTEIN_A_FACTOR * (emitting - ground)**3 * amplitude**2
        np.testing.assert_allclose(result.photon_energy_hartree, [emitting - ground, .2 - ground, .2 - ground])
        np.testing.assert_allclose(result.dipole_strength_au2, [amplitude**2, 0., 0.], rtol=1e-11, atol=1e-15)
        np.testing.assert_array_equal(result.sublevel_populations, [1., 0., 0.])
        np.testing.assert_allclose(result.vacuum_radiative_rate_s_inverse, rate, rtol=1e-12)
        self.assertAlmostEqual(result.vacuum_radiative_lifetime_s * rate, 1.)

    def test_phase_and_triplet_basis_rotation_preserve_the_ensemble_rate(self):
        energy, spin_orbit, dipole = prepared_model()
        phases = np.exp(1j * np.linspace(-.92, 1.41, len(energy)))
        basis = np.diag(phases)
        basis[2:4, 2:4] = basis[2:4, 2:4] @ np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
        # The spin-free Hamiltonian is unchanged by this unitary rotation
        # because the two triplet components have the same spin-free energy.
        reference = calculate(energy, spin_orbit, dipole, 298.)
        dipole += np.eye(len(energy))[:, :, None] * [.2, -.35, .7]
        changed = calculate(
            energy - 200, basis.conj().T @ spin_orbit @ basis,
            np.einsum("ai,abc,bj->ijc", basis.conj(), dipole, basis), 298.,
        )
        np.testing.assert_allclose(changed.photon_energy_hartree, reference.photon_energy_hartree, atol=1e-13)
        np.testing.assert_allclose(changed.sublevel_populations, reference.sublevel_populations, atol=1e-12)
        np.testing.assert_allclose(changed.vacuum_radiative_rate_s_inverse,
                                   reference.vacuum_radiative_rate_s_inverse, rtol=1e-10)

    def test_rejects_a_nonhermitian_operator_or_an_unisolated_triplet(self):
        energy, spin_orbit, dipole = prepared_model()
        spin_orbit[0, 5] += 1j * .02
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            calculate(energy, spin_orbit, dipole)
        energy[1] = energy[2]
        with self.assertRaisesRegex(ValueError, "isolated"):
            calculate(energy, np.zeros_like(spin_orbit), dipole)

    def test_sublevel_rates_are_available_without_an_ensemble_assumption(self):
        result = calculate(*prepared_model())
        self.assertTrue(np.max(result.sublevel_vacuum_rates_s_inverse) > 0)
        self.assertIsNone(result.sublevel_populations)
        self.assertIsNone(result.vacuum_radiative_rate_s_inverse)

    def test_rejects_spin_mixed_data_in_the_spin_free_inputs(self):
        energy, spin_orbit, dipole = prepared_model()
        energy[3] += .01
        with self.assertRaisesRegex(ValueError, "share one energy"):
            calculate(energy, spin_orbit, dipole)
        energy[3] = energy[2]
        dipole[0, 2, 0] = dipole[2, 0, 0] = .01
        with self.assertRaisesRegex(ValueError, "dipoles must be spin-free"):
            calculate(energy, spin_orbit, dipole)

    def test_a_small_nonzero_splitting_retains_its_finite_temperature_weight(self):
        energy, spin_orbit, dipole = prepared_model()
        spin_orbit.fill(0)
        spin_orbit[3, 3] = 5e-11
        spin_orbit[4, 4] = 1e-2
        result = calculate(energy, spin_orbit, dipole, 1e-6)
        self.assertLess(result.sublevel_populations[1], 1e-6)
        self.assertGreater(result.sublevel_populations[1], 1e-8)
        np.testing.assert_array_equal(result.sublevel_rates_resolved, [False, False, True])
