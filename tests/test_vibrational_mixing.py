"""Analytic checks for mixing vibrational energies and signed dipoles."""

import unittest

import numpy as np

from pyscf_esd import VibrationalMixingResult, mix_vibrational_states


class VibrationalMixingTests(unittest.TestCase):
    def test_detuned_two_state_block_has_the_exact_energies_and_borrowed_strength(self):
        center, half_detuning, coupling = -0.5, 0.001, 0.0007
        hamiltonian = np.array([
            [center - half_detuning, coupling],
            [coupling, center + half_detuning],
        ])
        result = mix_vibrational_states(hamiltonian, [[2, 0, 0], [0, 0, 0]])
        splitting = np.hypot(half_detuning, coupling)
        bright_fractions = (1 + np.array([1, -1]) * half_detuning / splitting) / 2

        self.assertIsInstance(result, VibrationalMixingResult)
        np.testing.assert_allclose(
            result.energies_hartree, center + np.array([-1, 1]) * splitting, rtol=0, atol=1e-15,
        )
        np.testing.assert_allclose(result.dipole_strength_au2, 4 * bright_fractions, rtol=1e-13)
        np.testing.assert_allclose(
            hamiltonian @ result.eigenvectors,
            result.eigenvectors * result.energies_hartree,
            rtol=0, atol=1e-15,
        )
        np.testing.assert_allclose(
            result.transition_dipole_au[:, 0], 2 * result.eigenvectors[0], rtol=0, atol=1e-15,
        )

    def test_zero_couplings_only_reorder_batched_amplitudes(self):
        energies = np.array([0.3, 0.1, 0.2])
        dipoles = np.arange(-18.0, 18.0).reshape(2, 2, 3, 3)
        order = np.argsort(energies)
        result = mix_vibrational_states(np.diag(energies), dipoles)

        np.testing.assert_array_equal(result.energies_hartree, energies[order])
        np.testing.assert_array_equal(np.abs(result.eigenvectors), np.eye(3)[:, order])
        phases = result.eigenvectors[order, np.arange(len(order))]
        np.testing.assert_array_equal(result.transition_dipole_au, dipoles[..., order, :] * phases[:, None])
        np.testing.assert_array_equal(result.dipole_strength_au2, np.sum(dipoles**2, axis=-1)[..., order])

    def test_equal_dipoles_cancel_coherently_and_covary_under_a_basis_phase_change(self):
        hamiltonian = np.array([[0.02, 0.001], [0.001, 0.02]])
        dipoles = np.array([[1, -2, 0], [1, -2, 0]])
        result = mix_vibrational_states(hamiltonian, dipoles)
        phases = np.diag([1, -1])
        changed = mix_vibrational_states(phases @ hamiltonian @ phases, phases @ dipoles)

        np.testing.assert_allclose(result.dipole_strength_au2, [0, 10], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(changed.energies_hartree, result.energies_hartree)
        np.testing.assert_allclose(changed.dipole_strength_au2, result.dipole_strength_au2, atol=1e-14)
        gauge = np.sign(np.sum(result.eigenvectors * (phases @ changed.eigenvectors), axis=0))
        np.testing.assert_allclose(
            changed.transition_dipole_au, gauge[:, None] * result.transition_dipole_au, atol=1e-14,
        )

    def test_common_energy_offset_preserves_resonant_mixing(self):
        coupling = 1e-10
        hamiltonian = [[1e8, coupling], [coupling, 1e8]]
        result = mix_vibrational_states(hamiltonian, [[1, 0, 0], [0, 0, 0]])

        np.testing.assert_allclose(result.dipole_strength_au2, [0.5, 0.5], atol=1e-14)

    def test_invalid_hamiltonian_and_dipole_inputs(self):
        hamiltonian = np.eye(2)
        dipoles = np.zeros((2, 3))
        for invalid, message in (
            (np.empty((0, 0)), "nonempty square"),
            ([[1, 0, 0], [0, 1, 0]], "nonempty square"),
            ([[0, 0], [1e-20, 0]], "symmetric"),
            ([[0, 1j], [-1j, 0]], "real numeric"),
            ([[np.nan, 0], [0, 0]], "finite"),
        ):
            with self.subTest(hamiltonian=invalid), self.assertRaisesRegex(ValueError, message):
                mix_vibrational_states(invalid, dipoles)
        for invalid, message in (
            ([0, 1, 0], "shape"),
            (np.zeros((2, 4)), "shape"),
            (np.ones((2, 3), dtype=complex), "real numeric"),
            ([[0, 0, 0], [0, np.inf, 0]], "finite"),
        ):
            with self.subTest(dipoles=invalid), self.assertRaisesRegex(ValueError, message):
                mix_vibrational_states(hamiltonian, invalid)

    def test_unrepresentable_output_raises(self):
        with self.assertRaisesRegex(ValueError, "numerical range"):
            mix_vibrational_states([[0]], [[1e200, 0, 0]])
        with self.assertRaisesRegex(ValueError, "numerical range"):
            mix_vibrational_states(np.full((2, 2), 1e308), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
