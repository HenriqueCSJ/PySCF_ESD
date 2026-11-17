"""Synthetic checks of the restricted TDA one-electron matrix elements."""

import unittest

import numpy as np

from pyscf_esd.tda_state_interaction import assemble_tda_state_interaction


_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def _excite(bitstring, created, removed):
    """Apply a†_created a_removed to an occupation-number bitstring."""
    if not bitstring & (1 << removed):
        return bitstring, 0
    parity = (bitstring & ((1 << removed) - 1)).bit_count()
    bitstring ^= 1 << removed
    if bitstring & (1 << created):
        return bitstring, 0
    parity += (bitstring & ((1 << created) - 1)).bit_count()
    return bitstring | (1 << created), (-1) ** parity


def _fock_states(singlets, triplets):
    """Form tiny synthetic states directly in the occupation-number basis."""
    nocc, nvir = singlets.shape[1:]
    nspinorb = 2 * (nocc + nvir)
    reference = (1 << (2 * nocc)) - 1
    states = np.zeros((1 << nspinorb, 1 + len(singlets) + 3 * len(triplets)), dtype=complex)
    states[reference, 0] = 1
    excitations = [(singlet, np.eye(2)) for singlet in singlets]
    excitations += [(triplet, pauli) for triplet in triplets for pauli in _PAULI]
    for state, (amplitude, spin) in enumerate(excitations, 1):
        for i, a, alpha, beta in np.ndindex(nocc, nvir, 2, 2):
            bit, sign = _excite(reference, 2 * (nocc + a) + alpha, 2 * i + beta)
            states[bit, state] += sign * amplitude[i, a] * spin[alpha, beta] / np.sqrt(2)
    return states


def _fock_one_body(states, operator):
    """Apply the second-quantized operator without a particle-hole formula."""
    transformed = np.zeros_like(states)
    for bit in np.flatnonzero(np.any(states, axis=1)):
        for p, q in np.ndindex(operator.shape):
            dest, sign = _excite(int(bit), p, q)
            transformed[dest] += sign * operator[p, q] * states[bit]
    return states.conj().T @ transformed


class TDAStateInteractionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(34)
        self.singlets = np.linalg.qr(rng.normal(size=(4, 2)))[0].T.reshape(2, 2, 2)
        self.triplets = np.linalg.qr(rng.normal(size=(4, 2)))[0].T.reshape(2, 2, 2)
        dipole = rng.normal(size=(3, 4, 4))
        soc = rng.normal(size=(3, 4, 4))
        self.dipole_mo = (dipole + dipole.swapaxes(1, 2)) / 2
        self.soc_mo = 1j * (soc - soc.swapaxes(1, 2)) / 2

    def test_spin_orbital_oracle(self):
        states = _fock_states(self.singlets, self.triplets)
        np.testing.assert_allclose(states.conj().T @ states, np.eye(states.shape[1]), atol=1e-14)
        expected_dipole = [_fock_one_body(states, np.kron(position, np.eye(2))) for position in self.dipole_mo]
        soc_spin_orbital = sum(np.kron(spatial, spin / 2) for spatial, spin in zip(self.soc_mo, _PAULI))
        expected_soc = _fock_one_body(states, soc_spin_orbital)

        result = assemble_tda_state_interaction(self.singlets, self.triplets, self.dipole_mo, self.soc_mo)

        np.testing.assert_allclose(result.dipole, expected_dipole, atol=1e-14)
        np.testing.assert_allclose(result.soc, expected_soc, atol=1e-14)
        np.testing.assert_array_equal(result.dipole[:, :3, 3:], 0)
        # Singlets are time-reversal even and Cartesian triplets are odd.
        time_reversal_phase = np.r_[np.ones(3), -np.ones(6)]
        np.testing.assert_allclose(
            result.soc, time_reversal_phase[:, None] * result.soc.conj() * time_reversal_phase, atol=1e-14,
        )
        for i in (3, 6):
            triplet = result.dipole[:, i:i + 3, i:i + 3]
            np.testing.assert_array_equal(triplet, triplet[:, :1, :1] * np.eye(3))
            np.testing.assert_allclose(result.soc[i:i + 3, i:i + 3], 0, atol=1e-14)

    def test_reference_trace_and_orbital_identity_shift(self):
        result = assemble_tda_state_interaction(self.singlets, self.triplets, self.dipole_mo, self.soc_mo)
        ground = 2 * np.trace(self.dipole_mo[:, :2, :2], axis1=1, axis2=2)
        np.testing.assert_array_equal(result.dipole[:, 0, 0], ground)
        identity = np.eye(result.soc.shape[0])
        shift = np.array([-0.25, 0.5, 2.5])[:, None, None]
        shifted_mo = self.dipole_mo + shift * np.eye(4)

        shifted_ground = assemble_tda_state_interaction(self.singlets, self.triplets, shifted_mo, self.soc_mo)
        fixed_ground = assemble_tda_state_interaction(
            self.singlets, self.triplets, shifted_mo, self.soc_mo, dipole_reference=ground,
        )

        np.testing.assert_allclose(shifted_ground.dipole - result.dipole, 4 * shift * identity, atol=1e-14)
        np.testing.assert_allclose(fixed_ground.dipole, result.dipole, atol=1e-14)
        np.testing.assert_array_equal(shifted_ground.soc, result.soc)

    def test_ground_transition_spin_factors_and_phase_congruence(self):
        result = assemble_tda_state_interaction(self.singlets, self.triplets, self.dipole_mo, self.soc_mo)
        expected_singlet = np.sqrt(2) * np.einsum("kia,sia->ks", self.dipole_mo[:, :2, 2:], self.singlets)
        expected_triplet = np.einsum("kia,tia->tk", self.soc_mo[:, :2, 2:], self.triplets) / np.sqrt(2)
        np.testing.assert_allclose(result.dipole[:, 0, 1:3], expected_singlet)
        np.testing.assert_allclose(result.soc[0, 3:].reshape(2, 3), expected_triplet)
        np.testing.assert_array_equal(result.soc[:3, :3], 0)

        singlet_sign = np.array([-1, 1])
        triplet_sign = np.array([1, -1])
        signs = np.r_[1, singlet_sign, np.repeat(triplet_sign, 3)]
        phase = signs[:, None] * signs
        changed = assemble_tda_state_interaction(
            singlet_sign[:, None, None] * self.singlets, triplet_sign[:, None, None] * self.triplets,
            self.dipole_mo, self.soc_mo,
        )
        np.testing.assert_allclose(changed.dipole, result.dipole * phase, atol=1e-14)
        np.testing.assert_allclose(changed.soc, result.soc * phase, atol=1e-14)

    def test_empty_manifolds_and_input_validation(self):
        ground = assemble_tda_state_interaction(self.singlets[:0], self.triplets[:0], self.dipole_mo, self.soc_mo)
        self.assertEqual(ground.dipole.shape, (3, 1, 1))
        np.testing.assert_array_equal(ground.soc, [[0]])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            assemble_tda_state_interaction(self.singlets / np.sqrt(2), self.triplets, self.dipole_mo, self.soc_mo)
        with self.assertRaisesRegex(ValueError, "SOC MO operator must be Hermitian"):
            assemble_tda_state_interaction(self.singlets, self.triplets, self.dipole_mo, self.soc_mo.imag)


if __name__ == "__main__":
    unittest.main()
