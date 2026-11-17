"""Check coherent complex FC/HT amplitudes and real oscillator closure."""

import unittest
from unittest.mock import patch

import numpy as np

import pyscf_esd.harmonic as harmonic
from pyscf_esd.harmonic import harmonic_spectrum


class ComplexHarmonicTests(unittest.TestCase):
    def test_assigned_complex_amplitudes_match_real_and_imaginary_oracles(self):
        omega_i = [0.004, 0.007]
        omega_f = [0.006, 0.009]
        angle = 0.4
        rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        dipole = np.array([0.4 + 0.2j, -0.3 - 0.1j, 0.2j])
        derivatives = np.array([
            [0.03 - 0.07j, 0.12 + 0.09j, 0.03j],
            [-0.02 + 0.01j, 0.01, 0.04 - 0.02j],
        ])
        phase = np.exp(0.73j)
        for temperature in (0, 1000):
            with self.subTest(temperature=temperature):
                options = dict(
                    gap_00_hartree=-0.12, process="emission", initial_max_quanta=2,
                    final_max_quanta=3, shift_au=[5.0, -8.0], rotation=rotation,
                    temperature_kelvin=temperature,
                )

                def spectrum(moment, slope):
                    return harmonic_spectrum(
                        omega_i, omega_f, transition_dipole_au=moment,
                        dipole_derivatives_au=slope, **options,
                    )

                with patch.object(harmonic, "_fc_parameters", wraps=harmonic._fc_parameters) as overlaps:
                    complex_lines = spectrum(dipole, derivatives)
                    overlaps.assert_called_once()
                real_lines = spectrum(dipole.real, derivatives.real)
                imaginary_lines = spectrum(dipole.imag, derivatives.imag)
                rotated_lines = spectrum(phase * dipole, phase * derivatives)

                for name in ("initial_quanta", "final_quanta", "population", "photon_energy_hartree"):
                    np.testing.assert_array_equal(getattr(complex_lines, name), getattr(real_lines, name))
                for name in ("transition_dipole_au", "condon_transition_dipole_au"):
                    expected = getattr(real_lines, name) + 1j * getattr(imaginary_lines, name)
                    self.assertEqual(getattr(real_lines, name).dtype, np.dtype(float))
                    np.testing.assert_allclose(getattr(complex_lines, name), expected, atol=2e-15)
                    np.testing.assert_allclose(getattr(rotated_lines, name), phase * expected, atol=2e-15)
                expected_strengths = real_lines.dipole_strength_au2 + imaginary_lines.dipole_strength_au2
                self.assertEqual(complex_lines.dipole_strength_au2.dtype, np.dtype(float))
                np.testing.assert_allclose(complex_lines.dipole_strength_au2, expected_strengths, atol=2e-15)
                np.testing.assert_allclose(rotated_lines.dipole_strength_au2, expected_strengths, atol=2e-15)
                self.assertAlmostEqual(
                    complex_lines.final_dipole_strength_coverage,
                    rotated_lines.final_dipole_strength_coverage,
                )

    def test_condon_and_ht_can_cancel_with_a_complex_dipole(self):
        spectrum = harmonic_spectrum(
            [0.004], [0.004], gap_00_hartree=0.1, shift_au=[10.0],
            transition_dipole_au=[1 + 2j, 0, 0], dipole_derivatives_au=[[0.2 + 0.4j, 0, 0]],
            initial_max_quanta=0, final_max_quanta=1,
        )
        # The common ground-state Gaussian has mean Q_i=-shift/2, so its
        # integrated linear dipole cancels the constant term on the 0-0 line.
        self.assertGreater(abs(spectrum.condon_transition_dipole_au[0, 0]), 1.0)
        self.assertLess(spectrum.dipole_strength_au2[0], 1e-27)
        self.assertGreater(spectrum.dipole_strength_au2[1], 0.0)

    def test_complex_condon_only_dipole(self):
        for temperature in (0, 1000):
            with self.subTest(temperature=temperature):
                spectrum = harmonic_spectrum(
                    [0.004], [0.004], gap_00_hartree=0.1, transition_dipole_au=[2j, 0, 0],
                    initial_max_quanta=0, final_max_quanta=1, temperature_kelvin=temperature,
                )
                np.testing.assert_allclose(spectrum.transition_dipole_au[:, 0], [2j, 0])
                np.testing.assert_allclose(spectrum.dipole_strength_au2, [4, 0])
                self.assertIsNone(spectrum.condon_transition_dipole_au)
                self.assertAlmostEqual(spectrum.final_dipole_strength_coverage, 1.0)

    def test_thermal_closure_with_real_dipole_and_complex_derivatives(self):
        omega = 0.004
        dipole = [0.4, -0.3, 0.2]
        derivatives = np.array([[0.03 - 0.07j, 0.12 + 0.09j, 0.03j]])
        for temperature in (0, 1000):
            with self.subTest(temperature=temperature):
                options = dict(
                    gap_00_hartree=0.1, transition_dipole_au=dipole,
                    dipole_derivatives_au=derivatives, initial_max_quanta=3,
                    temperature_kelvin=temperature,
                )
                complete = harmonic_spectrum([omega], [omega], final_max_quanta=4, **options)
                self.assertTrue(np.iscomplexobj(complete.transition_dipole_au))
                self.assertTrue(np.iscomplexobj(complete.condon_transition_dipole_au))
                np.testing.assert_array_equal(complete.condon_transition_dipole_au.imag, 0.0)
                by_initial = np.bincount(complete.initial_quanta[:, 0], complete.dipole_strength_au2)
                variances = (2 * np.arange(by_initial.size) + 1) / (2 * omega)
                expected = np.dot(dipole, dipole) + variances * np.sum(abs(derivatives)**2)
                np.testing.assert_allclose(by_initial, expected)
                total = complete.thermal_strength_au2.sum()
                self.assertAlmostEqual(complete.final_dipole_strength_coverage, 1.0)
                truncated = harmonic_spectrum([omega], [omega], final_max_quanta=0, **options)
                self.assertAlmostEqual(truncated.final_dipole_strength_coverage, truncated.thermal_strength_au2.sum() / total)

    def test_absorption_m1_bound_with_duschinsky_mixing_and_photon_clipping(self):
        omega_i, omega_f = [0.004, 0.007], [0.006, 0.009]
        angle = 0.4
        options = dict(
            gap_00_hartree=0.003, temperature_kelvin=700, shift_au=[5.0, -8.0],
            rotation=[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            transition_dipole_au=[0.4 + 0.2j, -0.3 - 0.1j, 0.2j],
            dipole_derivatives_au=[
                [0.03 - 0.07j, 0.12 + 0.09j, 0.03j],
                [-0.02 + 0.01j, 0.01, 0.04 - 0.02j],
            ],
        )
        truncated = harmonic_spectrum(omega_i, omega_f, initial_max_quanta=2, final_max_quanta=2, **options)
        # Reversing the sign of the photon energy keeps the same vibrational
        # populations and amplitudes while exposing the clipped sticks.
        negative = harmonic_spectrum(
            omega_i, omega_f, initial_max_quanta=2, final_max_quanta=2, process="emission", **options,
        )
        self.assertGreater(negative.thermal_strength_au2 @ negative.photon_energy_hartree, 1e-4)
        wider = harmonic_spectrum(omega_i, omega_f, initial_max_quanta=8, final_max_quanta=14, **options)
        weighted = wider.thermal_strength_au2 * wider.photon_energy_hartree
        omitted_initial = wider.initial_quanta.sum(axis=1) > 2
        omitted_final = (~omitted_initial) & (wider.final_quanta.sum(axis=1) > 2)
        moment = truncated.positive_photon_moments[1]
        self.assertAlmostEqual(moment.retained, weighted[~(omitted_initial | omitted_final)].sum())
        self.assertGreaterEqual(moment.omitted_initial_upper_bound, weighted[omitted_initial].sum())
        self.assertGreaterEqual(moment.omitted_final_upper_bound, weighted[omitted_final].sum())
        # All clipped transitions from the retained initial states fit in the
        # small final basis, so the analytic final-tail bound is tight.
        np.testing.assert_allclose(moment.omitted_final_upper_bound, weighted[omitted_final].sum(), rtol=3e-5)

    def test_rejects_nonfinite_complex_components_and_complex_coordinates(self):
        options = dict(gap_00_hartree=0.1, initial_max_quanta=0, final_max_quanta=0)
        with self.assertRaisesRegex(ValueError, "transition_dipole_au must be finite"):
            harmonic_spectrum([0.004], [0.004], transition_dipole_au=[complex(0, np.inf), 0, 0], **options)
        with self.assertRaisesRegex(ValueError, "dipole_derivatives_au must be finite"):
            harmonic_spectrum(
                [0.004], [0.004], transition_dipole_au=[1, 0, 0],
                dipole_derivatives_au=[[complex(0, np.nan), 0, 0]], **options,
            )
        with self.assertRaisesRegex(ValueError, "shift_au must be real"):
            harmonic_spectrum([0.004], [0.004], transition_dipole_au=[1, 0, 0], shift_au=[1j], **options)


if __name__ == "__main__":
    unittest.main()
