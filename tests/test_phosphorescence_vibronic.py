"""Analytic controls for SOC response and prepared vibronic phosphorescence."""

from dataclasses import replace
import json
import unittest

import numpy as np

from pyscf_esd import run_prepared_phosphorescence, spin_orbit_dipole_derivatives
from pyscf_esd.harmonic import _KB_HARTREE_PER_K
from pyscf_esd.spectral_properties import _EINSTEIN_A_FACTOR
from pyscf_esd.vibronic_properties import condon_ht_line_strengths
from test_phosphorescence import calculate, prepared_model
from test_prepared_ah import _prepared_oscillator


def isolated_model():
    energy, soc, dipole = prepared_model()
    soc[0, 5] = soc[5, 0] = 0.
    soc[3, 3], soc[4, 4] = .003, .004
    # An imaginary coupling tests the bra conjugation in the response.
    soc[1, 2], soc[2, 1] = .006j, -.006j
    return energy, soc, dipole


class VibronicPhosphorescenceTests(unittest.TestCase):
    def test_analytic_mixed_dipole_derivative(self):
        energy, soc, dipole = isolated_model()
        result = calculate(energy, soc, dipole)
        response = np.zeros(soc.shape + (1,), dtype=complex)
        response[1, 2, 0], response[2, 1, 0] = 1j, -1j
        muprime = np.zeros((*soc.shape, 1, 3), dtype=complex)
        muprime[0, 1, 0, 0] = muprime[1, 0, 0, 0] = .17
        derivative = spin_orbit_dipole_derivatives(result, dipole, response, muprime)
        intensity_slope = 2 * np.einsum("ic,ic->i", result.transition_dipole_au.conj(), derivative[:, 0]).real
        delta, coupling, moment = .1, .006, .4
        separation = np.hypot(delta, 2 * coupling)
        singlet_fraction = (1 - delta / separation) / 2
        expected = moment**2 * 2 * delta * coupling / separation**3 + 2 * moment * .17 * singlet_fraction
        np.testing.assert_allclose(intensity_slope, [expected, 0., 0.], rtol=2e-13, atol=1e-14)

    def test_individual_response_requires_resolved_sublevels(self):
        energy, soc, dipole = prepared_model()
        with self.assertRaisesRegex(ValueError, "isolated electronic sublevels"):
            spin_orbit_dipole_derivatives(
                calculate(energy, soc, dipole), dipole,
                np.zeros((*soc.shape, 1)), np.zeros((*soc.shape, 1, 3)),
            )

    def test_prepared_ah_emission_uses_adiabatic_origin_and_complex_strengths(self):
        electronic = calculate(*isolated_model())
        parameters = replace(_prepared_oscillator(), shift_au=np.zeros(1))
        slopes = np.zeros((3, 1, 3), dtype=complex)
        slopes[:, 0, 1] = 1e-3j
        result = run_prepared_phosphorescence(
            parameters, electronic, lowest_sublevel_00_hartree=.09, dipole_derivatives_au=slopes,
        )
        np.testing.assert_allclose(result.sublevel_00_hartree, .09 + electronic.sublevel_splittings_hartree)
        np.testing.assert_array_equal(result.sublevel_populations, [1., 0., 0.])
        gap, omega = result.sublevel_00_hartree[0], parameters.ground_frequencies_hartree[0]
        expected = _EINSTEIN_A_FACTOR * (
            gap**3 * electronic.dipole_strength_au2[0] + (gap - omega)**3 * abs(slopes[0, 0, 1])**2 / (2 * omega)
        )
        report = result.physical_properties
        np.testing.assert_allclose(report["vacuum_radiative_rate_s_inverse"], expected, rtol=1e-12)
        self.assertAlmostEqual(report["vacuum_radiative_lifetime_s"] * expected, 1.)
        fractions = report["sublevels"][0]["condon_herzberg_teller"]["fractions_of_total_signal"]
        self.assertAlmostEqual(sum(fractions.values()), 1.)
        moment = report["sublevels"][0]["dominant_lines"][0]["transition_dipole_au"]
        self.assertEqual(set(moment), {"real", "imaginary"})
        json.dumps(report, allow_nan=False)

    def test_displaced_thermal_spectra_interfere_and_weight_spin_once(self):
        electronic = calculate(*isolated_model())
        parameters = replace(_prepared_oscillator(), shift_au=np.array([4.]))
        slopes = .02j * electronic.transition_dipole_au[:, None, :]
        slopes[0] *= 1 + 1j
        temperature = 298.
        result = run_prepared_phosphorescence(
            parameters, electronic, lowest_sublevel_00_hartree=.09,
            dipole_derivatives_au=slopes, temperature_kelvin=temperature,
        )
        population = np.exp(-electronic.sublevel_splittings_hartree / (_KB_HARTREE_PER_K * temperature))
        population /= population.sum()
        np.testing.assert_allclose(result.sublevel_populations, population)
        spectrum = result.sublevel_spectra[0]
        terms = condon_ht_line_strengths(spectrum)
        self.assertGreater(np.max(np.abs(terms["condon_ht_interference_au2"])), 0.)
        np.testing.assert_allclose(sum(terms.values()), spectrum.dipole_strength_au2, atol=1e-16)
        per_sublevel = [_EINSTEIN_A_FACTOR * np.sum(
            line.population * line.dipole_strength_au2 * line.photon_energy_hartree**3
        ) for line in result.sublevel_spectra]
        np.testing.assert_allclose(result.physical_properties["vacuum_radiative_rate_s_inverse"],
                                   population @ per_sublevel, rtol=1e-12)
