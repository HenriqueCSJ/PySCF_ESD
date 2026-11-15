"""Fourier interpolation tests that require no electronic calculations."""

import unittest

import numpy as np

from pyscf_esd.anharmonic_sampling import resample_periodic


class PeriodicSamplingTests(unittest.TestCase):
    def test_interpolates_potential_and_dipole_channels(self):
        def values(angle):
            return np.stack((0.02 * (1 - np.cos(3 * angle)), np.sin(angle), 2 * np.cos(2 * angle)), axis=1)

        coarse = values(2 * np.pi * np.arange(9) / 9)
        fine = values(2 * np.pi * np.arange(72) / 72)
        np.testing.assert_allclose(resample_periodic(coarse, 72), fine, atol=2e-14)
        np.testing.assert_allclose(resample_periodic(coarse, 9), coarse, atol=1e-14)

    def test_even_grid_nyquist_mode_and_explicit_cutoff(self):
        coarse = np.array([1., -1.] * 4) + 2
        fine = 2 + np.cos(2 * np.pi * 4 * np.arange(32) / 32)
        np.testing.assert_allclose(resample_periodic(coarse, 32), fine, atol=5e-15)
        np.testing.assert_array_equal(resample_periodic(coarse, 32, max_harmonic=0), np.full(32, 2.0))
        np.testing.assert_array_equal(resample_periodic(coarse, 8), coarse)

    def test_rejects_invalid_grid_or_bandwidth(self):
        for value, n in (([1, 2], 8), ([0, 1, np.nan], 8), ([0, 1, 2j], 8), ([0, 1, 2], 2)):
            with self.subTest(value=value, n=n), self.assertRaises(ValueError):
                resample_periodic(value, n)
        with self.assertRaisesRegex(ValueError, "max_harmonic"):
            resample_periodic([0, 1, 2], 8, max_harmonic=2)


if __name__ == "__main__":
    unittest.main()
