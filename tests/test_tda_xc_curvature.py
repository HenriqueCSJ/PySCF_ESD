import itertools
import os
import unittest
from unittest import mock

import numpy as np

from pyscf_esd.tda_xc_curvature import (
    require_semilocal_xc_fourth,
    semilocal_xc_derivatives,
    semilocal_xc_fourth,
)


class TDAXCCurvatureTests(unittest.TestCase):
    def setUp(self):
        try:
            from pyscf.dft import libxc
        except ImportError:
            self.skipTest("PySCF is unavailable")
        self.libxc = libxc

    def require_fourth(self, xc_code):
        try:
            require_semilocal_xc_fourth(xc_code)
        except NotImplementedError as error:
            if os.environ.get("PYSCF_ESD_LIBXC_LXC"):
                raise
            self.skipTest(str(error))

    def test_missing_fourth_derivative_fails_before_grid_integration(self):
        if self.libxc.test_deriv_order("LDA_X", 4):
            self.skipTest("native PySCF already supports LXC")
        with mock.patch.dict(os.environ, {"PYSCF_ESD_LIBXC_LXC": ""}):
            with self.assertRaisesRegex(NotImplementedError, "MAXORDER=4"):
                require_semilocal_xc_fourth("LDA_X")

    def test_slater_fourth_matches_density_power_law(self):
        self.require_fourth("LDA_X")
        density = np.array([.007, .41, 3.7])
        cx = .75 * (3 / np.pi)**(1/3)
        expected = -cx * (40/81) * density**(-8/3)
        result = semilocal_xc_fourth("LDA_X", density)
        self.assertEqual(result.shape, (1, 1, 1, 1, 3))
        np.testing.assert_allclose(result[0, 0, 0, 0], expected, rtol=2e-13)

    def test_pbe0_small_gradient_coefficients(self):
        self.require_fourth("PBE0")
        density = np.array([.008, .61, 3.7])
        rho = np.vstack((density, np.zeros((3, density.size))))
        result = semilocal_xc_fourth("PBE0", rho)

        # PBE exchange enhancement and the PBE correlation logarithm,
        # expanded in sigma = |grad rho|^2 at fixed rho.
        cx, kappa, mu = .75 * (3 / np.pi)**(1/3), .804, .2195149727645171
        beta, gamma = .06672455060314922, (1 - np.log(2)) / np.pi**2
        kf = (3 * np.pi**2)**(1/3)
        s_sigma = 1 / (4 * kf**2 * density**(8/3))
        t_sigma = np.pi / (16 * kf * density**(7/3))
        f_sigma_sigma = (.75 * cx * density**(4/3) * 2 * mu**2 / kappa * s_sigma**2
                         - density * beta**2 / gamma * t_sigma**2)
        f_rho_rho_sigma = (28/9) * density**(-10/3) * (
            -.75 * cx * mu / (4 * kf**2) + beta * np.pi / (16 * kf)
        )
        np.testing.assert_allclose(result[1, 1, 1, 1], 12 * f_sigma_sigma, rtol=2e-11)
        np.testing.assert_allclose(result[1, 1, 2, 2], 4 * f_sigma_sigma, rtol=2e-11)
        np.testing.assert_allclose(result[0, 1, 1, 0], 2 * f_rho_rho_sigma, rtol=2e-11)
        np.testing.assert_array_equal(result[0, 0, 0, 1], 0)

    def test_pbe0_tensor_permutation_and_rotation(self):
        self.require_fourth("PBE0")
        rho = np.array([[.5, 1.2], [.32, -.3], [.12, .6], [-.25, .8]])
        transform = np.eye(4)
        transform[1:, 1:] = np.array([[3/5, -4/5, 0], [0, 0, -1], [4/5, 3/5, 0]])
        reference = semilocal_xc_fourth("PBE0", rho)
        for order in itertools.permutations(range(4)):
            np.testing.assert_allclose(reference.transpose(*order, 4), reference, atol=2e-13)
        rotated = semilocal_xc_fourth("PBE0", transform @ rho)
        expected = np.einsum(
            "ai,bj,ck,dl,ijklg->abcdg", transform, transform, transform, transform, reference,
            optimize=True,
        )
        np.testing.assert_allclose(rotated, expected, rtol=2e-12, atol=2e-13)

    def test_pbe0_components_match_builtin_hybrid(self):
        self.require_fourth("PBE0")
        pbe0_components = "0.25*HF+0.75*GGA_X_PBE,GGA_C_PBE"
        self.require_fourth(pbe0_components)
        rho = np.array([[.5, 1.2], [.32, -.3], [.12, .6], [-.25, .8]])
        np.testing.assert_allclose(
            semilocal_xc_fourth("PBE0", rho), semilocal_xc_fourth(pbe0_components, rho),
            rtol=2e-13, atol=2e-13,
        )

    def test_pbe0_matches_native_pyscf_at_conditioned_points(self):
        rho = np.array([[.05, .17, 1.2, 1.3], [.0032, .01, .16, .37],
                        [-.0012, -.06, .07, -.11], [-.0025, -.025, .03, -.26]])
        derivatives = semilocal_xc_derivatives("PBE0", rho)
        max_order = min(self.libxc.max_deriv_order("PBE0"), 4)
        for order in range(max_order + 1):
            expected = self.libxc.eval_xc_eff("PBE0", rho, deriv=order, spin=0)
            np.testing.assert_allclose(derivatives[order], expected, rtol=2e-10, atol=2e-13)

    def test_pbe0_tail_screening_matches_native_pyscf(self):
        density = np.array([0, 1e-15, 1.5e-15, 2e-15, 3e-15, 9e-13, 1e-12, 2e-12])
        gradient = np.array([[.5], [-.12], [.065]]) * density**(4/3)
        rho = np.vstack((density, gradient))
        actual = semilocal_xc_derivatives("PBE0", rho)
        expected = self.libxc.eval_xc_eff("PBE0", rho, deriv=1, spin=0)
        np.testing.assert_allclose(actual[1], expected, rtol=2e-10, atol=1e-15)
        np.testing.assert_array_equal(actual[4][..., :4], 0)

    def test_lda_component_in_gga_only_contributes_to_density_partial(self):
        self.require_fourth("GGA_X_PBE")
        self.require_fourth("LDA_X")
        xc_code = ".4*GGA_X_PBE+.6*LDA_X"
        self.require_fourth(xc_code)
        rho = np.array([[.5, 1.2], [.32, -.3], [.12, .6], [-.25, .8]])
        expected = .4 * semilocal_xc_fourth("GGA_X_PBE", rho)
        expected[0, 0, 0, 0] += .6 * semilocal_xc_fourth("LDA_X", rho[0])[0, 0, 0, 0]
        np.testing.assert_allclose(
            semilocal_xc_fourth(xc_code, rho), expected, rtol=2e-13, atol=2e-13,
        )
