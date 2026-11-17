import unittest

import numpy as np

from pyscf_esd.tda_xc_grid import _density_at_nodes, _lagrangian_grid_integrand


def gaussian_ao(points, exponent):
    values = np.exp(-exponent * np.einsum("ag,ag->g", points, points))
    first = -2 * exponent * points * values
    second = values * (4 * exponent**2 * points[:, None] * points[None, :]
                       - 2 * exponent * np.eye(3)[:, :, None])
    i, j = np.triu_indices(3)
    return np.concatenate((values[None, :], first, second[i, j]))[:, None, :]


class TDAXCGridTests(unittest.TestCase):
    def test_gga_density_hessian_has_the_native_ao_derivative_order(self):
        points = np.array([[.13, -.18], [-.6, .46], [.7, -.23]])
        exponent = .71
        density, nodes = _density_at_nodes(gaussian_ao(points, exponent), np.ones((1, 1)), gga=True, xp=np)
        expected_rho = np.exp(-2 * exponent * np.einsum("ag,ag->g", points, points))
        expected_gradient = -4 * exponent * points * expected_rho
        expected_hessian = expected_rho * (16 * exponent**2 * points[:, None] * points[None, :]
                                           - 4 * exponent * np.eye(3)[:, :, None])
        np.testing.assert_allclose(density[0], expected_rho, atol=1e-14)
        np.testing.assert_allclose(density[1:], expected_gradient, atol=1e-14)
        np.testing.assert_allclose(nodes[:, 0], expected_gradient, atol=1e-14)
        np.testing.assert_allclose(nodes[:, 1:], expected_hessian, atol=1e-14)


    def test_grid_integrand_derivative_includes_the_third_xc_derivative(self):
        points = np.array([[.1, -.32], [-.42, .66], [.3, -.53]])
        exponent, p, t = .8, -.7, .37
        ground, ground_node = _density_at_nodes(gaussian_ao(points, exponent), np.ones((1, 1)), gga=False, xp=np)
        # The exact LDA energy density rho cubed / 6 gives v=rho²/2, f=rho, k=1.
        integrand, derivative = _lagrangian_grid_integrand(
            ground**2 / 2, ground[None, :], np.ones((1, 1, 1, points.shape[1])),
            ground_node, p * ground, p * ground_node, t * ground, t * ground_node, np,
        )
        expected = (p / 2 + t*t) * ground[0]**3
        np.testing.assert_allclose(integrand, expected, atol=1e-14)
        np.testing.assert_allclose(derivative, -12 * exponent * points * expected, atol=1e-14)
