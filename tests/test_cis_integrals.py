"""Check analytic AO integral derivatives against displaced PySCF integrals."""

from dataclasses import fields
from importlib.util import find_spec
import unittest

import numpy as np

from pyscf_esd.cis_integrals import IntegralDerivativeEngine, IntegralValues


@unittest.skipUnless(find_spec("pyscf"), "PySCF is required for AO derivative checks")
class CISIntegralDerivativeTests(unittest.TestCase):
    def setUp(self):
        from pyscf import gto, lib

        self.addCleanup(lib.num_threads, lib.num_threads())
        lib.num_threads(1)
        self.mol = gto.M(
            atom="O 0 0 0; H 0.13 1.4 -0.9; H 0 -1.6 -0.8",
            unit="Bohr", basis="6-31g*", verbose=0,
        )
        self.coords = self.mol.atom_coords()
        self.engine = IntegralDerivativeEngine(self.mol)
        self.left, self.right = np.random.default_rng(318).normal(size=(2, self.mol.natm, 3))

    def displaced_integrals(self, s, t):
        from pyscf.scf.hf import get_hcore

        coords = self.coords + s * self.left + t * self.right
        displaced = self.mol.set_geom_(coords, unit="Bohr", inplace=False)
        return IntegralValues(
            displaced.intor_symmetric("int1e_ovlp"), get_hcore(displaced),
            displaced.intor("int2e", aosym="s1"), displaced.energy_nuc(),
        )

    def test_first_and_mixed_derivatives(self):
        step = 1e-4
        first = self.engine.first(self.left)
        second = self.engine.second(self.left, self.right)
        first_plus = self.displaced_integrals(step, 0)
        first_minus = self.displaced_integrals(-step, 0)
        corners = [
            self.displaced_integrals(s * step, t * step)
            for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]

        for field in fields(IntegralValues):
            with self.subTest(integral=field.name):
                forward = getattr(first_plus, field.name)
                backward = getattr(first_minus, field.name)
                pp, pm, mp, mm = (getattr(corner, field.name) for corner in corners)
                np.testing.assert_allclose(
                    getattr(first, field.name), (forward - backward) / (2 * step),
                    rtol=0, atol=3e-7,
                )
                np.testing.assert_allclose(
                    getattr(second, field.name), (pp - pm - mp + mm) / (4 * step**2),
                    rtol=0, atol=2e-6,
                )

    def test_mixed_symmetry_and_rigid_translations(self):
        translation = np.broadcast_to([0.3, 0.5, -0.7], self.coords.shape)
        first_translation = self.engine.first(translation)
        mixed_translation = self.engine.second(translation, self.right)
        forward = self.engine.second(self.left, self.right)
        reverse = self.engine.second(self.right, self.left)

        for field in fields(IntegralValues):
            with self.subTest(integral=field.name):
                np.testing.assert_allclose(getattr(first_translation, field.name), 0, atol=5e-12)
                np.testing.assert_allclose(getattr(mixed_translation, field.name), 0, atol=5e-12)
                np.testing.assert_allclose(
                    getattr(forward, field.name), getattr(reverse, field.name), rtol=0, atol=5e-12,
                )
