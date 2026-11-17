"""Prepared UKS/TDA electronic observables without SCF or TDA solves."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from pyscf_esd.cpu import ElectronicConvergenceError
from pyscf_esd.unrestricted_tda import unrestricted_tda_vertical_spectrum


class _DeviceArray(np.ndarray):
    __cuda_array_interface__ = {}
    device = nullcontext()

    def get(self):
        return np.asarray(self)


def _prepared(backend="pyscf", *, n_alpha=2, n_beta=1):
    mf = type("UKS", (), {"__module__": f"{backend}.dft.uks"})()
    mf.converged = True
    mf.xc = "PBE0"
    mf._numint = SimpleNamespace(_xc_type=Mock(return_value="GGA"))
    mf.nelec = n_alpha, n_beta
    mf.mo_coeff = np.stack([np.eye(4), np.eye(4)])
    mf.mo_occ = np.array([np.arange(4) < n_alpha, np.arange(4) < n_beta], dtype=float)
    mf.e_tot = -7.5
    spin = abs(n_alpha - n_beta) / 2
    mf.spin_square = Mock(return_value=(spin * (spin + 1) + 0.1, 2.05))
    mf.kernel = Mock(side_effect=AssertionError("would solve a new UKS reference"))

    td = type("TDA", (), {"__module__": f"{backend}.tdscf.uks"})()
    td._scf = mf
    td.e = np.array([0.1, 0.2, 0.3])
    td.converged = np.array([True, True, True])
    td.xy = [object() for _ in td.e]
    td.kernel = Mock(side_effect=AssertionError("would solve new TDA roots"))
    dipoles = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.4], [-0.1, 0.2, 0.2]])
    if backend == "gpu4pyscf":
        mf.mo_coeff = mf.mo_coeff.view(_DeviceArray)
        mf.mo_occ = mf.mo_occ.view(_DeviceArray)
        td.e = td.e.view(_DeviceArray)
        td.converged = td.converged.view(_DeviceArray)

    def transition_dipole(xy):
        return dipoles[[td.xy.index(root) for root in xy]]

    td.transition_dipole = Mock(side_effect=transition_dipole)
    return td


class UnrestrictedTDAVerticalTests(unittest.TestCase):
    def test_cpu_selected_roots_preserve_phases_and_absolute_observables(self):
        td = _prepared()
        result = unrestricted_tda_vertical_spectrum(td, roots=[2, 0])
        self.assertEqual(result.root_indices, (2, 0))
        self.assertEqual(result.backend, "pyscf")
        self.assertEqual(result.reference_energy_hartree, -7.5)
        self.assertEqual(result.reference_ms, 0.5)
        self.assertAlmostEqual(result.reference_spin_squared, 0.85)
        self.assertIsNone(result.root_spin_squared)
        np.testing.assert_array_equal(result.photon_energy_hartree, [0.3, 0.1])
        np.testing.assert_array_equal(result.transition_dipole_au, [[-0.1, 0.2, 0.2], [0, 0, 0]])
        np.testing.assert_allclose(result.dipole_strength_au2, [0.09, 0])
        np.testing.assert_allclose(result.oscillator_strength, [0.018, 0])
        expected_a = 4 / 3 * (7.2973525643e-3)**3 * 0.3**3 * 0.09 / 2.4188843265864e-17
        np.testing.assert_allclose(result.einstein_a_s_inverse, [expected_a, 0])
        self.assertFalse(result.transition_dipole_au.flags.writeable)
        td.transition_dipole.assert_called_once_with(xy=[td.xy[2], td.xy[0]])
        td._scf.kernel.assert_not_called()
        td.kernel.assert_not_called()

    def test_gpu_quartet_and_native_order(self):
        result = unrestricted_tda_vertical_spectrum(_prepared("gpu4pyscf", n_alpha=4, n_beta=1))
        self.assertEqual(result.backend, "gpu4pyscf")
        self.assertEqual(result.root_indices, (0, 1, 2))
        self.assertEqual(result.reference_ms, 1.5)
        self.assertAlmostEqual(result.reference_spin_squared, 3.85)
        self.assertIsNone(result.root_spin_squared)
        self.assertIs(type(result.photon_energy_hartree), np.ndarray)
        np.testing.assert_array_equal(result.photon_energy_hartree, [0.1, 0.2, 0.3])

    def test_selected_roots_must_have_converged_and_be_positive(self):
        td = _prepared()
        td.converged[0] = False
        td.e[0] = np.nan
        td.e[1] = -0.2
        with self.assertRaisesRegex(ElectronicConvergenceError, "root must be converged"):
            unrestricted_tda_vertical_spectrum(td)
        with self.assertRaisesRegex(ElectronicConvergenceError, "positive excitation energy"):
            unrestricted_tda_vertical_spectrum(td, roots=[1])
        td.transition_dipole.assert_not_called()
        self.assertEqual(unrestricted_tda_vertical_spectrum(td, roots=[2]).root_indices, (2,))

    def test_requires_prepared_dft_tda_and_unambiguous_selection(self):
        td = _prepared(n_alpha=3, n_beta=1)
        plain_tda = type(td)
        td.__class__ = type("CasidaTDDFT", (type("TDDFT", (), {
            "__module__": "pyscf.tdscf.uks",
        }), plain_tda), {})
        with self.assertRaisesRegex(TypeError, "spin-conserving molecular"):
            unrestricted_tda_vertical_spectrum(td)
        td.__class__ = plain_tda
        for invalid in ([], [0, 0], [3], [True], [-1], [0.5]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "distinct zero-based"):
                unrestricted_tda_vertical_spectrum(td, roots=invalid)
        td._scf.converged = False
        with self.assertRaisesRegex(ElectronicConvergenceError, "reference must already be converged"):
            unrestricted_tda_vertical_spectrum(td)
        td._scf.converged = True
        td._scf._numint._xc_type.return_value = "HF"
        with self.assertRaisesRegex(ValueError, "requires a density functional"):
            unrestricted_tda_vertical_spectrum(td)
        td._scf._numint._xc_type.return_value = "GGA"
        td.e = None
        with self.assertRaisesRegex(ElectronicConvergenceError, "contain solved roots"):
            unrestricted_tda_vertical_spectrum(td)
        td.transition_dipole.assert_not_called()


if __name__ == "__main__":
    unittest.main()
