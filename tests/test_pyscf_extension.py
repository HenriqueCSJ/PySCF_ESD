"""Test the PySCF entry point without running an electronic calculation."""

from importlib.util import find_spec
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(find_spec("pyscf") and find_spec("pyscf.gto"), "PySCF is required")
class PySCFExtensionTests(unittest.TestCase):
    def test_namespace_exports_share_the_original_api(self):
        from pyscf import esd
        import pyscf_esd
        from pyscf_esd.cpu import Settings

        self.assertIs(esd.Settings, Settings)
        for name in pyscf_esd.__all__:
            self.assertIs(getattr(esd, name), getattr(pyscf_esd, name))

    def test_run_uses_the_molecule_and_selected_options(self):
        from pyscf import esd, gto
        from pyscf.data.nist import BOHR

        mol = gto.M(atom="He 0 0 0; H 0 0 0.8", unit="Angstrom", basis="def2-SVP",
                    charge=1, verbose=0)
        with patch("pyscf.esd.run_vibronic") as workflow:
            result = esd.run(mol, xc="b3lyp", target_index=5, model="ah", temperature_kelvin=298)
        workflow.assert_called_once()
        symbols, coords, settings = workflow.call_args.args
        self.assertEqual(symbols, ("He", "H"))
        np.testing.assert_allclose(coords, [[0, 0, 0], [0, 0, 0.8 / BOHR]])
        self.assertEqual((settings.basis, settings.charge, settings.xc, settings.nstates),
                         ("def2-SVP", 1, "b3lyp", 7))
        self.assertEqual(workflow.call_args.kwargs,
                         {"target_index": 5, "model": "ah", "temperature_kelvin": 298})
        self.assertIs(result, workflow.return_value)

    def test_unsupported_molecules_fail_before_running_the_workflow(self):
        from pyscf import esd, gto

        open_shell = gto.M(atom="H 0 0 0", spin=1, verbose=0)
        mixed_basis = gto.M(atom="He 0 0 0; H 0 0 0.8", charge=1, verbose=0,
                            basis={"He": "sto-3g", "H": "def2-SVP"})
        with patch("pyscf.esd.run_vibronic") as workflow:
            for mol, message in ((gto.Mole(), "built"), (open_shell, "closed-shell"),
                                 (mixed_basis, "named basis")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    esd.run(mol)
        workflow.assert_not_called()
