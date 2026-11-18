"""Check the PySCF entry point and the isotope response of a minimal VG model."""

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
        mol.nucprop = {"He": {"mass": 3.016029}, 2: {"mass": 2.014101778}}
        with patch("pyscf.esd.run_vibronic") as workflow:
            result = esd.run(mol, xc="b3lyp", target_index=5, model="ah", temperature_kelvin=298,
                             backend="gpu", solvent="cyclohexane")
        workflow.assert_called_once()
        symbols, coords, settings = workflow.call_args.args
        self.assertEqual(symbols, ("He", "H"))
        np.testing.assert_allclose(coords, [[0, 0, 0], [0, 0, 0.8 / BOHR]])
        self.assertEqual((settings.basis, settings.charge, settings.xc, settings.nstates),
                         ("def2-SVP", 1, "b3lyp", 7))
        options = workflow.call_args.kwargs.copy()
        np.testing.assert_array_equal(options.pop("masses_amu"), [3.016029, 2.014101778])
        self.assertEqual(options,
                         {"target_index": 5, "model": "ah", "temperature_kelvin": 298,
                          "backend": "gpu", "solvent": "cyclohexane"})
        self.assertIs(result, workflow.return_value)

    def test_all_models_use_isotopic_masses_in_their_electronic_sessions(self):
        from pyscf import esd, gto
        from pyscf_esd.electronic_workflow import _ElectronicSession

        mol = gto.M(atom="H1 0 0 0; H2 0 0 .74", basis="sto-3g", verbose=0)
        mol.nucprop = {"H2": {"mass": 2.014101778}}
        expected_masses = mol.atom_mass_list(isotope_avg=True)

        class GeometryReached(Exception):
            pass

        def observe(session, *args, **kwargs):
            np.testing.assert_array_equal(session.masses, expected_masses)
            raise GeometryReached

        with patch.object(_ElectronicSession, "ground", observe):
            for backend in ("cpu", "gpu"):
                for model in ("auto", "ah", "vg"):
                    with self.subTest(backend=backend, model=model), self.assertRaises(GeometryReached):
                        esd.run(mol, backend=backend, model=model)

    def test_single_root_vg_scales_with_isotope_mass(self):
        from pyscf import esd, gto

        mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
        mass_h = mol.atom_mass_list(isotope_avg=True)[0]
        mass_d = 2.014101778
        options = dict(model="vg", process="absorption", max_quanta=50)
        hydrogen = esd.run(mol, **options)
        mol.nucprop = {"H": {"mass": mass_d}}
        deuterium = esd.run(mol, **options)

        def frequency(result):
            modes = result.diagnostics["harmonic_reference"]["ground_at_excited"]["modes"]
            mode, = modes
            return mode["frequency_cm^-1"]

        np.testing.assert_allclose(frequency(deuterium) / frequency(hydrogen), np.sqrt(mass_h / mass_d), rtol=1e-10)

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
