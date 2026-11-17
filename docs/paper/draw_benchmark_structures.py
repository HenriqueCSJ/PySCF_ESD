"""Render the benchmark-molecule plate as vector PDF (RDKit and CairoSVG)."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent / "benchmarks"
WIDTH_MM, HEIGHT_MM = 178, 96
PANELS = (
    ("Benzene", "c1ccccc1", "C6H6", "benzene.svg"),
    ("CF₃NO", "FC(F)(F)N=O", "CF3NO", None),
    ("Pyridine", "c1ccncc1", "C5H5N", "pyridine.svg"),
    ("9-Cyanoanthracene", "N#Cc1c2ccccc2cc2ccccc12", "C15H9N", "cyanoanthracene.svg"),
    (
        "Quinine dication",
        "C=C[C@H]1C[NH+]2CC[C@H]1C[C@H]2[C@H](O)c1cc[nH+]c2ccc(OC)cc12",
        "C20H26N2O2+2",
        "quinine.svg",
    ),
    ("LiH", "[Li][H]", "HLi", None),
    ("Formaldehyde", "[H]C([H])=O", "CH2O", None),
    ("Water", "[H]O[H]", "H2O", None),
    ("Naphthalene", "c1ccc2ccccc2c1", "C10H8", None),
)


def panel_molecule(name, smiles, formula, archived_svg):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    expected = Chem.MolFromSmiles(smiles, params)
    if archived_svg:
        source = (BENCHMARKS / archived_svg).read_text(encoding="utf-8")
        molecule = Chem.MolFromRDKitSVG(source, removeHs=False)
    else:
        molecule = expected
        rdDepictor.Compute2DCoords(molecule)

    if Chem.MolToSmiles(molecule) != Chem.MolToSmiles(expected):
        raise ValueError(f"{name}: archived graph or stereochemistry differs from the reference")
    if rdMolDescriptors.CalcMolFormula(molecule) != formula:
        raise ValueError(f"{name}: unexpected molecular formula")
    if name == "Quinine dication":
        nitrogen_states = [
            (atom.GetFormalCharge(), atom.GetTotalNumHs())
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() == 7
        ]
        carbon_configurations = [
            configuration
            for index, configuration in Chem.FindMolChiralCenters(molecule)
            if molecule.GetAtomWithIdx(index).GetAtomicNum() == 6
        ]
        if nitrogen_states != [(1, 1), (1, 1)] or sorted(carbon_configurations) != ["R", "R", "S", "S"]:
            raise ValueError(f"{name}: unexpected nitrogen protonation or carbon stereochemistry")
    return molecule


def draw_molecule(name, molecule, width, height):
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.useBWAtomPalette()
    options.bondLineWidth = 1.3
    options.fixedBondLength = 19.5
    options.fixedFontSize = 10
    options.padding = 0.04
    options.prepareMolsBeforeDrawing = False
    if name == "LiH":
        options.rotate = 180
    prepared = rdMolDraw2D.PrepareMolForDrawing(molecule, addChiralHs=False)
    if name == "Quinine dication":
        from_drawn_bonds = Chem.Mol(prepared)
        for atom in from_drawn_bonds.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignChiralTypesFromBondDirs(from_drawn_bonds)
        Chem.SanitizeMol(from_drawn_bonds)
        Chem.AssignStereochemistry(from_drawn_bonds, cleanIt=True, force=True)
        if Chem.MolToSmiles(from_drawn_bonds) != Chem.MolToSmiles(molecule):
            raise ValueError(f"{name}: drawn wedge and dash directions change stereochemistry")
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    return ET.fromstring(drawer.GetDrawingText())


def build_svg():
    # Use PDF points as SVG user units to keep atom and panel labels at known print sizes.
    width = WIDTH_MM * 72 / 25.4
    height = HEIGHT_MM * 72 / 25.4
    column_width, row_height = width / 3, height / 3
    drawing_width, drawing_height = int(column_width - 10), int(row_height - 20)
    svg_namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_namespace)
    plate = ET.Element(
        f"{{{svg_namespace}}}svg",
        width=f"{WIDTH_MM}mm",
        height=f"{HEIGHT_MM}mm",
        viewBox=f"0 0 {width} {height}",
    )
    for index, (name, smiles, formula, archived_svg) in enumerate(PANELS):
        row, column = divmod(index, 3)
        center_x = (column + 0.5) * column_width
        row_y = row * row_height
        molecule = panel_molecule(name, smiles, formula, archived_svg)
        drawing = draw_molecule(name, molecule, drawing_width, drawing_height)
        drawing.set("x", str(center_x - drawing_width / 2))
        drawing.set("y", str(row_y + 3))
        plate.append(drawing)

        label = ET.SubElement(
            plate,
            f"{{{svg_namespace}}}text",
            {
                "x": str(center_x),
                "y": str(row_y + row_height - 5),
                "font-family": "DejaVu Sans, sans-serif",
                "font-size": "9.5",
                "text-anchor": "middle",
                "fill": "#191919",
            },
        )
        label.text = f"({chr(ord('a') + index)})  {name}"
    return ET.tostring(plate, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=HERE / "benchmark-structures.pdf")
    parser.add_argument("--svg", type=Path, help="also write the editable SVG artwork")
    args = parser.parse_args()
    artwork = build_svg()
    cairosvg.svg2pdf(bytestring=artwork, write_to=str(args.pdf))
    if args.svg:
        args.svg.write_bytes(artwork)


if __name__ == "__main__":
    main()
