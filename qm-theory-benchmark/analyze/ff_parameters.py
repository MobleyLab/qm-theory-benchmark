import io
from collections import OrderedDict, defaultdict
from textwrap import wrap

import click
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from reportlab.pdfgen import canvas


def get_oemol_struc(oemol, torsions=False, atom_indices=[], width=500, height=300):
    """
    a routine to visualize the chemical structures using oemol backend and highlight the atoms involved in torsion
    that is being driven
    :param oemol: pass an oemol object
    :param torsions: boolean to highlight or not
    :param atom_indices: atom_indices to highlight
    :param width: width of the png output
    :param height: height of the png output
    :return: a png string
    """
    from openeye import oechem, oedepict
    from PIL import Image

    # Highlight element of interest
    class NoAtom(oechem.OEUnaryAtomPred):
        def __call__(self, atom):
            return False

    class AtomInTorsion(oechem.OEUnaryAtomPred):
        def __call__(self, atom):
            return atom.GetIdx() in atom_indices

    class NoBond(oechem.OEUnaryBondPred):
        def __call__(self, bond):
            return False

    class BondInTorsion(oechem.OEUnaryBondPred):
        def __call__(self, bond):
            return (bond.GetBgn().GetIdx() in atom_indices) and (
                bond.GetEnd().GetIdx() in atom_indices
            )

    class CentralBondInTorsion(oechem.OEUnaryBondPred):
        def __call__(self, bond):
            return (bond.GetBgn().GetIdx() in atom_indices[1:3]) and (
                bond.GetEnd().GetIdx() in atom_indices[1:3]
            )

    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomIdx())

    oedepict.OEPrepareDepiction(oemol)
    img = oedepict.OEImage(width, height)
    display = oedepict.OE2DMolDisplay(oemol, opts)
    if torsions:
        atoms = oemol.GetAtoms(AtomInTorsion())
        bonds = oemol.GetBonds(NoBond())
        abset = oechem.OEAtomBondSet(atoms, bonds)
        oedepict.OEAddHighlighting(
            display,
            oechem.OEColor(oechem.OEYellow),
            oedepict.OEHighlightStyle_BallAndStick,
            abset,
        )

    oedepict.OERenderMolecule(img, display)
    png = oedepict.OEWriteImageToString("png", img)
    buf = io.BytesIO(png)
    return Image.open(buf)


# @click.command()
# @click.option(
#     "--file",
#     "file",
#     type=click.STRING,
#     required=True,
#     default='../qcainspect/urea.sdf',
#     help="filename to be read"
# )


def get_assigned_parameters(offmol, forcefield):
    topology = Topology.from_molecules([offmol])
    molecule_force_list = forcefield.label_molecules(topology)

    parameter_list = defaultdict(list)
    para_id = []
    for mol_idx, mol_forces in enumerate(molecule_force_list):
        for force_tag, force_dict in mol_forces.items():
            if force_tag == "ProperTorsions":
                for (atom_indices, parameter) in force_dict.items():
                    parameter_list[parameter.id].append(atom_indices)
                    para_id.append(parameter.id)
    return (sorted(set(para_id)), OrderedDict(sorted(parameter_list.items())))


@click.command()
@click.option(
    "--file",
    "file",
    type=click.STRING,
    required=True,
    default=None,
    help="first geometry file filename with location",
)
def main(file):
    mol = Molecule.from_file(file, allow_undefined_stereo=True)
    forcefield = ForceField("openff_unconstrained-2.0.0-rc.1.offxml")
    assigned_params, params_indices = get_assigned_parameters(mol, forcefield)
    oemol = mol.to_openeye()
    img = get_oemol_struc(oemol)
    c = canvas.Canvas(file + ".pdf")
    c.setFont("Helvetica", 6)
    c.drawInlineImage(img, 100, 700, width=img.size[0] / 2.5, height=img.size[1] / 2.5)
    text = c.beginText(20, 700)
    text.textLines(["Assigned torsion parameters and atom indices"])
    for key, values in params_indices.items():
        wrapped_text = "\n".join(wrap("- " + key + " : " + str(values), 230))
        text.textLines(wrapped_text)
    c.drawText(text)

    c.save()


if __name__ == "__main__":
    main()
