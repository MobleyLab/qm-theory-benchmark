def show_oemol_struc(oemol, torsions=False, atom_indices=[], width=500, height=300):
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
    from IPython.display import Image
    from openeye import oechem, oedepict

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
    return Image(png)
