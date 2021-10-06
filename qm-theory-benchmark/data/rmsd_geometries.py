import io
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from forcebalance.molecule import Molecule as fb_molecule
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit.topology import Molecule
from PIL import Image
from simtk import unit
from visualization import show_oemol_struc

PARTICLE = unit.mole.create_unit(
    6.02214076e23 ** -1,
    "particle",
    "particle",
)
HARTREE_PER_PARTICLE = unit.hartree / PARTICLE
HARTREE_TO_KCALMOL = HARTREE_PER_PARTICLE.conversion_factor_to(
    unit.kilocalorie_per_mole
)
BOLTZMANN_CONSTANT = unit.constants.BOLTZMANN_CONSTANT_kB
REF_SPEC = "MP2/heavy-aug-cc-pVTZ"
BOHR_TO_ANG = unit.bohr.conversion_factor_to(unit.angstrom)


def get_rmsd(initial, final):
    """
    Evaluate the RMSD between two arrays
    """
    assert len(initial) == len(final)
    n = len(initial)
    if n == 0:
        return 0.0
    diff = np.subtract(initial, final)
    rmsd = np.sqrt(np.sum(diff ** 2) / n)
    return rmsd


@click.command()
@click.option(
    "--data_pickle",
    "data_pickle",
    type=click.STRING,
    required=False,
    default="./torsiondrive_data.pkl",
    help="pickle file in which the energies dict is stored",
)
def main(data_pickle):
    df = pd.read_pickle(data_pickle)

    rcParams.update({"font.size": 14})
    KELLYS_COLORS = [
        "#ebce2b",
        "#db6917",
        "#96cde6",
        "#ba1c30",
        "#c0bd7f",
        "#7f7e80",
        "#5fa641",
        "#d485b2",
        "#4277b6",
        "#92ae31",
        "#463397",
        "#e1a11a",
        "#6f340d",
        "#e8e948",
        "#d32b1e",
        "#df8461",
        "#2b3514",
        "#702c8c",
        "#7e1510",
        "#91218c",
    ]

    pdf = PdfPages("../outputs/rmsds_of_geoms.pdf")

    specs = list(df.columns)

    rmsd_dict = defaultdict(dict)
    geom_dict = defaultdict(dict)
    for i, index in enumerate(df.index):
        ref_geoms = []
        ref_angles = []
        for key, item in df.loc[index, REF_SPEC][0]["final_molecules"].items():
            ref_angles.append(key[0])
            key_geo = np.array(item.geometry) * BOHR_TO_ANG
            ref_geoms.append(key_geo)
        if ref_angles[0] == -180:
            ref_angles[0] = ref_angles[0] + 360
        ref_angles, ref_geoms = zip(*sorted(zip(ref_angles, ref_geoms)))

        geom_dict[index + REF_SPEC] = dict(zip(ref_angles, ref_geoms))
        mapped_smiles = df.loc[index, REF_SPEC][0]["mapped_smiles"]
        offmol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
        with NamedTemporaryFile(suffix=".xyz") as iofile:
            offmol.to_file(iofile.name, "xyz")
            fbmol = fb_molecule(iofile.name)
        for item in ref_geoms:
            offmol.add_conformer(item * unit.angstrom)

        Path("../outputs/init_mols/" + str(i)).mkdir(parents=True, exist_ok=True)
        fname = (
            "../outputs/init_mols/"
            + str(i)
            + "/"
            + str(i)
            + "_"
            + REF_SPEC.replace("/", "-")
            + ".xyz"
        )
        offmol.to_file(fname, file_format="xyz")
        dihedrals = df.loc[index, REF_SPEC][0]["dihedrals"]
        fig, ax = plt.subplots(figsize=[10, 8])

        for j, spec in enumerate(specs):
            if spec == REF_SPEC:
                continue
            try:
                angles = []
                geoms = []
                rmsds = []
                for key, item in df.loc[index, spec][0]["final_molecules"].items():
                    angles.append(key[0])
                    key_geo = np.array(item.geometry) * BOHR_TO_ANG
                    copy_of_geom = deepcopy(key_geo)
                    fbmol.xyzs = [geom_dict[index + REF_SPEC][key[0]], copy_of_geom]
                    fbmol.align(center=False)
                    rms = get_rmsd(fbmol.xyzs[0], fbmol.xyzs[1])
                    rmsds.append(rms)
                    geoms.append(fbmol.xyzs[1])
                rmsd_dict[index][spec] = np.sqrt(
                    np.sum([t * t / len(rmsds) for t in rmsds])
                )
                geom_dict[index + spec] = dict(zip(angles, geoms))
                angles, rmsds, geoms = zip(*sorted(zip(angles, rmsds, geoms)))

                if spec == "default":
                    spec = "B3LYP-D3BJ/DZVP"
                    ax.plot(
                        angles,
                        rmsds,
                        "-v",
                        label=spec,
                        linewidth=2.0,
                        c=KELLYS_COLORS[j + 1],
                        markersize=10,
                    )
                else:
                    ax.plot(
                        angles,
                        rmsds,
                        "-o",
                        label=spec,
                        linewidth=2.0,
                        c=KELLYS_COLORS[j + 1],
                    )

                offmol_out = Molecule.from_mapped_smiles(
                    mapped_smiles, allow_undefined_stereo=True
                )
                for item in geoms:
                    offmol_out.add_conformer(item * unit.angstrom)
                fname = (
                    "../outputs/init_mols/"
                    + str(i)
                    + "/"
                    + str(i)
                    + "_"
                    + spec.replace("/", "-")
                    + ".xyz"
                )
                offmol_out.to_file(fname, file_format="xyz")
            except Exception:
                print(i, "and ", spec, "failed, skipping it")
                continue

        plt.xlabel(
            "Dihedral angle in degrees",
        )
        plt.ylabel("RMSD wrt REF_SPEC in Ang.")
        plt.legend(loc="lower left", bbox_to_anchor=(1.04, 0), fontsize=12)

        oemol = offmol.to_openeye()
        image = show_oemol_struc(
            oemol, torsions=True, atom_indices=dihedrals, width=600, height=500
        )
        img = Image.open(io.BytesIO(image.data))
        im_arr = np.asarray(img)
        newax = fig.add_axes([0.9, 0.6, 0.35, 0.35], anchor="SW", zorder=-1)
        newax.imshow(im_arr)
        newax.axis("off")
        plt.show()
        pdf.savefig(fig, dpi=600, bbox_inches="tight")

    tmp_df = pd.DataFrame(rmsd_dict)
    df = tmp_df.T
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, size=3, color="0.25")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.get_label().set_fontsize(12)
    ax.set(ylabel="RMSD wrt reference over torsion profile in angstrom")
    ax.yaxis.get_label().set_fontsize(10)
    plt.show()
    fig = ax.get_figure()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")
    pdf.close()


if __name__ == "__main__":
    main()
