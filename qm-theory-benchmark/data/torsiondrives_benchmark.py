import copy
import io
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit.topology import Molecule
from PIL import Image
from simtk import unit
from tabulate import tabulate
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


def get_mae_score(spec_ener_dict):
    """
    Gives the RMSE of each key/spec
    :param spec_ener_dict:
    :return:
    """

    ref_ener = spec_ener_dict[REF_SPEC]
    other_ener = copy.deepcopy(spec_ener_dict)
    other_ener.pop(REF_SPEC)
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            n_grid = len(value[1])
            ener_diff_abs = np.abs(np.subtract(value[1], ref_ener[i][1]))
            score[key] += np.sum(ener_diff_abs / n_grid)
        score[key] = score[key] / n_val
    return score


def get_rmse_score(spec_ener_dict):
    """
    Gives the RMSE of each key/spec
    :param spec_ener_dict:
    :return:
    """

    ref_ener = spec_ener_dict[REF_SPEC]
    other_ener = copy.deepcopy(spec_ener_dict)
    other_ener.pop(REF_SPEC)
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            n_grid = len(value[1])
            ener_diff_sqrd = np.multiply(
                np.subtract(value[1], ref_ener[i][1]),
                np.subtract(value[1], ref_ener[i][1]),
            )
            score[key] += np.sqrt(np.sum(ener_diff_sqrd / n_grid))
        score[key] = score[key] / n_val
    return score


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

    pdf = PdfPages("../outputs/torsiondrives_benchmark.pdf")

    specs = [
        "default",
        "B3LYP-D3BJ/DEF2-TZVP",
        "B3LYP-D3BJ/DEF2-QZVP",
        "B3LYP-D3BJ/DEF2-TZVPP",
        "B3LYP-D3BJ/DEF2-TZVPD",
        "B3LYP-D3BJ/DEF2-TZVPPD",
        "B3LYP-D3MBJ/DZVP",
        "WB97X-D3BJ/DZVP",
        "PW6B95-D3BJ/DZVP",
        "MP2/aug-cc-pVTZ",
        "MP2/heavy-aug-cc-pVTZ",
        "B3LYP-D3BJ/6-31+G**",
        "B3LYP-D3BJ/6-311+G**",
    ]

    rmse = defaultdict(dict)
    mae = defaultdict(dict)
    for i, index in enumerate(df.index):
        ref_angles = np.array(df.loc[index, REF_SPEC][0]["angles"])
        ref_energies = (
            np.array(df.loc[index, REF_SPEC][0]["relative_energies"])
            * HARTREE_TO_KCALMOL
        )
        if ref_angles[0] == -180:
            ref_angles = np.append(ref_angles[1:], ref_angles[0] + 360)
            ref_energies = np.append(ref_energies[1:], ref_energies[0])
        mapped_smiles = df.loc[index, REF_SPEC][0]["mapped_smiles"]
        dihedrals = df.loc[index, REF_SPEC][0]["dihedrals"]
        fig, ax = plt.subplots(figsize=[10, 8])
        ax.plot(
            ref_angles,
            ref_energies,
            "-D",
            label=REF_SPEC,
            linewidth=3.0,
            c="k",
            markersize=10,
        )

        for j, spec in enumerate(specs):
            if spec == REF_SPEC:
                continue
            try:
                angles = np.array(df.loc[index, spec][0]["angles"])
                energies = (
                    np.array(df.loc[index, spec][0]["relative_energies"])
                    * HARTREE_TO_KCALMOL
                )
                if angles[0] == -180:
                    angles = np.append(angles[1:], angles[0] + 360)
                    energies = np.append(energies[1:], energies[0])
                rmse_energies = np.sqrt(np.mean((energies - ref_energies) ** 2))
                mae_energies = np.mean(np.abs(energies - ref_energies))
                if spec == "default":
                    print("rmse:", rmse_energies, " mae:", mae_energies)
                rmse[index][spec] = rmse_energies
                mae[index][spec] = mae_energies
                if spec == "default":
                    spec = "B3LYP-D3BJ/DZVP"
                    ax.plot(
                        angles,
                        energies,
                        "-v",
                        label=spec,
                        linewidth=2.0,
                        c=KELLYS_COLORS[j + 1],
                        markersize=10,
                    )
                elif spec == REF_SPEC:
                    ax.plot(
                        angles,
                        energies,
                        "-o",
                        label=REF_SPEC,
                        linewidth=2.0,
                        c=KELLYS_COLORS[j + 1],
                    )
                else:
                    ax.plot(
                        angles,
                        energies,
                        "-o",
                        label=spec,
                        linewidth=2.0,
                        c=KELLYS_COLORS[j + 1],
                    )
            except Exception:
                print("Failed ", i, index, "in spec: ", spec)
                continue

        plt.xlabel(
            "Dihedral angle in degrees",
        )
        plt.ylabel("Relative energies in kcal/mol")
        plt.legend(loc="lower left", bbox_to_anchor=(1.04, 0), fontsize=12)
        offmol = Molecule.from_mapped_smiles(mapped_smiles)
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

    table = []
    xlabels = []
    rmse_vals = []
    mae_vals = []

    for spec in specs:
        if spec == REF_SPEC:
            continue
        else:
            xlabels.append(spec)
            tmp1 = 0
            tmp2 = 0
            nvals = 0
            for index, value in rmse.items():
                if spec in rmse[index].keys():
                    tmp1 += rmse[index][spec] * rmse[index][spec]
                    tmp2 += mae[index][spec]
                    nvals += 1

            tmp1 = np.sqrt(tmp1 / nvals)
            tmp2 = tmp2 / nvals

        rmse_vals.append(tmp1)
        mae_vals.append(tmp2)
        table.append([spec, "%.4f" % tmp1, "%.4f" % tmp2])
    # RMSE, MAE
    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25

    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))

    plt.bar(x_pos, rmse_vals, width, color=KELLYS_COLORS[1], label="RMSE")
    plt.bar(x_pos + width, mae_vals, width, color=KELLYS_COLORS[2], label="MAE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha="right")
    plt.xlabel("Scores of various methods wrt " + REF_SPEC)
    plt.ylabel("RMSE, MAE in kcal/mol")
    plt.legend(loc="upper left", fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

    tmp_df = pd.DataFrame(rmse)
    df = tmp_df.T
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, size=3, color="0.25")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.get_label().set_fontsize(12)
    ax.set(ylabel="RMSE in kcal/mol")
    ax.yaxis.get_label().set_fontsize(10)
    fig = ax.get_figure()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

    tmp_df = pd.DataFrame(mae)
    df = tmp_df.T
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, size=3, color="0.25")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.get_label().set_fontsize(12)
    ax.set(ylabel="MAE in kcal/mol")
    ax.yaxis.get_label().set_fontsize(10)
    fig = ax.get_figure()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")
    pdf.close()

    print(
        tabulate(
            table,
            headers=["Specification", "RMSE in kcal/mol", "MAE in kcal/mol"],
            tablefmt="orgtbl",
        )
    )
    print("* closer to zero the better")

    with open("../outputs/torsion_analysis_scores_v5.txt", "w") as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(
            tabulate(
                table,
                headers=["Specification", "RMSE in kcal/mol", "MAE in kcal/mol"],
                tablefmt="orgtbl",
            )
        )
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
