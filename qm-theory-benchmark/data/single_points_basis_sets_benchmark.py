import copy
import io
import json
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
REF_SPEC = (
    "df-ccsd(t)/cbs"  # B3LYP-D3BJ/DEF2-TZVPPD' #'MP2/heavy-aug-cc-pVTZ-constrained'
)


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
    default="/home/maverick/Desktop/OpenFF/dev-dir/qm-theory-benchmark/qm-theory-benchmark/data/qm-single-point"
            "-energies.pkl",
    help="pickle file in which the energies dict is stored",
)
def main(data_pickle):
    df_spe = pd.read_pickle(data_pickle)
    df_mp2 = pd.read_pickle("./torsiondrive_data.pkl")
    rcParams.update({"font.size": 12})
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
        "#01a263",
        "#fecb00",
        "#cd0d2d",
        "#00247d"
    ]
    pdf = PdfPages("../outputs/torsions_basis_sets_ccsd_cbs_ref.pdf")
    # for the sake of querying convenience listing out the keywords, methods and basis_sets sets explicitly
    keywords_list = [
        "default",
        "b3lyp-nl/dzvp",
        "b3lyp-d3bj/def2-tzvp",
        "b3lyp-d3bj/def2-tzvpd",
        "b3lyp-d3bj/def2-tzvpp",
        "b3lyp-d3bj/def2-tzvppd",
        "b3lyp-d3bj/def2-qzvp",
        "b3lyp-d3bj/6-31+g**",
        "b3lyp-d3bj/6-311+g**",
        "df-ccsd(t)/cbs",
    ]

    methods = [
        "b3lyp-d3bj",
        "b3lyp-nl",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "b3lyp-d3bj",
        "mp2/heavy-aug-cc-pv[tq]z + d:ccsd(t)/heavy-aug-cc-pvdz",
    ]

    basis_sets = [
        "dzvp",
        "dzvp",
        "def2-tzvp",
        "def2-tzvpd",
        "def2-tzvpp",
        "def2-tzvppd",
        "def2-qzvp",
        "6-31+g**",
        "6-311+g**",
        None,
    ]

    rmse = defaultdict(dict)
    mae = defaultdict(dict)
    angle_dict = json.load(
        open("./angle_indices_for_single_points.txt")
    )
    for i, index in enumerate(df_mp2.index):
        ref_angles = angle_dict[str(i)]
        energies_spe = []
        for kk in range(24):
            energies_spe.append(
                df_spe[
                    (
                        "-".join([str(i), str(kk)]),
                        "mp2/heavy-aug-cc-pv[tq]z + d:ccsd(" "t)/heavy-aug-cc-pvdz",
                        None,
                        "df-ccsd(t)/cbs",
                    )
                ]["method_returned_energy"]
            )

        energy_min = min(energies_spe)
        relative_energies = [
            (x - energy_min) * HARTREE_TO_KCALMOL for x in energies_spe
        ]
        ref_angles, relative_energies = zip(*sorted(zip(ref_angles, relative_energies)))
        ref_angles = np.array(ref_angles)
        relative_energies = np.array(relative_energies)
        ref_energies = np.array(relative_energies)
        mapped_smiles = df_mp2.loc[index, "MP2/heavy-aug-cc-pVTZ"][0]["mapped_smiles"]
        dihedrals = df_mp2.loc[index, "MP2/heavy-aug-cc-pVTZ"][0]["dihedrals"]
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

        for j, spec in enumerate(keywords_list):
            if spec == "df-ccsd(t)/cbs":
                continue

            if j != len(keywords_list) - 1:
                angles = angle_dict[str(i)]
                energies_spe = []
                for kk in range(24):
                    if "method_returned_energy" in list(
                        df_spe[
                            ("-".join([str(i), str(kk)]), methods[j], basis_sets[j], spec)
                        ].keys()
                    ):
                        energies_spe.append(
                            df_spe[
                                (
                                    "-".join([str(i), str(kk)]),
                                    methods[j],
                                    basis_sets[j],
                                    spec,
                                )
                            ]["method_returned_energy"]
                        )
                    else:
                        energies_spe.append(
                            df_spe[
                                (
                                    "-".join([str(i), str(kk)]),
                                    methods[j],
                                    basis_sets[j],
                                    spec,
                                )
                            ]["scf_plus_dispersion_energy"]
                        )

                energy_min = min(energies_spe)
                energies = [(x - energy_min) * HARTREE_TO_KCALMOL for x in energies_spe]
                angles, energies = zip(*sorted(zip(angles, energies)))
                angles = np.array(angles)
                energies = np.array(energies)
            elif j == len(keywords_list) - 1:
                angles = np.array(df_mp2.loc[index, spec][0]["angles"])
                energies = (
                    np.array(df_mp2.loc[index, spec][0]["relative_energies"])
                    * HARTREE_TO_KCALMOL
                )
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

    for spec in keywords_list:
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
    ax.set_ylim(0, 2.5)
    ax.yaxis.get_label().set_fontsize(10)
    ax.set_ylim(-0.1, 2.6)
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

    with open("../outputs/torsion_basis_sets_analysis_scores.txt", "w") as f:
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
