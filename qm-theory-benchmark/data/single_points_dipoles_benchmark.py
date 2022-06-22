from collections import defaultdict
import json
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from simtk import unit
from tabulate import tabulate

PARTICLE = unit.mole.create_unit(
    6.02214076e23 ** -1,
    "particle",
    "particle",
)
HARTREE_PER_PARTICLE = unit.hartree / PARTICLE
HARTREE_TO_KCALMOL = HARTREE_PER_PARTICLE.conversion_factor_to(
    unit.kilocalorie_per_mole
)
ESU_BOHR = unit.elementary_charge * unit.bohr
ESU_BOHR_TO_DEBYE = ESU_BOHR.conversion_factor_to(unit.debye)
BOLTZMANN_CONSTANT = unit.constants.BOLTZMANN_CONSTANT_kB
REF_SPEC = "MP2/heavy-aug-cc-pVTZ"


def diff_between_vectors(vec1, vec2):
    """
    gives out the magnitude difference and angle between vectors
    :param vec1:
    :param vec2:
    :return:
    """
    mu1 = np.linalg.norm(vec1)
    mu2 = np.linalg.norm(vec2)
    mu_diff = mu1 - mu2
    unit_vector_1 = vec1 / mu1
    unit_vector_2 = vec2 / mu2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.pi * np.arccos(np.round(dot_product, decimals=8))  # * unit.degrees
    return mu_diff, angle


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
    keywords_list = [
        "default",
        "b3lyp-nl/dzvp",
        # "b3lyp-d3bj/def2-tzvp",
        # "b3lyp-d3bj/def2-tzvpd",
        # "b3lyp-d3bj/def2-tzvpp",
        # "b3lyp-d3bj/def2-tzvppd",
        # "b3lyp-d3bj/def2-qzvp",
        # "b3lyp-d3bj/6-31+g**",
        # "b3lyp-d3bj/6-311+g**",
        "b97-d3bj/def2-tzvp",
        "m05-2x-d3/dzvp",
        "m06-2x-d3/dzvp",
        "m08-hx-d3/dzvp",
        # "wb97x-d3bj/dzvp", commented out since the dispersion energies are not handled properly in current qcf
        # version
        # for this functional
        "wb97m-d3bj/dzvp",
        "wb97m-v/dzvp",
        "pw6b95-d3bj/dzvp",
        "pw6b95-d3/dzvp",
        "b3lyp-d3mbj/dzvp",
        "mp2/aug-cc-pvtz",
        "mp2/heavy-aug-cc-pv(t+d)z",
        "dsd-blyp-d3bj/heavy-aug-cc-pvtz",
    ]

    methods = [
        "b3lyp-d3bj",
        "b3lyp-nl",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        # "b3lyp-d3bj",
        "b97-d3bj",
        "m05-2x-d3",
        "m06-2x-d3",
        "m08-hx-d3",
        # "wb97x-d3bj",
        "wb97m-d3bj",
        "wb97m-v",
        "pw6b95-d3bj",
        "pw6b95-d3",
        "b3lyp-d3mbj",
        "mp2",
        "mp2",
        "dsd-blyp-d3bj",
    ]

    basis_sets = [
        "dzvp",
        "dzvp",
        # "def2-tzvp",
        # "def2-tzvpd",
        # "def2-tzvpp",
        # "def2-tzvppd",
        # "def2-qzvp",
        # "6-31+g**",
        # "6-311+g**",
        "def2-tzvp",
        "dzvp",
        "dzvp",
        "dzvp",
        # "dzvp",
        "dzvp",
        "dzvp",
        "dzvp",
        "dzvp",
        "dzvp",
        "aug-cc-pvtz",
        "heavy-aug-cc-pv(t+d)z",
        "heavy-aug-cc-pvtz",
    ]

    pdf = PdfPages("../outputs/dipoles_alltogther_same_geo_v1.1.pdf")
    rmse_dipole_dict = defaultdict(dict)
    rmse_angle_dict = defaultdict(dict)
    angle_dict = json.load(open("./angle_indices_for_single_points.txt"))
    for i, index in enumerate(df_mp2.index):
        ref_dipoles = (
            np.array(df_mp2.loc[index, REF_SPEC][0]["dipoles"]) * ESU_BOHR_TO_DEBYE
        )

        mu_diff_with_ref = defaultdict(float)
        angle_diff_with_ref = defaultdict(float)

        for j, spec in enumerate(keywords_list):
            if spec == REF_SPEC:
                continue
            dipoles_spe = []
            angles = angle_dict[str(i)]
            print(spec)
            for kk in range(24):
                dipoles_spe.append(
                    df_spe[("-".join([str(i), str(kk)]), methods[j], basis_sets[j], spec)][
                        "properties"
                    ]["scf_dipole_moment"]
                )
            dipoles = np.array(dipoles_spe) * ESU_BOHR_TO_DEBYE
            angles, dipoles = zip(*sorted(zip(angles, dipoles)))

            n_grid = len(dipoles)
            if spec == "mp2/aug-cc-pvtz":
                print(i)
            for k, item in enumerate(dipoles):
                if spec == "mp2/aug-cc-pvtz":
                    print(k, item, ref_dipoles[k])
                mu_diff, angle_diff = diff_between_vectors(item, ref_dipoles[k])
                mu_diff_with_ref[spec] += mu_diff * mu_diff
                angle_diff_with_ref[spec] += angle_diff * angle_diff
            angle_diff_with_ref[spec] = np.sqrt(angle_diff_with_ref[spec] / n_grid)
            mu_diff_with_ref[spec] = np.sqrt(mu_diff_with_ref[spec] / n_grid)

            rmse_dipole_dict[index][spec] = mu_diff_with_ref[spec]
            rmse_angle_dict[index][spec] = angle_diff_with_ref[spec]

    table = []
    xlabels = []
    spec_angle_rmse = []
    spec_mu_rmse = []
    for spec in keywords_list:
        xlabels.append(spec)
        mu_tmp = 0
        ang_tmp = 0
        nvals = 0

        for ind in df_mp2.index:
            mu_tmp += rmse_dipole_dict[ind][spec] * rmse_dipole_dict[ind][spec]
            ang_tmp += rmse_angle_dict[ind][spec] * rmse_angle_dict[ind][spec]
            nvals += 1
        mu_tmp = np.sqrt(mu_tmp / nvals)
        ang_tmp = np.sqrt(ang_tmp / nvals)
        spec_angle_rmse.append(ang_tmp)
        spec_mu_rmse.append(mu_tmp)
        table.append([spec, "%.4f" % mu_tmp, "%.4f" % ang_tmp])

    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25
    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))
    plt.bar(x_pos, spec_angle_rmse, width, label="RMSE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha="right")
    plt.xlabel("RMSE of deviations in dipole directions wrt " + REF_SPEC)
    plt.ylabel("RMSE in dipole vector alignment (degrees)")
    plt.legend(loc="upper left", fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25
    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))
    plt.bar(x_pos, spec_mu_rmse, width, label="RMSE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha="right")
    plt.xlabel("RMSE of dipole moments wrt " + REF_SPEC)
    plt.ylabel("RMSE in Debye")
    plt.legend(loc="upper left", fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

    tmp_df = pd.DataFrame(rmse_dipole_dict)
    df = tmp_df.T
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, size=3, color="0.25")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.get_label().set_fontsize(12)
    ax.set(ylabel="RMSE over torsion profile dipole moments in debye")
    ax.yaxis.get_label().set_fontsize(10)
    plt.show()
    fig = ax.get_figure()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

    pdf.close()

    print(
        tabulate(
            table,
            headers=[
                "Specification",
                "RMSE in dipole moments (Debye)",
                "RMSE in dipole vector angle wrt " "ref (degrees)",
            ],
            tablefmt="orgtbl",
        )
    )
    print("* closer to zero the better")
    #
    with open("../outputs/dipoles_analysis_scores_same_geo_v1.1.txt", "w") as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(
            tabulate(
                table,
                headers=[
                    "Specification",
                    "RMSE in dipole moments (Debye)",
                    "RMSE in dipole vector " "angle wrt " "ref (degrees)" "",
                ],
                tablefmt="orgtbl",
            )
        )
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
