from collections import defaultdict
import pandas as pd
import click
import matplotlib.pyplot as plt
import numpy as np
import qcportal as ptl
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from simtk import unit
from tabulate import tabulate

PARTICLE = unit.mole.create_unit(6.02214076e23 ** -1, "particle", "particle", )
HARTREE_PER_PARTICLE = unit.hartree / PARTICLE
HARTREE_TO_KCALMOL = HARTREE_PER_PARTICLE.conversion_factor_to(unit.kilocalorie_per_mole)
ESU_BOHR = unit.elementary_charge * unit.bohr
ESU_BOHR_TO_DEBYE = ESU_BOHR.conversion_factor_to(unit.debye)
BOLTZMANN_CONSTANT = unit.constants.BOLTZMANN_CONSTANT_kB
REF_SPEC = 'MP2/heavy-aug-cc-pVTZ-constrained'


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
    angle = np.pi * np.arccos(dot_product)  # * unit.degrees
    return mu_diff, angle

@click.command()
@click.option(
    "--data_pickle",
    "data_pickle",
    type=click.STRING,
    required=False,
    default='./torsiondrive_data.pkl',
    help="pickle file in which the energies dict is stored"
)
def main(data_pickle):
    df = pd.read_pickle(data_pickle)
    rcParams.update({'font.size': 14})
    KELLYS_COLORS = ["#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f", "#7f7e80", "#5fa641", "#d485b2",
                     "#4277b6", "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", "#6f340d",
                     "#d32b1e", "#2b3514",
                     ]
    specs = list(df.columns)
    specs.remove('MP2/aug-cc-pVTZ')
    specs.remove('MP2/heavy-aug-cc-pVTZ')
    specs.remove('WB97X-D3BJ/DZVP')
    specs.remove('PW6B95-D3BJ/DZVP')
    specs.remove('B3LYP-D3MBJ/DZVP')
    pdf = PdfPages('../outputs/dipoles_alltogther.pdf')
    spec_dipole_angle_diff_dict = defaultdict(list)
    spec_dipole_mu_diff_dict = defaultdict(list)
    for i, index in enumerate(df.index):
        try:
            ref_dipoles = np.array(df.loc[index, REF_SPEC][0]['dipoles']) * ESU_BOHR_TO_DEBYE
            ref_angles = np.array(df.loc[index, REF_SPEC][0]['angles'])
            if ref_angles[0] == -180:
                ref_angles = np.append(ref_angles[1:], ref_angles[0] + 360)
                ref_dipoles = np.append(ref_dipoles[1:], ref_dipoles[0])
        except:
            continue
        for j, spec in enumerate(specs):
            mu_diff_with_ref = defaultdict(float)
            angle_diff_with_ref = defaultdict(float)
            if spec == REF_SPEC:
                continue
            try:
                dipoles = np.array(df.loc[index, spec][0]['dipoles']) * ESU_BOHR_TO_DEBYE
            except:
                continue
            n_grid = len(dipoles)
            for k,item in enumerate(dipoles):
                mu_diff, angle_diff = diff_between_vectors(item, ref_dipoles[k])
                mu_diff_with_ref[spec] += mu_diff * mu_diff
                angle_diff_with_ref[spec] += angle_diff * angle_diff
            angle_diff_with_ref[spec] = np.sqrt(angle_diff_with_ref[spec] / n_grid)
            mu_diff_with_ref[spec] = np.sqrt(mu_diff_with_ref[spec] / n_grid)
            spec_dipole_angle_diff_dict[spec] = np.mean(list(angle_diff_with_ref.values()))
            spec_dipole_mu_diff_dict[spec] = np.mean(list(mu_diff_with_ref.values()))
    table = []
    xlabels = []
    angle_vals = []
    mu_vals = []
    for key, value in spec_dipole_angle_diff_dict.items():
        table.append([key, "%.4f" % spec_dipole_mu_diff_dict[key], "%.4f" % value])
        xlabels.append(key)
        angle_vals.append(spec_dipole_angle_diff_dict[key])
        mu_vals.append(spec_dipole_mu_diff_dict[key])

    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25
    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))
    plt.bar(x_pos, angle_vals, width, label="RMSE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha='right')
    plt.xlabel('RMSE of deviations in dipole directions wrt ' + REF_SPEC)
    plt.ylabel('RMSE in angles (degrees)')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25
    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))
    plt.bar(x_pos, mu_vals, width, label="RMSE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha='right')
    plt.xlabel('RMSE of dipole moments wrt ' + REF_SPEC)
    plt.ylabel('RMSE in Debye')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches='tight')
    pdf.close()

    print(tabulate(table, headers=['Specification', 'RMSE in dipole moments (Debye)', 'RMSE in dipole vector angle wrt '
                                                                                  'ref (degrees)'],
                   tablefmt='orgtbl'))
    print("* closer to zero the better")
    #
    with open('../outputs/dipoles_analysis_scores.txt', 'w') as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(tabulate(table, headers=['Specification', 'RMSE in dipole moments (Debye)', 'RMSE in dipole vector '
                                                                                           'angle wrt '
                                                                                    'ref (degrees)'
                                                                                    ''],
                         tablefmt='orgtbl'))
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
