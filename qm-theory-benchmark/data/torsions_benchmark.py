import copy
import io
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit.topology import Molecule
from simtk import unit
from tabulate import tabulate

from visualization import show_oemol_struc

PARTICLE = unit.mole.create_unit(6.02214076e23 ** -1, "particle", "particle", )
HARTREE_PER_PARTICLE = unit.hartree / PARTICLE
HARTREE_TO_KCALMOL = HARTREE_PER_PARTICLE.conversion_factor_to(unit.kilocalorie_per_mole)
BOLTZMANN_CONSTANT = unit.constants.BOLTZMANN_CONSTANT_kB
REF_SPEC = 'MP2/heavy-aug-cc-pVTZ-constrained'


def boltzmann_weight(value, temp=None):
    """
    returns exponential(-value/kbT), value is in kcal/mol
    :param value:
    :param temp:
    :return:
    """
    if temp == None:
        temp = np.inf * unit.kelvin
    kbt = BOLTZMANN_CONSTANT * temp
    kbt_in_kcalmol = kbt.value_in_unit(unit.kilocalorie)
    factor = np.exp(-value / kbt_in_kcalmol)
    return factor


def weight_fn(value):
    """
        Weights as per forcebalance when the relative energy falls into different ranges, the thresholds are in kcal/mol
        :param value:
        :return:
        """
    if value < 1:
        return 1
    elif value >= 1 or value < 5:
        return np.sqrt(1 + (value - 1) * (value - 1))
    elif value >= 5:
        return 0


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
            ener_diff_sqrd = np.multiply(np.subtract(value[1], ref_ener[i][1]), np.subtract(value[1], ref_ener[i][1]))
            score[key] += np.sqrt(np.sum(ener_diff_sqrd / n_grid))
        score[key] = score[key] / n_val
    return score


def get_qb_score(spec_ener_dict):
    """
    Define the score of method wrt the reference method in the same fashion as qubekit defines its
        objective function for torsions fitting
    :param spec_ener_dict:
    :return:
    """

    ref_ener = spec_ener_dict[REF_SPEC]
    other_ener = copy.deepcopy(spec_ener_dict)
    other_ener.pop(REF_SPEC)
    boltzmann_factor = [list(map(boltzmann_weight, sub_list[1])) for sub_list in ref_ener]
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            n_grid = len(value[1])
            ener_diff_sqrd = np.multiply(np.subtract(value[1], ref_ener[i][1]), np.subtract(value[1], ref_ener[i][1]))
            score[key] += np.sqrt(np.sum(np.multiply(boltzmann_factor[i], ener_diff_sqrd)) / n_grid)
        score[key] = score[key] / n_val
    return score


def get_fb_score(spec_ener_dict):
    """
        Define the score of method wrt the reference method in the same fashion as forcebalance defines its
        objective function for torsions fitting
        function. Energies are in units of kcal/mol here.
        :param spec_ener_dict: dict of energy lists per specification
        :return score: a list of cumulative scores for all molecules per specification
        """

    ref_ener = df.loc[REF_SPEC]
    other_ener = copy.deepcopy(spec_ener_dict)
    other_ener.pop(REF_SPEC)
    ref_weights = [list(map(weight_fn, sub_list[1])) for sub_list in ref_ener]
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            score[key] += np.divide(np.sum(np.multiply(ref_weights[i], np.subtract(value[1], ref_ener[i][1]) *
                                                       np.subtract(
                                                           value[1], ref_ener[i][1]))),
                                    np.sum(ref_weights[i]))
        score[key] = score[key] / n_val
    return score


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

    pdf = PdfPages('../outputs/test_torsions_alltogther.pdf')

    specs = list(df.columns)
    specs.remove('MP2/aug-cc-pVTZ')
    specs.remove('MP2/heavy-aug-cc-pVTZ')
    specs.remove('WB97X-D3BJ/DZVP')
    specs.remove('PW6B95-D3BJ/DZVP')
    specs.remove('B3LYP-D3MBJ/DZVP')

    rmse = defaultdict(dict)
    mae = defaultdict(dict)
    for i, index in enumerate(df.index):
        rmse['index'] = {}
        mae['index'] = {}

        try:
            ref_angles = np.array(df.loc[index, REF_SPEC][0]['angles'])
            ref_energies = np.array(df.loc[index, REF_SPEC][0]['relative_energies']) * HARTREE_TO_KCALMOL
            if ref_angles[0] == -180:
                ref_angles = np.append(ref_angles[1:], ref_angles[0] + 360)
                ref_energies = np.append(ref_energies[1:], ref_energies[0])
            mapped_smiles = df.loc[index, REF_SPEC][0]['mapped_smiles']
            dihedrals = df.loc[index, REF_SPEC][0]['dihedrals']
            fig, ax = plt.subplots(figsize=[10, 8])
            ax.plot(ref_angles, ref_energies, '-D', label=REF_SPEC, linewidth=3.0, c='k',
                    markersize=10)
        except:
            continue
        for j, spec in enumerate(specs):
            if spec == REF_SPEC:
                continue
            try:
                angles = np.array(df.loc[index, spec][0]['angles'])
                energies = np.array(df.loc[index, spec][0]['relative_energies']) * HARTREE_TO_KCALMOL
                rmse_energies = np.sqrt(np.mean((energies - ref_energies) ** 2))
                mae_energies = np.mean(np.abs(energies - ref_energies))
                rmse['index'][spec] = rmse_energies
                mae['index'][spec] = mae_energies
                if spec == 'default':
                    spec = 'B3LYP-D3BJ/DZVP'
                ax.plot(angles, energies, '-o', label=spec, linewidth=2.0, c=KELLYS_COLORS[j + 1])
            except:
                continue


        plt.xlabel('Dihedral angle in degrees', )
        plt.ylabel('Relative energies in kcal/mol')
        plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0), fontsize=12)
        offmol = Molecule.from_mapped_smiles(mapped_smiles)
        oemol = offmol.to_openeye()
        image = show_oemol_struc(oemol, torsions=True, atom_indices=dihedrals, width=600, height=500)
        img = Image.open(io.BytesIO(image.data))
        im_arr = np.asarray(img)
        newax = fig.add_axes([0.9, 0.4, 0.5, 0.5], anchor='SW', zorder=-1)
        newax.imshow(im_arr)
        newax.axis('off')
        plt.show()
        pdf.savefig(fig, dpi=600, bbox_inches='tight')

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
            try:
                for key, value in rmse.items():
                    tmp1 += rmse[key][spec]
                    tmp2 += mae[key][spec]
            except:
                pass
        rmse_vals.append(tmp1)
        mae_vals.append(tmp2)
        table.append([spec, "%.4f" % tmp1, "%.4f" % tmp2])
    # RMSE, MAE
    fig, ax = plt.subplots(figsize=[10, 8])
    # Width of a bar
    width = 0.25

    # Position of bars on x-axis
    x_pos = np.arange(len(xlabels))

    plt.bar(x_pos, rmse_vals, width, label="RMSE")
    plt.bar(x_pos + width, mae_vals, width, label="MAE")
    # Rotation of the bars names
    plt.xticks(x_pos + width / 2, xlabels, rotation=60, ha='right')
    plt.xlabel('Scores of various methods wrt ' + REF_SPEC)
    plt.ylabel('RMSE, MAE in kcal/mol')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches='tight')
    pdf.close()

    print(tabulate(table, headers=['Mol index', 'RMSE in kcal/mol', 'MAE in kcal/mol'],
                   tablefmt='orgtbl'))
    print("* closer to zero the better")

    with open('../outputs/torsion_analysis_scores.txt', 'w') as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(tabulate(table, headers=['Specification', 'FB score', 'QK score'], tablefmt='orgtbl'))
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
