import copy
import io
from ast import literal_eval
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import qcportal as ptl
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
REF_SPEC = 'B3LYP-D3BJ/DEF2-QZVP'


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
        weights as per forcebalance when the energy falls into different ranges, the thresholds are in kcal/mol
        :param value:
        :return:
        """
    if value < 1:
        return 1
    elif value >= 1 or value < 5:
        return np.sqrt(1 + (value - 1) * (value - 1))
    elif value >= 5:
        return 0


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
    boltzmann_factor = [list(map(boltzmann_weight, sub_list)) for sub_list in ref_ener]
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            n_grid = len(value)
            ener_diff_sqrd = np.multiply(np.subtract(value, ref_ener[i]), np.subtract(value, ref_ener[i]))
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
    ref_ener = spec_ener_dict[REF_SPEC]
    other_ener = copy.deepcopy(spec_ener_dict)
    other_ener.pop(REF_SPEC)
    ref_weights = [list(map(weight_fn, sub_list)) for sub_list in ref_ener]
    score = defaultdict(float)
    for key, values in other_ener.items():
        n_val = len(values)
        for i, value in enumerate(values):
            score[key] += np.divide(np.sum(np.multiply(ref_weights[i], np.subtract(value, ref_ener[i]))),
                                    np.sum(ref_weights[i]))
        score[key] = score[key] / n_val
    return score


def main():
    client = ptl.FractalClient()
    ds_list = client.list_collections('TorsionDriveDataset')
    matches = [x[1] for x in ds_list.index if (isinstance(x[1], str) and 'Theory Benchmark' in x[1])]
    print("\n".join(matches))
    ds = client.get_collection("TorsionDriveDataset", 'OpenFF Theory Benchmarking Set v1.0')
    ds.status()
    specifications = ds.list_specifications().index.to_list()
    print(specifications)
    rcParams.update({'font.size': 14})
    KELLYS_COLORS = ["#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f", "#7f7e80", "#5fa641", "#d485b2",
                     "#4277b6", "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", "#6f340d",
                     "#d32b1e", "#2b3514",
                     ]

    pdf = PdfPages('../outputs/torsions_alltogther.pdf')
    spec_ener_dict = defaultdict(list)
    for i, entry in enumerate(ds.data.records.values()):
        fig, ax = plt.subplots(figsize=[10, 8])
        for j, spec in enumerate(specifications):
            td_record = ds.get_record(name=entry.name, specification=spec)
            final_energy_dict = td_record.dict()['final_energy_dict']
            dihedrals = td_record.dict()['keywords']['dihedrals'][0]
            angles = []
            energies = []
            if not bool(final_energy_dict):
                print(spec, entry.dict()['object_map'][spec], " is empty")
                continue
            for key, value in final_energy_dict.items():
                angles.append(literal_eval(key)[0])
                energies.append(value)
            angles, energies = zip(*sorted(zip(angles, energies)))
            energy_min = min(energies)
            relative_energies = [(x - energy_min) * HARTREE_TO_KCALMOL for x in energies]
            spec_ener_dict[spec].append(relative_energies)
            if spec == 'B3LYP-D3BJ/DEF2-QZVP':
                ax.plot(angles, relative_energies, '-D', label=spec, linewidth=3.0, c='k', markersize=10)
            else:
                ax.plot(angles, relative_energies, '-o', label=spec, linewidth=2.0, c=KELLYS_COLORS[j])

        plt.xlabel('Dihedral angle in degrees', )
        plt.ylabel('Relative energies in kcal/mol')
        plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0), fontsize=12)
        mapped_smiles = ds.get_entry(entry.name).attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
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
    pdf.close()

    fb_score = get_fb_score(spec_ener_dict)
    qb_score = get_qb_score(spec_ener_dict)
    table = []
    for key, value in qb_score.items():
        table.append([key, "%.4f" % fb_score[key], "%.4f" % value])
    print(tabulate(table, headers=['Specification', 'FB score', 'QK score'], tablefmt='orgtbl'))
    print("* closer to zero the better")

    with open('../outputs/torsion_analysis_scores.txt', 'w') as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(tabulate(table, headers=['Specification', 'FB score', 'QK score'], tablefmt='orgtbl'))
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
