from collections import defaultdict

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
BOLTZMANN_CONSTANT = unit.constants.BOLTZMANN_CONSTANT_kB
REF_SPEC = 'B3LYP-D3BJ/DEF2-QZVP'

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
    angle = np.pi * np.arccos(dot_product) #* unit.degrees
    return mu_diff, angle



def main():

    #create a qcfractal client instance to connect to QCA server
    client = ptl.FractalClient()
    ds_list = client.list_collections('TorsionDriveDataset')
    matches = [x[1] for x in ds_list.index if (isinstance(x[1], str) and 'Theory Benchmark' in x[1])]
    print("\n".join(matches))
    # download the torsiondrive dataset
    ds = client.get_collection("TorsionDriveDataset", 'OpenFF Theory Benchmarking Set v1.0')
    ds.status()

    specifications = ['default', 'B3LYP-D3BJ/DEF2-TZVP', 'B3LYP-D3BJ/DEF2-TZVPP',
                       'B3LYP-D3BJ/DEF2-QZVP', 'B3LYP-D3BJ/6-31+G**',
                      'B3LYP-D3BJ/6-311+G**']
    #, 'B3LYP-D3BJ/DEF2-TZVPD', 'B3LYP-D3BJ/DEF2-TZVPPD', 'WB97X-D3BJ/DZVP', 'PW6B95-D3BJ/DZVP', 'B3LYP-D3MBJ/DZVP']
    # ds.list_specifications().index.to_list()
    print(specifications)
    rcParams.update({'font.size': 14})
    KELLYS_COLORS = ["#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f", "#7f7e80", "#5fa641", "#d485b2",
                     "#4277b6", "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948", "#7e1510", "#92ae31", "#6f340d",
                     "#d32b1e", "#2b3514",
                     ]

    pdf = PdfPages('../outputs/dipoles_alltogther.pdf')
    spec_dipole_angle_diff_dict = defaultdict(list)
    spec_dipole_mu_diff_dict = defaultdict(list)
    for i, entry in enumerate(ds.data.records.values()):
        if i>0:
            break
        # getting the reference method dipoles
        td_record = ds.get_record(name=entry.name, specification=REF_SPEC)
        ref_geo_dipole = defaultdict(list)
        angle_keys = td_record.dict()['final_energy_dict'].keys()
        for key in angle_keys:
            grid_final_opt_id = td_record.dict()['optimization_history'][key][-1]
            optrec = client.query_procedures(id=grid_final_opt_id)[0]
            ref_geo_dipole[key] = optrec.get_trajectory()[-1].properties.scf_dipole_moment
        mu_diff_with_ref = defaultdict(float)
        angle_diff_with_ref = defaultdict(float)
        for j, spec in enumerate(specifications):
            if spec == REF_SPEC:
                continue
            td_record = ds.get_record(name=entry.name, specification=spec)
            angle_keys = td_record.dict()['final_energy_dict'].keys()
            n_grid = len(angle_keys)
            for key in angle_keys:
                grid_final_opt_id = td_record.dict()['optimization_history'][key][-1]
                optrec = client.query_procedures(id=grid_final_opt_id)[0]
                final_geo_dipole = optrec.get_trajectory()[-1].properties.scf_dipole_moment
                mu_diff, angle_diff = diff_between_vectors(final_geo_dipole, ref_geo_dipole[key])
                mu_diff_with_ref[key] += mu_diff * mu_diff
                angle_diff_with_ref[key] += angle_diff * angle_diff
            angle_diff_with_ref[key] = np.sqrt(angle_diff_with_ref[key]/n_grid)
            mu_diff_with_ref[key] = np.sqrt(mu_diff_with_ref[key]/n_grid)
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
    plt.xticks(x_pos + width/2, xlabels, rotation=60, ha='right')
    plt.xlabel('RMSE of deviations in dipole of various methods wrt ' + REF_SPEC)
    plt.ylabel('RMSE')
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
    plt.xticks(x_pos + width/2, xlabels, rotation=60, ha='right')
    plt.xlabel('RMSE of dipole moments wrt ' + REF_SPEC)
    plt.ylabel('RMSE')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches='tight')
    pdf.close()

    print(tabulate(table, headers=['Specification', 'RMSE in deviations'], tablefmt='orgtbl'))
    print("* closer to zero the better")
    #
    with open('../outputs/dipoles_analysis_scores.txt', 'w') as f:
        f.write("Using " + REF_SPEC + " as a reference method the scores are: \n")
        f.write(tabulate(table, headers=['Specification', 'RMSE in dipole moments','RMSE in dipole vector angle wrt '
                                                                                   'ref'
                                                                                   ''],
                         tablefmt='orgtbl'))
        f.write("\n")
        f.write("* closer to zero the better")


if __name__ == "__main__":
    main()
