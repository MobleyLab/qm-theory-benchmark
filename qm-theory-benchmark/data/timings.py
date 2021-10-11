import io
import pickle
from collections import defaultdict

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from openff.toolkit.topology import Molecule

from visualization import show_oemol_struc

with open("../qca-util-scripts/spe-recs.pkl", "rb") as outfile:
    spe_data = pickle.load(outfile)


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
    # 'b97-d3bj/def2-tzvp',
    "m05-2x-d3/dzvp",
    "m06-2x-d3/dzvp",
    "m08-hx-d3/dzvp",
    # 'wb97x-d3bj/dzvp',
    # 'wb97m-d3bj/dzvp',
    "wb97m-v/dzvp",
    "pw6b95-d3bj/dzvp",
    "pw6b95-d3/dzvp",
    "b3lyp-d3mbj/dzvp",
    "mp2/aug-cc-pvtz",
    "dsd-blyp-d3bj/heavy-aug-cc-pvtz",
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
    # 'b97-d3bj',
    "m05-2x-d3",
    "m06-2x-d3",
    "m08-hx-d3",
    # 'wb97x-d3bj',
    # 'wb97m-d3bj',
    "wb97m-v",
    "pw6b95-d3bj",
    "pw6b95-d3",
    "b3lyp-d3mbj",
    "mp2",
    "dsd-blyp-d3bj",
    "mp2/heavy-aug-cc-pv[tq]z + d:ccsd(t)/heavy-aug-cc-pvdz",
]
basis = [
    "dzvp",
    "dzvp",
    "def2-tzvp",
    "def2-tzvpd",
    "def2-tzvpp",
    "def2-tzvppd",
    "def2-qzvp",
    "6-31+g**",
    "6-311+g**",
    # 'def2-tzvp',
    "dzvp",
    "dzvp",
    "dzvp",
    # 'dzvp',
    # 'dzvp',
    "dzvp",
    "dzvp",
    "dzvp",
    "dzvp",
    "aug-cc-pvtz",
    "heavy-aug-cc-pvtz",
    None,
]

same_cpu = defaultdict(list)
wall_time = defaultdict(float)
cpu_name = defaultdict(str)
threads = defaultdict(str)
for j in range(59):
    cpu = spe_data[(str(j) + '-' + str(0), methods[0], basis[0], keywords_list[0])]['provenance']['cpu']
    nthreads = spe_data[(str(j) + '-' + str(0), methods[0], basis[0], keywords_list[0])]['provenance']['nthreads']
    for i, key in enumerate(keywords_list):
        for k in range(24):
            cpu_n = spe_data[(str(j) + '-' + str(k), methods[i], basis[i], key)]['provenance']['cpu']
            nthreads_n = spe_data[(str(j) + '-' + str(k), methods[i], basis[i], key)]['provenance']['nthreads']
            if cpu_n == cpu and nthreads == nthreads_n:
                same_cpu[str(j)].append(True)
                wall_time[(j, key)] = spe_data[(str(j) + '-' + str(k), methods[i], basis[i], key)]['provenance'][
                    'wall_time']
                cpu_name[str(j)] = cpu
                threads[str(j)] = nthreads
                break

same_cpu_keys = []
for key, val in same_cpu.items():
    if len(val) == 19:
        same_cpu_keys.append(key)
        print(key, all(val))

pdf = PdfPages("../outputs/timings_spe_benchmark.pdf")
df_mp2 = pd.read_pickle("./torsiondrive_data.pkl")
indices = df_mp2.index
for same_cpu_key in same_cpu_keys:
    mapped_smiles = df_mp2.loc[indices[int(same_cpu_key)], "MP2/heavy-aug-cc-pVTZ"][0]["mapped_smiles"]
    dihedrals = df_mp2.loc[indices[int(same_cpu_key)], "MP2/heavy-aug-cc-pVTZ"][0]["dihedrals"]
    offmol = Molecule.from_mapped_smiles(mapped_smiles)
    oemol = offmol.to_openeye()
    image = show_oemol_struc(
        oemol, torsions=True, atom_indices=dihedrals, width=600, height=500
    )
    img = Image.open(io.BytesIO(image.data))
    im_arr = np.asarray(img)

    times = []
    for key in keywords_list:
        times.append(wall_time[(int(same_cpu_key), key)])
    tims, keys = zip(*sorted(zip(times, keywords_list)))
    x_pos = np.arange(len(keys))
    width = 0.8
    fig, ax = plt.subplots(figsize=[10, 8])
    plt.yscale('log')
    plt.bar(x_pos, tims, width)
    plt.xticks(x_pos + width / 2, keys, rotation=60, ha="right")
    plt.ylabel("Wall time in seconds (in log scale)")
    plt.title("mol: " + str(same_cpu_key) +", CPU: "+ cpu_name[str(same_cpu_key)] +", # of threads: "+ str(threads[str(
        same_cpu_key)]))
    newax = fig.add_axes([0.2, 0.5, 0.25, 0.25], anchor="SW", zorder=1)
    newax.imshow(im_arr)
    newax.axis("off")
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

pdf.close()
