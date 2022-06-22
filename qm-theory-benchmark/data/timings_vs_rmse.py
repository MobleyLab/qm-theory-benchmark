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

with open("/home/maverick/Desktop/OpenFF/dev-dir/qm-theory-benchmark/qm-theory-benchmark/data/qm-single-point-energies.pkl", "rb") as outfile:
    spe_data = pickle.load(outfile)

keywords_list = [
    "default",
    "wb97m-d3bj/dzvp",

]

methods = [
    "b3lyp-d3bj",
    "wb97m-d3bj",
]

basis_sets = [
    "dzvp",
    "dzvp"
]

same_cpu = defaultdict(list)
wall_time = defaultdict(float)
cpu_name = defaultdict(str)
threads = defaultdict(str)
for j in range(59):
    cpu = spe_data[(str(j) + "-" + str(0), methods[0], basis_sets[0], keywords_list[0])][
        "provenance"
    ]["cpu"]
    nthreads = spe_data[
        (str(j) + "-" + str(0), methods[0], basis_sets[0], keywords_list[0])
    ]["provenance"]["nthreads"]
    for i, key in enumerate(keywords_list):
        for k in range(24):
            cpu_n = spe_data[(str(j) + "-" + str(k), methods[i], basis_sets[i], key)][
                "provenance"
            ]["cpu"]
            nthreads_n = spe_data[(str(j) + "-" + str(k), methods[i], basis_sets[i], key)][
                "provenance"
            ]["nthreads"]
            if cpu_n == cpu and nthreads == nthreads_n:
                same_cpu[str(j)].append(True)
                wall_time[(j, key)] = spe_data[
                    (str(j) + "-" + str(k), methods[i], basis_sets[i], key)
                ]["provenance"]["wall_time"]
                cpu_name[str(j)] = cpu
                threads[str(j)] = nthreads
                break

same_cpu_keys = []
for key, val in same_cpu.items():
    if len(val) == 19:
        same_cpu_keys.append(key)
        print(key, all(val))

pdf = PdfPages("../outputs/timings_wb97_spe_benchmark_color.pdf")
df_mp2 = pd.read_pickle("./torsiondrive_data.pkl")
indices = df_mp2.index
for same_cpu_key in same_cpu_keys:
    mapped_smiles = df_mp2.loc[indices[int(same_cpu_key)], "MP2/heavy-aug-cc-pVTZ"][0][
        "mapped_smiles"
    ]
    dihedrals = df_mp2.loc[indices[int(same_cpu_key)], "MP2/heavy-aug-cc-pVTZ"][0][
        "dihedrals"
    ]
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
    plt.yscale("log")
    plt.bar(x_pos, tims, width, color="#d9863a")
    plt.xticks(x_pos, keys, rotation=60, ha="right")
    plt.ylabel("Wall time in seconds (in log scale)")
    plt.title(
        "mol: "
        + str(same_cpu_key)
        + ", CPU: "
        + cpu_name[str(same_cpu_key)]
        + ", # of threads: "
        + str(threads[str(same_cpu_key)])
    )
    newax = fig.add_axes([0.2, 0.5, 0.25, 0.25], anchor="SW", zorder=1)
    newax.imshow(im_arr)
    newax.axis("off")
    plt.show()
    pdf.savefig(fig, dpi=600, bbox_inches="tight")

pdf.close()
