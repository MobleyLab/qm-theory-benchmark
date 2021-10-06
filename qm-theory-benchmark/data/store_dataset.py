import pickle
from collections import defaultdict

from qcportal import FractalClient
from tqdm import tqdm

client = FractalClient(verify=False)
ds = client.get_collection(
    "Dataset", "OpenFF Theory Benchmarking Single Point Energies v1.0"
)
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
    "wb97x-d3bj/dzvp",
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
    "wb97x-d3bj",
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
    "dzvp",
    # 'dzvp',
    "dzvp",
    "dzvp",
    "dzvp",
    "dzvp",
    "aug-cc-pvtz",
    "heavy-aug-cc-pvtz",
    None,
]

num_recs = len(ds.get_index())
spe_data = defaultdict(dict)
for i, key in tqdm(enumerate(keywords_list)):
    print(key)
    recs = ds.get_records(method=methods[i], basis=basis[i], keywords=key)
    length_of_recs = len(recs)
    if length_of_recs == num_recs and not isinstance(recs, list):
        for index in tqdm(recs.index):
            if recs.loc[index].record.status == "COMPLETE":
                spe_data[(index, methods[i], basis[i], key)] = recs.loc[
                    index
                ].record.dict()
                spe_data[(index, methods[i], basis[i], key)]["stdout"] = recs.loc[
                    index
                ].record.get_stdout()
                spe_data[(index, methods[i], basis[i], key)][
                    "method_return_energy"
                ] = recs.loc[index].record.properties.return_energy
                print(
                    key,
                    spe_data[(index, methods[i], basis[i], key)][
                        "method_return_energy"
                    ],
                )
    elif length_of_recs == 2 and isinstance(recs, list):
        for index in tqdm(recs[1].index):
            if (
                recs[0].loc[index].record.status == "COMPLETE"
                and recs[1].loc[index].record.status == "COMPLETE"
            ):
                spe_data[(index, methods[i], basis[i], key)][
                    "dispersion_correction"
                ] = (recs[0].loc[index].record.dict())
                spe_data[(index, methods[i], basis[i], key)]["dispersion_stdout"] = (
                    recs[0].loc[index].record.get_stdout()
                )
                spe_data[(index, methods[i], basis[i], key)] = (
                    recs[1].loc[index].record.dict()
                )
                spe_data[(index, methods[i], basis[i], key)]["stdout"] = (
                    recs[1].loc[index].record.get_stdout()
                )
                spe_data[(index, methods[i], basis[i], key)][
                    "scf_plus_disp_return_energy"
                ] = (
                    recs[0].loc[index].record.properties.return_energy
                    + recs[1].loc[index].record.properties.return_energy
                )
                print(
                    key,
                    spe_data[(index, methods[i], basis[i], key)][
                        "scf_plus_disp_return_energy"
                    ],
                )

with open("spe-recs.pkl", "wb") as outfile:
    pickle.dump(spe_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
