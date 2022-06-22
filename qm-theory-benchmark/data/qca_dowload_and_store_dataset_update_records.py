import pickle
from collections import defaultdict
from qcportal import FractalClient
from tqdm import tqdm

# initializing a client instance to connect to QCArchive and getting the dataset
client = FractalClient.from_file()
ds = client.get_collection("Dataset", "OpenFF Theory Benchmarking Single Point Energies v1.0")

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
    "b97-d3bj/def2-tzvp",
    "m05-2x-d3/dzvp",
    "m06-2x-d3/dzvp",
    "m08-hx-d3/dzvp",
    # "wb97x-d3bj/dzvp", commented out since the dispersion energies are not handled properly in current qcf version 
    # for this functional
    "wb97m-d3bj/dzvp",
    "wb97m-v/dzvp",
    "pw6b95-d3bj/dzvp",
    "pw6b95-d3/dzvp",
    "b3lyp-d3mbj/dzvp",
    "mp2/aug-cc-pvtz",
    "mp2/heavy-aug-cc-pv(t+d)z",
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
    None,
]

# Dowloading the data for each of the functional/basis-set combo
# - QCF handles the calculations of single points that contain dispersion terms in a different way, as in the
#   calculation is split into two parts the original functional energy calculation, done with psi4, and the additional
#   dispersion term, done with dftd3. This granularity facilitates reuse of already calculated dispersion energy in
#   case of another functional.
# - There are unique cases of wb97* functionals where the dispersion term is handled in conjunction with the functional
#   itself. These are identified and a fix was applied. For reference, checkout https://github.com/MolSSI/QCFractal/issues/688
# - One caveat here is that this results in the data being stored either as a single dataframe or a list of two
#   dataframes, first one being the psi4 calculation and second one being the dispersion energy term
# - In the following piece of code you can see an if..else part where this check is done and data is downloaded
#   accordingly

# there are 59 torsiondrives(molecules) with 24 grid points making the total number of single points to be 1416
number_of_records = len(ds.get_index())
single_points_data = defaultdict(dict)

# Iterating over each of those records and storing data required for analysis in a pickle file
for i, key in tqdm(enumerate(keywords_list)):
    print(key)
    records = ds.get_records(method=methods[i], basis=basis_sets[i], keywords=key)
    length_of_records = len(records)
    # checking whether it is a single calculation or a split calculation, where dispersion is evaluated by
    # dftd3 separate from the psi4 functional calculation
    if length_of_records == number_of_records and not isinstance(records, list):
        for index in tqdm(records.index):
            if records.loc[index].record.status == "COMPLETE":
                single_points_data[(index, methods[i], basis_sets[i], key)] = records.loc[
                    index
                ].record.dict()
                single_points_data[(index, methods[i], basis_sets[i], key)]["stdout"] = records.loc[
                    index
                ].record.get_stdout()
                single_points_data[(index, methods[i], basis_sets[i], key)][
                    "method_returned_energy"
                ] = records.loc[index].record.properties.return_energy
                print(
                    key,
                    single_points_data[(index, methods[i], basis_sets[i], key)][
                        "method_returned_energy"
                    ],
                )
    elif length_of_records == 2 and isinstance(records, list):
        for index in tqdm(records[1].index):
            if (
                    records[0].loc[index].record.status == "COMPLETE"
                    and records[1].loc[index].record.status == "COMPLETE"
            ):
                single_points_data[(index, methods[i], basis_sets[i], key)] = (
                    records[1].loc[index].record.dict()
                )
                single_points_data[(index, methods[i], basis_sets[i], key)][
                    "dispersion_correction"
                ] = (records[0].loc[index].record.dict())
                single_points_data[(index, methods[i], basis_sets[i], key)]["dispersion_stdout"] = (
                    records[0].loc[index].record.get_stdout()
                )

                single_points_data[(index, methods[i], basis_sets[i], key)]["stdout"] = (
                    records[1].loc[index].record.get_stdout()
                )
                single_points_data[(index, methods[i], basis_sets[i], key)][
                    "scf_plus_dispersion_energy"
                ] = (
                        records[0].loc[index].record.properties.return_energy
                        + records[1].loc[index].record.properties.return_energy
                )
                print(
                    key,
                    single_points_data[(index, methods[i], basis_sets[i], key)][
                        "scf_plus_dispersion_energy"
                    ],
                )

# Writing to pickle file
with open("qm-single-point-energies.pkl", "wb") as outfile:
    pickle.dump(single_points_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
