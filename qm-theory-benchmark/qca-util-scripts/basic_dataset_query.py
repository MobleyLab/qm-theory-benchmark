from collections import defaultdict

import pandas as pd
from openff.qcsubmit.datasets import BasicDataset
from qcfractal.interface import FractalClient

client = FractalClient()
ds = client.get_collection(
    "Dataset", "OpenFF Theory Benchmarking Single Point Energies v1.0"
)
bds = BasicDataset.parse_file(
    "/home/maverick/Desktop/OpenFF/qca-dataset-submission/submissions/2021-09-06-theory-bm-single-points/dataset.json"
    ".bz2"
)
keywords_list = list(bds.qc_specifications.keys())

methods = []
basis = []
for key in keywords_list:
    methods.append(bds.qc_specifications[key].method)
    basis.append(bds.qc_specifications[key].basis)

status_dict = defaultdict(dict)
for i, key in enumerate(keywords_list):
    recs = ds.get_records(method=methods[i], basis=basis[i], keywords=key)
    indx = len(recs)
    num_complete = 0
    num_error = 0
    num_incomplete = 0
    if indx == 1416 and not isinstance(recs, list):
        for ii in range(1416):
            if recs.iloc[ii].record.status == "COMPLETE":
                num_complete += 1
            elif recs.iloc[ii].record.status == "ERROR":
                num_error += 1
            elif recs.iloc[ii].record.status == "INCOMPLETE":
                num_incomplete += 1
    elif indx > 1 and isinstance(recs, list):
        for ii in range(1416):
            if recs[indx - 1].iloc[ii].record.status == "COMPLETE":
                num_complete += 1
            elif recs[indx - 1].iloc[ii].record.status == "ERROR":
                num_error += 1
            elif recs[indx - 1].iloc[ii].record.status == "INCOMPLETE":
                num_incomplete += 1
    status_dict[key] = {
        "COMPLETE": num_complete,
        "ERROR": num_error,
        "INCOMPLETE": num_incomplete,
    }

df = pd.DataFrame(status_dict)
print(df.to_markdown())
