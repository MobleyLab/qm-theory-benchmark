from qcfractal.interface import FractalClient

client = FractalClient("https://hpc3-22-04:7777", verify=False)
ds = client.get_collection(
    "Dataset", "OpenFF Theory Benchmarking Single Point Energies v1.0"
)

spec_name = "wb97m-d3bj/dzvp"  # 'df-ccsd(t)/cbs'
method = "wb97m-d3bj"  # 'mp2/heavy-aug-cc-pv[tq]z + d:ccsd(t)/heavy-aug-cc-pvdz'
basis = "dzvp"  # None
recs = ds.get_records(method=method, basis=basis, keywords=spec_name)

indx = len(recs)
for ii in range(1416):
    if recs.iloc[ii].record.status == "ERROR":
        client.modify_tasks(operation="restart", base_result=recs.iloc[ii].record.id)
