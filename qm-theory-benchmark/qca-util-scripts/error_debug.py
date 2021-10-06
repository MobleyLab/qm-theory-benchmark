import lzma

from qcfractal.interface import FractalClient

client = FractalClient.from_file()
ds = client.get_collection(
    "Dataset", "OpenFF Theory Benchmarking Single Point Energies v1.0"
)

spec_name = "wb97m-d3bj/dzvp"
method = "wb97m-d3bj"
basis = "dzvp"
recs = ds.get_records(method=method, basis=basis, keywords=spec_name)

indx = len(recs)
print(
    "Record ID: ",
    recs[indx - 1].iloc[0].record.id,
    "Status: ",
    recs[indx - 1].iloc[0].record.status,
)
print(client.query_tasks(base_result=int(recs[indx - 1].iloc[0].record.id)))
kv = client.query_kvstore(recs[indx - 1].iloc[0].record.error)
out = lzma.decompress(kv[list(kv.keys())[0]].data).decode()
print("-----------")
print(bytes(str(out), "utf-8").decode("unicode_escape"))
