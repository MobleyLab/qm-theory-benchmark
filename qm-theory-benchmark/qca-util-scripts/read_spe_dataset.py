import pickle

with open("spe-recs.pkl", "rb") as outfile:
    spe_data = pickle.load(outfile)

print(spe_data.keys())
