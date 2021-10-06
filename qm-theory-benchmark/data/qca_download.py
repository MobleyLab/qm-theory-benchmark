import pandas as pd
import qcportal as ptl
from simtk import unit

PARTICLE = unit.mole.create_unit(
    6.02214076e23 ** -1,
    "particle",
    "particle",
)
HARTREE_PER_PARTICLE = unit.hartree / PARTICLE
HARTREE_TO_KCALMOL = HARTREE_PER_PARTICLE.conversion_factor_to(
    unit.kilocalorie_per_mole
)


def main():
    # Define the qcfractal server instance to download data from the datasets:
    # 1. OpenFF Theory Benchmarking Set v1.1 - which contain the torsiondrives at different levels of theory
    # link (https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-12-18-OpenFF-Theory
    # -Benchmarking-Set-v1.0)

    client = ptl.FractalClient()
    ds = client.get_collection(
        "TorsionDriveDataset", "OpenFF Theory Benchmarking Set v1.0"
    )
    specifications = ds.list_specifications().index.to_list()
    print(specifications)

    # Create a dataframe to store the torsiondrives data
    df = pd.DataFrame(columns=specifications)
    for i, entry_index in enumerate(ds.df.index):
        for spec_name in specifications:
            data_entry = ds.get_entry(entry_index)
            td_record_id = data_entry.object_map[spec_name]
            td_dict = {}
            td_dict["td_record_id"] = td_record_id
            td_dict["attributes"] = data_entry.attributes
            td_dict["mapped_smiles"] = data_entry.attributes[
                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
            ]
            df.loc[entry_index + str(i), spec_name] = [td_dict]
            td_record = client.query_procedures(td_record_id)[0]
            print(f"{i:5d} : {entry_index:50s} status {td_record.status}")

            if td_record.status == "COMPLETE":
                angles = []
                energies = []
                dipoles = []
                quadrupoles = []
                for key, value in td_record.get_final_energies().items():
                    angles.append(key[0])
                    energies.append(value)
                    dipoles.append(
                        td_record.get_final_results()[key].extras["qcvars"][
                            "SCF DIPOLE"
                        ]
                    )
                    quadrupoles.append(
                        td_record.get_final_results()[key].extras["qcvars"][
                            "SCF QUADRUPOLE"
                        ]
                    )
                angles, energies, dipoles, quadrupoles = zip(
                    *sorted(zip(angles, energies, dipoles, quadrupoles))
                )
                energy_min = min(energies)
                relative_energies = [(x - energy_min) for x in energies]
                dihedrals = td_record.keywords.dict()["dihedrals"][0]
                df.loc[entry_index + str(i), spec_name][0].update(
                    {
                        "initial_molecules": client.query_molecules(
                            td_record.initial_molecule
                        ),
                        "final_molecules": td_record.get_final_molecules(),
                        "final_energies": td_record.get_final_energies(),
                        "angles": angles,
                        "relative_energies": relative_energies,
                        "dipoles": dipoles,
                        "quadrupoles": quadrupoles,
                        "dihedrals": dihedrals,
                        "keywords": td_record.keywords.dict(),
                    }
                )

    # saving it to a pickle file
    df.to_pickle("./torsiondrive_data.pkl")


if __name__ == "__main__":
    main()
