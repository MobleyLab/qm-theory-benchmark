import pandas as pd
import qcportal as ptl
from openff.toolkit.topology import Molecule
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
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
    # 1. OpenFF Theory Benchmarking Set v1.0 - which contain the torsiondrives at different levels of theory
    # link (https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-12-18-OpenFF-Theory
    # -Benchmarking-Set-v1.0)
    # 2. OpenFF Theory Benchmarking Constrained Optimization Set MP2 heavy-aug-cc-pVTZ v1.1 - reference data of
    # constrained minimizations of the end geometries obtained from openff-default spec torsiondrives with a higher
    # level of theory MP2 and a larger basis set heavy-aug-cc-pVTZ
    # link (https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-11-25-theory-bm-set
    # -mp2-heavy-aug-cc-pvtz)

    client = ptl.FractalClient()
    ref_ds = client.get_collection(
        "OptimizationDataset",
        "OpenFF Theory Benchmarking Constrained Optimization Set MP2 heavy-aug-cc-pVTZ v1.1",
    )
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

    # REF_SPEC from constrained optimizations
    ref_spec = "MP2/heavy-aug-cc-pVTZ-constrained"
    df[ref_spec] = ""
    for i, entry_index in enumerate(df.index):
        ref_angles = []
        ref_energies = []
        init_mols = []
        final_mols = []
        ref_dipoles = []
        ref_quadrupoles = []
        optentry = ref_ds.get_entry(name=str(i) + "-0")
        attributes = optentry.attributes
        dihedrals = optentry.additional_keywords["constraints"]["freeze"][0]["indices"]
        df.loc[entry_index, ref_spec] = [[{"attributes": attributes}]]
        for id in range(24):
            opt_record = ref_ds.get_record(
                name=str(i) + "-" + str(id), specification="default"
            )
            init_mols.append(opt_record.get_initial_molecule())
            final_mols.append(opt_record.get_final_molecule())
            offmol = Molecule.from_mapped_smiles(
                attributes["canonical_isomeric_explicit_hydrogen_mapped_smiles"],
                allow_undefined_stereo=True,
            )
            offmol.add_conformer(opt_record.get_initial_molecule().geometry * unit.bohr)
            rdmol = offmol.to_rdkit()
            ref_angles.append(
                round(
                    GetDihedralDeg(
                        rdmol.GetConformer(0),
                        dihedrals[0],
                        dihedrals[1],
                        dihedrals[2],
                        dihedrals[3],
                    )
                )
            )
            ref_energies.append(opt_record.get_final_energy())
            ref_dipoles.append(
                opt_record.get_trajectory()[-1].extras["qcvars"]["SCF DIPOLE"]
            )
            ref_quadrupoles.append(
                opt_record.get_trajectory()[-1].extras["qcvars"]["SCF QUADRUPOLE"]
            )

        ref_angles, ref_energies, init_mols = zip(
            *sorted(zip(ref_angles, ref_energies, init_mols))
        )
        ref_energy_min = min(ref_energies)
        ref_relative_energies = [(x - ref_energy_min) for x in ref_energies]
        df.loc[entry_index, ref_spec][0].update(
            {
                "initial_molecules": init_mols,
                "final_molecules": final_mols,
                "final_energies": dict(zip(ref_angles, ref_energies)),
                "angles": ref_angles,
                "dipoles": ref_dipoles,
                "quadrupoles": ref_quadrupoles,
                "relative_energies": ref_relative_energies,
                "keywords": opt_record.keywords,
                "mapped_smiles": attributes[
                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                ],
                "dihedrals": dihedrals,
            }
        )

    df.to_pickle("./torsiondrive_data_test.pkl")


if __name__ == "__main__":
    main()
