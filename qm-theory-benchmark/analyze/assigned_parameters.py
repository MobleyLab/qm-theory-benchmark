from openff.toolkit.topology import Topology


def get_assigned_parameters(offmol, forcefield):
    topology = Topology.from_molecules([offmol])
    molecule_force_list = forcefield.label_molecules(topology)

    parameter_list = []
    for mol_idx, mol_forces in enumerate(molecule_force_list):
        for force_tag, force_dict in mol_forces.items():
            for (atom_indices, parameter) in force_dict.items():
                parameter_list.append(parameter.id)
    return sorted(set(parameter_list))
