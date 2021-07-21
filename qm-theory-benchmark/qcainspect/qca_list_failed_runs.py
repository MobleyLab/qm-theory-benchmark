import click
from openff.toolkit.topology import Molecule
from qcportal import FractalClient
from tqdm import tqdm
import sys

@click.command()
@click.option(
    "--dataset",
    type=click.STRING,
    default='None',
    help="name the dataset"
)
@click.option(
    "--type",
    type=click.STRING,
    default='None',
    help="type of dataset, OPT, TD, HESS"
)
@click.option(
    "--spec",
    type=click.STRING,
    default='default',
    help="compute spec to check"
)

def main(type, dataset, spec):
    if type == 'None' or dataset == 'None':
        print("ERROR: Enter the correct dataset name and type")
        sys.exit(1)
    cl = FractalClient.from_file()
    if type == 'OPT':
        type = 'OptimizationDataset'
    elif type == 'TD':
        type = 'TorsionDriveDataset'
    elif type == 'HESS':
        type = 'Dataset'
    ds = cl.get_collection(collection_type=type, name=dataset)

    failed_list = []
    for entry in tqdm(ds.data.records.values()):
        # accessing the optimization record which contains the full trajectory
        record = ds.get_record(name=entry.name, specification=spec)
        if record.status != "COMPLETE":
            print(record)
            failed_list.append((entry.attributes['canonical_isomeric_explicit_hydrogen_mapped_smiles'], record.status))
    for item in failed_list:
        print(item)
        td_rec = cl.query_procedures(id = item[0])
        offmol = Molecule.from_mapped_smiles(td_rec)
        offmol.visualize(backend='openeye')
    print("Done!")

if __name__ == "__main__":
    main()