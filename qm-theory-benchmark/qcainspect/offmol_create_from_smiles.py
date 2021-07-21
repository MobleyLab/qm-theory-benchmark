import click
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.toolkits import ToolkitRegistry, OpenEyeToolkitWrapper, RDKitToolkitWrapper

@click.command()
@click.option(
    "--smiles",
    "smiles",
    type=click.STRING,
    required=True,
    help="smiles string in quotes"
)
@click.option(
    "--file",
    "file",
    type=click.STRING,
    required=True,
    default="./file.sdf",
    help="file path in quotes, defaults to ./file.sdf"
)
@click.option(
    "--file_format",
    "file_format",
    type=click.STRING,
    required=True,
    default="sdf",
    help="file format, default is sdf"
)
def main(smiles, file, file_format):
    toolkit_precedence = [RDKitToolkitWrapper]
    toolkit_registry = ToolkitRegistry(toolkit_precedence)
    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True, toolkit_registry=toolkit_registry)
    offmol.generate_conformers(toolkit_registry=toolkit_registry)
    offmol.to_file(file_path=file, file_format=file_format)


if __name__ == "__main__":
    main()
