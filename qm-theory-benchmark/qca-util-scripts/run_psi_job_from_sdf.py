import json
from pathlib import Path

import click
import qcengine
from openff.toolkit.topology import Molecule
from qcelemental.models import OptimizationInput
from qcelemental.models.common_models import Model
from qcelemental.models.procedures import QCInputSpecification


@click.command()
@click.option(
    "--file",
    "file",
    type=click.STRING,
    required=True,
    default="methane.sdf",
    help="sdf filename with location",
)
def main(file):
    offmol = Molecule.from_file(file, allow_undefined_stereo=True)
    qcel_mol = offmol.to_qcschema()
    psi4_model = Model(method="B3LYP-D3BJ", basis="DZVP")
    geometric_input = OptimizationInput(
        initial_molecule=qcel_mol,
        input_specification=QCInputSpecification(
            model=psi4_model,
            keywords={
                "maxiter": 200,
                "scf_properties": [
                    "dipole",
                    "quadrupole",
                    "wiberg_lowdin_indices",
                    "mayer_indices",
                    "mulliken_charges",
                ],
            },
            driver="gradient",
        ),
        keywords={
            "coordsys": "dlc",
            "enforce": 0,
            "epsilon": 1e-05,
            "reset": True,
            "qccnv": False,
            "molcnv": False,
            "check": 0,
            "trust": 0.1,
            "tmax": 0.3,
            "maxiter": 300,
            "convergence_set": "gau",
            "program": "psi4",
        },
    )

    opt_result = qcengine.compute_procedure(
        input_data=geometric_input, procedure="geometric"
    )
    final_mol = opt_result.final_molecule
    off_final = Molecule.from_qcschema(final_mol)
    off_final.to_file("QM_opt_final" + file, file_format="sdf")
    Path("./QM_output").mkdir(parents=True, exist_ok=True)
    with open("./QM_output/optresult.json", "w") as f:
        json.dump(opt_result.json(), f)
    psi_out_list = [x.stdout for x in opt_result.trajectory]
    f = open("./QM_output/psi_out.out", "w")
    for item in psi_out_list:
        f.write(item)
    f.close()
    f = open("./QM_output/geometric_out.out", "w")
    f.write(opt_result.stdout)
    f.close()

    return


if __name__ == "__main__":
    main()
