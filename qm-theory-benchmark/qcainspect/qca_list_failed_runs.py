import click
from qcportal import FractalClient
from qcengine import compute_procedure

@click.command()
@click.option(
    "--qcaid",
    "qcaid",
    type=click.INT,
    required=True,
    help="id of the optimization record to pull information from"
)
@click.option(
    "--print_spec",
    type=click.BOOL,
    default=False,
    help="print the compute spec of the optimization record"
)
@click.option(
    "--print_last",
    type=click.BOOL,
    default=False,
    help="print the psi4 output of the last step (final optimized molecule)"
)

def main(qcaid, print_spec, print_last):
    cl = FractalClient.from_file()
    opt_record = cl.query_procedures(id=qcaid)
    if print_spec:
        print(opt_record[0].dict()['qc_spec'])
    if print_last:
        last_step = opt_record[0].get_trajectory()[-1]
        output = last_step.get_stdout()
        print(output)


if __name__ == "__main__":
    main()