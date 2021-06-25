import click
from qcengine import compute_procedure
from qcportal import FractalClient


@click.command()
@click.option(
    "--qcaid",
    "qcaid",
    type=click.INT,
    required=True,
)
def main(qcaid):
    """
    run the psi4 job using the same input as the qcarchive job id
    :param qcaid: QCArchive id of the job
    :return:
    """
    cl = FractalClient.from_file()
    task = cl.query_tasks(base_result=qcaid)
    # task[0].spec.args[0]['keywords']['coordsys'] = 'dlc'
    print(task)
    # then compute with qcengine.compute_procedure
    compute_result = compute_procedure(*task[0].spec.args)
    print(compute_result)
    return "job finished without error"

if __name__ == "__main__":
    main()
