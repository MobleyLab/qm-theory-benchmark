### QM theory benchmark for force field fitting data.

File manifest
-------------
- 1.download.ipynb - downloader script to get data from QCArchive, compatible with new qcfractal version \
- 2.plotting.ipynb - energy data plotter \
- 3.dipoles.ipynb - dipole data plotter \
- 4.local_runs.ipynb - notebook to run the single points locally for some functionals that were added later in the study \
- 5.local_run_for_timings.ipynb - notebook to aggregate the wall times on the same node configuration \
- ./data - directory with the data for each functional and the reference data in json format \
- ./output - plot outputs \
- ./non-qm-potentials - scripts to evaluate energies with SQM/ML potentials \

### Data for different functionals and reference data.

The data files are in ./data. For the MP2 geometries at which these single points were performed the data is in the file "MP2_heavy-aug-cc-pVTZ_torsiondrive_data.json", and the data structure follows:

```
{"molecule index": {"metadata": {"mapped_smiles": "String of mapped smiles",
                                 "mol_charge": "molecular charge"
                                 "mol_multiplicity": "molecular multiplicity"
                                 "dihedral scanned": "list of dihedrals scanned, the four atom indices of the dihedral"},
                    "angles": [list of dihedral angles],
                    "final_energies": [list of MP2 energies for each of the angle index in Hartree],
                    "final_geometries": [3D geometry in units of bohr],
                    "dipoles": [list of corresponding dipole vectors for each angle index in atomic units]]
```

Most of the files were named as `functional_basis_single_points_data.json`. The data per functional and basis set combination is arranged as follows:

```
{"molecule index": {"angles": [list of dihedral angles],
                   "total energies": [list of total energies (DFT+dispersion) for each of the angle index in Hartree],
                   "dft energies": [list of DFT energies in Hartree],
                   "dispersion energies": [list of dispersion energies wherever applicable in Hartree],
                   "dipoles": [list of corresponding dipole vectors for each angle index in atomic units]},
                   ...}
```

The data is in nested dictionary and can be read using
```
import json

data = json.load(open('file_name.json'))
```


### Copyright

Copyright (c) 2021, Pavan Behara


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
