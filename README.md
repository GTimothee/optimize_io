# DASK IO
[![Build Status](https://travis-ci.com/GTimothee/dask_io.svg?branch=master)](https://travis-ci.com/GTimothee/dask_io)
[![Coverage Status](https://coveralls.io/repos/github/GTimothee/dask_io/badge.svg?branch=master)](https://coveralls.io/github/GTimothee/dask_io?branch=master)

A module optimizing the task graph in Dask's threaded scheduler in order to faster I/O operations.

# Third-party libraries
Use requirements.txt or requirements_conda.txt to install the dependencies.

For a conda environment:
```
conda create --name <env> --file requirements_conda.txt
```
For a pip environment: 
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Troubleshooting
### If there are missing dependencies when creating the conda environment
- 1) Create a new conda environement ``` conda create --name <envname> ```
- 2) activate the environment ``` conda activate <envname> ```
- 3) 'cd' inside the dask_io directory (where the requirements_conda.txt file is)
- 4) install the dependencies that are not missing ``` while read requirement; do conda install --yes $requirement; done < requirements_conda.txt ```

## Note for developers
To create the requirements_conda file:
```
conda list -e > requirements_conda.txt
``` 
To create the ``requirements.txt" file from conda environment:
```
pip freeze > requirements.txt
``` 
Remove mkl dependencies from ``requirements.txt" to get rid of compatibility issues.