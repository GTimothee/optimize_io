# dask_io
A module optimizing the task graph in Dask's threaded scheduler in order to faster I/O operations.

# third parties
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