# dask_io
A module optimizing the task graph in Dask's threaded scheduler in order to faster I/O operations.

# logging
Logging formatting details can be found at dask_io/optimizer/logging_config.ini
An example of how to handle those logs can be found at 
- dask_io/tests/logging_config.ini (configuration file)
- dask_io/tests/conftest.py (declaration of the logger)