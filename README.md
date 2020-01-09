# dask_io
A module optimizing the task graph in Dask's threaded scheduler in order to faster I/O operations.


## setup
add custom_setup.py at `dask_io.tests.custom_setup.py` containing:

```
import os, sys

def setup_custon_dask():
    custom_dask_path = 
    sys.path.insert(0, custom_dask_path)
```

+ add a symbolic link/or direct folder called 'data' to data folder for the tests