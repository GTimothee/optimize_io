import os, pytest
import numpy as np
from dask_io.utils.get_arrays import create_random_dask_array, save_to_hdf5

def create_test_array_nochunk(file_path, shape):
    if not os.path.isfile(file_path):
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, file_path, physik_cs=None, key='/data', compression=None)
