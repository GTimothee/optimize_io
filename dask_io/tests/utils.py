import os, pytest
import numpy as np
from dask_io.utils.get_arrays import get_dask_array_from_hdf5, create_random_dask_array, save_to_hdf5

DATA_DIRPATH = 'data'
ARRAY_FILEPATH = os.path.join(DATA_DIRPATH, 'sample_array_nochunk.hdf5')
LOG_DIR = "dask_io/logs"

@pytest.fixture(autouse=True)
def setup_routine():
    if not os.path.isfile(ARRAY_FILEPATH):
        print(f'Test array does not exist, creating it...')
        shape = (1540, 1210, 1400)
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, ARRAY_FILEPATH, physik_cs=None, key='/data', compression=None)

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)