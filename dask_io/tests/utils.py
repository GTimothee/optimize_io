import os, pytest
import numpy as np
import shutil
from dask_io.utils.get_arrays import get_dask_array_from_hdf5, create_random_dask_array, save_to_hdf5

ARRAY_FILEPATH = './data/sample_array_nochunk.hdf5'
ARRAY_FILEPATH_SMALL = './data/small_array_nochunk.hdf5'
LOG_DIR = "dask_io/logs"

temp_dir_inialized = False

ARRAY_SHAPES = {
        ARRAY_FILEPATH: (1540, 1210, 1400),
        ARRAY_FILEPATH_SMALL: (500, 500, 500),
}

def create_test_array_nochunk(file_path):
    if not os.path.isfile(file_path):
        print(f'Test array does not exist, creating it...')

        shape = ARRAY_SHAPES[ARRAY_FILEPATH_SMALL] # ARRAY_SHAPES[file_path]
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, file_path, physik_cs=None, key='/data', compression=None)


@pytest.fixture() # autouse=True)
def test_array_path(): # setup_routine():
    """ Create a file at path ARRAY_FILEPATH for the tests.
    If the array does not exist, the array is created and saved.
    """ 
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()
    ARRAY_FILEPATH_SMALL = os.path.join(temp_dir.name, 'small_array_nochunk.hdf5')

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    create_test_array_nochunk(ARRAY_FILEPATH_SMALL)
    
    yield ARRAY_FILEPATH_SMALL

    shutil.rmtree(temp_dir.name)