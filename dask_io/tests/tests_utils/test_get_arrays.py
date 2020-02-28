import os 
import pytest
import h5py 

from dask_io.utils.get_arrays import get_dask_array_from_hdf5, get_dataset

from ..utils import ARRAY_FILEPATH_SMALL, ARRAY_SHAPES, setup_routine

def test_get_dask_array_from_hdf5():
    def sanity_check(arr):
        return True
    
    # 1. Exceptions
    with pytest.raises(TypeError):
        # behaviors with None parameter
        get_dask_array_from_hdf5(None, '/data', logic_cs="auto")
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, None, logic_cs="auto")
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, '/data', logic_cs=None)

    with pytest.raises(ValueError):
        # bad extension 
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0] + ".txt"
        with open(filepath, "w+") as f:  # create the file so that file not found error not raised
            get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

    with pytest.raises(FileNotFoundError):
        # no extension
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0]
        get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

        # file does not exist
        os.path.remove(ARRAY_FILEPATH_SMALL)
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, '/data', logic_cs="auto")


def test_file_in_list():
    dataset = get_dataset(ARRAY_FILEPATH_SMALL, '/data')
    assert isInstance(dataset, h5py.Dataset)