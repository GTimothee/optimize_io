import os 
import pytest
import h5py 
import dask

from dask_io.utils.get_arrays import get_dask_array_from_hdf5, get_dataset

from ..utils import ARRAY_FILEPATH_SMALL, ARRAY_SHAPES, create_test_array


def test_get_dask_array_from_hdf5(create_test_array):    
    
    dask_arr = get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, '/data', logic_cs="auto")
    assert isinstance(dask_arr, dask.array.Array)

    with pytest.raises(TypeError):
        get_dask_array_from_hdf5(None, '/data', logic_cs="auto")
    
    with pytest.raises(TypeError):
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, None, logic_cs="auto")

    with pytest.raises(ValueError):
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, '/data', logic_cs=None)

    with pytest.raises(ValueError):
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0] + ".badextension"
        with open(filepath, "w+") as f:  # create the file so that file not found error not raised
            get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

    with pytest.raises(FileNotFoundError):
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0]
        get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

    with pytest.raises(FileNotFoundError):
        os.remove(ARRAY_FILEPATH_SMALL)
        get_dask_array_from_hdf5(ARRAY_FILEPATH_SMALL, '/data', logic_cs="auto")


def test_get_dataset(create_test_array):
    dataset = get_dataset(ARRAY_FILEPATH_SMALL, '/data')
    assert isinstance(dataset, h5py.Dataset)

    with pytest.raises(ValueError):
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0] + ".badextension"
        with open(filepath, "w+") as f:  # create the file so that file not found error not raised
            get_dataset(filepath, '/data')

    with pytest.raises(FileNotFoundError):
        filepath = os.path.splitext(ARRAY_FILEPATH_SMALL)[0]
        get_dataset(filepath, '/data')

    with pytest.raises(KeyError):
        dataset = get_dataset(ARRAY_FILEPATH_SMALL, '/badkey')

    with pytest.raises(FileNotFoundError):
        os.remove(ARRAY_FILEPATH_SMALL)
        dataset = get_dataset(ARRAY_FILEPATH_SMALL, '/data')