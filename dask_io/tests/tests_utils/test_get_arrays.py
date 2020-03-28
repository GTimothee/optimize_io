import os 
import pytest

from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5, get_dataset
from dask_io.optimizer.configure import disable_clustering

from ..utils import create_test_array_nochunk


@pytest.fixture
def test_array_path():
    disable_clustering()
    array_filepath = './small_array_nochunk.hdf5'
    if not os.path.isfile(array_filepath):
        create_test_array_nochunk(array_filepath, (100, 100, 100))
    return array_filepath 


def test_get_dask_array_from_hdf5(test_array_path):    
    import dask

    dask_arr = get_dask_array_from_hdf5(test_array_path, '/data', logic_cs="auto")
    assert isinstance(dask_arr, dask.array.Array)

    with pytest.raises(TypeError):
        get_dask_array_from_hdf5(None, '/data', logic_cs="auto")
    
    with pytest.raises(TypeError):
        get_dask_array_from_hdf5(test_array_path, None, logic_cs="auto")

    with pytest.raises(ValueError):
        get_dask_array_from_hdf5(test_array_path, '/data', logic_cs=None)

    with pytest.raises(ValueError):
        filepath = os.path.splitext(test_array_path)[0] + ".badextension"
        with open(filepath, "w+") as f:  # create the file so that file not found error not raised
            get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

    with pytest.raises(FileNotFoundError):
        filepath = os.path.splitext(test_array_path)[0]
        get_dask_array_from_hdf5(filepath, '/data', logic_cs="auto")

    with pytest.raises(FileNotFoundError):
        os.remove(test_array_path)
        get_dask_array_from_hdf5(test_array_path, '/data', logic_cs="auto")


def test_get_dataset(test_array_path):
    import h5py 

    dataset = get_dataset(test_array_path, '/data')
    assert isinstance(dataset, h5py.Dataset)

    with pytest.raises(ValueError):
        filepath = os.path.splitext(test_array_path)[0] + ".badextension"
        with open(filepath, "w+") as f:  # create the file so that file not found error not raised
            get_dataset(filepath, '/data')

    with pytest.raises(FileNotFoundError):
        filepath = os.path.splitext(test_array_path)[0]
        get_dataset(filepath, '/data')

    with pytest.raises(KeyError):
        dataset = get_dataset(test_array_path, '/badkey')

    with pytest.raises(FileNotFoundError):
        os.remove(test_array_path)
        dataset = get_dataset(test_array_path, '/data')