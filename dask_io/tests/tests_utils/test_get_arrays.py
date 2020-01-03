import dask_io.utils as utils
from utils.get_arrays import get_dask_array_from_hdf5


def test_create_dask_array():
    print("test")


"""def test_get_dask_array_from_hdf5():
    file_path = 'data/test_utils.hdf5'
    dataset_key = 'data'
    logic_css = ["auto", "physical"]
    
    with logic_cs in logic_css:
        array = get_dask_array_from_hdf5(file_path, dataset_key, logic_cs="auto")"""