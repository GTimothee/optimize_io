import os
import h5py
import pytest
import numpy as np

with open('dask_io/config.json') as json_conffile:
    import json
    data = json.load(json_conffile)
    import sys
    sys.path.insert(0, data['dask_path'])

import dask
import dask.array as da

from dask_io.optimizer.cases.case_config import CaseConfig
from dask_io.optimizer.cases.case_creation import get_arr_chunks
from dask_io.optimizer.configure import enable_clustering, disable_clustering
from dask_io.optimizer.utils.utils import ONE_GIG, CHUNK_SHAPES_EXP1
from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5

from ..utils import create_test_array_nochunk, ONE_GIG

import logging 
logger = logging.getLogger(__name__)

pytest.test_array_path = None

buffer_size = 4 * ONE_GIG
path = './small_array_nochunk.hdf5'


# TODO: make tests with different chunk shapes
@pytest.fixture(autouse=True)
def create_test_array():
    if not pytest.test_array_path:
        create_test_array_nochunk(path, (100, 100, 100))
        pytest.test_array_path = path

    enable_clustering(buffer_size, mem_limit=True)


@pytest.fixture(params=[(20, 20, 20), (50, 50, 50), (20, 100, 100), (1, 100, 100), (50, 20, 10)])
def shape_to_test(request):
    """ Testing two block shapes, two slab shapes, one cuboid 
    """
    return request.param 


@pytest.fixture(params=[None]) # for the moment pass only None: all blocks are processed until solution find for get_used_proxies
def nb_chunks(request):
    return request.param 


@pytest.fixture(params=[False, True])
def optimized(request):
    return request.param 


def sum_tester(shape_to_test, nb_chunks):
    """ Test if the sum of two blocks yields the good result using our optimization function.
    """
    logger.info("testing shape %s", shape_to_test)

    # prepare test case
    case = CaseConfig(pytest.test_array_path, shape_to_test)
    case.sum(nb_chunks)

    # non optimized run
    disable_clustering()
    result_non_opti = case.get().compute()

    # optimized run
    enable_clustering(buffer_size)
    result_opti = case.get().compute()

    assert np.array_equal(result_non_opti, result_opti)


def test_split(optimized, nb_chunks, shape_to_test):
    def create_arrays_for_comparison():
        """ Get chunks as dask arrays to compare the chunks to the splitted files.
        """
        arr = get_dask_array_from_hdf5(pytest.test_array_path, '/data', logic_cs=shape_to_test)
        arr_list = get_arr_chunks(arr, nb_chunks=nb_chunks)
        return arr_list


    def apply_sanity_check(split_filepath):
        """ Check if splitted file not empty.
        """
        logger.info("Checking split file integrity...")
        with h5py.File(split_filepath, 'r') as f:
            keys_list = list(f.keys())
            logger.info("file : %s", f)
            logger.info("Number of datasets in hdf5 file : %s", len(keys_list))
            logger.info("First item: %s", keys_list[0])
            assert len(list(f.keys())) != 0
        logger.info("Integrity check passed.\n")


    def store_correct():
        """ Compare the real chunks to the splits to see if correctly splitted. 
        """
        logger.info("Testing %s matches...", len(arr_list))
        with h5py.File(split_filepath, 'r') as f:
            for i, a in enumerate(arr_list):
                stored_a = da.from_array(f['/data' + str(i)])
                # logger.info("split shape: %s", stored_a.shape)
                
                stored_a.rechunk(chunks=shape_to_test)
                # logger.info("split rechunked to: %s", stored_a.shape)
                # logger.info("will be compared to : %s ", a.shape)
                # logger.info("Testing all close...")
                test = da.allclose(stored_a, a)
                disable_clustering() # TODO: remove this, make it work even for all close
                assert test.compute()
        logger.info("Passed.\n")
        

    def split():
        # overwrite if split file already exists
        if os.path.isfile(split_filepath):
            os.remove(split_filepath)

        case = CaseConfig(pytest.test_array_path, shape_to_test)
        case.split_hdf5(split_filepath, nb_blocks=nb_chunks)
        case.get().compute()
        return 

    logger.info("PARAMETERS:")
    logger.info("Optimized: %s", optimized), nb_chunks, shape_to_test
    logger.info("Nb_chunk: %s", nb_chunks)
    logger.info("Shape: %s \n", shape_to_test)

    # setup config
    split_filepath = "./split_file.hdf5"

    if optimized:
        enable_clustering(buffer_size)
    else:
        disable_clustering()

    # test
    split()
    apply_sanity_check(split_filepath)

    # assert
    arr_list = create_arrays_for_comparison()
    store_correct()