import os
import h5py
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

from ..utils import ARRAY_FILEPATH, DATA_DIRPATH, setup_routine


def sum_tester(shapes_to_test):
    """ Test if the sum of two blocks yields the good result using our optimization function.
    """
    nb_chunks = 2

    for chunk_shape in shapes_to_test: 
        # prepare test case
        cs = CHUNK_SHAPES_EXP1[chunk_shape]
        case = CaseConfig(ARRAY_FILEPATH, cs)
        case.sum(nb_chunks)

        # non optimized run
        disable_clustering()
        result_non_opti = case.get().compute()

        # optimized run
        buffer_size = 4 * ONE_GIG
        enable_clustering(buffer_size)
        result_opti = case.get().compute()

        assert np.array_equal(result_non_opti, result_opti)


def test_sum_blocks():
    sum_tester(['blocks_previous_exp', 'blocks_dask_interpol'])


def test_sum_slabs():
    sum_tester(['slabs_previous_exp', 'slabs_dask_interpol'])


def split_test(optimized):
    def create_arrays_for_comparison():
        """ Get chunks as dask arrays to compare the chunks to the splitted files.
        """
        arr = get_dask_array_from_hdf5(ARRAY_FILEPATH, '/data', to_da=True, logic_cs=cs)
        arr_list = get_arr_chunks(arr, nb_chunks=nb_blocks)
        return arr_list


    def apply_sanity_check(split_filepath):
        """ Check if splitted file not empty.
        """
        print("Checking split file integrity...")
        with h5py.File(split_filepath, 'r') as f:
            print("file", f)
            print("keys", list(f.keys()))
            assert len(list(f.keys())) != 0
        print("Integrity check passed.")


    def store_correct():
        """ Compare the real chunks to the splits to see if correctly splitted. 
        """
        print("Testing", len(arr_list), "matches...")
        with h5py.File(split_filepath, 'r') as f:
            for i, a in enumerate(arr_list):
                stored_a = da.from_array(f['/data' + str(i)])
                print("split shape:", stored_a.shape)
                
                stored_a.rechunk(chunks=cs)
                print("split rechunked to:", stored_a.shape)
                print("will be compared to : ", a.shape)

                print("Testing all close...")
                test = da.allclose(stored_a, a)
                assert test.compute()
                print("Passed.")
        

    def split():
        # overwrite if split file already exists
        if os.path.isfile(split_filepath):
            os.remove(split_filepath)

        case = CaseConfig(ARRAY_FILEPATH, cs)
        case.split_hdf5(split_filepath, nb_blocks=nb_blocks)
        case.get().compute()
        return 


    # setup config
    split_filepath = os.path.join(DATA_DIRPATH, "split_file.hdf5")
    nb_blocks = 2

    if optimized:
        buffer_size = 4 * ONE_GIG
        enable_clustering(buffer_size)
    else:
        disable_clustering()

    for chunks_shape_name in CHUNK_SHAPES_EXP1.keys(): 
        # select chunk shape to use for the split (=split shape)
        cs = CHUNK_SHAPES_EXP1[chunks_shape_name]

        # do the test
        split()
        apply_sanity_check(split_filepath)

        # assert
        arr_list = create_arrays_for_comparison()
        store_correct()


def test_split_optimized():
    split_test(True)


def test_split_non_optimized():
    split_test(False)   