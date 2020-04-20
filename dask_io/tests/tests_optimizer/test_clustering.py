import os 
import sys
import pytest

from dask_io.optimizer.configure import enable_clustering
from dask_io.optimizer.cases.case_config import Split
from dask_io.optimizer.utils.array_utils import get_arr_shapes
from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.find_proxies import get_used_proxies
from dask_io.optimizer.clustering import *  # package being tested

from ..utils import create_test_array_nochunk, ONE_GIG

pytest.test_array_path = None

path = './small_array_nochunk.hdf5'

import logging 
logger = logging.getLogger('test')


@pytest.fixture(autouse=True)
def create_test_array():
    if not pytest.test_array_path:
        create_test_array_nochunk(path, (100, 100, 100))
        pytest.test_array_path = path


def test_get_covered_blocks():
    """
    Remainder: 
        function to test description: 
    """
    slice_tuple = (slice(0, 225, None), slice(242, 484, None), slice(500, 700, None))
    chunk_shape = (220, 242, 200)
    ranges = get_covered_blocks(slice_tuple, chunk_shape)
    assert [list(r) for r in ranges] == [[0, 1], [1], [2, 3]]

    slice_tuple = (slice(0, 220, None), slice(0, 242, None), slice(0, 200, None))
    x_range, y_range, z_range = get_covered_blocks(slice_tuple, chunk_shape)
    ranges = get_covered_blocks(slice_tuple, chunk_shape)
    assert [list(r) for r in ranges] == [[0], [0], [0]]


def test_get_blocks_used():
    cs = (20, 20, 20)
    case = Split(pytest.test_array_path, cs)
    case.split_hdf5("./split_file.hdf5", nb_blocks=None)
    arr = case.get()

    # routine to get the needed data
    # we assume those functions have been tested before get_blocks_used
    cs_confirmed, dicts = get_used_proxies(arr.dask.dicts)

    assert cs == cs_confirmed

    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    strategy, max_blocks_per_load = get_load_strategy(ONE_GIG, 
                                                      cs, 
                                                      (100,100,100)) 

    # actual test of the function
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj, cs)
    blocks_used.sort()
    expected = list(range(125))
    assert blocks_used == expected


def test_create_buffers_blocks():
    """ Test if the buffering works according to clustered writes in all 3 possible configurations.

    Data:
    -----
    input array shape: 100x100x100
    input arr created with 2 bytes per pixel
    block shape: 20x20x20
    
    Which gives us:
    ---------------
    - nb blocks per row = 5
    - nb blocks per slice = 25
    - block size in bytes : (20*20*20) * 2 bytes = 16000 
    """
    cs = (20, 20, 20)
    case = Split(pytest.test_array_path, cs)
    case.split_hdf5("./split_file.hdf5", nb_blocks=None)
    arr = case.get()

    _, dicts = get_used_proxies(arr.dask.dicts)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]

    # EXPECTED BEHAVIOR FOR CLUSTERED WRITES
    l1 = [[i] for i in range(125)] # 1 block
    l2 = list() # 3 blocks
    for i in range(25):
        o = (i*5)
        l2.append([0+o,1+o,2+o])
        l2.append([3+o,4+o])
    l3 = list() # 1 block column
    for i in range(25):
        l3.append(list(range(i*5,i*5+5)))
    l4 = list() # 2 block columns
    for i in range(5):
        o = i*25 # offset
        l4.append(list(range(0+o, 10+o)))
        l4.append(list(range(10+o, 20+o)))
        l4.append(list(range(20+o, 25+o)))
    l5 = list() # 1 block slice
    for i in range(5):
        l5.append(list(range((i*25),(i*25)+25)))
    l6 = list() # 3 block slices
    l6.append(list(range(0,25*3)))
    l6.append(list(range(75, 125)))
    l7 = [list(range(125))] # whole array

    nb_bytes_per_block = 20*20*20
    byte_size = 2
    experiment_params = {
        nb_bytes_per_block * byte_size : l1, # 1 block
        nb_bytes_per_block * byte_size *3: l2, # some blocks (3)
        nb_bytes_per_block * byte_size *5: l3, # 1 block column
        nb_bytes_per_block * byte_size *5*2: l4, # some block columns (2)
        nb_bytes_per_block * byte_size *5*5: l5, # 1 block slice
        nb_bytes_per_block * byte_size *5*5*3: l6, # some block slices (3)
        nb_bytes_per_block * byte_size *5*5*5: l7, # whole array
    }

    for buffer_size, expected in experiment_params.items():
        logging.info("\nTesting buffer %s", buffer_size)
        logging.debug("Expecting %s", expected)
        enable_clustering(buffer_size, mem_limit=True)
        buffers = create_buffers(origarr_name, dicts, cs)
        logging.debug("Got %s", buffers)
        assert buffers == expected


def test_create_buffers_slabs():
    """ Test if the buffering works according to clustered writes when processing slabs.
    The only strategy that should be used is ``block slices".
    """
    cs = (5, 100, 100) # 20 chunks
    case = Split(pytest.test_array_path, cs)
    case.split_hdf5("./split_file.hdf5", nb_blocks=None)
    arr = case.get()

    _, dicts = get_used_proxies(arr.dask.dicts)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]

    nb_bytes_per_block = 100*100*5
    byte_size = 2
    l1 = [[i] for i in range(20)]
    l2 = [list(range(10)), list(range(10, 20))]
    l3 = [list(range(7)), list(range(7, 14)), list(range(14,20))]

    experiment_params = {
        nb_bytes_per_block * byte_size: l1,
        nb_bytes_per_block * byte_size*10: l2,
        nb_bytes_per_block * byte_size*7: l3 
    }

    for buffer_size, expected in experiment_params.items():
        logging.info("\nTesting buffer %s", buffer_size)
        logging.debug("Expecting %s", expected)
        enable_clustering(buffer_size, mem_limit=True)
        buffers = create_buffers(origarr_name, dicts, cs)
        logging.debug("Got %s", buffers)
        assert buffers == expected


def test_create_buffer_node():
    # preparation
    cs = (20, 20, 20)
    case = Split(pytest.test_array_path, cs)
    case.split_hdf5("./split_file.hdf5", nb_blocks=None)
    arr = case.get()

    graph = arr.dask.dicts
    _, dicts = get_used_proxies(graph)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts, cs)
        
    # apply function
    keys = list()
    for buffer in buffers:            
        key = create_buffer_node(graph, origarr_name, dicts, buffer, cs)    
        keys.append(key)

    # test output
    buffers_key = origarr_name.split('-')[-1] + '-merged'

    indices = set()
    for buffer_key in graph[buffers_key].keys():
        _, start, end = buffer_key
        indices.add((start, end))

    buffers = set([(b[0], b[-1]) for b in buffers])

    assert buffers_key in graph.keys()
    assert len(indices) == len(buffers)
    assert buffers == indices

    
def test_get_load_strategy():
    strat, max_nb_blocks = get_load_strategy(4000, (10, 10, 1), None, nb=4)
    assert strat == 'blocks'
    assert max_nb_blocks == 10


def test_start_new_buffer():    
    # WARNING: only blocks strategy have been implemented so far

    strategy = "blocks"
    nb_blocks_per_row = 4
    max_nb_blocks_per_buffer = 10

    test_buffers = [[0,1,2,3,4,5,6,7,8,9],
                    [0,1,2,3],
                    [1,2],
                    [1,2]]

    blocks = [10,
              4,
              3,
              4] 

    expected = [True,
                True,
                False,
                True]

    i = 1
    for curr_buffer, b, e in zip(test_buffers, blocks, expected):
        print("case", i)
        
        res = start_new_buffer(curr_buffer, 
                            b, 
                            curr_buffer[-1], 
                            strategy, 
                            nb_blocks_per_row, 
                            max_nb_blocks_per_buffer)

        assert res == e
        i += 1


def test_overlap_slices():
    curr_buffers = [
        list(range(16)),
        list(range(8)),
        list(range(20, 24))
    ]
    buffers = [
        list(range(16, 20)),
        list(range(8, 12)),
        list(range(56, 60))
    ]
    expected = [
        True,
        False,
        True
    ]
    blocks_shape = (4, 4, 4)

    i = 1
    for curr_buff, buff, exp in zip(curr_buffers, buffers, expected):
        print("case", i)
        out = overlap_slice(curr_buff, buff, blocks_shape)
        i += 1
        assert out == exp


def test_merge_rows():
    blocks_shape = (4, 4, 4)
    nb_blocks_per_row = 4
    max_blocks_per_load = 21

    buffers_list = [
        [list(range(x*4, x*4 +4)) for x in range(8)]
    ]
    expected = [
        [list(range(16)), list(range(16, 32))]
    ]
    
    for buffers, exp in zip(buffers_list, expected):
        out = merge_rows(buffers, blocks_shape, nb_blocks_per_row, max_blocks_per_load)
        assert out == exp


def test_merge_slices():
    nb_blocks_per_slice = 16
    max_blocks_per_load = 35
    buffers_list = [
        [list(range(x * 16, x * 16 + 16)) for x in range(4)]
    ]
    expected = [
        [list(range(32)), list(range(32, 64))]
    ]

    for buffers, exp in zip(buffers_list, expected):    
        out = merge_slices(buffers, nb_blocks_per_slice, max_blocks_per_load)
        assert out == exp


def test_buffering():
    strategy = "blocks"
    blocks_shape = (4, 4, 4)
    max_nb_blocks_per_buffer = 9

    blocks = [0,1,2,3,6,7,8,9,10,11,12]
    row_concat, slices_concat = (False, False)
    exp = [[0,1,2,3], [6,7], [8, 9, 10, 11], [12]]
    out = buffering(blocks, strategy, blocks_shape, max_nb_blocks_per_buffer, row_concat=row_concat, slices_concat=slices_concat)
    assert out == exp