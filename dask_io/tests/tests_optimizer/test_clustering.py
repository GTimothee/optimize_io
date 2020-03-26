import os 
import sys
import pytest

from dask_io.optimizer.configure import enable_clustering
from dask_io.optimizer.cases.case_config import CaseConfig
from dask_io.optimizer.utils.array_utils import get_arr_shapes
from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.find_proxies import get_used_proxies
from dask_io.optimizer.clustering import *  # package being tested

from ..utils import create_test_array_nochunk, ONE_GIG

pytest.test_array_path = None

path = '/run/media/user/HDD 1TB/data/big_array_nochunk.hdf5'

import logging 
logger = logging.getLogger(__name__)

# TODO: make tests with different chunk shapes
@pytest.fixture(params=[(770, 605, 700)])
def case(request):
    buffer_size = 4 * ONE_GIG

    if not pytest.test_array_path:
        create_test_array_nochunk(path, (1540, 1210, 1400))
        pytest.test_array_path = path
        
    cs = request.param  
    arr = get_dask_array_from_hdf5(path, '/data', logic_cs=cs)

    # test case description below:
    # take a block of size half the size of array (4 blocks)
    _, chunks, blocks_dims = get_arr_shapes(arr)
    _3d_pos = numeric_to_3d_pos(5, blocks_dims, 'C')
    dims = [(_3d_pos[0]+1) * chunks[0],
            (_3d_pos[1]+1) * chunks[1],
            (_3d_pos[2]+1) * chunks[2]]
    arr = arr[0:dims[0], 0:dims[1], 0:dims[2]]
    arr = arr + 1

    enable_clustering(buffer_size, mem_limit=True)
    return (arr, cs)


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


def test_get_blocks_used(case):
    arr, cs = case

    # routine to get the needed data
    # we assume those functions have been tested before get_blocks_used
    _, dicts = get_used_proxies(arr.dask.dicts)

    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    arr_obj = dicts['origarr_to_obj'][origarr_name]
    strategy, max_blocks_per_load = get_load_strategy(4 * ONE_GIG, 
                                                      (770, 605, 700), 
                                                      cs) 

    # actual test of the function
    blocks_used, block_to_proxies = get_blocks_used(dicts, origarr_name, arr_obj, cs)
    expected = [0,1,4,5]
    assert blocks_used == expected


def test_create_buffers(case):
    """
    row size = 7 blocks
    slices size = 35 blocks

    default mem = 1 000 000 000

    # (220 * 242 * 200) = 10 648 000 
    # with 4 bytes per pixel, we have maximum 23 blocks that can be loaded
    # 21 blocks contiguous and not overlaping then the last 14 blocks
    => strategy: buffer=row_size max 
    """
    arr, cs = case

    _, dicts = get_used_proxies(arr.dask.dicts)
    origarr_name = list(dicts['origarr_to_obj'].keys())[0]
    buffers = create_buffers(origarr_name, dicts, cs)
    
    expected = [[0,1], [4,5]]
    
    print("buffers - ", buffers)
    print("expected - ", expected)
    assert buffers == expected


def test_create_buffer_node(case):
    # preparation
    arr, cs = case
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