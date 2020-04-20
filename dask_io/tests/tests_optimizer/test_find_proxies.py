import os, pytest

from dask_io.optimizer.configure import enable_clustering
from dask_io.optimizer.utils.utils import CHUNK_SHAPES_EXP1
from dask_io.optimizer.utils.array_utils import get_arr_shapes
from dask_io.optimizer.cases.case_config import Split
from dask_io.optimizer.find_proxies import *  # package to be tested

from ..utils import create_test_array_nochunk, ONE_GIG

import logging

logger = logging.getLogger('test')
pytest.test_array_path = None

buffer_size = 4 * ONE_GIG
path = './small_array_nochunk.hdf5'


# TODO: make tests with different chunk shapes
# TODO: use smaller test array
@pytest.fixture(autouse=True)
def create_test_array():
    if not pytest.test_array_path:
        create_test_array_nochunk(path, (100, 100, 100))
        pytest.test_array_path = path

    enable_clustering(buffer_size, mem_limit=True)


def test_get_array_block_dims():
    shape = (500, 1200, 300)
    chunks = (100, 300, 20)
    block_dims = get_array_block_dims(shape, chunks)
    expected = (5, 4, 15)
    assert block_dims == expected


def test_get_graph_from_dask():
    """ Test if it runs well.
    TODO: Better test function.
    """
    # create config for the test
    case = Split(pytest.test_array_path, "auto")
    case.sum(nb_chunks=None)
    dask_array = case.get()

    # test function
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)


def test_BFS():
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    values, depth = standard_BFS('a', graph)
    assert values == ['a', 'b', 'c', 'd', 'e']
    assert depth == 2

    values, depth = standard_BFS('f', graph)
    assert values == ['f', 'e']
    assert depth == 1


def test_get_root_nodes():
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    root_nodes = get_root_nodes(graph)
    assert root_nodes == ['a', 'f']


def test_BFS_integration():
    """ Test to see if get used proxies would work with BFS.
    TODO: Deprecated.
    For now, get_used_proxies is in standby and returns all proxies.
    """
    graph = {
        'a': ['b', 'c'],
        'b': [],
        'c': ['d', 'e'],
        'd': [],
        'e': [],
        'f': ['e']
    }
    root_nodes = get_root_nodes(graph)

    max_components = list()
    max_depth = 0
    for root in root_nodes:
        node_list, depth = standard_BFS(root, graph)
        if len(max_components) == 0 or depth > max_depth:
            max_components = [node_list]
            max_depth = depth
        elif depth == max_depth:
            max_components.append(node_list)

    assert max_depth == 2
    assert len(max_components) == 1
    assert max_components[0] == ['a', 'b', 'c', 'd', 'e']