import os, pytest

from dask_io.optimizer.configure import enable_clustering
from dask_io.optimizer.utils.utils import CHUNK_SHAPES_EXP1
from dask_io.optimizer.utils.array_utils import get_arr_shapes
from dask_io.optimizer.cases.case_config import CaseConfig
from dask_io.optimizer.find_proxies import *  # package to be tested

from ..utils import create_test_array_nochunk, ONE_GIG

import logging

logger = logging.getLogger(__name__)
pytest.test_array_path = None

buffer_size = 4 * ONE_GIG
path = '/run/media/user/HDD 1TB/data/big_array_nochunk.hdf5'


# TODO: make tests with different chunk shapes
# TODO: use smaller test array
@pytest.fixture(autouse=True)
def create_test_array():
    if not pytest.test_array_path:
        create_test_array_nochunk(path, (1540, 1210, 1400))
        pytest.test_array_path = path

    enable_clustering(buffer_size, mem_limit=True)


def test_get_array_block_dims():
    shape = (500, 1200, 300)
    chunks = (100, 300, 20)
    block_dims = get_array_block_dims(shape, chunks)
    expected = (5, 4, 15)
    assert block_dims == expected


def test_get_graph_from_dask():
    # create config for the test
    case = CaseConfig(pytest.test_array_path, "auto")
    case.sum(nb_chunks=None)
    dask_array = case.get()

    # test function
    dask_graph = dask_array.dask.dicts 
    graph = get_graph_from_dask(dask_graph, undirected=False)
    # with open(os.path.join(LOG_DIR, 'get_graph_from_dask.txt'), "w+") as f:
    #     for k, v in graph.items():    
    #         f.write("\n\n" + str(k))
    #         f.write("\n" + str(v))


# def used_proxies_tester(shapes_to_test):
#     for chunk_shape_key in shapes_to_test:
#         cs = CHUNK_SHAPES_EXP1[chunk_shape_key]

#         case = CaseConfig(pytest.test_array_path, cs)
#         case.sum(nb_chunks=2)
        
#         for use_BFS in [True]: #, False]:
#             dask_array = case.get()

#             # test function
#             dask_graph = dask_array.dask.dicts 
#             _, dicts = get_used_proxies(dask_graph)
            
#             # test slices values
#             slices = list(dicts['proxy_to_slices'].values())

#             if "blocks" in chunk_shape_key:
#                 s1 = (slice(0, cs[0], None), slice(0, cs[1], None), slice(0, cs[2], None))
#                 s2 = (slice(0, cs[0], None), slice(0, cs[1], None), slice(cs[2], 2 * cs[2], None))
#             else:
#                 s1 = (slice(0, cs[0], None), slice(0, cs[1], None), slice(0, cs[2], None))
#                 s2 = (slice(cs[0], 2 * cs[0], None), slice(0, cs[1], None), slice(0, cs[2], None))

#             logger.info("\nExpecting:")
#             logger.info(s1)
#             logger.info(s2)

#             logger.info("\nGot:")
#             logger.info(slices[0])
#             logger.info(slices[1])

#             assert slices == [s1, s2]


# def test_get_used_proxies_blocks():
#     used_proxies_tester(['blocks_previous_exp', 'blocks_dask_interpol'])


# def test_get_used_proxies_slabs():
#     used_proxies_tester(['slabs_previous_exp'])


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


def test_BFS_2():
    """ test to include bfs in the program
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


def test_BFS_3():
    """ test to include bfs in the program and test with rechunk case
    """

    # get test array with logical rechunking
    chunks_shape = (770, 605, 700)

    case = CaseConfig(pytest.test_array_path, chunks_shape)
    case.sum(nb_chunks=None)
    dask_array = case.get()
    # dask_array.visualize(filename='tests/outputs/img.png', optimize_graph=False)

    # get formatted graph for processing
    graph = get_graph_from_dask(dask_array.dask.dicts, undirected=False)  # we want a directed graph

    # with open(os.path.join(LOG_DIR, 'test_BFS_3.txt'), "w+") as f:
    #     for k, v in graph.items():
    #         f.write("\n\n" + str(k))
    #         f.write("\n" + str(v))

    # test the actual program
    root_nodes = get_root_nodes(graph)
    """logger.info('\nRoot nodes:')
    for root in root_nodes:
        logger.info(root)"""

    max_components = list()
    max_depth = 0
    for root in root_nodes:
        node_list, depth = standard_BFS(root, graph)
        if len(max_components) == 0 or depth > max_depth:
            max_components = [node_list]
            max_depth = depth
        elif depth == max_depth:
            max_components.append(node_list)


    logger.info("nb components found: %s", str(len(max_components)))
    #TODO: assertions