import numpy as np
import operator
import pytest

from dask_io.optimizer.cases.resplit_utils import Volume
from dask_io.optimizer.cases.resplit_case import *

import logging 
logger = logging.getLogger('test')


def test_add_offset():
    def get_rand(max, size):
        return np.random.randint(max, size=size)

    np.random.seed(43)  
    nb_iters = 5
    for _ in range(nb_iters):
        l = list()   # get random volumes
        for i in range(6):
            t = tuple(list(get_rand(100,3)))
            l.append(t)
        volumes_list = [
            Volume(1, l[0], l[1]),
            Volume(1, l[2], l[3]),
            Volume(1, l[4], l[5])
        ]
        logger.debug("Original list:")
        for e in l:
            logger.debug("\t%s", e)

        B = (5,6,3)
        _3d_index = (get_rand(B[0], 1)[0], 
                     get_rand(B[1], 1)[0], 
                     get_rand(B[2], 1)[0])
        add_offsets(volumes_list, _3d_index, B)
        logger.debug("3d position of buffer %s", _3d_index)

        l_test = list()
        for v in volumes_list:
            l_test.append(v.p1)
            l_test.append(v.p2)

        logger.debug("New list:")
        for e in l_test:
            logger.debug("\t%s", e)
        
        # offset as computed in add_offsets
        offset = [B[dim] * _3d_index[dim] for dim in range(len(_3d_index))]
        for i, t_test in enumerate(l_test):
            t = l[i]
            assert t_test == tuple(map(operator.add, t, offset)) 


d_arrays_expected = {
    0: [(slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))],
    1: [(slice(0, 1, None), slice(0, 40, None), slice(40, 80, None))],
    2: [(slice(0, 1, None), slice(0, 40, None), slice(80, 120, None))],
    3: [(slice(0, 1, None), slice(40, 60, None), slice(0, 40, None)),
        (slice(0, 1, None), slice(60, 80, None), slice(0, 40, None))],
    4: [(slice(0, 1, None), slice(40, 60, None), slice(40, 60, None)),
        (slice(0, 1, None), slice(40, 60, None), slice(60, 80, None)),
        (slice(0, 1, None), slice(60, 80, None), slice(40, 80, None))],
    5: [(slice(0, 1, None), slice(40, 60, None), slice(80, 120, None)),
        (slice(0, 1, None), slice(60, 80, None), slice(80, 120, None))],
    6: [
        (slice(0, 1, None), slice(80, 120, None), slice(0, 40, None))
    ],
    7: [
        (slice(0, 1, None), slice(80, 120, None), slice(40, 60, None)),
        (slice(0, 1, None), slice(80, 120, None), slice(60, 80, None))
    ],
    8: [(slice(0, 1, None), slice(80, 120, None), slice(80, 120, None))]
}

d_regions_expected = {
    0: [
        (slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))
    ],
    1:[
        (slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))
    ],
    2: [
        (slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))
    ],
    3: [
        (slice(0, 1, None), slice(0, 20, None), slice(0, 40, None)),
        (slice(0, 1, None), slice(20, 40, None), slice(0, 40, None))
    ],
    4: [
        (slice(0, 1, None), slice(0, 20, None), slice(0, 20, None)),
        (slice(0, 1, None), slice(0, 20, None),  slice(20, 40, None)),
        (slice(0, 1, None), slice(20, 40, None), slice(0, 40, None))
    ],
    5: [
        (slice(0, 1, None), slice(0, 20, None), slice(0, 40, None)),
        (slice(0, 1, None), slice(20, 40, None), slice(0, 40, None))
    ],
    6: [
        (slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))
    ],
    7: [
        (slice(0, 1, None), slice(0, 40, None), slice(0, 20, None)),
        (slice(0, 1, None), slice(0, 40, None), slice(20, 40, None))
    ],
    8: [
        (slice(0, 1, None), slice(0, 40, None), slice(0, 40, None))
    ]
}

R_test = (1,120,120)
B_test = (1,60,60)
O_test = (1,40,40)
volumes_to_keep_test = [1]


def neat_print(d):
    """ Utility function, not a test. Print a dict of Volumes.
    """
    for k, v in d.items():
        logger.debug("----output file: %s", k)
        for e in v:
            e.print()


def test_get_buff_to_vols():
    R = R_test 
    B = B_test 
    O = O_test 

    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)
    # TODO: asserts


def test_get_dirty_arrays_dict():
    """ by dirty we mean not cleaned -> see clean function
    """                
    R = R_test 
    B = B_test 
    O = O_test 

    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)
    
    test_arrays = get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes) 
    test_arrays_lengths = { k: len(v) for (k, v) in test_arrays.items()}
    expected = {
        0: 1,
        1: 2,
        2: 1,
        3: 2,
        4: 4,
        5: 2, 
        6: 1, 
        7: 2, 
        8: 1
    }

    logger.debug("----------Before merge:")
    neat_print(test_arrays)

    for k, v in expected.items():
        assert test_arrays_lengths[k] == v


def test_merge_cached_volumes():
    # prep case
    R = R_test 
    B = B_test 
    O = O_test 
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)
    test_arrays = get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes) 

    # do the merge
    merge_cached_volumes(test_arrays, volumes_to_keep_test)
    
    # assert
    expected = {
        0: 1,
        1: 1, # << modified 
        2: 1,
        3: 2,
        4: 3, # << modified
        5: 2, 
        6: 1, 
        7: 2, 
        8: 1
    }
    test_arrays_lengths = { k: len(v) for (k, v) in test_arrays.items()}
    # logger.debug("----------After merge:")
    # neat_print(test_arrays)
    for k, v in expected.items():
        assert test_arrays_lengths[k] == v


def test_clean_arrays_dict():
    # prep case
    R = R_test 
    B = B_test 
    O = O_test 
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)
    buff_to_vols = get_buff_to_vols(R, B, O, buffers_volumes, buffers_partition)
    test_arrays = get_arrays_dict(buff_to_vols, buffers_volumes, outfiles_volumes) 
    merge_cached_volumes(test_arrays, volumes_to_keep_test)

    # do the clean
    clean_arrays_dict(test_arrays)
    # logger.debug("----------After cleaning:")
    # logger.debug(test_arrays)
    for outputfile_key, expected_array_list in d_arrays_expected.items():
        arrays_list = test_arrays[outputfile_key]

        expected_array_list = list(map(lambda e: str(e), expected_array_list))
        arrays_list = list(map(lambda e: str(e), arrays_list))

        for e in expected_array_list:
            assert e in arrays_list  


def test_regions_dict():
    """ Given arrays_dict, does this function return the good regions_dict
    """
    R = R_test 
    O = O_test 
    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)

    regions_dict = get_regions_dict(d_arrays_expected, outfiles_volumes)
    for outputfile_key, expected_regions_list in regions_dict.items():
        regions_list = regions_dict[outputfile_key]
        expected_regions_list = list(map(lambda e: str(e), expected_regions_list))
        regions_list = list(map(lambda e: str(e), regions_list))

        for e in expected_regions_list:
            assert e in regions_list  


def test_get_volumes():
    """ test getmain and gethidden
    """
    R = (1,120,120)
    B = (1,60,60)
    O = (1,40,40)
    
    from dask_io.optimizer.utils.utils import numeric_to_3d_pos
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape

    buffers_partition = get_blocks_shape(R, B)

    for bufferindex in range(4):
        logger.debug("buffer %s", bufferindex)
        _3d_index = numeric_to_3d_pos(bufferindex, buffers_partition, order='F')
        T = list()
        for dim in range(3):
            nb = _3d_index[dim]+1
            logger.debug("nb:%s", nb)
            C = (nb * B[dim]) % O[dim]
            if C == 0 and B[dim] != O[dim]:
                C = O[dim]
            T.append(B[dim] - C)
            logger.debug("C: %s", C)
        logger.debug("T: %s", T)

        main_volumes = get_main_volumes(B, T)
        assert len(main_volumes) == 3

        hidden = compute_hidden_volumes(T, O)
        assert len(hidden) == 1


@pytest.fixture(autouse=True)
def get_BOR_cases():
    import json, os
    l = __file__.split('/')[:-1]
    with open(os.path.join('/', *l, 'BOR_test_cases.json')) as f:
        pytest.BOR_dict = json.load(f)


def test_compute_zones():
    for k, v in pytest.BOR_dict.items():
        BOIR = v['B'], v['O'], v['I'], v['R']
        BOIR = list(map(lambda l: tuple(l), BOIR))
        B, O, _, R = BOIR
        volumestokeep = v['keep']
        arrays_dict, regions_dict = compute_zones(B, O, R, volumestokeep)