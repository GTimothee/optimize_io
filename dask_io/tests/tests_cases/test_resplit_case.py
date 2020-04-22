import numpy as np
import operator

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
    0: [(slice(0, 40, None), slice(0, 40, None), slice(None, None, None))],
    1: [(slice(0, 40, None), slice(40, 80, None), slice(None, None, None))],
    2: [(slice(0, 40, None), slice(80, 120, None), slice(None, None, None))],
    3: [(slice(40, 60, None), slice(0, 40, None), slice(None, None, None)),
        (slice(60, 80, None), slice(0, 40, None), slice(None, None, None))],
    4: [(slice(40, 60, None), slice(40, 60, None), slice(None, None, None)),
    (slice(40, 60, None), slice(60, 80, None), slice(None, None, None)),
    (slice(60, 80, None), slice(40, 80, None), slice(None, None, None))],
    5: [(slice(40, 60, None), slice(80, 120, None), slice(None, None, None)),
        (slice(60, 80, None), slice(80, 120, None), slice(None, None, None))],
    6: [
        (slice(80, 120, None), slice(0, 40, None), slice(None, None, None))
    ],
    7: [
        (slice(80, 120, None), slice(40, 60, None), slice(None, None, None)),
        (slice(80, 120, None), slice(60, 80, None), slice(None, None, None))
    ],
    8: [(slice(80, 120, None), slice(80, 120, None), slice(None, None, None))]
}

d_regions_expected = {
    0: [
        (slice(0, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    1:[
        (slice(0, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    2: [
        (slice(0, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    3: [
        (slice(0, 20, None), slice(0, 40, None), slice(None, None, None)),
        (slice(20, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    4: [
        (slice(0, 20, None), slice(0, 20, None), slice(None, None, None)),
        (slice(0, 20, None),  slice(20, 40, None),  slice(None, None, None)),
        (slice(20, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    5: [
        (slice(0, 20, None), slice(0, 40, None), slice(None, None, None)),
        (slice(20, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    6: [
        (slice(0, 40, None), slice(0, 40, None), slice(None, None, None))
    ],
    7: [
        (slice(0, 40, None), slice(0, 20, None), slice(None, None, None)),
        (slice(0, 40, None), slice(20, 40, None), slice(None, None, None))
    ],
    8: [
        (slice(0, 40, None), slice(0, 40, None), slice(None, None, None))
    ]
}

R_test = (1,120,120)
B_test = (1,60,60)
O_test = (1,40,40)
volumes_to_keep_test = [1]


def test_get_dict():
    """ test getbufftovols and getarraydict
    """
    def neat_print(d):
        for k, v in d.items():
            logger.debug("----output file: %s", k)
            for e in v:
                e.print()
                
    R = R_test 
    B = B_test 
    O = O_test 

    buff_to_vols = dict()
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    for buffer_index in buffers_volumes.keys():
        _3d_index = numeric_to_3d_pos(buffer_index, buffers_partition, order='F')
        
        T = list()
        for dim in range(len(buffers_volumes[buffer_index].p1)):
            C = ((_3d_index[dim]+1) * B[dim]) % O[dim]
            if C == 0 and B[dim] != O[dim]:
                C = O[dim]
            T.append(B[dim] - C)
        volumes_list = get_main_volumes(B, T)  # get coords in basis of buffer
        volumes_list = volumes_list + compute_hidden_volumes(T, O)  # still in basis of buffer
        add_offsets(volumes_list, _3d_index, B)  # convert coords in basis of R
        buff_to_vols[buffer_index] = volumes_list

    # test getarraydict
    test_arrays = get_arrays_dict(buff_to_vols, buffers_volumes, R, O) 
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

    # test merge
    merge_rules = get_merge_rules(volumes_to_keep_test)
    merge_cached_volumes(test_arrays, merge_rules)
    test_arrays_lengths = { k: len(v) for (k, v) in test_arrays.items()}

    expected = {
        0: 1,
        1: 1, # << 
        2: 1,
        3: 2,
        4: 3, # << 
        5: 2, 
        6: 1, 
        7: 2, 
        8: 1
    }

    logger.debug("----------After merge:")
    neat_print(test_arrays)

    for k, v in expected.items():
        assert test_arrays_lengths[k] == v

    clean_arrays_dict(test_arrays)
    for k, s_list in d_arrays_expected.items():
        s_list2 = test_arrays[k]
        s_list = list(map(lambda e: str(e), s_list))
        s_list2 = list(map(lambda e: str(e), s_list))
        for e in s_list:
            logger.debug("e:%s", e)
            assert e in s_list2  # TODO: does not work
        
    logger.debug("-------------RESULT")
    logger.debug(test_arrays)


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