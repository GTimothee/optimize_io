import numpy as np
import operator

from dask_io.optimizer.cases.resplit_utils import Volume
from dask_io.optimizer.cases.resplit_case import add_offsets

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