import os
import sys
import math
import time
import datetime 
import logging
from logging.config import fileConfig

from dask_io.optimizer.clustered import apply_clustered_strategy
from dask_io.optimizer.find_proxies import get_used_proxies, get_array_block_dims
from dask_io.optimizer.utils import LOG_TIME, LOG_DIR

fileConfig('logging_config.ini')
logger = logging.getLogger(name).addHandler(logging.NullHandler())


def clustered_optimization(graph):
    """ Applies clustered IO optimization on a Dask graph.

    Arguments:
    ----------
        graph : dark_array.dask.dicts
    """
    logger.info("Finding proxies.")
    chunk_shape, dicts = get_used_proxies(graph)

    if chunk_shape == None or dicts == None:
        return graph

    logger.info("Launching optimization algorithm.") 
    apply_clustered_strategy(graph, dicts, chunk_shape)
    return graph


def optimize_func(dsk, keys):
    """ Apply an optimization on the dask graph.
    Main function of the library. 

    Arguments:
    ----------
        dsk: dask graph
        keys: 

    Returns: 
    ----------
        the optimized dask graph
    """
    t = time.time()
    dask_graph = dsk.dicts
    dask_graph = clustered_optimization(dask_graph)
    logger.info("Time spent to create the graph: {0:.2f} milliseconds.".format((time.time() - t) * 1000))
    return dask_graph