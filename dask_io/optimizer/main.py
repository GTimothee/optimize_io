import os
import sys
import math
import time
import datetime 
import logging

from dask_io.optimizer.clustered import apply_clustered_strategy
from dask_io.optimizer.modifiers import get_used_proxies, get_array_block_dims
from dask_io.utils.utils import LOG_TIME, LOG_DIR

logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_TIME + '.log'), level=logging.DEBUG) # to be set to WARNING

DEBUG_MODE = False

def clustered_optimization(graph):
    """ Applies clustered IO optimization on a Dask graph.

    Arguments:
    ----------
        graph : dark_array.dask.dicts
    """
    print("Finding proxies.")
    chunk_shape, dicts = get_used_proxies(graph)

    if chunk_shape == None or dicts == None:
        return graph

    print("Launching optimization algorithm.") 
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
    logging.info("Time spent to create the graph: {0:.2f} milliseconds.".format((time.time() - t) * 1000))

    def neat_print_graph(graph, log=True):
        for k, v in graph.items():
            if log: 
                logging.debug(f"\nkey: {k}")
            else:
                print(f"\nkey: {k}")

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if log: 
                        logging.debug(f"\tk: {k2}")
                        logging.debug(f"\t{v2} \n")
                    else:
                        print(f"\tk: {k2}")
                        print(f"\t{v2} \n")

            else:
                if log: 
                    logging.debug(f"\tv: {v}")
                else:
                    print(f"\tv: {v}")

    neat_print_graph(dsk, log=True)

    if DEBUG_MODE:
        raise ValueError("stop here")

    return dsk