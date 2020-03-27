import os
import dask
import collections
import numpy as np
import datetime
import logging

from collections import Hashable

from dask_io.optimizer.utils.utils import add_to_dict_of_lists, flatten_iterable
from dask_io.optimizer.utils.array_utils import get_array_block_dims

logger = logging.getLogger(__name__)


"""
    get_used_proxies:
        Main function of the module.
        Find the proxies that are used by other tasks in the task graph.

    standard BFS: 
        A standard breadth first search algorithm to find the different trees in the graph.

    get_graph_from_dask: 
        Convert a dask graph (output of array.dask.dicts) 
        into a graph in mathematical format i.e. a dictionary mapping 
        each node with its dependencies.

    search_dask_graph:
        Search proxies in the remade graph and fill in dictionaries to store information.
"""

def standard_BFS(root, graph):
    """ Apply a standard breadth first search algorithm on a graph stored in a dictionary.

    return: 
    -------
        visited: list of graph nodes reachable from root
        max_depth: the maximum depth reached during the process i.e. the depth of the tree.
    """
    nodes = list(graph.keys())
    queue = [(root, 0)]
    visited = [root]

    max_depth = 0
    while len(queue) > 0:
        node, depth = queue.pop(0)

        if depth > max_depth:
            max_depth = depth

        try:  # we want to stop digging when node is a value i.e. a leaf
            hash(node)
            if not node in graph:  
                continue
        except:
            continue

        neighbors = graph[node]
        for n in neighbors:
            if not n in visited:
                queue.append((n, depth + 1))
                visited.append(n)
    
    return visited, max_depth


def is_task(v):
    if isinstance(v, tuple) and callable(v[0]):
        return True 
    return False


def get_graph_from_dask(graph, undirected=False):
    """ Transform dask graph into a real graph in order to use graph algorithms on it.
    in the graph, each node is an element (tuple, list, object)
    when directed, each edge is the relation "is the result of a function on"
    {a: b} implies that a is the result of a function a = f(b or more)
    each key is mapped to a list of values (other keys used by this key as input)
    """

    def add_to_remade_graph(d, key, value, undir):
        """
        arg: 
            undir: do you want undirected graph
        """
        try:  # do not treat slices
            if (isinstance(key, tuple) and all([isinstance(s, slice) for s in key])) or (isinstance(value, tuple) and all([isinstance(s, slice) for s in value])):
                return 
        except:
            pass

        d = add_to_dict_of_lists(d, key, value, unique=True)
        if undir:
            d = add_to_dict_of_lists(d, value, key, unique=True)

    remade_graph = dict()
    for key, v in graph.items():  
        # if it is a subgraph, recurse
        if isinstance(v, dict):
            if isinstance(key, str) and "array-original" in key:
                add_to_remade_graph(remade_graph, key, v, undirected)
            else:
                subgraph = get_graph_from_dask(v, undirected=undirected)
                remade_graph.update(subgraph)

        # if it is a task, add its arguments
        elif is_task(v):  
            for arg in v[1:]:
                if isinstance(arg, tuple):
                    pass

                if isinstance(arg, list):
                    l = flatten_iterable(arg)
                    for e in l:
                        if isinstance(e, tuple):
                            pass 
                        if isinstance(key, collections.Hashable) and isinstance(e, collections.Hashable):                 
                            add_to_remade_graph(remade_graph, key, e, undirected)
                    continue

                if isinstance(key, collections.Hashable) and isinstance(arg, collections.Hashable):   
                    add_to_remade_graph(remade_graph, key, arg, undirected)

        # if it is an argument, add it
        elif isinstance(key, Hashable):
            if isinstance(v, Hashable):  
                if isinstance(v, tuple):
                    pass
                add_to_remade_graph(remade_graph, key, v, undirected)
        else:
            pass
    
    return remade_graph


#TODO : refactor
def search_dask_graph(graph, 
    main_components,
    proxy_to_slices,
    origarr_to_used_proxies,
    origarr_to_obj,
    origarr_to_blocks_shape,
    proxy_to_dict,
    unused_keys):
    """ Search proxies in the remade graph and fill in dictionaries to store information.
    """

    for key, v in graph.items():  

        # if it is a subgraph, recurse
        if isinstance(v, dict):
            search_dask_graph(v, main_components,
            proxy_to_slices,
            origarr_to_used_proxies,
            origarr_to_obj,
            origarr_to_blocks_shape,
            proxy_to_dict,
            unused_keys)

        # if it is an original array, store it
        elif isinstance(key, str) and "array-original" in key: # TODO: support other formats
            obj = v
            origarr_to_obj[key] = obj
            if not obj.shape:
                raise ValueError("Empty dataset!")
            continue

        # if it is a task, add its arguments
        elif is_task(v) and (key not in unused_keys): 
            if main_components:
                used_key = False
                for main_comp in main_components:
                    if key in main_comp:
                        used_key = True 
            else:
                used_key = True

            if used_key:
                try:
                    f, target, slices = v
                    # search for values that are array-original, meaning that key is proxy 
                    if "array-original" in target and all([isinstance(s, slice) for s in slices]):
                        add_to_dict_of_lists(origarr_to_used_proxies, target, key, unique=True)
                        proxy_to_slices[key] = slices
                        proxy_to_dict[key] = graph
                        continue
                except:
                    pass
        else:
            pass

    return 


def get_root_nodes(remade_graph):
    """ Find keys in the graph that are not used as values by another(other) key(s).
    Some of those keys are root nodes of the graph. 
    """
    keys = list(remade_graph.keys())
    vals = list(remade_graph.values())
    flatten = list()

    # flatten the values which is a list of lists 
    # because get_graph_from_dask which is using add_to_dict_of_lists
    for l in vals:
        for e in l:
            flatten.append(e)

    # do the actual job
    root_nodes = list()
    for key in keys:
        if key not in flatten:
            root_nodes.append(key)

    return root_nodes


def get_used_proxies(graph):
    """ Find the proxies that are used by other tasks in the task graph.
    We call ``proxy" a task that uses ``getitem" directly on the ``original-array".
    """
    unused_keys = list()
    proxy_to_slices = dict()
    origarr_to_used_proxies = dict()
    origarr_to_obj = dict()
    origarr_to_blocks_shape = dict()
    proxy_to_dict = dict()

    remade_graph = get_graph_from_dask(graph, undirected=False)
    
    log_file_path = os.path.join('/tmp', 'dask_io_input_graph.log')
    logger.info('Input graph log can be found at %s', log_file_path)
    with open(log_file_path, "w+") as f:
        for k, v in graph.items():    
            f.write("\n\n " + str(k))
            f.write("\n" + str(v))

    all_nodes = list()
    for root in get_root_nodes(remade_graph):
        nodes_used_list, depth = standard_BFS(root, remade_graph)
        all_nodes.append(nodes_used_list)

    search_dask_graph(graph, all_nodes,
    proxy_to_slices,
    origarr_to_used_proxies,
    origarr_to_obj,
    origarr_to_blocks_shape,
    proxy_to_dict,
    unused_keys)
  
    # find chunk shape (to be replaced)
    #-------------------------------------------------------------
    
    if not len(list(proxy_to_slices.keys())) > 0:
        return None, None

    first_key = list(proxy_to_slices.keys())[0]
    sample_slices = proxy_to_slices[first_key]
    logger.info("proxytoslices first_key %s: %s", first_key, sample_slices)    
    chunk_shape = list()
    for s in sample_slices:
        start = s.start
        stop = s.stop
        chunk_shape.append(stop - start)
    chunk_shape = tuple(chunk_shape)

    # create the new dictionary (to be replaced)
    origarr_to_blocks_shape = dict()
    for key, obj in origarr_to_obj.items():
        logger.info("supposed to be original array: %s -> %s", key, obj)
        blocks_dims = get_array_block_dims(obj.shape, chunk_shape)
        logger.debug('Found following block dimensions: %s', blocks_dims)
        origarr_to_blocks_shape[key] = blocks_dims 
        # warning on above line: 
        # if more than one original array and different chunk shapes it will not work
    #-------------------------------------------------------------

    return chunk_shape, {
        'proxy_to_slices': proxy_to_slices, 
        'origarr_to_used_proxies': origarr_to_used_proxies,
        'origarr_to_obj': origarr_to_obj,
        'origarr_to_blocks_shape': origarr_to_blocks_shape,
        'proxy_to_dict': proxy_to_dict
    }