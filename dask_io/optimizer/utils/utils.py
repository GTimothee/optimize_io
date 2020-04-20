import os
import datetime
import csv
import traceback


ONE_GIG = 1000000000
SUB_BIGBRAIN_SHAPE = (1540, 1610, 1400)
CHUNK_SHAPES_EXP1 = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}


def flush_cache():
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches') 


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print("\nk", k2)
                print(v2, "\n")


def create_csv_file(filepath, columns, delimiter=',', mode='w+'):
    """
    args:
        file_path: path of the file to be created
        columns: column names for the prefix
        delimiter
        mode

    returns:
        csv_out: file handler in order to close the file later in the program
        writer: writer to write rows into the csv file
    """
    try:
        csv_out = open(filepath, mode=mode)
        writer = csv.writer(csv_out, delimiter=delimiter)
        writer.writerow(columns)
        return csv_out, writer
    except OSError as e:
        print(traceback.format_exc())
        print("An error occured while attempting to create/write csv file.")
        exit(1)


def add_to_dict_of_lists(d, k, v, unique=False):
    """ if key does not exist, add a new list [value], else, 
    append value to existing list corresponding to the key
    """
    if k not in d:
        if v:
            d[k] = [v]
        else:
            d[k] = list()
    else:
        if v and (unique and v not in d[k]) or not unique:
            d[k].append(v)
    return d


def flatten_iterable(l, plain_list=list()):
    for e in l:
        if isinstance(e, list) and not isinstance(e, (str, bytes)):
            plain_list = flatten_iterable(e, plain_list)
        else:
            plain_list.append(e)
    return plain_list


def numeric_to_3d_pos(numeric_pos, blocks_shape, order):
    if order == 'F':
        nb_blocks_per_row = blocks_shape[0]
        nb_blocks_per_slice = blocks_shape[0] * blocks_shape[1]
    elif order == 'C':
        nb_blocks_per_row = blocks_shape[2]
        nb_blocks_per_slice = blocks_shape[1] * blocks_shape[2]
    else:
        raise ValueError("unsupported")

    i = math.floor(numeric_pos / nb_blocks_per_slice)
    numeric_pos -= i * nb_blocks_per_slice
    j = math.floor(numeric_pos / nb_blocks_per_row)
    numeric_pos -= j * nb_blocks_per_row
    k = numeric_pos
    return (i, j, k)


def _3d_to_numeric_pos(_3d_pos, shape, order):
    if order == 'F':
        nb_blocks_per_row = shape[0]
        nb_blocks_per_slice = shape[0] * shape[1]
    elif order == 'C':
        nb_blocks_per_row = shape[2]
        nb_blocks_per_slice = shape[1] * shape[2]
    else:
        raise ValueError("unsupported")

    return (_3d_pos[0] * nb_blocks_per_slice) + \
        (_3d_pos[1] * nb_blocks_per_row) + _3d_pos[2]