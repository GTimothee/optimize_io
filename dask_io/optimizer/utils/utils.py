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