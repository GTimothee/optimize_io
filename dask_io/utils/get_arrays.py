""" A list of utility functions for the tests

To be renamed "util.py"
"""
import dask
import dask.array as da
import math
import os
import h5py
import datetime
import csv

import dask_utils_perso
from dask_utils_perso.utils import get_random_slab, get_random_cubic_block, get_random_right_cuboid


ONE_GIG = 1000000000
SUB_BIGBRAIN_SHAPE = (1540, 1610, 1400)
LOG_TIME = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
CHUNK_SHAPES_EXP1 = {'slabs_dask_interpol': ('auto', (1210), (1400)), 
                    'slabs_previous_exp': (7, (1210), (1400)),
                    'blocks_dask_interpol': (220, 242, 200), 
                    'blocks_previous_exp': (770, 605, 700)}


def get_dask_array_from_hdf5(file_path, dataset_key, to_da=True, logic_cs="auto"):
    """ Extract a dask array from a hdf5 file using the dataset key.
    Dataset key: key of the dataset inside the hdf5 file.

    Arguments:
    ----------
        file path: path to hdf5 file (string)
        dataset_key: key of the dictionary to retrieve data

    Options:
    --------
        to_da: To cast the dataset into a dask array. True is default.
            Set it to False if you want to do it yourself (ex for adjusting the chunks).
        logic_cs:  if no physical chunked then should choose a chunks shape. 
            "auto" is automatic ~100MB chunk size.
            "physical" is to set logical chunks the same as physical chunks, if physical chunks.

    Returns: 
    --------
        dask array 
    """

    def check_extension(file_path, ext):
        if file_path.split('.')[-1] != ext:
            return False 
        return True

    def physically_chunked():
        if dataset.chunks:
            return True 
        return False

    if not check_extension(file_path, 'hdf5'):
        raise ValueError("This is not a hdf5 file.") 

    with h5py.File(file_path, 'r') as f:
        if not f.keys():
            raise ValueError('No dataset found in the input file. Aborting.')

        if not to_da:
            return f[key]

        dataset = f[key]

        if logic_cs == "physical":
            if physically_chunked:  
                logic_cs = dataset.chunks
            else:
                print("logic_cs set to `physical` but dataset not physically chunked. Using `auto` as logic_cs.")
                logic_cs = "auto"
        
        return da.from_array(dataset, chunks=logic_cs)


def load_array_parts(arr, geometry="slabs", nb_elements=0, shape=None, axis=0, random=True, seed=0, upper_corner=(0,0,0), as_numpy=False):
    """ Load part of array.
    Load 1 (or more parts -> one for the moment) of a too-big-for-memory array from file into memory.
    -given 1 or more parts (not necessarily close to each other)
    -take into account geometry
    -take into account storage type (unique_file or multiple_files) thanks to dask

    Arguments:
    ----------
        geometry: name of geometry to use
        nb_elements: nb elements wanted, not used for right_cuboid geometry
        shape: shape to use for right_cuboid
        axis: support axis for the slab
        random: if random cut or at a precise position. If set to False, upper_corner should be set.
        upper_corner: upper corner of the block/slab to be extracted (position from which to extract in the array).

    Returns a numpy array
    """
    if geometry not in ["slabs", "cubic_blocks", "right_cuboid"]:
        print("bad geometry type. Aborting.")
        return

    if geometry == "slabs":
        if random == False and len(upper_corner) != 2:
            print("Bad shape for upper corner: must be of dimension 2. Aborting.")
            return
        arr = get_random_slab(nb_elements, arr, axis, seed, random, pos=upper_corner)
    elif geometry == "cubic_blocks":
        arr = get_random_cubic_block(nb_elements, arr, seed, random, corner_index=upper_corner)
    elif geometry == "right_cuboid":
        arr = get_random_right_cuboid(arr, shape, seed, random, pos=upper_corner)

    if as_numpy:
        return arr.compute()
    else:
        return arr


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
    except OSError:
        print("An error occured while attempting to create/write csv file.")
        exit(1)


def create_random_cube(storage_type, file_path, shape, axis=0, physik_chunks_shape=None, dtype=None, distrib='uniform', compression=None):
    """ Generate random cubic array from normal distribution and store it on disk.
    Pass a dtype to cast the input dask array.

    Arguments:
        storage_type: string
        shape: tuple or integer
        file_path: has to contain the filename if hdf5 type, should not contain a filename if npy type.
        axis: for numpy: splitting dimension because numpy store matrices
        physik_chunks_shape: for hdf5 only (as far as i am concerned for the moment)

    Example usage:
    >> create_random_cube(storage_type="hdf5",
                   file_path="tests/data/bbsamplesize.hdf5",
                   shape=(1540,1210,1400),
                   physik_chunks_shape=None,
                   dtype="float16")
    """
    if distrib == 'normal':
        print('Drawing from normal distribution')
        arr = da.random.normal(size=shape)
    else:
        print('Drawing from uniform distribution')
        arr = da.random.random(size=shape)
    if dtype:
        arr = arr.astype(dtype)
    save_arr(arr, storage_type, file_path, key='/data', axis=axis, chunks_shape=physik_chunks_shape, compression=compression)


def save_arr(arr, storage_type, file_path, key='/data', axis=0, chunks_shape=None, compression=None):
    """ Save dask array to hdf5 dataset or numpy file stack.
    """

    if storage_type == "hdf5":
        if chunks_shape:
            print(f'Using chunk shape {chunks_shape}')
            da.to_hdf5(file_path, key, arr, chunks=chunks_shape)
        else:
            if compression == "gzip":
                print('Using gzip compression')
                da.to_hdf5(file_path, key, arr, chunks=None, compression="gzip")
            else:
                print('Without compression')
                da.to_hdf5(file_path, key, arr, chunks=None)
    elif storage_type == "numpy":
        da.to_npy_stack(os.path.join(file_path, "npy/"), arr, axis=axis)


def get_or_create_array(config, npy_stack_dir=None):
    """ Load or create Dask Array for tests. You can specify a test case too.

    If file exists the function returns the array.
    If chunk_shape given the function rechunk the array before returning it.
    If file does not exist it will be created using "shape" parameter.

    Arguments (from config object):
    ----------
        file_path: File containing the array, will be created if does not exist.
        chunk_shape: 
        shape: Shape of the array to create if does not exist.
        test_case: Test case. If None, returns the test array.
        nb_chunks: Number of chunks to treat in the test case.
        overwrite: Use the chunk_shape to create a new array, overwriting if file_path already used.
        split_file: for the test case 'split'
    """

    file_path = config.array_filepath
    if not os.path.isfile(file_path):
        raise FileNotFoundError()
    
    # get the file and rechunk logically using a chosen chunk shape, or dask default
    if npy_stack_dir:
        arr = da.from_npy_stack(dirname=npy_stack_dir, mmap_mode=None)
    else:
        if config.chunks_shape:
            arr = get_dask_array_from_hdf5(file_path, logic_chunks_shape=config.chunks_shape)
        else:
            arr = get_dask_array_from_hdf5(file_path) # TODO: see what happens
    return arr


def create_random_array():
    file_path = '../' + os.path.join(os.getenv('DATA_PATH'), 'sample_array_nochunk.hdf5')
    create_random_cube(storage_type="hdf5",
        file_path=file_path,
        shape=(1540, 1210, 1400),
        physik_chunks_shape=None,
        dtype="float16")
