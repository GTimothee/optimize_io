import dask
import dask.array as da
import os
import h5py
from dask_io.utils.array_utils import inspect_h5py_file


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

    if not os.path.isfile(file_path):
        raise FileNotFoundError()

    if not check_extension(file_path, 'hdf5'):
        raise ValueError("This is not a hdf5 file.") 

    f = h5py.File(file_path, 'r')

    """
    >> with h5py.File(file_path, 'r') as f:
    
    cannot do the above because it closes the file after 
    the context but the array has not yet been computed
    therefore the computation graph contains a "<Closed HDF5 dataset>" 

    TODO: The file is not closed however.
    """

    if not f.keys():
        raise ValueError('No dataset found in the input file. Aborting.')

    if not to_da:
        return f[dataset_key]

    inspect_h5py_file(f)

    dataset = f[dataset_key]

    if logic_cs == "physical":
        if physically_chunked():  
            logic_cs = dataset.chunks
        else:
            print("logic_cs set to `physical` but dataset not physically chunked. Using `auto` as logic_cs.")
            logic_cs = "auto"
    
    return da.from_array(dataset, chunks=logic_cs)


def create_random_dask_array(shape, distrib, dtype=None):
    """ Generate and return a random dask array.

    Arguments:
    ----------
        shape: array shape
        distrib: distribution from which to draw values. Supported: "normal" and "uniform".
        dtype: type of values
    """
    if distrib not in ["normal", "uniform"]:
        raise ValueError(f"The following distribution is not supported: {distrib}")

    print(f'Creating a new random array')
    print(f'- shape: {shape}')
    print(f'- distrib: {distrib}')
    print(f'- dtype: {dtype}')

    if distrib == 'normal':
        arr = da.random.normal(size=shape)
    elif distrib == 'uniform':
        arr = da.random.random(size=shape)
        
    if dtype:
        arr = arr.astype(dtype)
    return arr


def save_to_hdf5(arr, file_path, physik_cs=None, key='/data', compression=None):
    """ Save dask array to hdf5 dataset.

    Arguments: 
    ----------
        arr: dask array
        file_path
        physik_cs
        key
        compression: compression algorithm. If None then compression unabled.
    """

    print(f'Saving a dask array at {file_path}:')
    print(f'- physik_cs: {physik_cs}')
    print(f'- key: {key}')
    print(f'- compression: {compression}')

    da.to_hdf5(file_path, key, arr, chunks=physik_cs, compression=compression)
    print(f'Array successfully saved.\n')

    # TODO: sanity check
    # get_dask_array_from_hdf5(file_path, dataset_key, to_da=True, logic_cs="auto")
    # get_arr_shapes