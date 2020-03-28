import logging
import dask.array as da

from dask_io.optimizer.utils.array_utils import get_arr_shapes, get_arr_shapes

logger = logging.getLogger(__name__)


def get_arr_chunks(arr, nb_chunks=None, as_dict=False):
    """ Return the list of the chunks of the input array.

    Arguments:
    ----------
        arr: dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays

    Returns:
    --------
        a list of dask arrays.
    """
    data = get_arr_shapes(arr)
    if len(data) == 3:
        _, chunk_shape, dims = data
    elif len(data) == 4:
        _, chunk_shape, dims, _ = data
    else:
        raise ValueError()

    arr_list = list()
    positions = list()
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (nb_chunks and len(arr_list) < nb_chunks) or not nb_chunks:
                    upper_corner = (i * chunk_shape[0],
                                    j * chunk_shape[1],
                                    k * chunk_shape[2])

                    arr_list.append(
                        arr[
                            upper_corner[0]: upper_corner[0] + chunk_shape[0], 
                            upper_corner[1]: upper_corner[1] + chunk_shape[1], 
                            upper_corner[2]: upper_corner[2] + chunk_shape[2]
                        ]
                    )
                    if as_dict:
                        positions.append((i,j,k))

    if as_dict:
        return zip(positions, arr_list)
    else:
        return arr_list


def sum_chunks_case(arr, nb_chunks, compute=False):
    """ A test case where chunks of a dask array are summed up.

    Arguments:
    ----------
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
        compute: to compute directly
    """
    
    arr_list = get_arr_chunks(arr, nb_chunks)
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a

    if compute:
        return sum_arr.compute()
    return sum_arr


def split_to_hdf5(arr, f, nb_blocks):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.
    
    Arguments:
    ----------
        arr: array to split
        f: an open hdf5 file to store data in it.
        nb_blocks: nb blocks we want to extract
    """
    arr_list = get_arr_chunks(arr, nb_blocks)
    datasets = list()

    for i, a in enumerate(arr_list):
        key = '/data' + str(i)
        logger.debug("creating chunk in hdf5 dataset -> dataset path: %s", key)
        logger.debug("storing chunk of shape %s", a.shape)
        datasets.append(f.create_dataset(key, shape=a.shape))

    return da.store(arr_list, datasets, compute=False)


def split_hdf5_multiple(arr, file_list, nb_blocks):
    """
    Arguments:
    ----------
        arr: Array to split
        file_list: Empty list to store output files' objects.
        nb_blocks: Nb blocks we want to extract. None = all blocks.
    """
    # create output files

    # get array blocks
    arr_list = get_arr_chunks(arr, nb_blocks)

    # for each out file
        # create dataset wth key = key of block

    return da.store(arr_list, datasets, compute=False)