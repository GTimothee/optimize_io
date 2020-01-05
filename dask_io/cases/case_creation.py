import dask.array as da
from dask_io.utils.array_utils import get_arr_shapes


def get_arr_chunks(arr, nb_chunks=None):
    """ Return the list of the chunks of the input array.

    Arguments:
    ----------
        arr: dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays

    Returns:
    --------
        a list of dask arrays.
    """
    _, chunk_shape, dims = get_arr_shapes(arr)
    arr_list = list()
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


def split_to_hdf5(arr, f, nb_blocks=None):
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
        print("creating chunk in hdf5 dataset -> dataset path: ", key)
        print("storing chunk of shape", a.shape)
        datasets.append(f.create_dataset(key, shape=a.shape))

    return da.store(arr_list, datasets, compute=False)