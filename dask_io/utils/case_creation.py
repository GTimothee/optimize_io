

def get_arr_list(arr, nb_chunks=None):
    """ Return a list of dask arrays. Each dask array being a block of the input array.

    Arguments:
    ----------
        arr: dask array
        nb_chunks: if None then function returns all arrays, else function returns n=nb_chunks arrays
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
                    arr_list.append(load_array_parts(arr=arr,
                                                     geometry="right_cuboid",
                                                     shape=chunk_shape,
                                                     upper_corner=upper_corner,
                                                     random=False))
    return arr_list


def sum_chunks(arr, nb_chunks):
    """ Sum chunks together.

    Arguments:
    ----------
        arr: array from which blocks will be sum
        nb_chunks: number of chunks to sum
    """
    
    arr_list = get_arr_list(arr, nb_chunks)
    sum_arr = arr_list.pop(0)
    for a in arr_list:
        sum_arr = sum_arr + a
    return sum_arr


    def get_test_arr(config, npy_stack_dir=None):

    # create the dask array from input file path
    arr = get_or_create_array(config, npy_stack_dir=npy_stack_dir)
    
    # do dask arrays operations for the chosen test case
    case = getattr(config, 'test_case', None)
    # print("case in config", case)
    if case:
        if case == 'sum':
            arr = sum_chunks(arr, config.nb_chunks)
        elif case == 'split':
            arr = split_array(arr, config.out_file, config.nb_blocks)
    return arr


def split_array(arr, f, nb_blocks=None):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.
    
    Arguments:
    ----------
        nb_blocks: nb blocks we want to extract
    """
    # arr_list = get_arr_list(arr, nb_blocks)
    datasets = list()

    # for hdf5:
    """for i, a in enumerate(arr_list):
        # print("creating dataset in split file -> dataset path: ", '/data' + str(i))
        # print("storing data of shape", a.shape)
        datasets.append(f.create_dataset('/data' + str(i), shape=a.shape))
    return da.store(arr_list, datasets, compute=False)"""

    # for numpy storage
    return da.to_npy_stack('data/numpy_data', arr, axis=0)