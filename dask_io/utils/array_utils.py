from h5py import Dataset
import dask


def get_array_block_dims(shape, chunk_shape):
    """ from shape of image and size of chukns=blocks, return the dimensions of the array in terms of blocks
    i.e. number of blocks in each dimension
    """
    chunks = chunk_shape 
    logging.debug(f'Chunks for get_array_block_dims: {chunks}')
    if not len(shape) == len(chunks):
        raise ValueError(
            "chunks and shape should have the same dimension",
            shape,
            chunks)
    return tuple([int(s / c) for s, c in zip(shape, chunks)])


def get_arr_shapes(arr, dtype=False):
    """ Routine that returns shape information on from dask array.

    Arguments:
    ----------
        arr: dask array

    Returns:
    --------
        shape: shape of the dask array
        chunks: shape of one chunk
        chunk_dims: number of chunks in each dimension
    """
    if not isinstance(arr, dask.array.Array):
        raise TypeError('Not a dask array')

    shape = arr.shape
    chunks = tuple([c[0] for c in arr.chunks])
    chunk_dims = [len(c) for c in arr.chunks]  
    data = [shape, chunks, chunk_dims]
    if dtype:
        data.append(arr.dtype)
    return tuple(data)


def inspect_h5py_file(f):
    print(f'Inspecting h5py file...')
    for k, v in f.items():
        print(f'\tFound object {v.name} at key {k}')
        if isinstance(v, Dataset):
            print(f'\t - Object type: dataset')
            print(f'\t - Physical chunks shape: {v.chunks}')
            print(f'\t - Compression: {v.compression}')
            print(f'\t - Shape: {v.shape}')
            print(f'\t - Size: {v.size}')
            print(f'\t - Dtype: {v.dtype}')
        else:
            print(f'\t - Object type: group')