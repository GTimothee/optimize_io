

def get_arr_shapes(arr):
    """ Routine that returns shape information on from dask array.

    Arguments:
    ----------
        arr: dask array

    Returns:
    --------
        shape: shape of the dask array
        chunks: shape of one chunk
        chunk_dims: number of chunks in eah dimension
    """
    shape = arr.shape
    chunks = tuple([c[0] for c in arr.chunks])
    chunk_dims = [len(c) for c in arr.chunks]  
    return shape, chunks, chunk_dims


def get_dask_array_chunks_shape(dask_array):
    t = dask_array.chunks
    cs = list()
    for tupl in t:
        cs.append(tupl[0])
    return tuple(cs