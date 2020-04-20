class Axes(Enum):
    i: 0
    j: 1
    k: 2


class Volume:
    def __init__(self, index, p1, p2):
        if not isinstance(p1, tuple) \ 
            or not isinstance(p2, tuple):
            raise TypeError()

        self.index = index
        self.p1 = p1  # bottom left corner
        self.p2 = p2  # top right corner


    def add_offset(self, offset):
        """
        offset: a tuple
        """
        self.p1 = self._add_offset(self.p1, offset)
        self.p2 = self._add_offset(self.p2, offset)
            

    def _add_offset(self, p, offset):
        if isinstance(offset, list):
            offset = tuple(offset)
        elif not isinstance(offset, tuple):
            raise TypeError("Expected tuple")
        return tuple(map(operator.add, p, offset))


    def get_corners(self):
        return (self.p1, self.p2)


def hypercubes_overlap(hypercube1, hypercube2):
    """ Evaluate if two hypercubes cross each other.
    """
    if not isinstance(hypercube1, Volume) or \
        not isinstance(hypercube2, Volume):
        raise TypeError()

    lowercorner1, uppercorner1 = hypercube1
    lowercorner2, uppercorner2 = hypercube2
    nb_dims = len(uppercorner1)
    
    for i in range(nb_dims):
        if not uppercorner1[i] > lowercorner1[i] and \
            not uppercorner2[i] > lowercorner1[i]:
            return False

    return True


def get_blocks_shape(big_array, small_array):
    """ Return the number of small arrays in big array in all dimensions as a shape. 
    """
    return tuple([int(b/s) for b, s in zip(big_array, small_array)])


def get_crossed_outfiles(buffer_index, buffers, outfiles):
    """ Returns list of output files that are crossing buffer at buffer_index.

    Arguments: 
    ----------
        buffer_index: Integer indexing the buffer of interest in storage order.
        buffers: list of volumes representing the buffers.
        outfiles: list of volumes representing the output files.
    """
    crossing = list()
    buffer_of_interest = buffers[buffer_index]
    for outfile in outfiles:
        if hypercubes_overlap(buffer_of_interest.get_corners(), outfile.get_corners()):
            crossing.append(outfile)
    return crossing


def merge_volumes(volume1, volume2):
    """ Merge two volumes into one.
    """
    if not isinstance(volume1, Volume) or \
        not isinstance(volume2, Volume):
        raise TypeError()

    lowercorner1, uppercorner1 = volume1
    lowercorner2, uppercorner2 = volume2
    lowercorner = (min(lowercorner1[0], lowercorner2[0]), 
                   min(lowercorner1[1], lowercorner2[1]),
                   min(lowercorner1[2], lowercorner2[2]))
    uppercorner = (max(uppercorner1[0], uppercorner2[0]), 
                   max(uppercorner1[1], uppercorner2[1]),
                   max(uppercorner1[2], uppercorner2[2]))
    return Volume(None, lowercorner, uppercorner)


def included_in(volume, outfile):
    """ Alias of hypercubes_overlap. 
    We do not verify that it is included but by definition
    of the problem if volume crosses outfile then volume in outfile.

    Arguments: 
    ----------
        volume: Volume in buffer
        outfile: Volume representing an output file
    """
    return hypercubes_overlap(volume, outfile)


def add_to_array_dict(array_dict, outfile, volume):
    """ Add volume information to dictionary associating output file index to 
    """
    if not isinstance(outfile.index, int) or \ 
        not isinstance(volume, Volume) or \ 
        not isinstance(outfile, Volume):
        raise TypeError()

    if not outfile.index in array_dict.keys():
        array_dict[outfile.index] = list()
    array_dict[outfile.index].append(volume)


def convert_Volume_to_slices(v):
    if not isinstance(v, Volume):
        raise TypeError()
    p1, p2 = v.get_corners()
    return [slice(p1[dim], p2[dim], None) for dim in range(len(p1))]


def clean_arrays_dict(arrays_dict):
    for k in arrays_dict.keys():
        v = arrays_dict[k]
        slices = convert_Volume_to_slices(v)
        arrays_dict(k) = v