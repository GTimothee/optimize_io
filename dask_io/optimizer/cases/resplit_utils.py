class Axes(Enum):
    i: 0
    j: 1
    k: 2


class Volume:
    def __init__(self, index, p1, p2):
        if not isinstance(index, int):
            raise TypeError()
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


def merge_volumes():
    pass


def included_in(volume, outfile):
    pass


def add_to_array_dict(array_dict, outfile, volume):
    pass