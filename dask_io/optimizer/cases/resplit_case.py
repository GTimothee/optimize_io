from enum import Enum
from dask_io.optimizer.cases.resplit_utils import *


class Axes(Enum):
    i: 0,
    j: 1,
    k: 2


class Volume:
    def __init__(self, index, p1, p2):
        self.index = index
        self.p1 = p1
        self.p2 = p2


def get_main_volumes(B, T):
    """ I- Get a dictionary associating volume indices to volume positions in the buffer.
    Indexing following the keep algorithm indexing in storage order.
    Position following pillow indexing for rectangles i.e. (bottom left corner, top right corner)

    Arguments:
    ----------
        B: buffer shape
        T: Theta prime shape -> Theta value for C_x(n) (see paper)
    """
    return [
        Volume(1,
               (B[Axes.i], 0, T[Axes.k]),
               (0, T[Axes.j], B[Axes.k])),
        Volume(2,
               (B[Axes.i], T[Axes.j], T[Axes.k]),
               (0, B[Axes.j], B[Axes.k])),
        Volume(3,
               (T[Axes.i], T[Axes.j], 0),
               (0, B[Axes.j], T[Axes.k])),
        Volume(4,
               (B[Axes.i], 0, 0),
               (T[Axes.i], T[Axes.j], T[Axes.k])),
        Volume(5,
               (B[Axes.i], 0, T[Axes.k]),
               (T[Axes.i], T[Axes.j], B[Axes.k])),
        Volume(6,
               (B[Axes.i], T[Axes.j], 0),
               (T[Axes.i], B[Axes.j], T[Axes.k])),
        Volume(7,
               (B[Axes.i], T[Axes.j], T[Axes.k]),
               (T[Axes.i], B[Axes.j], B[Axes.k]))]


def compute_hidden_volumes(T, O, volumes):
    """ II- compute hidden output files' positions (in F0)

    Hidden volumes are output files inside the f0 volume (see paper).
    Those output files can be complete or uncomplete. 
    An uncomplete volume is some output file data that is not entirely contained in f0,
    such that it overlaps with an other buffer. 

    Arguments:
    ----------
        T: Theta shape for the buffer being treated (see paper)
        O: output file shape
        volumes: list of volumes
    """
    # a) get volumes' graduation on x, y, z axes
    # i.e. find the crosses in the 2D example drawing below:
    #   
    # k
    # ↑
    # -----------------
    # |     f1     |f3|
    # x------------|--|      ▲  ▲ ---------- T[dim==k]
    # |            |  |      |  | ← O[k]
    # |   hidden   |f2|      |  |
    # x------------|  |      |  ▼
    # |   hidden   |  |      | ← Theta[k]
    # x------------x--- → j  ▼   

    points = list()
    for dim in range(3):
        points_on_axis = list()
        nb_hidden_volumes = T[dim]/O[dim] 
        nb_complete_vols = floor(nb_hidden_volumes) 

        a = T[dim]
        points_on_axis.append(a)
        for _ in range(nb_complete_vols):
            b = a - O[dim]
            points_on_axis.append(b)
            a = b

        if not 0 in points_on_axis:
            points_on_axis.append(0)

        points_on_axis.sort()
        points.append(points_on_axis)

    # b) compute volumes' corners (bottom left and top right) from points
    blc_index = [0,0,0] # bottom left corner index
    trc_index = [1,1,1] # top right corner index

    # -----------------
    # |     f1     |f3|
    # |---------trc2--|      ▲  ▲ ---------- T[dim==k]
    # |            |  |      |  | ← O[k]
    # |   hidden   |f2|      |  |
    # blc2------trc1  |      |  ▼
    # |   hidden   |  |      | ← Theta[k]
    # blc1-------------      ▼

    index = 7 # key of the volume in the dictionary of volumes, [1 -> 7 included] already taken
    for i in range(len(points[0])-1):
        for j in range(len(points[1])-1):
            for k in range(len(points[2])-1):
                corners = [(points[0][blc_index[0]], points[1][blc_index[1]], points[2][blc_index[2]]),
                           (points[0][trc_index[0]], points[1][trc_index[1]], points[2][trc_index[2]])]
                index += 1
                volumes.append(Volume(index, corners[0], corners[1]))

                blc_index[Axis.k] += 1
                trc_index[Axis.k] += 1
            blc_index[Axis.j] += 1
            trc_index[Axis.j] += 1
        blc_index[Axis.i] += 1
        trc_index[Axis.i] += 1

return volumes


def add_offsets(volumes):
    """ III - Add offset to volumes positions to get positions in the reconstructed image.
    """
    return volumes


def get_array_dict(buff_to_vols):
    """ IV - Assigner les volumes à tous les output files, en gardant référence du type de volume que c'est
    """
    array_dict = dict()

    for buffer_index, buffer_volumes in buff_to_vols.items():
        crossed_outfiles = get_crossed_outfiles(buffer_index, outfiles) # refine search

        for volume in buffer_volumes:
            for outfile in crossed_outfiles:
                if included_in(volume, outfile):
                    add_to_array_dict(array_dict, outfile, volume)
                    break # a volume can belong to only one output file

    return array_dict


def merge_cached_volumes(arrays_dict):
    """ V - Pour chaque output file, pour chaque volume, si le volume doit être kept alors fusionner
    """
    for outfileindex in array_dict.keys():
        volumes = array_dict[outfileindex]
        for volume in volumes:
            if volume.index in volumestokeep:
                merge_volumes(volumes, volume.index)
        array_dict[outfileindex] = map_to_slices(volumes)
    return


def compute_zones(B, O):
    """ Main function of the module. Compute the "arrays" and "regions" dictionary for the resplit case.

    Arguments:
    ----------
        B: buffer shape
        O: output file shape
    """
    buff_to_vols = dict()

    for buffer_index in range(nb_buffers):
        _3d_index = get3dpos(buffer_index)
        T = list()
        for i in range(3):
            C = (_3d_index[i] * B[i]) % O[i]
            T.append(B[i] - C)

        volumes = get_main_volumes(B, T)
        hidden_volumes = compute_hidden_volumes(T, O, volumes)
        volumes.extends(hidden_volumes)

        add_offsets(volumes)
        buff_to_vols[buffer_index] = volumes
        
    arrays_dict = get_array_dict(buff_to_vols)
    merge_cached_volumes(arrays_dict)

    regions_dict = deepcopy(array_dict)
    offsets = get_offsets()
    regions_dict = remove_offset(regions_dict, offsets)

    return arrays_dict, regions_dict