from dask_io.optimizer.utils.utils import numeric_to_3d_pos, _3d_to_numeric_pos
from dask_io.optimizer.cases.resplit_utils import *


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
               (0,0,T[Axes.k]),
               (T[Axes.i], T[Axes.j], B[Axes.k])),
        Volume(2,
               (0, T[Axes.j], 0),
               (T[Axes.i], B[Axes.j], T[Axes.k])),
        Volume(3,
               (0, T[Axes.j], T[Axes.k]),
               (T[Axes.i], B[Axes.j], B[Axes.k])),
        Volume(4,
               (T[Axes.i], 0, 0),
               (B[Axes.i], T[Axes.j], T[Axes.k])),
        Volume(5,
               (T[Axes.i], 0, T[Axes.k]),
               (B[Axes.i], T[Axes.j], B[Axes.k])),
        Volume(6,
               (T[Axes.i], T[Axes.j], 0),
               (B[Axes.i], B[Axes.j], T[Axes.k])),
        Volume(7,
               (T[Axes.i], T[Axes.j], T[Axes.k]),
               (B[Axes.i], B[Axes.j], B[Axes.k]))]


def compute_hidden_volumes(T, O, volumes_list):
    """ II- compute hidden output files' positions (in F0)

    Hidden volumes are output files inside the f0 volume (see paper).
    Those output files can be complete or uncomplete. 
    An uncomplete volume is some output file data that is not entirely contained in f0,
    such that it overlaps with an other buffer. 

    Arguments:
    ----------
        T: Theta shape for the buffer being treated (see paper)
        O: output file shape
        volumes_list: list of volumes
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


def add_offsets(volumes_list, _3d_index, B):
    """ III - Add offset to volumes positions to get positions in the reconstructed image.
    """
    offset = [B[dim] * _3d_index[dim] for dim in range(len(_3d_index))]
    for volume in volumes_list:
        volume.add_offset(offset)


def get_arrays_dict(buff_to_vols, buffers, R, O):
    """ IV - Assigner les volumes à tous les output files, en gardant référence du type de volume que c'est
    """
    array_dict = dict()

    outfiles_partititon = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partititon, O)

    for buffer_index, buffer_volumes in buff_to_vols.items():
        crossed_outfiles = get_crossed_outfiles(buffer_index, buffers, outfiles_volumes) # refine search

        for volume in buffer_volumes:
            for outfile in crossed_outfiles:
                if included_in(volume, outfile):
                    add_to_array_dict(array_dict, outfile, volume)
                    break # a volume can belong to only one output file

    return array_dict


def merge_cached_volumes(arrays_dict, merge_rules):
    """ V - Pour chaque output file, pour chaque volume, si le volume doit être kept alors fusionner
    """
    for outfileindex in array_dict.keys():
        volumes = array_dict[outfileindex]
        for i in len(volumes):
            volume = volumes[i]
            if volume.index in merge_rules.keys():
                merge_directions = merge_rules[volume.index]
                for axis in merge_directions:
                    apply_merge(volume, volumes, axis)


def get_merge_rules(volumestokeep):
    """ Get merge rules corresponding to volumes to keep.
    See thesis for explanation of the rules.
    """
    rules = {
        1: [Axes.k] if 1 in volumestokeep else None,
        2: [Axes.j] if 2 in volumestokeep else None,
        3: [Axes.k] if 3 in volumestokeep else None,
        4: [Axes.i] if 4 in volumestokeep else None,
        5: [Axes.k] if 5 in volumestokeep else None,
        6: [Axes.j] if 6 in volumestokeep else None,
        7: [Axes.k, Axes.j] if 7 in volumestokeep else None
    }
    rules[3].append(Axes.j) if 2 in volumestokeep else pass 
    for i in [5,6,7]:
        rules[i].append(Axes.i) if 4 in volumestokeep
    for k in rules.keys():
        if rules[k] == None:
            del rules[k]  # see usage in merge_cached_volumes
    return rules


def compute_zones(B, O, R, volumestokeep):
    """ Main function of the module. Compute the "arrays" and "regions" dictionary for the resplit case.

    Arguments:
    ----------
        B: buffer shape
        O: output file shape
        R: shape of reconstructed image
        volumestokeep: volumes to be kept by keep strategy
    """
    # A/ associate each buffer to volumes contained in it
    buff_to_vols = dict()
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    for buffer in buffers_volumes:
        _3d_index = numeric_to_3d_pos(buffer.index, buffers_partition, order='F')
        
        T = list()
        for dim in range(len(buffer.p1)):
            C = (_3d_index[dim] * B[dim]) % O[dim]
            T.append(B[dim] - C)
        volumes_list = get_main_volumes(B, T)  # get coords in basis of buffer
        volumes_list = volumes_list + compute_hidden_volumes(T, O, volumes_list)  # still in basis of buffer
        
        buff_to_vols[buffer.index] = add_offsets(volumes_list, _3d_index, B)  # convert coords in basis of R

    # B/ Create arrays dict from buff_to_vols
    # arrays_dict associate each output file to parts of it to be stored at a time
    arrays_dict = get_arrays_dict(buff_to_vols, buffers_volumes, R, O) 
    merge_rules = get_merge_rules(volumestokeep)
    merge_cached_volumes(arrays_dict, merge_rules)
    clean_arrays_dict(arrays_dict)

    # C/ Create regions dict from arrays dict
    regions_dict = dict()
    # regions_dict = deepcopy(array_dict)
    # offsets = get_offsets()
    # regions_dict = remove_offset(regions_dict, offsets)

    return arrays_dict, regions_dict