def hypercubes_overlap(hypercube1, hypercube2):
    uppercorner1, lowercorner1 = hypercube1
    uppercorner2, lowercorner2 = hypercube2
    nb_dims = len(uppercorner1)
    
    for i in range(nb_dims):
        if not uppercorner1[i] > lowercorner1[i] and \
            not uppercorner2[i] > lowercorner1[i]:
            return False

    return True


def get_crossed_outfiles(buffer_index, outfiles):
    pass 


def merge_volumes():
    pass


def included_in(volume, outfile):
    pass


def add_to_array_dict(array_dict, outfile, volume):
    pass