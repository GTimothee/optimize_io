def hypercubes_overlap(rect1, rect2):
    corn1 = rect1.get_corners()
    corn2 = rect2.get_corners()
    vertical_cross, horizontal_cross = (False, False)
    
    if corn1["bot_left"].y > corn2["top_left"].y \
        or corn2["bot_left"].y > corn1["top_left"].y:
        return False
    else:
        vertical_cross = True
    
    if corn1["bot_right"].x < corn2["bot_left"].x \
        or corn2["bot_right"].x < corn1["bot_left"].x:
        return False
    else:
        horizontal_cross = True
    
    if vertical_cross and horizontal_cross:
        return True
    else:
        return False

def get_crossed_outfiles(buffer_index, outfiles):
    pass 

def merge_volumes():
    pass

def included_in(volume, outfile):
    pass

def add_to_array_dict(array_dict, outfile, volume):
    pass