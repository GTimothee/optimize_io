from dask_io.optimizer.cases.resplit_utils import *


def test_Volume_add_offset():
    v1 = Volume(1, (1,2,3), (4,5,6))
    v1.add_offset((5, 2, 3))
    assert v1.p1 == (6, 4, 6)
    assert v1.p2 == (9, 7, 9)


def test_Volume_get_corners():
    v1 = Volume(1, (1,2,3), (4,5,6))
    p1, p2 = v1.get_corners()
    assert p1 == (1,2,3)
    assert p2 == (4,5,6)


def test_hypercubes_overlap():
    hypercube1 = Volume(None, (0,0,0), (5,5,5))
    non_overlapping = Volume(None, (3,3,6), (8,8,11))
    overlapping = Volume(None, (3,3,3), (8,8,8))
    assert hypercubes_overlap(hypercube1, non_overlapping) == False
    assert hypercubes_overlap(hypercube1, overlapping) == True