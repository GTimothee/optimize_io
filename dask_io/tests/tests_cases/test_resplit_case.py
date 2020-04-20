from dask_io.optimizer.cases.resplit_case import Volume, add_offsets


# def test_compute_volumes():
#     vols = get_main_volumes(B, T)
#     hidd_vols = compute_hidden_volumes(T, O)
#     vols.extends(hidd_vols)
#     print(vols)


def test_Volume_add_offset():
    v1 = Volume(1, (1,2,3), (4,5,6))
    v1.add_offset((5, 2, 3))
    assert v1.p1 == (6, 4, 6)
    assert v1.p2 == (9, 7, 9)
    

# def test_add_offset():
#     add_offsets(volumes_dict, _3d_index, B)