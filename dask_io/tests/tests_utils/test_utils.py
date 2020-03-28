from dask_io.optimizer.utils.utils import add_to_dict_of_lists, flatten_iterable

def test_add_to_dict_of_lists():
    d = {'a': [1], 'c': [5, 6]}
    d = add_to_dict_of_lists(d, 'b', 2)
    d = add_to_dict_of_lists(d, 'b', 3)
    d = add_to_dict_of_lists(d, 'c', 4)
    expected = {'a': [1], 'b': [2, 3], 'c': [5, 6, 4]}
    assert expected == d


def test_flatten_iterable():
    l = [0, [1, 2, 3], 4, [5], [[6, 7]]]
    assert flatten_iterable(l, list()) == list(range(8))