import os
import dask.array as da
from dask_io.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.cases.case_creation import sum_chunks_case, split_to_hdf5


class CaseConfig():
    """ Contains the configuration for a test.
    """
    def __init__(self, array_filepath, chunks_shape):
        """ 
        Arguments:
        ----------
            array_filepath: to load input array from file
            chunks_shape: logical chunks shape to use when loading array
            dask_config: configuration to use
        """

        if not chunks_shape:
            raise ValueError(f'chunks_shape cannot be None.')

        self.array_filepath = array_filepath
        self.chunks_shape = chunks_shape
        self.case = None


    def sum(self, nb_chunks):
        """ Return a dask array of shape chukns_shape containing the sum of all chunks of the input array.
        """

        self.case = {
            'name': 'sum',
            'params': {
                'nb_chunks': nb_chunks
            }
        }


    def split_hdf5(self, out_filepath, nb_blocks=None):
        """ Split the input array into a hdf5 file.

        Arguments:
        ----------
            nb_blocks: nb_blocks to extract from the original array
        """

        if os.path.isfile(out_filepath):
            os.remove(out_filepath)

        self.case = {
            'name': 'split_hdf5',
            'params': {
                'out_filepath': out_filepath,
                'nb_blocks': nb_blocks,
                'out_file': None

            }
        }        


    def split_npy(self, out_dirpath):
        """ Split the input array into a numpy stack.
        """

        self.case = {
            'name': 'split_npy',
            'params': {
                'out_dirpath': out_dirpath
            }
        }   


    def get(self):
        """ Get the case to compute from the configuration.
        """

        if not self.case:
            print('No case defined, nothing to do.')
            return 
        
        arr = get_dask_array_from_hdf5(self.array_filepath, '/data', to_da=True, logic_cs=self.chunks_shape)

        case = self.case 
        if case['name'] == 'sum':
            return sum_chunks_case(arr, case['params']['nb_chunks'], compute=False)
        elif case['name'] == 'split_hdf5':
            case['params']['out_file'] = h5py.File(case['params']['out_filepath'], 'w')
            return split_to_hdf5(arr, case['params']['out_file'], nb_blocks=case['params']['nb_blocks'])
        elif case['name'] == 'split_npy':
            da.to_npy_stack(case['params']['out_dirpath'], arr)