import os, h5py 
import dask.array as da
from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.cases.case_creation import sum_chunks_case, split_to_hdf5

import logging
logger = logging.getLogger(__name__)

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
        self.case = {
            'name': 'split_hdf5',
            'params': {
                'out_filepath': out_filepath,
                'nb_blocks': nb_blocks,
                'out_file': None # h5py File object created from out_filepath

            }
        }        


    def split_hdf5_multiple(self, out_dirpath, nb_blocks=None):
        """ Split input array into several hdf5 files.
        Store one block per output file.
        Each hdf5 file is called according 
        to the index of the block in the reconstructed image.
        """
        self.case = {
            'name': 'split_hdf5_multiple',
            'params': {
                'out_dirpath': out_dirpath,
                'nb_blocks': nb_blocks,
                'out_files': list()
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
        arr = get_dask_array_from_hdf5(self.array_filepath, '/data', logic_cs=self.chunks_shape)
        logger.info('CS of array loaded: %s', arr.chunks)

        if self.case == None:
            logging.warning('No case defined, nothing to do.')
            return arr

        case = self.case 
        if case['name'] == 'sum':
            return sum_chunks_case(arr, case['params']['nb_chunks'], compute=False)
        elif case['name'] == 'split_hdf5':
            if os.path.isfile(case['params']['out_filepath']):
                os.remove(case['params']['out_filepath'])
            case['params']['out_file'] = h5py.File(case['params']['out_filepath'], 'w')
            return split_to_hdf5(arr, case['params']['out_file'], case['params']['nb_blocks'])
        elif case['name'] == 'split_npy':
            da.to_npy_stack(case['params']['out_dirpath'], arr)
        elif case['name'] == 'split_hdf5_multiple':
            return split_hdf5_multiple(arr, case['params']['out_files'], case['params']['nb_blocks'])

    
    def clean(self):
        name = self.case['name']
        if name and name != 'sum':
            try:
                f = self.case['params']['out_file']
                if f:
                    f.close()
            except:
                pass
