import traceback, sys, h5py
import dask.array as da 
import os, logging

from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.cases.case_creation import get_arr_chunks
from dask_io.optimizer.utils.array_utils import get_arr_shapes

logger = logging.getLogger(__name__)

def check_split_output_hdf5(input_filepath, output_filepath, logic_cs, input_dset_key='/data', output_dset_keyprefix='/data'):
    """ Compare the real chunks to the splits to see if split process went well. 
    Load the input array, successively extract one chunk and compare it to the corresponding dataset in output file.
    By default we suppose that dataset keys in output file are of the form: prefix + id (integer).

    Arguments:
    ----------
        input_filepath: hdf5 file with (at least) 1 dataset containing the multidim array that have been splitted.
        output_filepath: hdf5 file containing the splitted input_file (1 dataset per split).
        logic_cs: logic chunk shape used for the split
        input_dset_key: dataset key where input array is stored.
        output_dset_keyprefix: prefix for the keys of datasets containing splits in output file.
    """

    def sanity_check(file_path):
        logger.info(f"\nChecking file integrity: {file_path}")
        if os.path.isfile(file_path):
            logger.info(f'File has been found.')
        try:
            apply_sanity_check(file_path)
            logger.info("Sanity check passed.")
            return True
        except:
            logger.info('-' * 60)
            traceback.print_exc(file=sys.stdout)
            logger.info('-' * 60)
            logger.info("Sanity check failed, aborting goodness of split checking.")
            return False

    def apply_sanity_check(file_path):
        """ Check if splitted file not empty.
        """
        with h5py.File(file_path, 'r') as f:
            logger.info("file object %s", f)
            logger.info("keys %s", list(f.keys()))
            assert len(list(f.keys())) != 0
        logger.info("Integrity check passed.")
        
    if not sanity_check(input_filepath) or not sanity_check(output_filepath):
        return False

    logger.info(f"\nChecking files data...")
    input_arr = get_dask_array_from_hdf5(input_filepath, input_dset_key, to_da=True, logic_cs=logic_cs)
    input_arr_list = get_arr_chunks(input_arr)
    nb_chunks = len(input_arr_list)
    with h5py.File(output_filepath, 'r') as split_file:
        for i, a in enumerate(input_arr_list):
            stored_a = da.from_array(split_file[output_dset_keyprefix + str(i)])
            logger.info("Stored split shape: %s ", stored_a.shape)
            stored_a.rechunk(chunks=logic_cs)
            logger.info("Split rechunked to: %s", stored_a.shape)
            logger.info("Original data chunk: %s", a.shape)
            logger.info("Testing all close...")
            test = da.allclose(stored_a, a)
            if test.compute():
                logger.info(f"Test {i+1}/{nb_chunks} passed.")
            else:
                logger.info(f"Test {i+1}/{nb_chunks} failed.")
    return True