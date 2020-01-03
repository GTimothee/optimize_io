import os


class CaseConfig():
    """ Contains the configuration for a test.
    """
    def __init__(self, array_filepath, chunks_shape):
        self.array_filepath = array_filepath
        self.chunks_shape = chunks_shape

    def optimization(self, opti, scheduler_opti, buffer_size):
        self.opti = opti 
        self.scheduler_opti = scheduler_opti
        self.buffer_size = buffer_size

    def sum_case(self, nb_chunks):
        self.test_case = 'sum'
        self.nb_chunks = nb_chunks

    def split_case(self, in_filepath, out_filepath, nb_blocks=None):
        """
        nb_blocks: nb_blocks to extract from the original array
        """
        self.test_case = 'split'
        self.in_filepath = in_filepath  # TODO: remove this we already have it as array_filepath
        self.nb_blocks = nb_blocks if nb_blocks else None
        if os.path.isfile(out_filepath):
            os.remove(out_filepath)
        self.out_filepath = out_filepath
        # print("split file path stored in config:", self.out_filepath)
        self.out_file = h5py.File(self.out_filepath, 'w')

    def write_output(self, writer, out_file_path, t):
        if self.test_case == 'sum':
            data = [
                self.opti, 
                self.scheduler_opti, 
                self.chunk_shape, 
                self.nb_chunks, 
                self.buffer_size, 
                t,
                out_file_path
            ]
        else:
            raise ValueError("Unsupported test case.")
        writer.writerow(data)