import pytest
import os
import tempfile

import logging
from logging.config import fileConfig
import datetime 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

date_info = datetime.datetime.now().isoformat()
logfilename = '/tmp/dask_io_test' + date_info + '.log'
fh = logging.FileHandler(logfilename)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s -12s %(levelname)s -8s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


@pytest.fixture(scope="session", autouse=True)
def temporary_directory(): 
    """ Create a temporary directory to run all the tests into.
    """
    logger.info("Log file at: %s", logfilename)

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield
        os.chdir(old_cwd)