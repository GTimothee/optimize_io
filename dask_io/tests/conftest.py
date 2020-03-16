import pytest
import os
import tempfile

import logging
from logging.config import fileConfig
import datetime 
date_info = datetime.datetime.now().isoformat()
current_dir = os.path.dirname(os.path.abspath(__file__))
logfilename = '/tmp/dask_io_test' + date_info + '.log'
fileConfig(os.path.join(current_dir, 'logging_config.ini'), defaults={'logfilename': logfilename})
logger = logging.getLogger(__name__)

print("[dask_io] Log file at: ", logfilename)

@pytest.fixture(scope="session", autouse=True)
def temporary_directory(): 
    """ Create a temporary directory to run all the tests into.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield
        os.chdir(old_cwd)