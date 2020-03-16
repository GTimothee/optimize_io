import pytest
import os
import tempfile

import logging
from logging.config import fileConfig
from datetime import datetime 
date_info = datetime.datetime.now().isoformat()
fileConfig('logging_config.ini', defaults={'logfilename': os.path.join('/var/log/dask_io_test', date_info, '.log')})
logger = logging.getLogger(name)


@pytest.fixture(scope="session", autouse=True)
def temporary_directory(): 
    """ Create a temporary directory to run all the tests into.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield
        os.chdir(old_cwd)