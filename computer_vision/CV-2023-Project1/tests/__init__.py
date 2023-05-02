import logging
import os
import sys

from . import constants

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if not os.path.exists(constants.DATA_DIRECTORY):
    os.mkdir(constants.DATA_DIRECTORY)

if not os.path.exists(constants.TEST_OUTPUTS_DIRECTORY):
    os.mkdir(constants.TEST_OUTPUTS_DIRECTORY)

sys.path.insert(0, os.path.abspath(os.path.join(constants.SRC_DIRECTORY)))
sys.path.insert(0, os.path.abspath(os.path.join(constants.TESTS_DIRECTORY)))
