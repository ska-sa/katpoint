################################################################################
# Copyright (c) 2009-2016, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Unit test suite for katpoint."""

import logging
import sys
import unittest

from katpoint.test import test_target
from katpoint.test import test_antenna
from katpoint.test import test_catalogue
from katpoint.test import test_projection
from katpoint.test import test_timestamp
from katpoint.test import test_flux
from katpoint.test import test_conversion
from katpoint.test import test_pointing
from katpoint.test import test_refraction
from katpoint.test import test_delay

# Enable verbose logging to stdout for katpoint module - see output via nosetests -s flag
logger = logging.getLogger("katpoint")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("LOG: %(name)s %(levelname)s %(message)s"))
logger.addHandler(ch)


def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_target))
    testsuite.addTests(loader.loadTestsFromModule(test_antenna))
    testsuite.addTests(loader.loadTestsFromModule(test_catalogue))
    testsuite.addTests(loader.loadTestsFromModule(test_projection))
    testsuite.addTests(loader.loadTestsFromModule(test_timestamp))
    testsuite.addTests(loader.loadTestsFromModule(test_flux))
    testsuite.addTests(loader.loadTestsFromModule(test_conversion))
    testsuite.addTests(loader.loadTestsFromModule(test_pointing))
    testsuite.addTests(loader.loadTestsFromModule(test_refraction))
    testsuite.addTests(loader.loadTestsFromModule(test_delay))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
