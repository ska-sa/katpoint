"""Unit test suite for katpoint."""

import logging
import sys
import unittest
# pylint: disable-msg=W0403
import test_target
import test_antenna
import test_catalogue
import test_projection
import test_ephem_extra
import test_conversion

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
    testsuite.addTests(loader.loadTestsFromModule(test_ephem_extra))
    testsuite.addTests(loader.loadTestsFromModule(test_conversion))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
