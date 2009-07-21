"""Unit test suite for katpoint."""

import unittest
# pylint: disable-msg=W0403
import test_target
import test_antenna
import test_catalogue
import test_projection
import test_ephem_extra

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_target))
    testsuite.addTests(loader.loadTestsFromModule(test_antenna))
    testsuite.addTests(loader.loadTestsFromModule(test_catalogue))
    testsuite.addTests(loader.loadTestsFromModule(test_projection))
    testsuite.addTests(loader.loadTestsFromModule(test_ephem_extra))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
