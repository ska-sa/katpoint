"""Unit test suite for katpoint."""

import unittest
# pylint: disable-msg=W0403
import test_target
import test_projection

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_target))
    testsuite.addTests(loader.loadTestsFromModule(test_projection))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
