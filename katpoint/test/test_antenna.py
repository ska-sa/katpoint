"""Tests for the antenna module."""
# pylint: disable-msg=C0103,W0212

import unittest

from katpoint import antenna

class TestAntennaConstruction(unittest.TestCase):
    """Test construction of antennas from strings and vice versa."""
    def setUp(self):
        self.valid_antennas = ['XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0',
                               'FF1,  22:53:23.0, -60:41:03.3, 500.1, 10.0, 2300.0, 1000.0, 50.3']
        self.invalid_antennas = ['XDM, -25:53:23.05075, 1406.1086, 15.0',
                                 'FF1,  22:53:23.0, -60:41:03.3, 500.1, 10.0, 2300.0, 1000.0']
    
    def test_construct_antenna(self):
        """Test construction of antennas from strings and vice versa."""
        valid_antennas = [antenna.construct_antenna(descr) for descr in self.valid_antennas]
        valid_strings = [a.get_description() for a in valid_antennas]
        for descr in valid_strings:
            self.assertEqual(descr, antenna.construct_antenna(descr).get_description(),
                             'Antenna description differs from original string')
        for descr in self.invalid_antennas:
            self.assertRaises(ValueError, antenna.construct_antenna, descr)
