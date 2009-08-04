"""Tests for the antenna module."""
# pylint: disable-msg=C0103,W0212

import unittest
import time

import katpoint

class TestAntennaConstruction(unittest.TestCase):
    """Test construction of antennas from strings and vice versa."""
    def setUp(self):
        self.valid_antennas = ['XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0',
                               'FF1,  22:53:23.0, -60:41:03.3, 500.1, 10.0, 2300.0, 1000.0, 50.3']
        self.invalid_antennas = ['XDM, -25:53:23.05075, 1406.1086, 15.0',
                                 'FF1,  22:53:23.0, -60:41:03.3, 500.1, 10.0, 2300.0, 1000.0']
        self.timestamp = '2009/07/07 08:36:20'

    def test_construct_antenna(self):
        """Test construction of antennas from strings and vice versa."""
        valid_antennas = [katpoint.construct_antenna(descr) for descr in self.valid_antennas]
        valid_strings = [a.description for a in valid_antennas]
        for descr in valid_strings:
            self.assertEqual(descr, katpoint.construct_antenna(descr).description,
                             'Antenna description differs from original string')
        for descr in self.invalid_antennas:
            self.assertRaises(ValueError, katpoint.construct_antenna, descr)

    def test_sidereal_time(self):
        """Test sidereal time and the use of date/time strings vs floats as timestamps."""
        ant = katpoint.construct_antenna(self.valid_antennas[0])
        utc_secs = time.mktime(time.strptime(self.timestamp, '%Y/%m/%d %H:%M:%S')) - time.timezone
        sid1 = ant.sidereal_time(self.timestamp)
        sid2 = ant.sidereal_time(utc_secs)
        self.assertAlmostEqual(sid1, sid2, places=10, msg='Sidereal time differs for float and date/time string')
