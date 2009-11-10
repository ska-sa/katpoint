"""Tests for the ephem_extra module."""
# pylint: disable-msg=C0103,W0212

import unittest
import numpy as np

import ephem

import katpoint

def assert_angles_almost_equal(x, y, decimal):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), decimal=decimal)

class TestTimestamp(unittest.TestCase):
    """Test timestamp creation and conversion."""
    def setUp(self):
        self.valid_timestamps = [(1248186982.3980861, '2009-07-21 14:36:22.398'),
                                 (ephem.Date('2009/07/21 02:52:12.34'), '2009-07-21 02:52:12.340'),
                                 (0, '1970-01-01 00:00:00'),
                                 (-10, '1969-12-31 23:59:50'),
                                 ('2009-07-21 02:52:12.034', '2009-07-21 02:52:12.034'),
                                 ('2009-07-21 02:52:12.000', '2009-07-21 02:52:12'),
                                 ('2009-07-21 02:52:12', '2009-07-21 02:52:12'),
                                 ('2009-07-21 02:52', '2009-07-21 02:52:00'),
                                 ('2009-07-21 02', '2009-07-21 02:00:00'),
                                 ('2009-07-21', '2009-07-21 00:00:00'),
                                 ('2009-07', '2009-07-01 00:00:00'),
                                 ('2009', '2009-01-01 00:00:00'),
                                 ('2009/07/21 02:52:12.034', '2009-07-21 02:52:12.034'),
                                 ('2009/07/21 02:52:12.000', '2009-07-21 02:52:12'),
                                 ('2009/07/21 02:52:12', '2009-07-21 02:52:12'),
                                 ('2009/07/21 02:52', '2009-07-21 02:52:00'),
                                 ('2009/07/21 02', '2009-07-21 02:00:00'),
                                 ('2009/07/21', '2009-07-21 00:00:00'),
                                 ('2009/07', '2009-07-01 00:00:00'),
                                 ('2009', '2009-01-01 00:00:00'),
                                 ('2019-07-21 02:52:12', '2019-07-21 02:52:12')]
        self.invalid_timestamps = ['gielie', '03 Mar 2003']
        self.overflow_timestamps = ['2049-07-21 02:52:12']

    def test_construct_timestamp(self):
        """Test construction of timestamps."""
        for v, s in self.valid_timestamps:
            t = katpoint.Timestamp(v)
            self.assertEqual(str(t), s, "Timestamp string ('%s') differs from expected one ('%s')" % (str(t), s))
        for v in self.invalid_timestamps:
            self.assertRaises(ValueError, katpoint.Timestamp, v)
#        for v in self.overflow_timestamps:
#            self.assertRaises(OverflowError, katpoint.Timestamp, v)

    def test_numerical_timestamp(self):
        """Test numerical properties of timestamps."""
        t = katpoint.Timestamp(self.valid_timestamps[0][0])
        self.assertEqual(t, t + 0.0)
        self.assertNotEqual(t, t + 1.0)
        self.assertTrue(t > t - 1.0)
        self.assertTrue(t < t + 1.0)
        self.assertEqual(t, eval('katpoint.' + repr(t)))
        self.assertEqual(float(t), self.valid_timestamps[0][0])
        t = katpoint.Timestamp(self.valid_timestamps[1][0])
        self.assertAlmostEqual(t.to_ephem_date(), self.valid_timestamps[1][0], places=9)

class TestGeodetic(unittest.TestCase):
    """Closure tests for geodetic coordinate transformations."""
    def setUp(self):
        N = 1000
        self.lat = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        self.long = 2.0 * np.pi * np.random.rand(N)
        self.alt = 1000.0 * np.random.randn(N)

    def test_lla_to_ecef(self):
        """Closure tests for LLA to ECEF conversion and vice versa."""
        x, y, z = katpoint.lla_to_ecef(self.lat, self.long, self.alt)
        new_lat, new_long, new_alt = katpoint.ecef_to_lla(x, y, z)
        new_x, new_y, new_z = katpoint.lla_to_ecef(new_lat, new_long, new_alt)
        assert_angles_almost_equal(new_lat, self.lat, decimal=12)
        assert_angles_almost_equal(new_long, self.long, decimal=12)
        assert_angles_almost_equal(new_alt, self.alt, decimal=6)
        np.testing.assert_almost_equal(new_x, x, decimal=8)
        np.testing.assert_almost_equal(new_y, y, decimal=8)
        np.testing.assert_almost_equal(new_z, z, decimal=6)

    def test_ecef_to_enu(self):
        """Closure tests for ECEF to ENU conversion and vice versa."""
        x, y, z = katpoint.lla_to_ecef(self.lat, self.long, self.alt)
        e, n, u = katpoint.ecef_to_enu(self.lat[0], self.long[0], self.alt[0], x, y, z)
        new_x, new_y, new_z = katpoint.enu_to_ecef(self.lat[0], self.long[0], self.alt[0], e, n, u)
        np.testing.assert_almost_equal(new_x, x, decimal=8)
        np.testing.assert_almost_equal(new_y, y, decimal=8)
        np.testing.assert_almost_equal(new_z, z, decimal=8)
