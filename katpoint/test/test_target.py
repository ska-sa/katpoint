"""Tests for the target module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

import katpoint

class TestTargetConstruction(unittest.TestCase):
    """Test construction of targets from strings and vice versa."""
    def setUp(self):
        self.valid_targets = ['azel, -30.0, 90.0',
                              ', azel, 180, -45:00:00.0',
                              'Zenith, azel, 0, 90',
                              'radec J2000, 0, 0.0, (1000.0 2000.0 1.0 10.0)',
                              ', radec B1950, 14:23:45.6, -60:34:21.1',
                              'radec B1900, 14:23:45.6, -60:34:21.1',
                              'gal, 300.0, 0.0',
                              'Sag A, gal, 0.0, 0.0',
                              'Zizou, radec cal, 1.4, 30.0, (1000.0 2000.0 1.0 10.0)',
                              'Fluffy | *Dinky, radec, 12.5, -50.0, (1.0 2.0 1.0 2.0 3.0 4.0)',
                              'tle, GPS BIIA-21 (PRN 09)    \n' +
                              '1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054\n' +
                              '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n',
                              ', tle, GPS BIIA-22 (PRN 05)    \n' +
                              '1 22779U 93054A   07266.92814765  .00000062  00000-0  10000-3 0  2895\n' +
                              '2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055\n',
                              'Sun, special',
                              'Nothing, special',
                              'Moon | Luna, special solarbody',
                              'Aldebaran, star',
                              'Betelgeuse | Maitland, star orion',
                              'xephem star, Sadr~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0',
                              'Acamar | Theta Eridani, xephem, HIC 13847~f|S|A4~2:58:16.03~-40:18:17.1~2.906~2000~0',
                              'Kakkab | Alpha Lupi, xephem, HIC 71860 | SAO 225128~f|S|B1~14:41:55.768~-47:23:17.51~2.304~2000~0']
        self.invalid_targets = ['Sun',
                                'Sun, ',
                                '-30.0, 90.0',
                                ', azel, -45:00:00.0',
                                'Zenith, azel blah',
                                'radec J2000, 0.3',
                                'gal, 0.0',
                                'Zizou, radec cal, 1.4, 30.0, (1000.0, 2000.0, 1.0, 10.0)',
                                'tle, GPS BIIA-21 (PRN 09)    \n' +
                                '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n',
                                ', tle, GPS BIIA-22 (PRN 05)    \n' +
                                '1 93054A   07266.92814765  .00000062  00000-0  10000-3 0  2895\n' +
                                '2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055\n',
                                'Sunny, special',
                                'Slinky, star',
                                'xephem star, Sadr~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0',
                                'hotbody, 34.0, 45.0']
        self.azel_target = 'azel, 10.0, -10.0'
        self.radec_target = 'radec, 20.0, -20.0'
        self.gal_target = 'gal, 30.0, -30.0'
        self.tag_target = 'azel J2000 GPS, 40.0, -30.0'

    def test_construct_target(self):
        """Test construction of targets from strings and vice versa."""
        valid_targets = [katpoint.Target(descr) for descr in self.valid_targets]
        valid_strings = [t.description for t in valid_targets]
        for descr in valid_strings:
            t = katpoint.Target(descr)
            self.assertEqual(descr, t.description, "Target description ('%s') differs from original string ('%s')" %
                             (t.description, descr))
            print repr(t), t
        for descr in self.invalid_targets:
            self.assertRaises(ValueError, katpoint.Target, descr)
        azel1 = katpoint.Target(self.azel_target)
        azel2 = katpoint.construct_azel_target('10:00:00.0', '-10:00:00.0')
        self.assertEqual(azel1, azel2, 'Special azel constructor failed')
        radec1 = katpoint.Target(self.radec_target)
        radec2 = katpoint.construct_radec_target('20:00:00.0', '-20:00:00.0')
        self.assertEqual(radec1, radec2, 'Special radec constructor failed')
        # Check that description string updates when object is updated
        t1 = katpoint.Target('piet, azel, 20, 30')
        t2 = katpoint.Target('piet | bollie, azel, 20, 30')
        self.assertNotEqual(t1, t2, 'Targets should not be equal')
        t1.aliases += ['bollie']
        self.assertEqual(t1.description, t2.description, 'Target description string not updated')
        self.assertEqual(t1, t2.description, 'Equality with description string failed')
        self.assertEqual(t1, t2, 'Equality with target failed')
        self.assertEqual(t1, katpoint.Target(t2), 'Construction with target object failed')

    def test_constructed_coords(self):
        """Test whether calculated coordinates match those with which it is constructed."""
        azel = katpoint.Target(self.azel_target)
        calc_azel = azel.azel()
        calc_az, calc_el = katpoint.rad2deg(calc_azel[0]), katpoint.rad2deg(calc_azel[1])
        self.assertEqual(calc_az, 10.0, 'Calculated az does not match specified value in azel target')
        self.assertEqual(calc_el, -10.0, 'Calculated el does not match specified value in azel target')
        radec = katpoint.Target(self.radec_target)
        calc_radec = radec.radec()
        calc_ra, calc_dec = katpoint.rad2deg(calc_radec[0]), katpoint.rad2deg(calc_radec[1])
        # You would think that these could be made exactly equal, but the following assignment is inexact:
        # body = ephem.FixedBody()
        # body._ra = ra
        # Then body._ra != ra... Possibly due to double vs float? This problem goes all the way to libastro.
        np.testing.assert_almost_equal(calc_ra, 20.0 * 360 / 24., decimal=4)
        np.testing.assert_almost_equal(calc_dec, -20.0, decimal=4)
        lb = katpoint.Target(self.gal_target)
        calc_lb = lb.galactic()
        calc_l, calc_b = katpoint.rad2deg(calc_lb[0]), katpoint.rad2deg(calc_lb[1])
        np.testing.assert_almost_equal(calc_l, 30.0, decimal=4)
        np.testing.assert_almost_equal(calc_b, -30.0, decimal=4)

    def test_add_tags(self):
        """Test adding tags."""
        tag_target = katpoint.Target(self.tag_target)
        tag_target.add_tags(None)
        tag_target.add_tags('pulsar')
        tag_target.add_tags(['SNR', 'GPS'])
        self.assertEqual(tag_target.tags, ['azel', 'J2000', 'GPS', 'pulsar', 'SNR'], 'Added tags not correct')

class TestTargetCalculations(unittest.TestCase):
    """Test various calculations involving antennas and timestamps."""
    def setUp(self):
        self.target = katpoint.construct_azel_target('45:00:00.0', '75:00:00.0')
        self.ant1 = katpoint.Antenna('A1, -31.0, 18.0, 0.0, 12.0, 0.0 0.0 0.0')
        self.ant2 = katpoint.Antenna('A2, -31.0, 18.0, 0.0, 12.0, 10.0 -10.0 0.0')
        self.ts = katpoint.Timestamp('2013-08-14 08:25')

    def test_coords(self):
        """Test coordinate conversions for coverage."""
        self.target.azel(self.ts, self.ant1)
        self.target.apparent_radec(self.ts, self.ant1)
        self.target.astrometric_radec(self.ts, self.ant1)
        self.target.galactic(self.ts, self.ant1)
        self.target.parallactic_angle(self.ts, self.ant1)

    def test_delay(self):
        """Test geometric delay."""
        delay, delay_rate = self.target.geometric_delay(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(delay, 0.0, decimal=12)
        np.testing.assert_almost_equal(delay_rate, 0.0, decimal=12)
        delay, delay_rate = self.target.geometric_delay(self.ant2, [self.ts, self.ts], self.ant1)
        np.testing.assert_almost_equal(delay, np.array([0.0, 0.0]), decimal=12)
        np.testing.assert_almost_equal(delay_rate, np.array([0.0, 0.0]), decimal=12)

    def test_uvw(self):
        """Test uvw calculation."""
        u, v, w = self.target.uvw(self.ant2, self.ts, self.ant1)
        np.testing.assert_almost_equal(u, 10.821750916197391, decimal=5)
        np.testing.assert_almost_equal(v, -9.1043784587765906, decimal=5)
        np.testing.assert_almost_equal(w, 4.7781625336985198e-10, decimal=5)

    def test_projection(self):
        """Test projection."""
        az, el = katpoint.deg2rad(50.0), katpoint.deg2rad(80.0)
        x, y = self.target.sphere_to_plane(az, el, self.ts, self.ant1)
        re_az, re_el = self.target.plane_to_sphere(x, y, self.ts, self.ant1)
        np.testing.assert_almost_equal(re_az, az, decimal=12)
        np.testing.assert_almost_equal(re_el, el, decimal=12)
