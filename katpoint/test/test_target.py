"""Tests for the target module."""
# pylint: disable-msg=C0103,W0212

import unittest

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
                              'Zizou, radec cal, 1.4, 30.0, (1000.0 2000.0 1.0 10.0)',
                              'Fluffy | *Dinky, radec, 12.5, -50.0, (1.0 2.0 1.0 2.0 3.0 4.0)',
                              'tle, GPS BIIA-21 (PRN 09)    \n' +
                              '1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054\n' +
                              '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n',
                              ', tle, GPS BIIA-22 (PRN 05)    \n' +
                              '1 22779U 93054A   07266.92814765  .00000062  00000-0  10000-3 0  2895\n' +
                              '2 22779  53.8943 118.4708 0081407  68.2645 292.7207  2.00558015103055\n',
                              'Sun, special',
                              'Moon | Luna, special solarbody',
                              'Aldebaran, star',
                              'Betelgeuse | Maitland, star orion',
                              'xephem star, Sadr~f|S|F8~20:22:13.7|2.43~40:15:24|-0.93~2.23~2000~0',
                              'Acamar | Theta Eridani, xephem, HIC 13847~f|S|A4~2:58:16.03~-40:18:17.1~2.906~2000~',
                              'Kakkab | Alpha Lupi, xephem, HIC 71860 | SAO 225128~f|S|B1~14:41:55.768~-47:23:17.51~2.304~2000~']
        self.invalid_targets = ['Sun',
                                'Sun, ',
                                '-30.0, 90.0',
                                ', azel, -45:00:00.0',
                                'Zenith, azel blah',
                                'radec J2000, 0.3',
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
        self.radec_target = 'radec, 10.0, -10.0'
        self.tag_target = 'azel J2000 GPS, 40.0, -30.0'

    def test_construct_target(self):
        """Test construction of targets from strings and vice versa."""
        valid_targets = [katpoint.construct_target(descr) for descr in self.valid_targets]
        valid_strings = [t.description for t in valid_targets]
        for descr in valid_strings:
            self.assertEqual(descr, katpoint.construct_target(descr).description,
                             "Target description ('%s') differs from original string ('%s')" %
                             (katpoint.construct_target(descr).description, descr))
        for descr in self.invalid_targets:
            self.assertRaises(ValueError, katpoint.construct_target, descr)
        azel1 = katpoint.construct_target(self.azel_target)
        azel2 = katpoint.construct_azel_target('10:00:00.0', '-10:00:00.0')
        self.assertEqual(azel1.description, azel2.description, 'Special azel constructor failed')
        radec1 = katpoint.construct_target(self.radec_target)
        radec2 = katpoint.construct_radec_target('10:00:00.0', '-10:00:00.0')
        self.assertEqual(radec1.description, radec2.description, 'Special radec constructor failed')

    def test_add_tags(self):
        """Test adding tags."""
        tag_target = katpoint.construct_target(self.tag_target)
        tag_target.add_tags(None)
        tag_target.add_tags('pulsar')
        tag_target.add_tags(['SNR', 'GPS'])
        self.assertEqual(tag_target.tags, ['azel', 'J2000', 'GPS', 'pulsar', 'SNR'], 'Added tags not correct')

class TestFluxDensity(unittest.TestCase):
    """Test flux density calculation."""
    def setUp(self):
        self.flux_target = katpoint.construct_target('radec, 0.0, 0.0, (1.0 2.0 2.0 0.0 0.0)')

    def test_flux_density(self):
        """Test flux density calculation."""
        self.assertEqual(self.flux_target.flux_density(1.5), 100.0, 'Flux calculation wrong')
