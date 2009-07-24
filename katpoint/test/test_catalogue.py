"""Tests for the catalogue module."""
# pylint: disable-msg=C0103,W0212

import unittest
import time

from katpoint import catalogue, target, antenna

class TestCatalogueConstruction(unittest.TestCase):
    """Test construction of catalogues."""
    def setUp(self):
        self.tle_lines = ['GPS BIIA-21 (PRN 09)    \n',
                          '1 22700U 93042A   07266.32333151  .00000012  00000-0  10000-3 0  8054\n',
                          '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n']

    def test_construct_catalogue(self):
        """Test construction of catalogues."""
        cat = catalogue.Catalogue(add_specials=True, add_stars=True)
        cat.add(target.construct_target('Sun, special'))
        num_targets = len(cat.targets)
        self.assertEqual(num_targets, len(catalogue.specials) + 1 + 94, 'Number of targets incorrect')
        test_target = cat.targets[0]
        self.assertEqual(test_target.description, cat[test_target.name].description, 'Lookup failed')
        self.assertEqual(cat['Non-existent'], None, 'Lookup of non-existent target failed')
        cat.add_tle(self.tle_lines, 'tle')
        self.assertEqual(len(cat.targets), num_targets + 1, 'Number of targets incorrect')
        cat.remove(cat.targets[-1].name)
        self.assertEqual(len(cat.targets), num_targets, 'Number of targets incorrect')

class TestCatalogueFilterSort(unittest.TestCase):
    """Test filtering and sorting of catalogues."""
    def setUp(self):
        self.flux_target = target.construct_target('radec, 0.0, 0.0, (1.0 2.0 2.0 0.0 0.0)')
        self.antenna = 'XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0'
        self.timestamp = time.mktime(time.strptime('2009/06/14 12:34:56', '%Y/%m/%d %H:%M:%S'))

    def test_filter_catalogue(self):
        """Test filtering of catalogues."""
        cat = catalogue.Catalogue(add_specials=True, add_stars=True)
        cat = cat.filter(tags=['special', '~radec'])
        self.assertEqual(len(cat.targets), len(catalogue.specials), 'Number of targets incorrect')
        cat.add(self.flux_target)
        cat2 = cat.filter(flux_limit_Jy=50.0, flux_freq_MHz=1.5)
        self.assertEqual(len(cat2.targets), 1, 'Number of targets with sufficient flux should be 1')
        cat.add(target.construct_target('Zenith, azel, 0, 90'))
        ant = antenna.construct_antenna(self.antenna)
        cat3 = cat.filter(el_limit_deg=85, timestamp=self.timestamp, antenna=ant)
        self.assertEqual(len(cat3.targets), 1, 'Number of targets close to zenith should be 1')
        sun = target.construct_target('Sun, special')
        cat4 = cat.filter(dist_limit_deg=[0.0, 1.0], proximity_targets=sun, timestamp=self.timestamp, antenna=ant)
        self.assertEqual(len(cat4.targets), 1, 'Number of targets close to Sun should be 1')

    def test_sort_catalogue(self):
        """Test sorting of catalogues."""
        cat = catalogue.Catalogue(add_specials=True, add_stars=True)
        self.assertEqual(len(cat.targets), len(catalogue.specials) + 1 + 94, 'Number of targets incorrect')
        cat1 = cat.sort(key='name')
        self.assertEqual(cat1.targets[0].name, 'Achernar', 'Sorting on name failed')
        ant = antenna.construct_antenna(self.antenna)
        cat2 = cat.sort(key='ra', timestamp=self.timestamp, antenna=ant)
        self.assertEqual(str(cat2.targets[0].body.ra), '0:08:53.09', 'Sorting on ra failed')
        cat3 = cat.sort(key='dec', timestamp=self.timestamp, antenna=ant)
        self.assertEqual(str(cat3.targets[0].body.dec), '-60:25:27.3', 'Sorting on dec failed')
        cat4 = cat.sort(key='az', timestamp=self.timestamp, antenna=ant, ascending=False)
        self.assertEqual(str(cat4.targets[0].body.az), '359:25:07.3', 'Sorting on az failed')
        cat5 = cat.sort(key='el', timestamp=self.timestamp, antenna=ant)
        self.assertEqual(str(cat5.targets[0].body.alt), '-76:13:14.2', 'Sorting on el failed')
        cat.add(self.flux_target)
        cat6 = cat.sort(key='flux', ascending=False, flux_freq_MHz=1.5)
        self.assertEqual(cat6.targets[0].flux_density(1.5), 100.0, 'Sorting on flux failed')

    def test_visibility_list(self):
        """Test output of visibility list."""
        cat = catalogue.Catalogue(add_specials=True, add_stars=True)
        cat.add(self.flux_target)
        ant = antenna.construct_antenna(self.antenna)
        cat.visibility_list(timestamp=self.timestamp, antenna=ant, flux_freq_MHz=1.5)
