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

"""Tests for the catalogue module."""
# pylint: disable-msg=C0103,W0212

import unittest
import time

import katpoint


# Use the current year in TLE epochs to avoid pyephem crash due to expired TLEs
YY = time.localtime().tm_year % 100

class TestCatalogueConstruction(unittest.TestCase):
    """Test construction of catalogues."""
    def setUp(self):
        self.tle_lines = ['GPS BIIA-21 (PRN 09)    \n',
                          '1 22700U 93042A   %02d266.32333151  .00000012  00000-0  10000-3 0  805%1d\n' %
                           (YY, (YY // 10 + YY - 7 + 4) % 10),
                          '2 22700  55.4408  61.3790 0191986  78.1802 283.9935  2.00561720104282\n']
        self.edb_lines = ['HIC 13847,f|S|A4,2:58:16.03,-40:18:17.1,2.906,2000,\n']
        self.antenna = katpoint.Antenna('XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0')

    def test_construct_catalogue(self):
        """Test construction of catalogues."""
        cat = katpoint.Catalogue(add_specials=True, add_stars=True, antenna=self.antenna)
        cat.add(katpoint.Target('Sun, special'))
        num_targets = len(cat)
        self.assertEqual(num_targets, len(katpoint.specials) + 1 + 94, 'Number of targets incorrect')
        cat2 = katpoint.Catalogue(add_specials=True, add_stars=True)
        cat2.add(katpoint.Target('Sun, special'))
        self.assertEqual(cat, cat2, 'Catalogues not equal')
        try:
            self.assertEqual(hash(cat), hash(cat2), 'Catalogue hashes not equal')
        except TypeError:
            self.fail('Catalogue object not hashable')
        test_target = cat.targets[0]
        self.assertEqual(test_target.description, cat[test_target.name].description, 'Lookup failed')
        self.assertEqual(cat['Non-existent'], None, 'Lookup of non-existent target failed')
        cat.add_tle(self.tle_lines, 'tle')
        cat.add_edb(self.edb_lines, 'edb')
        self.assertEqual(len(cat.targets), num_targets + 2, 'Number of targets incorrect')
        cat.remove(cat.targets[-1].name)
        self.assertEqual(len(cat.targets), num_targets + 1, 'Number of targets incorrect')
        closest_target, dist = cat.closest_to(test_target)
        self.assertEqual(closest_target.description, test_target.description, 'Closest target incorrect')
# Reinstate this test once separation() can handle angles on top of each other (currently produces NaNs)
#        self.assertAlmostEqual(dist, 0.0, places=5, msg='Target should be on top of itself')

class TestCatalogueFilterSort(unittest.TestCase):
    """Test filtering and sorting of catalogues."""
    def setUp(self):
        self.flux_target = katpoint.Target('flux, radec, 0.0, 0.0, (1.0 2.0 2.0 0.0 0.0)')
        self.antenna = katpoint.Antenna('XDM, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0')
        self.antenna2 = katpoint.Antenna('XDM2, -25:53:23.05075, 27:41:03.36453, 1406.1086, 15.0, 100.0 0.0 0.0')
        self.timestamp = time.mktime(time.strptime('2009/06/14 12:34:56', '%Y/%m/%d %H:%M:%S'))

    def test_filter_catalogue(self):
        """Test filtering of catalogues."""
        cat = katpoint.Catalogue(add_specials=True, add_stars=True)
        cat = cat.filter(tags=['special', '~radec'])
        self.assertEqual(len(cat.targets), len(katpoint.specials), 'Number of targets incorrect')
        cat.add(self.flux_target)
        cat2 = cat.filter(flux_limit_Jy=50.0, flux_freq_MHz=1.5)
        self.assertEqual(len(cat2.targets), 1, 'Number of targets with sufficient flux should be 1')
        self.assertNotEqual(cat, cat2, 'Catalogues should be inequal')
        cat.add(katpoint.Target('Zenith, azel, 0, 90'))
        cat3 = cat.filter(az_limit_deg=[0, 180], timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(len(cat3.targets), 2, 'Number of targets rising should be 2')
        cat4 = cat.filter(az_limit_deg=[180, 0], timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(len(cat4.targets), 10, 'Number of targets setting should be 10')
        cat5 = cat.filter(el_limit_deg=85, timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(len(cat5.targets), 1, 'Number of targets close to zenith should be 1')
        sun = katpoint.Target('Sun, special')
        cat6 = cat.filter(dist_limit_deg=[0.0, 1.0], proximity_targets=sun,
                          timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(len(cat6.targets), 1, 'Number of targets close to Sun should be 1')

    def test_sort_catalogue(self):
        """Test sorting of catalogues."""
        cat = katpoint.Catalogue(add_specials=True, add_stars=True)
        self.assertEqual(len(cat.targets), len(katpoint.specials) + 1 + 94, 'Number of targets incorrect')
        cat1 = cat.sort(key='name')
        self.assertEqual(cat1, cat, 'Catalogue equality failed')
        self.assertEqual(cat1.targets[0].name, 'Achernar', 'Sorting on name failed')
        cat2 = cat.sort(key='ra', timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(cat2.targets[0].name, 'Sirrah', 'Sorting on ra failed') # RA: 0:08:53.09
        cat3 = cat.sort(key='dec', timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(cat3.targets[0].name, 'Agena', 'Sorting on dec failed') # DEC: -60:25:27.3
        cat4 = cat.sort(key='az', timestamp=self.timestamp, antenna=self.antenna, ascending=False)
        self.assertEqual(cat4.targets[0].name, 'Polaris', 'Sorting on az failed') # az: 359:25:07.3
        cat5 = cat.sort(key='el', timestamp=self.timestamp, antenna=self.antenna)
        self.assertEqual(cat5.targets[-1].name, 'Zenith', 'Sorting on el failed') # el: 90:00:00.0
        cat.add(self.flux_target)
        cat6 = cat.sort(key='flux', ascending=False, flux_freq_MHz=1.5)
        self.assertTrue('flux' in (cat6.targets[0].name, cat6.targets[-1].name),
                        'Flux target should be at start or end of catalogue after sorting')
        self.assertTrue((cat6.targets[0].flux_density(1.5) == 100.0) or
                        (cat6.targets[-1].flux_density(1.5) == 100.0), 'Sorting on flux failed')

    def test_visibility_list(self):
        """Test output of visibility list."""
        cat = katpoint.Catalogue(add_specials=True, add_stars=True)
        cat.add(self.flux_target)
        cat.remove('Zenith')
        cat.visibility_list(timestamp=self.timestamp, antenna=self.antenna, flux_freq_MHz=1.5, antenna2=self.antenna2)
        cat.antenna = self.antenna
        cat.flux_freq_MHz = 1.5
        cat.visibility_list(timestamp=self.timestamp)

    def test_completer(self):
        """Test IPython tab completer."""
        # pylint: disable-msg=W0201,W0612,R0903
        cat = katpoint.Catalogue(add_stars=True)
        # Set up dummy object containing user namespace and line to be completed
        class Dummy(object):
            pass
        event = Dummy()
        event.shell = Dummy()
        event.shell.user_ns = locals()
        event.line = "t = cat['Rasal"
        names = katpoint._catalogue_completer(event, event)
        self.assertEqual(names, ['Rasalgethi', 'Rasalhague'], 'Tab completer failed')
