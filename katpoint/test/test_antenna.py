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

"""Tests for the antenna module."""
# pylint: disable-msg=C0103,W0212

import unittest
import time
import cPickle

import katpoint
import numpy as np

def assert_angles_almost_equal(x, y, decimal):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), decimal=decimal)

class TestAntennaConstruction(unittest.TestCase):
    """Test construction of antennas from strings and vice versa."""
    def setUp(self):
        self.valid_antennas = [
            'XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0',
            'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0',
            'FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, '
                  '-0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16'
            ]
        self.invalid_antennas = [
            'XDM, -25:53:23.05075, 27:41:03.0',
            '',
            # Delay model can now have any number of terms (not 3 minimum)
#           'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7',
#           'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4, -8.7, 0.0'
            ]
        self.timestamp = '2009/07/07 08:36:20'

    def test_construct_antenna(self):
        """Test construction of antennas from strings and vice versa."""
        valid_antennas = [katpoint.Antenna(descr) for descr in self.valid_antennas]
        valid_strings = [a.description for a in valid_antennas]
        for descr in valid_strings:
            ant = katpoint.Antenna(descr)
            print str(ant), repr(ant)
            self.assertEqual(descr, ant.description, 'Antenna description differs from original string')
            self.assertEqual(ant.description, ant.format_katcp(), 'Antenna description differs from KATCP format')
        for descr in self.invalid_antennas:
            self.assertRaises(ValueError, katpoint.Antenna, descr)
        descr = valid_antennas[0].description
        self.assertEqual(descr, katpoint.Antenna(*descr.split(', ')).description)
        self.assertRaises(ValueError, katpoint.Antenna, descr, *descr.split(', ')[1:])
        # Check that description string updates when object is updated
        a1 = katpoint.Antenna('FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0')
        a2 = katpoint.Antenna('FF2, -30:43:17.3, 21:24:38.5, 1038.0, 13.0, 18.4 -8.7 0.0, 0.1, 1.22')
        self.assertNotEqual(a1, a2, 'Antennas should be inequal')
        a1.name = 'FF2'
        a1.diameter = 13.0
        a1.pointing_model = katpoint.PointingModel('0.1')
        a1.beamwidth = 1.22
        self.assertEqual(a1.description, a2.description, 'Antenna description string not updated')
        self.assertEqual(a1, a2.description, 'Antenna not equal to description string')
        self.assertEqual(a1, a2, 'Antennas not equal')
        self.assertEqual(a1, katpoint.Antenna(a2), 'Construction with antenna object failed')
        self.assertEqual(a1, cPickle.loads(cPickle.dumps(a1)), 'Pickling failed')
        try:
            self.assertEqual(hash(a1), hash(a2), 'Antenna hashes not equal')
        except TypeError:
            self.fail('Antenna object not hashable')

    def test_local_sidereal_time(self):
        """Test sidereal time and the use of date/time strings vs floats as timestamps."""
        ant = katpoint.Antenna(self.valid_antennas[0])
        utc_secs = time.mktime(time.strptime(self.timestamp, '%Y/%m/%d %H:%M:%S')) - time.timezone
        sid1 = ant.local_sidereal_time(self.timestamp)
        sid2 = ant.local_sidereal_time(utc_secs)
        self.assertAlmostEqual(sid1, sid2, places=10, msg='Sidereal time differs for float and date/time string')
        sid3 = ant.local_sidereal_time([self.timestamp, self.timestamp])
        sid4 = ant.local_sidereal_time([utc_secs, utc_secs])
        assert_angles_almost_equal(sid3, sid4, decimal=12)
