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

"""Tests for the refraction module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

import katpoint

def assert_angles_almost_equal(x, y, **kwargs):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), **kwargs)


class TestRefractionCorrection(unittest.TestCase):
    """Test refraction correction."""
    def setUp(self):
        self.rc = katpoint.RefractionCorrection()
        self.el = katpoint.deg2rad(np.arange(0.0, 90.1, 0.1))

    def test_refraction_basic(self):
        """Test basic refraction correction properties."""
        print repr(self.rc)
        self.assertRaises(ValueError, katpoint.RefractionCorrection, 'unknown')
        rc2 = katpoint.RefractionCorrection()
        self.assertEqual(self.rc, rc2, 'Refraction models should be equal')
        try:
            self.assertEqual(hash(self.rc), hash(rc2), 'Refraction model hashes should be equal')
        except TypeError:
            self.fail('RefractionCorrection object not hashable')

    def test_refraction_closure(self):
        """Test closure between refraction correction and its reverse operation."""
        # Generate random meteorological data (hopefully sensible) - first only a single weather measurement
        temp = -10. + 50. * np.random.rand()
        pressure = 900. + 200. * np.random.rand()
        humidity = 5. + 90. * np.random.rand()
        # Test closure on el grid
        refracted_el = self.rc.apply(self.el, temp, pressure, humidity)
        reversed_el = self.rc.reverse(refracted_el, temp, pressure, humidity)
        assert_angles_almost_equal(reversed_el, self.el, decimal=7,
                                   err_msg='Elevation closure error for temp=%f, pressure=%f, humidity=%f' %
                                           (temp, pressure, humidity))
        # Generate random meteorological data, now one weather measurement per elevation value
        temp = -10. + 50. * np.random.rand(len(self.el))
        pressure = 900. + 200. * np.random.rand(len(self.el))
        humidity = 5. + 90. * np.random.rand(len(self.el))
        # Test closure on el grid
        refracted_el = self.rc.apply(self.el, temp, pressure, humidity)
        reversed_el = self.rc.reverse(refracted_el, temp, pressure, humidity)
        assert_angles_almost_equal(reversed_el, self.el, decimal=7,
                                   err_msg='Elevation closure error for temp=%s, pressure=%s, humidity=%s' %
                                           (temp, pressure, humidity))
