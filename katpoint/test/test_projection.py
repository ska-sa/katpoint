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

"""Tests for the projection module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

import katpoint

try:
    from .aips_projection import newpos, dircos
    found_aips = True
except ImportError:
    found_aips = False

def skip(reason=''):
    """Use nose to skip a test."""
    try:
        import nose
        raise nose.SkipTest(reason)
    except ImportError:
        pass


def assert_angles_almost_equal(x, y, decimal):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), decimal=decimal)


class TestProjectionSIN(unittest.TestCase):
    """Test orthographic projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['SIN']
        self.sphere_to_plane = katpoint.sphere_to_plane['SIN']
        N = 100
        max_theta = np.pi / 2.0
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        # (x, y) points within unit circle
        theta = max_theta * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        self.x = np.sin(theta) * np.cos(phi)
        self.y = np.sin(theta) * np.sin(phi)

    def test_random_closure(self):
        """SIN projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=10)
        np.testing.assert_almost_equal(self.y, yy, decimal=10)
        assert_angles_almost_equal(az, aa, decimal=10)
        assert_angles_almost_equal(el, ee, decimal=10)

    def test_aips_compatibility(self):
        """SIN projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
        x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
        for n in xrange(len(az)):
            az_aips[n], el_aips[n], ierr = \
            newpos(2, self.az0[n], self.el0[n], self.x[n], self.y[n])
            x_aips[n], y_aips[n], ierr = \
            dircos(2, self.az0[n], self.el0[n], az[n], el[n])
        self.assertEqual(ierr, 0)
        assert_angles_almost_equal(az, az_aips, decimal=9)
        assert_angles_almost_equal(el, el_aips, decimal=9)
        np.testing.assert_almost_equal(xx, x_aips, decimal=9)
        np.testing.assert_almost_equal(yy, y_aips, decimal=9)

    def test_corner_cases(self):
        """SIN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(self.sphere_to_plane(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi, 1e-8))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(self.plane_to_sphere(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 0.0, 2.0)


class TestProjectionTAN(unittest.TestCase):
    """Test gnomonic projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['TAN']
        self.sphere_to_plane = katpoint.sphere_to_plane['TAN']
        N = 100
        # Stay away from edge of hemisphere
        max_theta = np.pi / 2.0 - 0.01
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        theta = max_theta * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        # Perform inverse TAN mapping to spread out points on plane
        self.x = np.tan(theta) * np.cos(phi)
        self.y = np.tan(theta) * np.sin(phi)

    def test_random_closure(self):
        """TAN projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=8)
        np.testing.assert_almost_equal(self.y, yy, decimal=8)
        assert_angles_almost_equal(az, aa, decimal=8)
        assert_angles_almost_equal(el, ee, decimal=8)

    def test_aips_compatibility(self):
        """TAN projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        # AIPS TAN only deprojects (x, y) coordinates within unit circle
        r = self.x * self.x + self.y * self.y
        az0, el0 = self.az0[r <= 1.0], self.el0[r <= 1.0]
        x, y = self.x[r <= 1.0], self.y[r <= 1.0]
        az, el = self.plane_to_sphere(az0, el0, x, y)
        xx, yy = self.sphere_to_plane(az0, el0, az, el)
        az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
        x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
        for n in xrange(len(az)):
            az_aips[n], el_aips[n], ierr = \
            newpos(3, az0[n], el0[n], x[n], y[n])
            x_aips[n], y_aips[n], ierr = \
            dircos(3, az0[n], el0[n], az[n], el[n])
        self.assertEqual(ierr, 0)
        assert_angles_almost_equal(az, az_aips, decimal=10)
        assert_angles_almost_equal(el, el_aips, decimal=10)
        np.testing.assert_almost_equal(xx, x_aips, decimal=10)
        np.testing.assert_almost_equal(yy, y_aips, decimal=10)

    def test_corner_cases(self):
        """TAN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 45 degrees from reference point on sphere
        xy = np.array(self.sphere_to_plane(0.0, 0.0, np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, -np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, -np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 4.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 4.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 4.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(self.plane_to_sphere(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, -np.pi / 4.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, -np.pi / 4.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, -np.pi / 4.0], decimal=12)


class TestProjectionARC(unittest.TestCase):
    """Test zenithal equidistant projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['ARC']
        self.sphere_to_plane = katpoint.sphere_to_plane['ARC']
        N = 100
        # Stay away from edge of circle
        max_theta = np.pi - 0.01
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        # (x, y) points within circle of radius pi
        theta = max_theta * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        self.x = theta * np.cos(phi)
        self.y = theta * np.sin(phi)

    def test_random_closure(self):
        """ARC projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=8)
        np.testing.assert_almost_equal(self.y, yy, decimal=8)
        assert_angles_almost_equal(az, aa, decimal=8)
        assert_angles_almost_equal(el, ee, decimal=8)

    def test_aips_compatibility(self):
        """ARC projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
        x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
        for n in xrange(len(az)):
            az_aips[n], el_aips[n], ierr = \
            newpos(4, self.az0[n], self.el0[n], self.x[n], self.y[n])
            x_aips[n], y_aips[n], ierr = \
            dircos(4, self.az0[n], self.el0[n], az[n], el[n])
        self.assertEqual(ierr, 0)
        assert_angles_almost_equal(az, az_aips, decimal=8)
        assert_angles_almost_equal(el, el_aips, decimal=8)
        np.testing.assert_almost_equal(xx, x_aips, decimal=8)
        np.testing.assert_almost_equal(yy, y_aips, decimal=8)

    def test_corner_cases(self):
        """ARC projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(self.sphere_to_plane(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)
        # Point diametrically opposite the reference point on sphere
        xy = np.array(self.sphere_to_plane(np.pi, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(np.abs(xy), [np.pi, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [1.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-1.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 1.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -1.0], decimal=12)
        # Points on circle with radius pi in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, np.pi, 0.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -np.pi, 0.0))
        assert_angles_almost_equal(ae, [-np.pi, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(self.plane_to_sphere(0.0, -np.pi / 2.0, np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, -np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, np.pi / 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, -np.pi / 2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 4.0, 0.0)
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 0.0, 4.0)


class TestProjectionSTG(unittest.TestCase):
    """Test stereographic projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['STG']
        self.sphere_to_plane = katpoint.sphere_to_plane['STG']
        N = 100
        # Stay well away from point of projection
        max_theta = 0.8 * np.pi
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        # Perform inverse STG mapping to spread out points on plane
        theta = max_theta * np.random.rand(N)
        r = 2.0 * np.sin(theta) / (1.0 + np.cos(theta))
        phi = 2 * np.pi * np.random.rand(N)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)

    def test_random_closure(self):
        """STG projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=9)
        np.testing.assert_almost_equal(self.y, yy, decimal=9)
        assert_angles_almost_equal(az, aa, decimal=9)
        assert_angles_almost_equal(el, ee, decimal=9)

    def test_aips_compatibility(self):
        """STG projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
        x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
        for n in xrange(len(az)):
            az_aips[n], el_aips[n], ierr = \
            newpos(6, self.az0[n], self.el0[n], self.x[n], self.y[n])
            x_aips[n], y_aips[n], ierr = \
            dircos(6, self.az0[n], self.el0[n], az[n], el[n])
        self.assertEqual(ierr, 0)
        # AIPS NEWPOS STG has poor accuracy on azimuth angle (large closure errors by itself)
        # assert_angles_almost_equal(az, az_aips, decimal=9)
        assert_angles_almost_equal(el, el_aips, decimal=9)
        np.testing.assert_almost_equal(xx, x_aips, decimal=9)
        np.testing.assert_almost_equal(yy, y_aips, decimal=9)

    def test_corner_cases(self):
        """STG projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(self.sphere_to_plane(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on circle of radius 2.0 in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(self.plane_to_sphere(0.0, -np.pi / 2.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0,  -np.pi / 2.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)


class TestProjectionCAR(unittest.TestCase):
    """Test plate carree projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['CAR']
        self.sphere_to_plane = katpoint.sphere_to_plane['CAR']
        N = 100
        # Unrestricted (az0, el0) points on sphere
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # Unrestricted (x, y) points on corresponding plane
        self.x = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.y = np.pi * (np.random.rand(N) - 0.5)

    def test_random_closure(self):
        """CAR projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=12)
        np.testing.assert_almost_equal(self.y, yy, decimal=12)
        assert_angles_almost_equal(az, aa, decimal=12)
        assert_angles_almost_equal(el, ee, decimal=12)


def sphere_to_plane_mattieu(targetaz,targetel,scanaz,scanel):
    #produces direction cosine coordinates from scanning antenna azimuth,elevation coordinates
    #see _coordinate options.py for derivation
    ll=np.cos(targetel)*np.sin(targetaz-scanaz)
    mm=np.cos(targetel)*np.sin(scanel)*np.cos(targetaz-scanaz)-np.cos(scanel)*np.sin(targetel)
    return ll,mm

def plane_to_sphere_mattieu(targetaz,targetel,ll,mm):
    scanaz=targetaz-np.arcsin(np.clip(ll/np.cos(targetel),-1.0,1.0))
    scanel=np.arcsin(np.clip((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.sqrt(np.cos(targetel)**2-ll**2)*mm)/(1.0-ll**2),-1.0,1.0))
    #alternate equations which gives same result
    # scanel_alternate1=np.arcsin((np.sqrt(1.0-ll**2-mm**2)*np.sin(targetel)+np.cos(targetel)*np.cos(targetaz-scanaz)*mm)/(1.0-ll**2))
    # num=np.cos(targetel)*np.cos(targetaz-scanaz)#or num=np.sqrt(np.cos(targetel)**2-ll**2)
    # den=np.sin(targetel)**2+num**2
    # scanel_alternate2=np.arcsin((np.sqrt(((den-mm**2)*(den-num**2)))+num*mm)/den)
    return scanaz,scanel


class TestProjectionSSN(unittest.TestCase):
    """Test swapped orthographic projection."""
    def setUp(self):
        self.plane_to_sphere = katpoint.plane_to_sphere['SSN']
        self.sphere_to_plane = katpoint.sphere_to_plane['SSN']
        N = 100
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        # (x, y) points within complicated SSN domain - clipped unit circle
        cos_el0 = np.cos(self.el0)
        # The x coordinate is bounded by +- cos(el0)
        self.x = (2 * np.random.rand(N) - 1) * cos_el0
        # The y coordinate ranges between two (semi-)circles centred on origin:
        # the unit circle on one side and circle of radius cos(el0) on other side
        y_offset = -np.sqrt(cos_el0 ** 2 - self.x ** 2)
        y_range = -y_offset + np.sqrt(1.0 - self.x ** 2)
        self.y = (y_range * np.random.rand(N) + y_offset) * np.sign(self.el0)

    def test_random_closure(self):
        """SSN projection: do random projections and check closure."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        xx, yy = self.sphere_to_plane(self.az0, self.el0, az, el)
        aa, ee = self.plane_to_sphere(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=10)
        np.testing.assert_almost_equal(self.y, yy, decimal=10)
        assert_angles_almost_equal(az, aa, decimal=10)
        assert_angles_almost_equal(el, ee, decimal=10)

    def test_vs_mattieu(self):
        """SSN projection: compare against Mattieu's original version."""
        az, el = self.plane_to_sphere(self.az0, self.el0, self.x, self.y)
        ll, mm = sphere_to_plane_mattieu(self.az0, self.el0, az, el)
        aa, ee = plane_to_sphere_mattieu(self.az0, self.el0, ll, mm)
        np.testing.assert_almost_equal(self.x, ll, decimal=10)
        np.testing.assert_almost_equal(self.y, -mm, decimal=10)
        assert_angles_almost_equal(az, aa, decimal=10)
        assert_angles_almost_equal(el, ee, decimal=10)

    def test_corner_cases(self):
        """SSN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(self.sphere_to_plane(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi, 1e-8))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(self.sphere_to_plane(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, self.sphere_to_plane, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(self.plane_to_sphere(0.0, np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Test valid (x, y) domain
        ae = np.array(self.plane_to_sphere(0.0, 1.0, 0.0, -np.cos(1.0)))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        ae = np.array(self.plane_to_sphere(0.0, -1.0, 0.0, np.cos(1.0)))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, self.plane_to_sphere, 0.0, 0.0, 0.0, 2.0)
