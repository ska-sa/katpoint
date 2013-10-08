"""Tests for the projection module."""
# pylint: disable-msg=C0103,W0212

import unittest

import numpy as np

from katpoint import _projection as projection
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
        az, el = projection.plane_to_sphere_sin(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_sin(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_sin(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=10)
        np.testing.assert_almost_equal(self.y, yy, decimal=10)
        assert_angles_almost_equal(az, aa, decimal=10)
        assert_angles_almost_equal(el, ee, decimal=10)

    def test_aips_compatibility(self):
        """SIN projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = projection.plane_to_sphere_sin(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_sin(self.az0, self.el0, az, el)
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
        xy = np.array(projection.sphere_to_plane_sin(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_sin(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_sin(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, np.pi / 2.0, np.pi, 1e-8))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_sin(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_sin, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_sin, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_sin(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_sin(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_sin(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_sin(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_sin, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_sin, 0.0, 0.0, 0.0, 2.0)


class TestProjectionTAN(unittest.TestCase):
    """Test gnomonic projection."""
    def setUp(self):
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
        az, el = projection.plane_to_sphere_tan(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_tan(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_tan(self.az0, self.el0, xx, yy)
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
        az, el = projection.plane_to_sphere_tan(az0, el0, x, y)
        xx, yy = projection.sphere_to_plane_tan(az0, el0, az, el)
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
        xy = np.array(projection.sphere_to_plane_tan(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 45 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_tan(0.0, 0.0, np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, 0.0, -np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, 0.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, 0.0, 0.0, -np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_tan(0.0, np.pi / 2.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, np.pi / 2.0, np.pi, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_tan(0.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_tan, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_tan, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_tan(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_tan(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 4.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 4.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 4.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_tan(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, -np.pi / 4.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, -np.pi / 4.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_tan(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, -np.pi / 4.0], decimal=12)


class TestProjectionARC(unittest.TestCase):
    """Test zenithal equidistant projection."""
    def setUp(self):
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
        az, el = projection.plane_to_sphere_arc(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_arc(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_arc(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=8)
        np.testing.assert_almost_equal(self.y, yy, decimal=8)
        assert_angles_almost_equal(az, aa, decimal=8)
        assert_angles_almost_equal(el, ee, decimal=8)

    def test_aips_compatibility(self):
        """ARC projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = projection.plane_to_sphere_arc(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_arc(self.az0, self.el0, az, el)
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
        xy = np.array(projection.sphere_to_plane_arc(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_arc(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_arc(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_arc(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)
        # Point diametrically opposite the reference point on sphere
        xy = np.array(projection.sphere_to_plane_arc(np.pi, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(np.abs(xy), [np.pi, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_arc, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [1.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-1.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 1.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -1.0], decimal=12)
        # Points on circle with radius pi in plane
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, np.pi, 0.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, -np.pi, 0.0))
        assert_angles_almost_equal(ae, [-np.pi, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 0.0, np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0, 0.0, 0.0, -np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_arc(0.0, -np.pi / 2.0, np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0,  -np.pi / 2.0, -np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0,  -np.pi / 2.0, 0.0, np.pi / 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_arc(0.0,  -np.pi / 2.0, 0.0, -np.pi / 2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_arc, 0.0, 0.0, 4.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_arc, 0.0, 0.0, 0.0, 4.0)


class TestProjectionSTG(unittest.TestCase):
    """Test stereographic projection."""
    def setUp(self):
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
        az, el = projection.plane_to_sphere_stg(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_stg(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_stg(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=9)
        np.testing.assert_almost_equal(self.y, yy, decimal=9)
        assert_angles_almost_equal(az, aa, decimal=9)
        assert_angles_almost_equal(el, ee, decimal=9)

    def test_aips_compatibility(self):
        """STG projection: compare with original AIPS routine."""
        if not found_aips:
            skip("AIPS projection module not found")
            return
        az, el = projection.plane_to_sphere_stg(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_stg(self.az0, self.el0, az, el)
        az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
        x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
        for n in xrange(len(az)):
            az_aips[n], el_aips[n], ierr = \
            newpos(9, self.az0[n], self.el0[n], self.x[n], self.y[n])
            x_aips[n], y_aips[n], ierr = \
            dircos(9, self.az0[n], self.el0[n], az[n], el[n])
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
        xy = np.array(projection.sphere_to_plane_stg(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_stg(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_stg(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_stg(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_stg, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_stg, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_stg(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on circle of radius 2.0 in plane
        ae = np.array(projection.plane_to_sphere_stg(0.0, 0.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0, 0.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0, 0.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0, 0.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_stg(0.0, -np.pi / 2.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0,  -np.pi / 2.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0,  -np.pi / 2.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_stg(0.0,  -np.pi / 2.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)


class TestProjectionCAR(unittest.TestCase):
    """Test plate carree projection."""
    def setUp(self):
        N = 100
        # Unrestricted (az0, el0) points on sphere
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # Unrestricted (x, y) points on corresponding plane
        self.x = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.y = np.pi * (np.random.rand(N) - 0.5)

    def test_random_closure(self):
        """CAR projection: do random projections and check closure."""
        az, el = projection.plane_to_sphere_car(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_car(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_car(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=12)
        np.testing.assert_almost_equal(self.y, yy, decimal=12)
        assert_angles_almost_equal(az, aa, decimal=12)
        assert_angles_almost_equal(el, ee, decimal=12)


class TestProjectionSSN(unittest.TestCase):
    """Test swapped orthographic projection."""
    def setUp(self):
        N = 100
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        # Keep away from poles (leave them as corner cases)
        self.el0 = 0.999 * np.pi * (np.random.rand(N) - 0.5)
        max_theta = np.pi / 2.0
        # (x, y) points within unit circle
        theta = max_theta * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        sinx = np.sin(theta) * np.cos(phi)
        siny = np.sin(theta) * np.sin(phi)
        self.az, self.el = projection.plane_to_sphere_sin(self.az0, self.el0, sinx, siny)

    def test_random_closure(self):
        """SSN projection: do random projections and check closure."""
        x, y = projection.sphere_to_plane_ssn(self.az0, self.el0, self.az, self.el)
        az, el = projection.plane_to_sphere_ssn(self.az0, self.el0, x, y)
        xx, yy = projection.sphere_to_plane_ssn(self.az0, self.el0, az, el)
        np.testing.assert_almost_equal(x, xx, decimal=10)
        np.testing.assert_almost_equal(y, yy, decimal=10)
        assert_angles_almost_equal(self.az, az, decimal=10)
        assert_angles_almost_equal(self.el, el, decimal=10)

    def test_corner_cases(self):
        """SSN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(projection.sphere_to_plane_ssn(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_ssn(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_ssn(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, np.pi / 2.0, np.pi, 1e-8))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ssn(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_ssn, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_ssn, 0.0, 0.0, 0.0, np.pi)

        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_ssn(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_ssn(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_ssn(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_ssn(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_ssn(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_ssn(0.0,  np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        ae = np.array(projection.plane_to_sphere_ssn(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_ssn, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_ssn, 0.0, 0.0, 0.0, 2.0)
