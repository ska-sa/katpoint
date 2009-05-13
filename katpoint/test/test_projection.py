"""Tests for the projection module."""

import unittest
import numpy as np
from scape import projection
try:
    import aips_projection
    found_aips = True
except ImportError:
    found_aips = False

def assert_angles_almost_equal(x, y, decimal):
    primary_angle = lambda x: x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), decimal=decimal)

class TestProjectionSIN(unittest.TestCase):
    
    def setUp(self):
        N = 100
        max_radius = 1.0
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # (x,y) points within unit circle
        r = max_radius * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)
        
    def test_random_closure(self):
        """SIN projection: do random projections and check closure."""
        az, el = projection.plane_to_sphere_SIN(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_SIN(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_SIN(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=12)
        np.testing.assert_almost_equal(self.y, yy, decimal=12)
        assert_angles_almost_equal(az, aa, decimal=12)
        assert_angles_almost_equal(el, ee, decimal=12)
    
    def test_aips_compatibility(self):
        """SIN projection: compare with original AIPS routine."""
        if found_aips:
            az, el = projection.plane_to_sphere_SIN(self.az0, self.el0, self.x, self.y)
            xx, yy = projection.sphere_to_plane_SIN(self.az0, self.el0, az, el)
            az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
            x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
            for n in xrange(len(az_aips)):
                az_aips[n], el_aips[n], ierr = \
                aips_projection.newpos(2, self.az0[n], self.el0[n], self.x[n], self.y[n])
                x_aips[n], y_aips[n], ierr = \
                aips_projection.dircos(2, self.az0[n], self.el0[n], az[n], el[n])
            assert_angles_almost_equal(az, az_aips, decimal=12)
            assert_angles_almost_equal(el, el_aips, decimal=12)
            np.testing.assert_almost_equal(xx, x_aips, decimal=12)
            np.testing.assert_almost_equal(yy, y_aips, decimal=12)
            
    def test_corner_cases(self):
        """SIN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(projection.sphere_to_plane_SIN(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_SIN(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_SIN(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, np.pi / 2.0, np.pi, 1e-8))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_SIN(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)        
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_SIN, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_SIN, 0.0, 0.0, 0.0, np.pi)
        
        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_SIN(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_SIN, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_SIN, 0.0, 0.0, 0.0, 2.0)

class TestProjectionTAN(unittest.TestCase):
    
    def setUp(self):
        N = 100
        max_radius = 1.0
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # (x,y) points within unit circle
        r = max_radius * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)
        
    def test_random_closure(self):
        """TAN projection: do random projections and check closure."""
        az, el = projection.plane_to_sphere_TAN(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_TAN(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_TAN(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=10)
        np.testing.assert_almost_equal(self.y, yy, decimal=10)
        assert_angles_almost_equal(az, aa, decimal=10)
        assert_angles_almost_equal(el, ee, decimal=10)

    def test_aips_compatibility(self):
        """TAN projection: compare with original AIPS routine."""
        if found_aips:
            az, el = projection.plane_to_sphere_TAN(self.az0, self.el0, self.x, self.y)
            xx, yy = projection.sphere_to_plane_TAN(self.az0, self.el0, az, el)
            az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
            x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
            for n in xrange(len(az_aips)):
                az_aips[n], el_aips[n], ierr = \
                aips_projection.newpos(3, self.az0[n], self.el0[n], self.x[n], self.y[n])
                x_aips[n], y_aips[n], ierr = \
                aips_projection.dircos(3, self.az0[n], self.el0[n], az[n], el[n])
            assert_angles_almost_equal(az, az_aips, decimal=10)
            assert_angles_almost_equal(el, el_aips, decimal=10)
            np.testing.assert_almost_equal(xx, x_aips, decimal=10)
            np.testing.assert_almost_equal(yy, y_aips, decimal=10)
    
    def test_corner_cases(self):
        """TAN projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(projection.sphere_to_plane_TAN(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 45 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_TAN(0.0, 0.0, np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, 0.0, -np.pi / 4.0, 0.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, 0.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, 0.0, 0.0, -np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_TAN(0.0, np.pi / 2.0, 0.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, -1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, np.pi / 2.0, np.pi, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [0.0, 1.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [1.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_TAN(0.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 4.0))
        np.testing.assert_almost_equal(xy, [-1.0, 0.0], decimal=12)        
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_TAN, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_TAN, 0.0, 0.0, 0.0, np.pi)
        
        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 4.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 4.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_TAN(0.0, -np.pi / 2.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [np.pi, -np.pi / 4.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_TAN, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_TAN, 0.0, 0.0, 0.0, 2.0)

class TestProjectionARC(unittest.TestCase):
    
    def setUp(self):
        N = 100
        max_radius = np.pi - 1e-2
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # (x,y) points within circle of radius pi
        r = max_radius * np.random.rand(N)
        phi = 2 * np.pi * np.random.rand(N)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)
        
    def test_random_closure(self):
        """ARC projection: do random projections and check closure."""
        az, el = projection.plane_to_sphere_ARC(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_ARC(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_ARC(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=8)
        np.testing.assert_almost_equal(self.y, yy, decimal=8)
        assert_angles_almost_equal(az, aa, decimal=8)
        assert_angles_almost_equal(el, ee, decimal=8)
    
    def test_aips_compatibility(self):
        """ARC projection: compare with original AIPS routine."""
        if found_aips:
            az, el = projection.plane_to_sphere_ARC(self.az0, self.el0, self.x, self.y)
            xx, yy = projection.sphere_to_plane_ARC(self.az0, self.el0, az, el)
            az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
            x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
            for n in xrange(len(az_aips)):
                az_aips[n], el_aips[n], ierr = \
                aips_projection.newpos(4, self.az0[n], self.el0[n], self.x[n], self.y[n])
                x_aips[n], y_aips[n], ierr = \
                aips_projection.dircos(4, self.az0[n], self.el0[n], az[n], el[n])
            assert_angles_almost_equal(az, az_aips, decimal=8)
            assert_angles_almost_equal(el, el_aips, decimal=8)
            np.testing.assert_almost_equal(xx, x_aips, decimal=8)
            np.testing.assert_almost_equal(yy, y_aips, decimal=8)
    
    def test_corner_cases(self):
        """ARC projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(projection.sphere_to_plane_ARC(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_ARC(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_ARC(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, np.pi / 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [np.pi / 2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_ARC(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-np.pi / 2.0, 0.0], decimal=12)        
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_ARC, 0.0, 0.0, 0.0, np.pi)
        
        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 1.0, 0.0))
        assert_angles_almost_equal(ae, [1.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, -1.0, 0.0))
        assert_angles_almost_equal(ae, [-1.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, 1.0))
        assert_angles_almost_equal(ae, [0.0, 1.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, -1.0))
        assert_angles_almost_equal(ae, [0.0, -1.0], decimal=12)        
        # Points on circle with radius pi in plane
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, np.pi, 0.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, -np.pi, 0.0))
        assert_angles_almost_equal(ae, [-np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, -np.pi))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_ARC(0.0, -np.pi / 2.0, np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, -np.pi / 2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, 0.0, np.pi / 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, 0.0, -np.pi / 2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_ARC, 0.0, 0.0, 4.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_ARC, 0.0, 0.0, 0.0, 4.0)

class TestProjectionSTG(unittest.TestCase):
    
    def setUp(self):
        N = 100
        self.az0 = np.pi * (2.0 * np.random.rand(N) - 1.0)
        self.el0 = np.pi * (np.random.rand(N) - 0.5)
        # Perform inverse STG mapping to spread out points on plane
        # Stay well away from projection point
        theta = 0.8 * np.pi * np.random.rand(N)
        r = 2.0 * np.sin(theta) / (1.0 + np.cos(theta))
        phi = 2 * np.pi * np.random.rand(N)
        self.x = r * np.cos(phi)
        self.y = r * np.sin(phi)
        
    def test_random_closure(self):
        """STG projection: do random projections and check closure."""
        az, el = projection.plane_to_sphere_STG(self.az0, self.el0, self.x, self.y)
        xx, yy = projection.sphere_to_plane_STG(self.az0, self.el0, az, el)
        aa, ee = projection.plane_to_sphere_STG(self.az0, self.el0, xx, yy)
        np.testing.assert_almost_equal(self.x, xx, decimal=9)
        np.testing.assert_almost_equal(self.y, yy, decimal=9)
        assert_angles_almost_equal(az, aa, decimal=9)
        assert_angles_almost_equal(el, ee, decimal=9)
    
    def test_aips_compatibility(self):
        """STG projection: compare with original AIPS routine."""
        if found_aips:
            az, el = projection.plane_to_sphere_STG(self.az0, self.el0, self.x, self.y)
            xx, yy = projection.sphere_to_plane_STG(self.az0, self.el0, az, el)
            az_aips, el_aips = np.zeros(az.shape), np.zeros(el.shape)
            x_aips, y_aips = np.zeros(xx.shape), np.zeros(yy.shape)
            for n in xrange(len(az)):
                az_aips[n], el_aips[n], ierr = \
                aips_projection.newpos(9, self.az0[n], self.el0[n], self.x[n], self.y[n])
                x_aips[n], y_aips[n], ierr = \
                aips_projection.dircos(9, self.az0[n], self.el0[n], az[n], el[n])
            # AIPS NEWPOS STG has poor accuracy on azimuth angle (large closure errors by itself)
            # assert_angles_almost_equal(az, az_aips, decimal=9)
            assert_angles_almost_equal(el, el_aips, decimal=9)
            np.testing.assert_almost_equal(xx, x_aips, decimal=9)
            np.testing.assert_almost_equal(yy, y_aips, decimal=9)
    
    def test_corner_cases(self):
        """STG projection: test special corner cases."""
        # SPHERE TO PLANE
        # Origin
        xy = np.array(projection.sphere_to_plane_STG(0.0, 0.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 0.0], decimal=12)
        # Points 90 degrees from reference point on sphere
        xy = np.array(projection.sphere_to_plane_STG(0.0, 0.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, 0.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, 0.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, 0.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        # Reference point at pole on sphere
        xy = np.array(projection.sphere_to_plane_STG(0.0, np.pi / 2.0, 0.0, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, -2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, np.pi / 2.0, np.pi, 0.0))
        np.testing.assert_almost_equal(xy, [0.0, 2.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [2.0, 0.0], decimal=12)
        xy = np.array(projection.sphere_to_plane_STG(0.0, np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(xy, [-2.0, 0.0], decimal=12)        
        # Points outside allowed domain on sphere
        self.assertRaises(ValueError, projection.sphere_to_plane_STG, 0.0, 0.0, np.pi, 0.0)
        self.assertRaises(ValueError, projection.sphere_to_plane_STG, 0.0, 0.0, 0.0, np.pi)
        
        # PLANE TO SPHERE
        # Origin
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 0.0, 0.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on circle of radius 2.0 in plane
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_STG(0.0, -np.pi / 2.0, 2.0, 0.0))
        assert_angles_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, -2.0, 0.0))
        assert_angles_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, 0.0, 2.0))
        assert_angles_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, 0.0, -2.0))
        assert_angles_almost_equal(ae, [np.pi, 0.0], decimal=12)        
