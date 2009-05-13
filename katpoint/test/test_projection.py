"""Tests for the projection module."""

import unittest
import numpy as np
from scape import projection

class TestProjectionSIN(unittest.TestCase):
    
    def test_random_closure(self):
        """SIN projection: do 100 random projections and check closure."""
        az0 = np.pi * (2.0 * np.random.rand(100) - 1.0)
        el0 = np.pi * (np.random.rand(100) - 0.5)
        # (x,y) points within unit circle
        r = np.random.rand(100)
        phi = 2 * np.pi * np.random.rand(100)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        az, el = projection.plane_to_sphere_SIN(az0, el0, x, y)
        xx, yy = projection.sphere_to_plane_SIN(az0, el0, az, el)
        aa, ee = projection.plane_to_sphere_SIN(az0, el0, xx, yy)
        np.testing.assert_almost_equal(x, xx, decimal=12)
        np.testing.assert_almost_equal(y, yy, decimal=12)
        np.testing.assert_almost_equal(az, aa, decimal=12)
        np.testing.assert_almost_equal(el, ee, decimal=12)
    
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
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 1.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, -1.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 0.0, 1.0))
        np.testing.assert_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0, 0.0, 0.0, -1.0))
        np.testing.assert_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_SIN(0.0, -np.pi / 2.0, 1.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, -1.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, 0.0, 1.0))
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_SIN(0.0,  -np.pi / 2.0, 0.0, -1.0))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_SIN, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_SIN, 0.0, 0.0, 0.0, 2.0)

class TestProjectionTAN(unittest.TestCase):
    
    def test_random_closure(self):
        """TAN projection: do 100 random projections and check closure."""
        az0 = np.pi * (2.0 * np.random.rand(100) - 1.0)
        el0 = np.pi * (np.random.rand(100) - 0.5)
        # (x,y) points within unit circle
        r = np.random.rand(100)
        phi = 2 * np.pi * np.random.rand(100)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        az, el = projection.plane_to_sphere_TAN(az0, el0, x, y)
        xx, yy = projection.sphere_to_plane_TAN(az0, el0, az, el)
        aa, ee = projection.plane_to_sphere_TAN(az0, el0, xx, yy)
        np.testing.assert_almost_equal(x, xx, decimal=12)
        np.testing.assert_almost_equal(y, yy, decimal=12)
        np.testing.assert_almost_equal(az, aa, decimal=12)
        np.testing.assert_almost_equal(el, ee, decimal=12)
    
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
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 1.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 4.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, -1.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 4.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 0.0, 1.0))
        np.testing.assert_almost_equal(ae, [0.0, np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0, 0.0, 0.0, -1.0))
        np.testing.assert_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_TAN(0.0, -np.pi / 2.0, 1.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, -1.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, 0.0, 1.0))
        np.testing.assert_almost_equal(ae, [0.0, -np.pi / 4.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_TAN(0.0,  -np.pi / 2.0, 0.0, -1.0))
        np.testing.assert_almost_equal(ae, [np.pi, -np.pi / 4.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_TAN, 0.0, 0.0, 2.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_TAN, 0.0, 0.0, 0.0, 2.0)

class TestProjectionARC(unittest.TestCase):
    
    def test_random_closure(self):
        """ARC projection: do 100 random projections and check closure."""
        az0 = np.pi * (2.0 * np.random.rand(100) - 1.0)
        el0 = np.pi * (np.random.rand(100) - 0.5)
        # (x,y) points within circle of radius pi
        r = np.clip(np.pi * np.random.rand(100), 0.0, np.pi - 1e-5)
        phi = 2 * np.pi * np.random.rand(100)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        az, el = projection.plane_to_sphere_ARC(az0, el0, x, y)
        xx, yy = projection.sphere_to_plane_ARC(az0, el0, az, el)
        aa, ee = projection.plane_to_sphere_ARC(az0, el0, xx, yy)
        np.testing.assert_almost_equal(x, xx, decimal=9)
        np.testing.assert_almost_equal(y, yy, decimal=9)
        np.testing.assert_almost_equal(az, aa, decimal=9)
        np.testing.assert_almost_equal(el, ee, decimal=9)
    
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
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        # Points on unit circle in plane
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 1.0, 0.0))
        np.testing.assert_almost_equal(ae, [1.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, -1.0, 0.0))
        np.testing.assert_almost_equal(ae, [-1.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, 1.0))
        np.testing.assert_almost_equal(ae, [0.0, 1.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, -1.0))
        np.testing.assert_almost_equal(ae, [0.0, -1.0], decimal=12)        
        # Points on circle with radius pi in plane
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, np.pi, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, -np.pi, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, np.pi))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0, 0.0, 0.0, -np.pi))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_ARC(0.0, -np.pi / 2.0, np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, -np.pi / 2.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, 0.0, np.pi / 2.0))
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_ARC(0.0,  -np.pi / 2.0, 0.0, -np.pi / 2.0))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
        # Points outside allowed domain in plane
        self.assertRaises(ValueError, projection.plane_to_sphere_ARC, 0.0, 0.0, 4.0, 0.0)
        self.assertRaises(ValueError, projection.plane_to_sphere_ARC, 0.0, 0.0, 0.0, 4.0)

class TestProjectionSTG(unittest.TestCase):
    
    def test_random_closure(self):
        """STG projection: do 100 random projections and check closure."""
        az0 = np.pi * (2.0 * np.random.rand(100) - 1.0)
        el0 = np.pi * (np.random.rand(100) - 0.5)
        # Perform inverse STG mapping to spread out points on plane
        theta = np.pi * np.random.rand(100)
        r = 2.0 * np.sin(theta) / np.clip(1.0 + np.cos(theta), 1e-5, 2.0)
        phi = 2 * np.pi * np.random.rand(100)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        az, el = projection.plane_to_sphere_STG(az0, el0, x, y)
        xx, yy = projection.sphere_to_plane_STG(az0, el0, az, el)
        aa, ee = projection.plane_to_sphere_STG(az0, el0, xx, yy)
        np.testing.assert_almost_equal(x, xx, decimal=8)
        np.testing.assert_almost_equal(y, yy, decimal=8)
        np.testing.assert_almost_equal(az, aa, decimal=8)
        np.testing.assert_almost_equal(el, ee, decimal=8)
    
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
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)
        # Points on circle of radius 2.0 in plane
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 2.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, -2.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 0.0, 2.0))
        np.testing.assert_almost_equal(ae, [0.0, np.pi / 2.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0, 0.0, 0.0, -2.0))
        np.testing.assert_almost_equal(ae, [0.0, -np.pi / 2.0], decimal=12)        
        # Reference point at pole on sphere
        ae = np.array(projection.plane_to_sphere_STG(0.0, -np.pi / 2.0, 2.0, 0.0))
        np.testing.assert_almost_equal(ae, [np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, -2.0, 0.0))
        np.testing.assert_almost_equal(ae, [-np.pi / 2.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, 0.0, 2.0))
        np.testing.assert_almost_equal(ae, [0.0, 0.0], decimal=12)        
        ae = np.array(projection.plane_to_sphere_STG(0.0,  -np.pi / 2.0, 0.0, -2.0))
        np.testing.assert_almost_equal(ae, [np.pi, 0.0], decimal=12)        
