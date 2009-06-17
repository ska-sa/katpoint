"""Enhancements to PyEphem."""

import time

import numpy as np
import ephem

#--------------------------------------------------------------------------------------------------
#--- Helper functions
#--------------------------------------------------------------------------------------------------

# The speed of light, in metres per second
lightspeed = ephem.c

def rad2deg(x):
    return x * 180.0 / np.pi

def deg2rad(x):
    return x * np.pi / 180.0

def unix_to_ephem_time(secs_since_epoch):
    """Convert seconds-since-Unix-epoch local time to a UTC PyEphem Date object.
    
    Parameters
    ----------
    secs_since_epoch : float
        Local time in seconds since Unix epoch (may have fractional part)
    
    Returns
    -------
    ephem_time : :class:`ephem.Date` object
        PyEphem UTC timestamp object, required for ephem calculations
    
    """
    timestamp = list(time.gmtime(np.floor(secs_since_epoch))[:6])
    timestamp[5] += secs_since_epoch - np.floor(secs_since_epoch)
    return ephem.Date(tuple(timestamp))

#--------------------------------------------------------------------------------------------------
#--- CLASS :  StationaryBody
#--------------------------------------------------------------------------------------------------

class StationaryBody(object):
    """Stationary body with fixed (az, el) coordinates.
    
    This is a simplified :class:`ephem.Body` that is useful to specify targets
    such as zenith and geostationary satellites.
    
    Parameters
    ----------
    az : string or float
        Azimuth, either in 'D:M:S' string format, or as a float in radians
    el : string or float
        Elevation, either in 'D:M:S' string format, or as a float in radians
    name : string, optional
        Name of body
    
    """
    def __init__(self, az, el, name=None):
        self.az = ephem.degrees(az)
        self.el = ephem.degrees(el)
        self.alt = self.el # alternative terminology
        if not name:
            name = "Az: %s El: %s" % (self.az, self.el)
        self.name = name
    
    def compute(self, observer):
        """Update target coordinates for given observer.
        
        This updates the (ra, dec) coordinates of the target, as seen from the
        given *observer*, while its (az, el) coordinates remain unchanged.
        
        """
        # pylint: disable-msg=W0201
        if isinstance(observer, ephem.Observer):
            ra, dec = observer.radec_of(self.az, self.el)
            self.ra = ra
            self.dec = dec
            # This is a kludge, as XEphem provides no way to convert apparent
            # (ra, dec) back to astrometric (ra, dec)
            self.a_ra = ra
            self.a_dec = dec

#--------------------------------------------------------------------------------------------------
#--- Geodetic coordinate transformations
#--------------------------------------------------------------------------------------------------

def lla_to_ecef(lat_rad, long_rad, alt_m):
    """Convert WGS84 spherical coordinates to ECEF cartesian coordinates.
    
    This converts a position on the Earth specified in latitude, longitude and
    altitude to earth-centered, earth-fixed (ECEF) cartesian coordinates. This
    code assumes the WGS84 earth model, described in [1]_.
    
    Parameters
    ----------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    long_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid
    
    Returns
    -------
    x_m : float or array
        X coordinate, in metres
    y_m : float or array
        Y coordinate, in metres
    z_m : float or array
        Z coordinate, in metres
    
    .. [1] National Imagery and Mapping Agency, "Department of Defense World
       Geodetic System 1984," NIMA TR8350.2, Page 4-4, last updated June, 2004.
    
    """
    # WGS84 ellipsoid constants
    a = 6378137.0                # semi-major axis of Earth in m 
    e = 8.1819190842622e-2       # eccentricity of Earth
    
    # intermediate calculation
    # (normal, or prime vertical radius of curvature)
    R = a / np.sqrt(1.0 - e**2 * np.sin(lat_rad)**2)
    
    x_m = (R + alt_m) * np.cos(lat_rad) * np.cos(long_rad)
    y_m = (R + alt_m) * np.cos(lat_rad) * np.sin(long_rad)
    z_m = ((1.0 - e**2) * R + alt_m) * np.sin(lat_rad)
    
    return x_m, y_m, z_m

def ecef_to_lla(x_m, y_m, z_m):
    """Convert ECEF cartesian coordinates to WGS84 spherical coordinates.
    
    This converts an earth-centered, earth-fixed (ECEF) cartesian position to a
    position on the Earth specified in latitude, longitude and altitude. This
    code assumes the WGS84 earth model.
    
    Parameters
    ----------
    x_m : float or array
        X coordinate, in metres
    y_m : float or array
        Y coordinate, in metres
    z_m : float or array
        Z coordinate, in metres
    
    Returns
    -------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    long_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid
    
    """
    # WGS84 ellipsoid constants
    a = 6378137.0                    # semi-major axis of Earth in m 
    e = 8.1819190842622e-2           # eccentricity of Earth
    
    b = np.sqrt(a**2 * (1.0 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x_m**2 + y_m**2)
    th  = np.arctan2(a * z_m, b * p)
    long_rad = np.arctan2(y_m, x_m)
    lat_rad = np.arctan2((z_m + ep**2 * b * np.sin(th)**3), (p - e**2 * a * np.cos(th)**3))
    N = a / np.sqrt(1.0 - e**2 * np.sin(lat_rad)**2)
    alt_m = p / np.cos(lat_rad) - N
    
    # Return long_rad in range [0, 2*pi)
    long_rad = np.mod(long_rad, 2.0 * np.pi)
    
    # Correct for numerical instability in altitude near exact poles
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
    if np.isscalar(alt_m):
        if (abs(x_m) < 1.0) and (abs(y_m) < 1.0):
            alt_m = abs(z_m) - b
    else:
        near_poles = (np.abs(x_m) < 1.0) & (np.abs(y_m) < 1.0)
        alt_m[near_poles] = np.abs(z_m[near_poles]) - b
    
    return lat_rad, long_rad, alt_m

def enu_to_ecef(ref_lat_rad, ref_long_rad, ref_alt_m, e_m, n_m, u_m):
    """Convert ENU coordinates relative to reference location to ECEF coordinates.
    
    This converts local east-north-up (ENU) coordinates relative to a given
    reference position to earth-centered, earth-fixed (ECEF) cartesian
    coordinates. The reference position is specified by its latitude, longitude
    and altitude.
    
    Parameters
    ----------
    ref_lat_rad : float or array
        Latitude of reference position, in radians
    ref_long_rad : float or array
        Longitude of reference position, in radians
    ref_alt_m : float or array
        Altitude of reference position, in metres above WGS84 ellipsoid
    e_m : float or array
        East coordinate, in metres
    n_m : float or array
        North coordinate, in metres
    u_m : float or array
        Up coordinate, in metres
    
    Returns
    -------
    x_m : float or array
        X coordinate, in metres
    y_m : float or array
        Y coordinate, in metres
    z_m : float or array
        Z coordinate, in metres
    
    """
    # ECEF coordinates of reference point
    ref_x_m, ref_y_m, ref_z_m = lla_to_ecef(ref_lat_rad, ref_long_rad, ref_alt_m)
    # Geocentric latitude
    gc_lat_rad = np.arctan2(ref_z_m, np.sqrt(ref_x_m**2 + ref_y_m**2))
    
    x_m = ref_x_m - np.sin(ref_long_rad) * e_m - \
                    np.cos(ref_long_rad) * np.sin(gc_lat_rad) * n_m + \
                    np.cos(ref_long_rad) * np.cos(gc_lat_rad) * u_m
    y_m = ref_y_m + np.cos(ref_long_rad) * e_m - \
                    np.sin(ref_long_rad) * np.sin(gc_lat_rad) * n_m + \
                    np.cos(gc_lat_rad) * np.sin(ref_long_rad) * u_m
    z_m = ref_z_m + np.cos(gc_lat_rad) * n_m + np.sin(gc_lat_rad) * u_m
    
    return x_m, y_m, z_m
