"""Enhancements to PyEphem."""

import time

import numpy as np
import ephem

#--------------------------------------------------------------------------------------------------
#--- Helper functions
#--------------------------------------------------------------------------------------------------

# The speed of light, in metres per second
lightspeed = ephem.c

def is_iterable(x):
    """Checks if object is iterable (but not a string)."""
    return hasattr(x, '__iter__')

def rad2deg(x):
    """Converts radians to degrees (also works for arrays)."""
    return x * (180.0 / np.pi)

def deg2rad(x):
    """Converts degrees to radians (also works for arrays)."""
    return x * (np.pi / 180.0)

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Timestamp
#--------------------------------------------------------------------------------------------------

class Timestamp(object):
    """Basic representation of time, in UTC seconds since Unix epoch.

    This is loosely based on :class:`ephem.Date`. Its base representation
    of time is UTC seconds since the Unix epoch, i.e. the standard Posix
    timestamp (:class:`ephem.Date` uses UTC days since noon on 1899/12/31, or
    the *Dublin Julian Day*). Fractional seconds are allowed, as the basic data
    type is a Python (double-precision) float.

    The following input formats are accepted for a timestamp:

    - None, which uses the current time (the default).

    - A floating-point number, directly representing the number of UTC seconds
      since the Unix epoch. Fractional seconds are allowed.

    - A string with format 'YYYY-MM-DD HH:MM:SS.SSS' or 'YYYY/MM/DD HH:MM:SS.SSS',
      or any prefix thereof. Examples are '1999-12-31 12:34:56.789', '1999-12-31',
      '1999-12-31 12:34:56' and even '1999'. The input string is always in UTC.

    - A :class:`ephem.Date` object, which is the standard time representation
      in PyEphem.

    Parameters
    ----------
    timestamp : float, string, :class:`ephem.Date` object or None
        Timestamp, in various formats (if None, defaults to now)

    Arguments
    ---------
    secs : float
        Timestamp as UTC seconds since Unix epoch

    """
    def __init__(self, timestamp=None):
        if isinstance(timestamp, basestring):
            try:
                timestamp = ephem.Date(timestamp.strip().replace('-', '/'))
            except ValueError:
                raise ValueError("Timestamp string '%s' not in correct format - " % (timestamp,) +
                                 "should be 'YYYY-MM-DD HH:MM:SS' or 'YYYY/MM/DD HH:MM:SS' or prefix thereof " +
                                 "(all UTC, fractional seconds allowed)")
        if timestamp is None:
            self.secs = time.time()
        elif isinstance(timestamp, ephem.Date):
            timestamp = list(timestamp.tuple()) + [0, 0, 0]
            frac_secs = timestamp[5] - np.floor(timestamp[5])
            timestamp[5] = int(np.floor(timestamp[5]))
            self.secs = time.mktime(timestamp) - time.timezone + frac_secs
        else:
            self.secs = float(timestamp)

    # Keep object small by using __slots__ instead of __dict__
    __slots__ = 'secs'

    def __repr__(self):
        """Short machine-friendly string representation of timestamp object."""
        return 'Timestamp(%s)' % repr(self.secs)

    def __str__(self):
        """Verbose human-friendly string representation of timestamp object."""
        return self.to_string()

    def __cmp__(self, other):
        """Compare timestamps based on chronological order."""
        return np.sign(self.secs - other.secs)

    def __add__(self, other):
        """Add seconds (as floating-point number) to timestamp and return result."""
        return Timestamp(self.secs + other)

    def __sub__(self, other):
        """Subtract seconds (or another timestamp) from timestamp and return result."""
        if isinstance(other, Timestamp):
            return self.secs - other.secs
        else:
            return Timestamp(self.secs - other)

    def __mul__(self, other):
        """Multiply timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs * other)

    def __div__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __truediv__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __radd__(self, other):
        """Add timestamp to seconds (as floating-point number) and return result."""
        return Timestamp(other + self.secs)

    def __iadd__(self, other):
        """Add seconds (as floating-point number) to timestamp in-place."""
        self.secs += other
        return self

    def __isub__(self, other):
        """Subtract seconds (as floating-point number) from timestamp in-place."""
        self.secs -= other
        return self

    def __float__(self):
        """Convert to floating-point UTC seconds."""
        return self.secs

    def local(self):
        """Convert timestamp to local time string representation (for display only)."""
        int_secs = np.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int_secs))
        timezone = time.strftime('%Z', time.localtime(int_secs))
        if frac_secs == 0.0:
            return '%s %s' % (datetime, timezone)
        else:
            return '%s%5.3f %s' % (datetime[:-1], float(datetime[-1]) + frac_secs, timezone)

    def to_string(self):
        """Convert timestamp to UTC string representation."""
        int_secs = np.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int_secs))
        if frac_secs == 0.0:
            return datetime
        else:
            return '%s%5.3f' % (datetime[:-1], float(datetime[-1]) + frac_secs)

    def to_ephem_date(self):
        """Convert timestamp to :class:`ephem.Date` object."""
        timetuple = list(time.gmtime(np.floor(self.secs))[:6])
        timetuple[5] += self.secs - np.floor(self.secs)
        return ephem.Date(tuple(timetuple))

#--------------------------------------------------------------------------------------------------
#--- CLASS :  StationaryBody
#--------------------------------------------------------------------------------------------------

class StationaryBody(object):
    """Stationary body with fixed (az, el) coordinates.

    This is a simplified :class:`ephem.Body` that is useful to specify targets
    such as zenith and geostationary satellites.

    Parameters
    ----------
    az, el : string or float
        Azimuth and elevation, either in 'D:M:S' string format, or float in rads
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

    This converts a position on the Earth specified in geodetic latitude,
    longitude and altitude to earth-centered, earth-fixed (ECEF) cartesian
    coordinates. This code assumes the WGS84 earth model, described in [1]_.

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
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    .. [1] National Imagery and Mapping Agency, "Department of Defense World
       Geodetic System 1984," NIMA TR8350.2, Page 4-4, last updated June, 2004.

    """
    # pylint: disable-msg=C0103
    # WGS84 Defining Parameters
    a = 6378137.0                           # semi-major axis of Earth in m
    f = 1.0 / 298.257223563                 # flattening of Earth

    # WGS84 derived geometric constants
    e2 = 2 * f - f ** 2                     # first eccentricity squared

    # intermediate calculation
    # (normal, or prime vertical radius of curvature)
    R = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)

    x_m = (R + alt_m) * np.cos(lat_rad) * np.cos(long_rad)
    y_m = (R + alt_m) * np.cos(lat_rad) * np.sin(long_rad)
    z_m = ((1.0 - e2) * R + alt_m) * np.sin(lat_rad)

    return x_m, y_m, z_m

def ecef_to_lla(x_m, y_m, z_m):
    """Convert ECEF cartesian coordinates to WGS84 spherical coordinates.

    This converts an earth-centered, earth-fixed (ECEF) cartesian position to a
    position on the Earth specified in geodetic latitude, longitude and altitude.
    This code assumes the WGS84 earth model.

    Parameters
    ----------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    long_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid

    Notes
    -----
    Based on the most accurate algorithm according to Zhu [zhu]_, which is
    summarised by Kaplan [kaplan]_ and described in the Wikipedia entry [geo]_.

    .. [zhu] J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to
       geodetic coordinates," Aerospace and Electronic Systems, IEEE Transactions
       on, vol. 30, pp. 957-961, 1994.
    .. [kaplan] Kaplan, "Understanding GPS: principles and applications," 1 ed.,
       Norwood, MA 02062, USA: Artech House, Inc, 1996.
    .. [geo] Wikipedia entry, "Geodetic system", 2009.

    """
    # pylint: disable-msg=C0103
    # WGS84 Defining Parameters
    a = 6378137.0                           # semi-major axis of Earth in m
    f = 1.0 / 298.257223563                 # flattening of Earth

    # WGS84 derived geometric constants
    b = a * (1.0 - f)                       # semi-minor axis in m
    e2 = 2 * f - f ** 2                     # first eccentricity squared
    ep2 = f * (2.0 - f) / (1.0 - f) ** 2    # second eccentricity squared

    # Define squared terms for convenience
    a2, b2 = a ** 2, b ** 2
    x2, y2, z2 = x_m ** 2, y_m ** 2, z_m ** 2

    r = np.sqrt(x2 + y2)
    E2 = a2 - b2
    F = 54.0 * b2 * z2
    G = r ** 2 + (1 - e2) * z2 - e2 * E2
    C = (e2 ** 2 * F * r ** 2) / (G ** 3)
    S = (1.0 + C + np.sqrt(C ** 2 + 2 * C)) ** (1. / 3.)
    P = F / (3.0 * (S + 1.0 / S + 1.0) ** 2 * G ** 2)
    Q = np.sqrt(1.0 + 2.0 * e2 ** 2 * P)
    r0 = - P * e2 * r / (1.0 + Q) + \
         np.sqrt(0.5 * a2 * (1.0 + 1.0 / Q) - P * (1 - e2) * z2 / (Q * (1.0 + Q)) - 0.5 * P * r ** 2)
    U = np.sqrt((r - e2 * r0) ** 2 + z2)
    V = np.sqrt((r - e2 * r0) ** 2 + (1.0 - e2) * z2)
    z0 = (b2 * z_m) / (a * V)
    alt_m = U * (1.0 - b2 / (a * V))
    lat_rad = np.arctan2(z_m + ep2 * z0, r)
    long_rad = np.arctan2(y_m, x_m)

    return lat_rad, long_rad, alt_m

def ecef_to_lla2(x_m, y_m, z_m):
    """Convert ECEF cartesian coordinates to WGS84 spherical coordinates.

    This converts an earth-centered, earth-fixed (ECEF) cartesian position to a
    position on the Earth specified in geodetic latitude, longitude and altitude.
    This code assumes the WGS84 earth model.

    Parameters
    ----------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    lat_rad : float or array
        Latitude (customary geodetic, not geocentric), in radians
    long_rad : float or array
        Longitude, in radians
    alt_m : float or array
        Altitude, in metres above WGS84 ellipsoid

    Notes
    -----
    This is a copy of the algorithm in the CONRAD codebase (from conradmisclib).
    It's nearly identical to :func:`ecef_to_lla`, but returns long/lat in
    different ranges.

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
    coordinates. The reference position is specified by its geodetic latitude,
    longitude and altitude.

    Parameters
    ----------
    ref_lat_rad, ref_long_rad : float or array
        Geodetic latitude and longitude of reference position, in radians
    ref_alt_m : float or array
        Geodetic altitude of reference position, in metres above WGS84 ellipsoid
    e_m, n_m, u_m : float or array
        East, North, Up coordinates, in metres

    Returns
    -------
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    """
    # ECEF coordinates of reference point
    ref_x_m, ref_y_m, ref_z_m = lla_to_ecef(ref_lat_rad, ref_long_rad, ref_alt_m)
    sin_lat, cos_lat = np.sin(ref_lat_rad), np.cos(ref_lat_rad)
    sin_long, cos_long = np.sin(ref_long_rad), np.cos(ref_long_rad)

    x_m = ref_x_m - sin_long*e_m - sin_lat*cos_long*n_m + cos_lat*cos_long*u_m
    y_m = ref_y_m + cos_long*e_m - sin_lat*sin_long*n_m + cos_lat*sin_long*u_m
    z_m = ref_z_m +                         cos_lat*n_m +          sin_lat*u_m

    return x_m, y_m, z_m

def ecef_to_enu(ref_lat_rad, ref_long_rad, ref_alt_m, x_m, y_m, z_m):
    """Convert ECEF coordinates to ENU coordinates relative to reference location.

    This converts earth-centered, earth-fixed (ECEF) cartesian coordinates to
    local east-north-up (ENU) coordinates relative to a given reference position.
    The reference position is specified by its geodetic latitude, longitude and
    altitude.

    Parameters
    ----------
    ref_lat_rad, ref_long_rad : float or array
        Geodetic latitude and longitude of reference position, in radians
    ref_alt_m : float or array
        Geodetic altitude of reference position, in metres above WGS84 ellipsoid
    x_m, y_m, z_m : float or array
        X, Y, Z coordinates, in metres

    Returns
    -------
    e_m, n_m, u_m : float or array
        East, North, Up coordinates, in metres

    """
    # ECEF coordinates of reference point
    ref_x_m, ref_y_m, ref_z_m = lla_to_ecef(ref_lat_rad, ref_long_rad, ref_alt_m)
    delta_x_m, delta_y_m, delta_z_m = x_m - ref_x_m, y_m - ref_y_m, z_m - ref_z_m
    sin_lat, cos_lat = np.sin(ref_lat_rad), np.cos(ref_lat_rad)
    sin_long, cos_long = np.sin(ref_long_rad), np.cos(ref_long_rad)

    e_m =         -sin_long*delta_x_m +         cos_long*delta_y_m
    n_m = -sin_lat*cos_long*delta_x_m - sin_lat*sin_long*delta_y_m + cos_lat*delta_z_m
    u_m =  cos_lat*cos_long*delta_x_m + cos_lat*sin_long*delta_y_m + sin_lat*delta_z_m

    return e_m, n_m, u_m

#--------------------------------------------------------------------------------------------------
#--- Spherical coordinate transformations
#--------------------------------------------------------------------------------------------------

def azel_to_enu(az_rad, el_rad):
    """Convert (az, el) spherical coordinates to unit vector in ENU coordinates.

    This converts horizontal spherical coordinates (azimuth and elevation angle)
    to a unit vector in the corresponding local east-north-up (ENU) coordinate
    system.

    Parameters
    ----------
    az_rad, el_rad : float or array
        Azimuth and elevation angle, in radians

    Returns
    -------
    e, n, u : float or array
        East, North, Up coordinates of unit vector

    """
    sin_az, cos_az = np.sin(az_rad), np.cos(az_rad)
    sin_el, cos_el = np.sin(el_rad), np.cos(el_rad)
    return sin_az * cos_el, cos_az * cos_el, sin_el

def hadec_to_enu(ha_rad, dec_rad, lat_rad):
    """Convert (ha, dec) spherical coordinates to unit vector in ENU coordinates.

    This converts equatorial spherical coordinates (hour angle and declination)
    to a unit vector in the corresponding local east-north-up (ENU) coordinate
    system. The geodetic latitude of the observer is also required.

    Parameters
    ----------
    ha_rad, dec_rad, lat_rad : float or array
        Hour angle, declination and geodetic latitude, in radians

    Returns
    -------
    e, n, u : float or array
        East, North, Up coordinates of unit vector

    """
    sin_ha, cos_ha = np.sin(ha_rad), np.cos(ha_rad)
    sin_dec, cos_dec = np.sin(dec_rad), np.cos(dec_rad)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    return -cos_dec * sin_ha, \
            cos_lat * sin_dec - sin_lat * cos_dec * cos_ha, \
            sin_lat * sin_dec + cos_lat * cos_dec * cos_ha

def enu_to_xyz(e, n, u, lat_rad):
    """Convert ENU to XYZ coordinates.

    This converts a vector in the local east-north-up (ENU) coordinate system to
    the XYZ coordinate system used in radio astronomy (see e.g. [TMS]_). The X
    axis is the intersection of the equatorial plane and the meridian plane
    through the reference point of the ENU system (and therefore is similar to
    'up'). The Y axis also lies in the equatorial plane to the east of X, and
    coincides with 'east'. The Z axis points toward the north pole, and therefore
    is similar to 'north'. The XYZ system is therefore a local version of the
    Earth-centred Earth-fixed (ECEF) system.

    Parameters
    ----------
    e, n, u : float or array
        East, North, Up coordinates of input vector
    lat_rad : float or array
        Geodetic latitude of ENU / XYZ reference point, in radians

    Returns
    -------
    x, y, z : float or array
        X, Y, Z coordinates of output vector

    References
    ----------
    .. [TMS] Thompson, Moran, Swenson, "Interferometry and Synthesis in Radio
       Astronomy," 2nd ed., Wiley-VCH, 2004, pp. 86-89.

    """
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    return -sin_lat * n + cos_lat * u, e, cos_lat * n + sin_lat * u
