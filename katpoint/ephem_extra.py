"""Enhancements to PyEphem."""

import time

import numpy as np
import ephem
import math

#--------------------------------------------------------------------------------------------------
#--- Helper functions
#--------------------------------------------------------------------------------------------------

# The speed of light, in metres per second
lightspeed = ephem.c

def is_iterable(x):
    """Checks if object is iterable (but not a string or 0-dimensional array)."""
    return hasattr(x, '__iter__') and not (getattr(x, 'shape', None) == ())

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
            int_secs = math.floor(timestamp[5])
            frac_secs = timestamp[5] - int_secs
            timestamp[5] = int(int_secs)
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
        int_secs = math.floor(self.secs)
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
        int_secs = math.floor(self.secs)
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
        int_secs = math.floor(self.secs)
        timetuple = list(time.gmtime(int_secs)[:6])
        timetuple[5] += self.secs - int_secs
        return ephem.Date(tuple(timetuple))

    def to_mjd(self):
        """Convert timestamp to Modified Julian Day (MJD)."""
        # Ephem dates are in Dublin Julian Days
        djd = self.to_ephem_date()
        return djd + 2415020 - 2400000.5

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
#--- CLASS :  NullBody
#--------------------------------------------------------------------------------------------------

class NullBody(object):
    """Body with no position, used as a placeholder.

    This body has the expected methods of :class:`Body`, but always returns NaNs
    for all coordinates. It is intended for use as a placeholder when no proper
    target object is available, i.e. as a dummy target.

    """
    def __init__(self):
        self.name = 'Nothing'
        self.az = self.alt = self.el = np.nan
        self.ra = self.dec = self.a_ra = self.a_dec = np.nan

    def compute(self, observer):
        pass
