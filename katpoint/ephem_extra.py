"""Enhancements to PyEphem."""

import numpy as np
import ephem

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

def wrap_angle(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.

    """
    return (angle + 0.5 * period) % period - 0.5 * period

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
