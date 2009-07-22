"""Antenna object wrapping its location and dish diameter."""

import numpy as np
import ephem

from .ephem_extra import Timestamp, is_iterable, enu_to_ecef, ecef_to_lla

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Antenna
#--------------------------------------------------------------------------------------------------

class Antenna(object):
    """An antenna that can point at a target.
    
    This is a wrapper around a PyEphem :class:`ephem.Observer` that adds a dish
    diameter. It has two variants: a stand-alone single dish, or an antenna
    that is part of an array. The first variant is initialised with the antenna
    location in lat-long-alt form, while the second variant is initialised with
    the array reference location in lat-long-alt form and an ENU (east-north-up)
    offset for the specific antenna.
    
    Parameters
    ----------
    name : string
        Name of antenna
    latitude : string or float
        Latitude, either in 'D:M:S' string format, or as a float in radians
    longitude : string or float
        Longitude, either in 'D:M:S' string format, or as a float in radians
    altitude : string or float
        Altitude, in meters
    diameter : string or float
        Dish diameter, in meters
    offset : sequence of 3 strings or floats, optional
        East-North-Up offset from lat-long-alt reference position, in meters
    
    Arguments
    ---------
    observer : :class:`ephem.Observer` object
        Underlying object used for pointing calculations
    ref_observer : :class:`ephem.Observer` object
        Array reference location for antenna in an array (same as *observer* for
        a stand-alone antenna)
    
    """
    def __init__(self, name, latitude, longitude, altitude, diameter, offset=None):
        self.name = name
        self.diameter = float(diameter)
        self.ref_observer = ephem.Observer()
        self.ref_observer.lat = latitude
        self.ref_observer.long = longitude
        self.ref_observer.elevation = float(altitude)
        # All astrometric ra/dec coordinates will be in J2000 epoch
        self.ref_observer.epoch = ephem.J2000
        # Disable ephem's built-in refraction model, since it's for optical wavelengths
        self.ref_observer.pressure = 0.0
        if offset:
            self.offset = [float(off) for off in offset]
            self.observer = ephem.Observer()
            # Convert ENU offset to lat-long-alt position required by Observer
            lat, lon, alt = ecef_to_lla(*enu_to_ecef(self.ref_observer.lat, self.ref_observer.long,
                                                     self.ref_observer.elevation, *self.offset))
            self.observer.lat = lat
            self.observer.long = lon
            self.observer.elevation = alt
            self.observer.epoch = ephem.J2000
            self.observer.pressure = 0.0
        else:
            self.offset = None
            self.observer = self.ref_observer
    
    def __str__(self):
        """Verbose human-friendly string representation of antenna object."""
        if self.offset:
            return "%s: %d-m dish at ENU offset %s m from lat %s, long %s, alt %s m" % \
                   (self.name, self.diameter, self.offset,
                    self.ref_observer.lat, self.ref_observer.long, self.ref_observer.elevation)
        else:
            return "%s: %d-m dish at lat %s, long %s, alt %s m" % (self.name, self.diameter,
                   self.observer.lat, self.observer.long, self.observer.elevation)
    
    def __repr__(self):
        """Short human-friendly string representation of antenna object."""
        return "<katpoint.Antenna '%s' diam=%sm at 0x%x>" % (self.name, self.diameter, id(self))
    
    # Provide description string as a read-only property, which is more compact than a method
    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0212
    def description():
        """Class method which creates description property."""
        doc = 'Complete string representation of antenna object, sufficient to reconstruct it.'
        def fget(self):
            if self.offset:
                return "%s, %s, %s, %s, %s, %s, %s, %s" % (self.name, self.ref_observer.lat,
                       self.ref_observer.long, self.ref_observer.elevation, self.diameter,
                       self.offset[0], self.offset[1], self.offset[2])
            else:    
                return "%s, %s, %s, %s, %s" % (self.name, self.observer.lat,
                       self.observer.long, self.observer.elevation, self.diameter)
        
        return locals()
    description = property(**description())
    
    def sidereal_time(self, timestamp=None):
        """Calculate sidereal time for local timestamp(s).
        
        This is a vectorised function that returns the local sidereal time at
        the antenna for a given timestamp in seconds-since-Unix-epoch.
        
        Parameters
        ----------
        timestamp : float or string or sequence, optional
            UTC timestamp(s) in seconds since Unix epoch, or string date/time
            (defaults to now)
        
        Returns
        -------
        lst : :class:`ephem.Angle` object, or sequence of objects
            Local sidereal time(s)
        
        """
        def _scalar_sidereal_time(t):
            """Calculate sidereal time at a single time instant."""
            self.observer.date = Timestamp(t).to_ephem_date()
            # pylint: disable-msg=E1101
            return self.observer.sidereal_time()
        if is_iterable(timestamp):
            return np.array([_scalar_sidereal_time(t) for t in timestamp])
        else:
            return _scalar_sidereal_time(timestamp)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  construct_antenna
#--------------------------------------------------------------------------------------------------

def construct_antenna(description):
    """Construct Antenna object from string representation.
    
    The description string contains a number of comma-separated fields, in one
    of the following formats:
    
    - name, latitude (D:M:S), longitude (D:M:S), altitude (m), diameter (m)
    - name, latitude (D:M:S), longitude (D:M:S), altitude (m), diameter (m),
      east offset (m), north offset (m), up offset (m)
    
    The first format is meant for stand-alone dishes, while the second format
    is useful for antennas that form part of an array. In the latter case the
    lat-long-alt location is the array reference location, with the antenna
    location specified as an ENU offset.
    
    Parameters
    ----------
    description : string
        String containing antenna name, location and dish diameter
    
    Returns
    -------
    ant : :class:`Antenna` object
        Constructed Antenna object
    
    Raises
    ------
    ValueError
        If *description* has the wrong format
    
    """
    fields = [s.strip() for s in description.split(',')]
    if not len(fields) in [5, 8]:
        raise ValueError("Antenna description string '%s' has wrong number of fields" % description)
    if len(fields) == 8:
        fields = fields[:5] + [fields[5:]]
    return Antenna(*fields)
