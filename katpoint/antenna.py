"""Antenna object wrapping its location and dish diameter."""

import numpy as np
import ephem

from .ephem_extra import Timestamp, is_iterable, enu_to_ecef, ecef_to_lla, lla_to_ecef, ecef_to_enu

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Antenna
#--------------------------------------------------------------------------------------------------

class Antenna(object):
    """An antenna that can point at a target.

    This is a wrapper around a PyEphem :class:`ephem.Observer` that adds a dish
    diameter. It has two variants: a stand-alone single dish, or an antenna
    that is part of an array. The first variant is initialised with the antenna
    location in WGS84 (lat-long-alt) form, while the second variant is
    initialised with the array reference location in WGS84 form and an ENU
    (east-north-up) offset for the specific antenna.

    Parameters
    ----------
    name : string
        Name of antenna
    latitude : string or float
        Geodetic latitude, either in 'D:M:S' string format or a float in radians
    longitude : string or float
        Longitude, either in 'D:M:S' string format or a float in radians
    altitude : string or float
        Altitude above WGS84 geoid, in meters
    diameter : string or float
        Dish diameter, in meters
    offset : sequence of 3 strings or floats, optional
        East-North-Up offset from WGS84 reference position, in meters

    Arguments
    ---------
    description : string
        Description string of antenna, used to reconstruct the object
    position_enu : tuple of 3 floats
        East-North-Up offset from WGS84 reference position, in meters
    position_wgs84 : tuple of 3 floats
        WGS84 position of antenna (latitude and longitude in radians, and altitude
        in meters)
    position_ecef : tuple of 3 floats
        ECEF (Earth-centred Earth-fixed) position of antenna (in meters)
    ref_position_wgs84 : tuple of 3 floats
        WGS84 reference position (latitude and longitude in radians, and altitude
        in meters)
    observer : :class:`ephem.Observer` object
        Underlying object used for pointing calculations
    ref_observer : :class:`ephem.Observer` object
        Array reference location for antenna in an array (same as *observer* for
        a stand-alone antenna)

    Notes
    -----
    The :class:`ephem.Observer` objects are abused for their ability to convert
    latitude and longitude to and from string representations via
    :class:`ephem.Angle`. The only reason for the existence of *ref_observer* is
    that it is a nice container for the reference latitude, longitude and altitude.

    It is a bad idea to edit the coordinates of the antenna in-place, as the
    various position tuples and the description string will not be updated -
    reconstruct a new antenna object instead.

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
        self.ref_position_wgs84 = self.ref_observer.lat, self.ref_observer.long, self.ref_observer.elevation
        if offset is not None:
            self.position_enu = tuple([float(off) for off in offset])
            # Convert ENU offset to ECEF coordinates of antenna, and then to WGS84 coordinates
            self.position_ecef = enu_to_ecef(self.ref_observer.lat, self.ref_observer.long,
                                             self.ref_observer.elevation, *self.position_enu)
            self.observer = ephem.Observer()
            self.observer.lat, self.observer.long, self.observer.elevation = ecef_to_lla(*self.position_ecef)
            self.observer.epoch = ephem.J2000
            self.observer.pressure = 0.0
            self.position_wgs84 = self.observer.lat, self.observer.long, self.observer.elevation
            self.description = "%s, %s, %s, %s, %s, %s, %s, %s" % tuple([self.name] + list(self.ref_position_wgs84) +
                                                                        [self.diameter] + list(self.position_enu))
        else:
            self.observer = self.ref_observer
            self.position_enu = (0.0, 0.0, 0.0)
            self.position_wgs84 = lat, lon, alt = self.observer.lat, self.observer.long, self.observer.elevation
            self.position_ecef = enu_to_ecef(lat, lon, alt, *self.position_enu)
            self.description = "%s, %s, %s, %s, %s" % tuple([self.name] + list(self.position_wgs84) + [self.diameter])

    def __str__(self):
        """Verbose human-friendly string representation of antenna object."""
        if np.any(self.position_enu):
            return "%s: %d-m dish at ENU offset %s m from lat %s, long %s, alt %s m" % \
                   tuple([self.name, self.diameter, np.array(self.position_enu)] + list(self.ref_position_wgs84))
        else:
            return "%s: %d-m dish at lat %s, long %s, alt %s m" % \
                   tuple([self.name, self.diameter] + list(self.position_wgs84))

    def __repr__(self):
        """Short human-friendly string representation of antenna object."""
        return "<katpoint.Antenna '%s' diam=%sm at 0x%x>" % (self.name, self.diameter, id(self))

    def baseline_toward(self, antenna2):
        """Baseline vector pointing toward second antenna, in ENU coordinates.

        This calculates the baseline vector pointing from this antenna toward a
        second antenna, *antenna2*, in local East-North-Up (ENU) coordinates
        relative to this antenna's geodetic location.

        Parameters
        ----------
        antenna2 : :class:`Antenna` object
            Second antenna of baseline pair (baseline vector points toward it)

        Returns
        -------
        e_m, n_m, u_m : float or array
            East, North, Up coordinates of baseline vector, in metres

        """
        # If this antenna is at reference position of second antenna, simply return its ENU offset
        if self.position_wgs84 == antenna2.ref_position_wgs84:
            return antenna2.position_enu
        else:
            lat, lon, alt = self.position_wgs84
            return ecef_to_enu(lat, lon, alt, *lla_to_ecef(*antenna2.position_wgs84))

    def local_sidereal_time(self, timestamp=None):
        """Calculate local sidereal time at antenna for timestamp(s).

        This is a vectorised function that returns the local sidereal time at
        the antenna for a given UTC timestamp.

        Parameters
        ----------
        timestamp : :class:`Timestamp` object or equivalent, or sequence, optional
            Timestamp(s) in UTC seconds since Unix epoch (defaults to now)

        Returns
        -------
        lst : :class:`ephem.Angle` object, or sequence of objects
            Local sidereal time(s), in radians

        """
        def _scalar_local_sidereal_time(t):
            """Calculate local sidereal time at a single time instant."""
            self.observer.date = Timestamp(t).to_ephem_date()
            # pylint: disable-msg=E1101
            return self.observer.sidereal_time()
        if is_iterable(timestamp):
            return np.array([_scalar_local_sidereal_time(t) for t in timestamp])
        else:
            return _scalar_local_sidereal_time(timestamp)

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
