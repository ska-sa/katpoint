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

"""Antenna object containing sufficient information to point at a target and correct delays.

An *antenna* is considered to be a steerable parabolic dish containing multiple
feeds. The :class:`Antenna` object wraps the antenna's location, dish diameter
and other parameters that affect pointing and delay calculations.

"""

import numpy as np
import ephem

from .timestamp import Timestamp
from .ephem_extra import is_iterable
from .conversion import enu_to_ecef, ecef_to_lla, lla_to_ecef, ecef_to_enu
from .pointing import PointingModel
from .delay import DelayModel

# --------------------------------------------------------------------------------------------------
# --- CLASS :  Antenna
# --------------------------------------------------------------------------------------------------


class Antenna(object):
    """An antenna that can point at a target.

    This is a wrapper around a PyEphem :class:`ephem.Observer` that adds a dish
    diameter and other parameters related to pointing and delay calculations.
    It has two variants: a stand-alone single dish, or an antenna that is part
    of an array. The first variant is initialised with the antenna location in
    WGS84 (lat-long-alt) form, while the second variant is initialised with the
    array reference location in WGS84 form and an ENU (east-north-up) offset
    for the specific antenna which also doubles as the first part of a broader
    delay model for the antenna.

    Additionally, a diameter, a pointing model and a beamwidth factor may be
    specified. These parameters are collected for convenience, and the pointing
    model is not applied by default when calculating pointing or delays.

    The Antenna object is typically passed around in string form, and is fully
    described by its *description string*, which has the following format::

     name, latitude (D:M:S), longitude (D:M:S), altitude (m), diameter (m),
     east-north-up offset (m) / delay model, pointing model, beamwidth

    A stand-alone dish has the antenna location as lat-long-alt and the ENU
    offset as an empty string, while an antenna that is part of an array has
    the array reference location as lat-long-alt and the ENU offset as a
    space-separated string of 3 numbers (followed by any additional delay model
    terms). The pointing model is a space-separated string of model parameters
    (or empty string if there is no pointing model). The beamwidth is a single
    floating-point number.

    Any empty fields at the end of the description string may be omitted, as
    they will be replaced by defaults. The first four fields are required.

    Here are some examples of description strings::

     - Single dish
       'XDM, -25:53:23.0, 27:41:03.0, 1406.1086, 15.0'

     - Simple array antenna
       'FF1, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 18.4 -8.7 0.0'

     - Fully-specified antenna
       'FF2, -30:43:17.3, 21:24:38.5, 1038.0, 12.0, 86.2 25.5 0.0, -0:06:39.6 0 0 0 0 0 0:09:48.9, 1.16'

    Parameters
    ----------
    name : string or :class:`Antenna` object
        Name of antenna, or full description string or existing antenna object
    latitude : string or float, optional
        Geodetic latitude, either in 'D:M:S' string format or float in radians
    longitude : string or float, optional
        Longitude, either in 'D:M:S' string format or a float in radians
    altitude : string or float, optional
        Altitude above WGS84 geoid, in metres
    diameter : string or float, optional
        Dish diameter, in metres
    delay_model : :class:`DelayModel` object or equivalent, optional
        Delay model for antenna, either as a direct object, a file-like object
        representing a parameter file, or a string or sequence of float params.
        The first three parameters form an East-North-Up offset from WGS84
        reference position, in metres.
    pointing_model : :class:`PointingModel` object or equivalent, optional
        Pointing model for antenna, either as a direct object, a file-like
        object representing a parameter file, or a string or sequence of
        float parameters from which the :class:`PointingModel` object can
        be instantiated
    beamwidth : string or float, optional
        Full width at half maximum (FWHM) average beamwidth, as a multiple of
        lambda / D (wavelength / dish diameter). This depends on the dish
        illumination pattern, and ranges from 1.03 for a uniformly illuminated
        circular dish to 1.22 for a Gaussian-tapered circular dish (the
        default).

    Arguments
    ---------
    position_enu : tuple of 3 floats
        East-North-Up offset from WGS84 reference position, in metres
    position_wgs84 : tuple of 3 floats
        WGS84 position of antenna (latitude and longitude in radians, and
        altitude in metres)
    position_ecef : tuple of 3 floats
        ECEF (Earth-centred Earth-fixed) position of antenna (in metres)
    ref_position_wgs84 : tuple of 3 floats
        WGS84 reference position (latitude and longitude in radians, and
        altitude in metres)
    observer : :class:`ephem.Observer` object
        Underlying object used for pointing calculations
    ref_observer : :class:`ephem.Observer` object
        Array reference location for antenna in an array (same as *observer*
        for a stand-alone antenna)

    Raises
    ------
    ValueError
        If description string has wrong format or parameters are incorrect

    Notes
    -----
    The :class:`ephem.Observer` objects are abused for their ability to convert
    latitude and longitude to and from string representations via
    :class:`ephem.Angle`. The only reason for the existence of *ref_observer*
    is that it is a nice container for the reference latitude, longitude and
    altitude.

    It is a bad idea to edit the coordinates of the antenna in-place, as the
    various position tuples will not be updated - reconstruct a new antenna
    object instead.

    Also note that the description string of the new Antenna could differ from
    the original description string if the original string had higher precision
    in its latitude and longitude coordinates than what ephem can handle
    internally. Generally the latitude and longitude should be specified up to
    0.1 arcsecond precision, while altitude should be in metres and East, North
    and Up offsets are generally specified up to millimetres.

    """
    def __init__(self, name, latitude=None, longitude=None, altitude=None,
                 diameter=0.0, delay_model=None, pointing_model=None,
                 beamwidth=1.22):
        if isinstance(name, Antenna):
            name = name.description
        if not name and latitude is None:
            raise ValueError('Empty antenna description string %r' % (name,))
        # The presence of a comma indicates that a description string is passed in - parse this string into parameters
        if name.find(',') >= 0:
            try:
                name.encode('ascii')
            except UnicodeError:
                raise ValueError("Antenna description string %r contains non-ASCII characters" % (name,))
            # Cannot have other parameters if description string is given - this is a safety check
            if not (latitude is None and longitude is None and altitude is None):
                raise ValueError("First parameter '%s' contains comma" % (name,) +
                                 'and is assumed to be description string - cannot have other parameters')
            # Split description string on commas
            fields = [s.strip() for s in name.split(',')]
            # Extract required fields
            if len(fields) < 4:
                raise ValueError("Antenna description string '%s' has less than four fields" % (name,))
            name, latitude, longitude, altitude = fields[:4]
            # Extract optional fields
            try:
                diameter = fields.pop(4)
                delay_model = fields.pop(4)
                pointing_model = fields.pop(4)
                beamwidth = fields.pop(4)
            except IndexError:
                pass

        self.name = name
        self.diameter = float(diameter)
        self.delay_model = DelayModel(delay_model)
        self.pointing_model = PointingModel(pointing_model)
        self.beamwidth = float(beamwidth)

        # Set up reference observer first
        self.ref_observer = ephem.Observer()
        self.ref_observer.lat = latitude
        self.ref_observer.long = longitude
        self.ref_observer.elevation = float(altitude)
        # All astrometric ra/dec coordinates will be in J2000 epoch
        self.ref_observer.epoch = ephem.J2000
        # Disable ephem's built-in refraction model, since it's for optical wavelengths
        self.ref_observer.pressure = 0.0
        self.ref_position_wgs84 = self.ref_observer.lat, self.ref_observer.long, self.ref_observer.elevation

        if self.delay_model:
            dm = self.delay_model
            self.position_enu = (dm['POS_E'], dm['POS_N'], dm['POS_U'])
            # Convert ENU offset to ECEF coordinates of antenna, and then to WGS84 coordinates
            self.position_ecef = enu_to_ecef(self.ref_observer.lat, self.ref_observer.long,
                                             self.ref_observer.elevation, *self.position_enu)
            self.observer = ephem.Observer()
            self.observer.lat, self.observer.long, self.observer.elevation = ecef_to_lla(*self.position_ecef)
            self.observer.epoch = ephem.J2000
            self.observer.pressure = 0.0
            self.position_wgs84 = self.observer.lat, self.observer.long, self.observer.elevation
        else:
            self.observer = self.ref_observer
            self.position_enu = (0.0, 0.0, 0.0)
            self.position_wgs84 = lat, lon, alt = self.observer.lat, self.observer.long, self.observer.elevation
            self.position_ecef = enu_to_ecef(lat, lon, alt, *self.position_enu)

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

    def __reduce__(self):
        """Custom pickling routine based on description string."""
        return (self.__class__, (self.description,))

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (other.description if isinstance(other, Antenna) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __lt__(self, other):
        """Less-than comparison operator (needed for sorting and np.unique)."""
        return self.description < (other.description if isinstance(other, Antenna) else other)

    def __hash__(self):
        """Base hash on description string, just like equality operator."""
        return hash(self.description)

    @property
    def description(self):
        """Complete string representation of antenna object, sufficient to reconstruct it."""
        # These fields are used to build up the antenna description string
        fields = [self.name]
        pos = self.ref_position_wgs84 if self.delay_model else self.position_wgs84
        fields += [str(coord) for coord in pos]
        fields += [str(self.diameter)]
        fields += [self.delay_model.description]
        fields += [self.pointing_model.description]
        fields += [str(self.beamwidth)]
        return ', '.join(fields)

    def format_katcp(self):
        """String representation if object is passed as parameter to KATCP command."""
        return self.description

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
