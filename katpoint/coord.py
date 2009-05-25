"""Wrapper for underlying coordinate library.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Observer class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

import time
import csv
import os.path
import re

import numpy as np
import ephem

import projection

#--------------------------------------------------------------------------------------------------
#--- Helper functions
#--------------------------------------------------------------------------------------------------

def rad2deg(x):
    return x * 180.0 / np.pi

def deg2rad(x):
    return x * np.pi / 180.0

def _unix_to_ephem_time(secs_since_epoch):
    """Convert seconds-since-Unix-epoch local time to a UTC PyEphem Date object."""
    timestamp = list(time.gmtime(np.floor(secs_since_epoch))[:6])
    timestamp[5] += secs_since_epoch - np.floor(secs_since_epoch)
    return ephem.Date(tuple(timestamp))

#--------------------------------------------------------------------------------------------------
#--- Antenna
#--------------------------------------------------------------------------------------------------

class Antenna(object):
    """Class describing the antenna doing the scanning.
    
    Parameters
    ----------
    name : string
        Name of antenna
    lat : string or float
        Latitude, either in 'D:M:S' string format, or as a float in radians
    long : string or float
        Longitude, either in 'D:M:S' string format, or as a float in radians
    alt : string or float
        Altitude, in meters
    diam : string or float
        Dish diameter, in meters
    
    """
    def __init__(self, name, lat, long, alt, diam):
        self.name = name
        self.observer = ephem.Observer()
        self.observer.lat = lat
        self.observer.long = long
        self.observer.elevation = float(alt)
        self.diam = float(diam)
    
    def __str__(self):
        return "%s: %d-m dish at %s %s, elevation %f m" % (self.name, self.diam,
               self.observer.lat, self.observer.long, self.observer.elevation)
    
    def sidereal_time(self, secs_since_epoch):
        """Calculate sidereal time for local timestamp(s)."""
        def _scalar_sidereal_time(t):
            """Calculate sidereal time at a single time instant."""
            self.observer.date = unix_to_ephem_time(t)
            return self.observer.sidereal_time()
        if np.isscalar(secs_since_epoch):
            return _scalar_sidereal_time(secs_since_epoch)
        else:
            return np.array([_scalar_sidereal_time(t) for t in secs_since_epoch])

# Dict used to look up antennas by name -> de facto catalogue
antenna_catalogue = {}

def add_to_antenna_catalogue(filename):
    """Add contents of antenna CSV file to existing catalogue of antennas.
    
    The CSV file should have the following columns:
    <name>, <latitude>, <longitude>, <altitude>, <dish diameter>
    
    Parameters
    ----------
    filename : string
        Name of CSV file containing list of antennas
    
    """
    cat_file = file(filename)
    for row in csv.reader(cat_file, skipinitialspace=True):
        if (len(row) == 5) and (row[0][0] != '#'):
            antenna_catalogue[row[0]] = Antenna(*row)

#--------------------------------------------------------------------------------------------------
#--- Source
#--------------------------------------------------------------------------------------------------

class Source(object):
    """Class describing the radio source being scanned.
    
    Parameters
    ----------
    body : ephem.Body object
        Pre-constructed ephem.Body object to embed in source object
    min_freq_Hz : float
        Minimum frequency for which flux density estimate is valid, in Hz
    max_freq_Hz : float
        Maximum frequency for which flux density estimate is valid, in Hz
    coefs : sequence of floats
        Coefficients of Baars polynomial used to estimate flux density
    
    """
    def __init__(self, body, min_freq_Hz=None, max_freq_Hz=None, coefs=None):
        self.body = body
        self.name = self.body.name
        self.min_freq_Hz = min_freq_Hz
        self.max_freq_Hz = max_freq_Hz
        self.coefs = coefs
    
    def __str__(self):
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            return "%s: %s, no flux info" % (self.name, self.body.__class__.__name__)
        else:
            return "%s: %s, flux defined for %.3f - %.3f GHz" % \
                   (self.name, self.body.__class__.__name__,
                    self.min_freq_Hz * 1e-9, self.max_freq_Hz * 1e-9)
            
    def pointing(self, antenna, timestamps):
        """Calculate source (az, el) coordinates as seen from antenna at timestamp(s)."""
        def _scalar_pointing(t):
            """Calculate (az, el) coordinates for a single time instant."""
            antenna.observer.date = _unix_to_ephem_time(t)
            self.body.compute(antenna.observer)
            return self.body.az, self.body.alt
        if np.isscalar(timestamps):
            return _scalar_pointing(timestamps)
        else:
            azel = np.array([_scalar_pointing(t) for t in timestamps])
            return azel[:, 0], azel[:, 1]
    
    def flux_density_Jy(self, obs_freq_Hz):
        """Calculate flux density for given observation frequency.
        
        This uses a polynomial flux model of the form
        
        log10 S[Jy] = a + b*log10(f[MHz]) + c*(log10(f[MHz]))^2
        
        as used in Baars 1977.
        
        Parameters
        ----------
        obs_freq_Hz : float
            Frequency at which to evaluate flux density
        
        Returns
        -------
        flux_density_Jy : float
            Flux density in Jy, or None if frequency is out of range or source
            does not have flux info
        
        """
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            # Source has no specified flux density
            return None
        if (obs_freq_Hz < self.min_freq_Hz) or (obs_freq_Hz > self.max_freq_Hz):
            # Frequency out of range for flux calculation of source
            return None
        log10_freq = np.log10(obs_freq_Hz * 1e-6)
        log10_flux = 0.0
        acc = 1.0
        for coeff in self.coefs:
            log10_flux += float(coeff) * acc
            acc *= log10_freq
        return 10.0 ** log10_flux

class StationaryBody(object):
    """Stationary body with fixed (az, el) coordinates."""
    def __init__(self, az, el, name=None):
        self.az = ephem.degrees(az)
        self.el = ephem.degrees(el)
        self.alt = self.el # alternative terminology
        if name is None:
            name = "Az: %s El: %s" % (self.az, self.el)
        self.name = name

    def compute(self, observer):
        pass
         # az / el is fixed :)
    
# Dict used to look up sources by name -> de facto catalogue
source_catalogue = {}
# Add special PyEphem bodies, such as solar system objects
specials = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for name in specials:
    source_catalogue[name.lower()] = Source(eval('ephem.%s()' % name))
source_catalogue['zenith'] = Source(StationaryBody('0.0', '90.0', 'Zenith'))

def add_to_source_catalogue(filename):
    """Add contents of source CSV or TLE file to existing catalogue of sources.
    
    A source CSV file has the extension '.csv' and the following columns:
    <name>, <epoch>, <ra>, <dec>, <min_freq_MHz>, <max_freq_MHz>, <(coefs list)>
    
    A source TLE file has the extension '.tle' and contains two-line elements.
    
    Parameters
    ----------
    filename : string
        Name of CSV or TLE file containing list of sources
    
    """
    cat_file = file(filename)
    ext = os.path.splitext(filename)[1]
    if ext == '.csv':
        epoch_map = {"J2000": ephem.J2000, "B1950": ephem.B1950, "B1900": ephem.B1900}
        # Load FixedBody sources from CSV file
        for row in csv.reader(cat_file, skipinitialspace=True):
            if (len(row) == 7) and (row[0][0] != '#'):
                names, epoch, ra, dec, min_freq, max_freq, coefs = row
                names = names.split()
                body = ephem.FixedBody()
                body.name = names[0]
                body._epoch = epoch_map[epoch]
                body._ra = ra
                body._dec = dec
                coefs = tuple([float(num) for num in coefs.strip(' ()').split()])
                min_freq, max_freq = 1e6 * float(min_freq), 1e6 * float(max_freq)
                if (min_freq <= 0.0) and (max_freq <= 0.0):
                    source = Source(body)
                    for name in names:
                        source_catalogue[name.lower().replace(' ', '')] = source
                else:
                    source = Source(body, min_freq, max_freq, coefs)
                    for name in names:
                        source_catalogue[name.lower().replace(' ', '')] = source
    elif ext == '.tle':
        lines = cat_file.readlines()
        if len(lines) % 3 > 0:
            raise ValueError("Source TLE file is malformed: wrong number of lines")
        for n in range(0, len(lines), 3):
            tle = lines[n:n + 3]
            tle[0] = tle[0].strip()
            body = ephem.readtle(*tle)
            source_catalogue[body.name.lower().replace(' ', '')] = Source(body)
    else:
        raise ValueError('Unrecognised source file extension (need .csv or .tle)')

def construct_source(name):
    clean_name = name.strip().lower()
    # Look for fixed (ra, dec) sources
    match = re.match('ra: (.+) dec: (.+)', clean_name)
    if match:
        body = ephem.FixedBody()
        body.name = name
        body._epoch = ephem.J2000
        body._ra = ephem.hours(match.group(1))
        body._dec = ephem.degrees(match.group(2))
        return Source(body)
    # Look for stationary (az, el) sources
    match = re.match('az: (.+) el: (.+)', clean_name)
    if match:
        return Source(StationaryBody(match.group(1), match.group(2)))
    # Do a named source lookup
    try:
        return source_catalogue[clean_name.replace(' ', '')]
    except KeyError:
        raise KeyError("Unknown source '%s'" % name)

#--------------------------------------------------------------------------------------------------
#--- Projections
#--------------------------------------------------------------------------------------------------

def sphere_to_plane(source, antenna, az, el, timestamps, projection_type='ARC'):
    # The source (az, el) coordinates will serve as reference point on the sphere
    ref_az, ref_el = source.pointing(antenna, timestamps)
    return projection.sphere_to_plane[projection_type](ref_az, ref_el, az, el)

def plane_to_sphere(source, antenna, x, y, timestamps, projection_type='ARC'):
    # The source (az, el) coordinates will serve as reference point on the sphere
    ref_az, ref_el = source.pointing(antenna, timestamps)
    return projection.plane_to_sphere[projection_type](ref_az, ref_el, x, y)
