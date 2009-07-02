"""Target object used for pointing and flux density calculation."""

import time

import numpy as np
import ephem

from .ephem_extra import unix_to_ephem_time, StationaryBody
from .projection import sphere_to_plane, plane_to_sphere

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Target
#--------------------------------------------------------------------------------------------------

class Target(object):
    """A target which can be pointed at by an antenna.
    
    This is a wrapper around a PyEphem :class:`ephem.Body` that adds flux
    density, alternate names and descriptive tags. For convenience, a default
    antenna and flux frequency can be set, to simplify the calling of pointing
    and flux density methods. These are not stored as part of the target object,
    however.
    
    Parameters
    ----------
    body : ephem.Body object
        Pre-constructed :class:`ephem.Body` object to embed in target object
    tags : list of strings
        Descriptive tags associated with target, starting with its body type
    aliases : list of strings, optional
        Alternate names of target
    min_freq_Hz : float, optional
        Minimum frequency for which flux density estimate is valid, in Hz
    max_freq_Hz : float, optional
        Maximum frequency for which flux density estimate is valid, in Hz
    coefs : sequence of floats, optional
        Coefficients of Baars polynomial used to estimate flux density
    antenna : :class:`Antenna` object, optional
        Default antenna to use for position calculations
    flux_freq_Hz : float, optional
        Default frequency at which to evaluate flux density, in Hz
        
    Arguments
    ---------
    name : string
        Name of target
    
    """
    def __init__(self, body, tags, aliases=None, min_freq_Hz=None, max_freq_Hz=None, coefs=None,
                 antenna=None, flux_freq_Hz=None):
        self.body = body
        self.name = self.body.name
        self.tags = tags
        if aliases is None:
            self.aliases = []
        else:
            self.aliases = aliases
        self.min_freq_Hz = min_freq_Hz
        self.max_freq_Hz = max_freq_Hz
        self.coefs = coefs
        self.antenna = antenna
        self.flux_freq_Hz = flux_freq_Hz
    
    def __str__(self):
        """Verbose human-friendly string representation of target object."""
        descr = str(self.name)
        if self.aliases:
            descr += ' (%s)' % (', '.join(self.aliases),)
        descr += ': [%s]' % (self.tags[0],)
        if self.tags[1:]:
            descr += ', tags=%s' % (','.join(self.tags[1:]),)
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            descr += ', no flux info'
        else:
            descr += ", flux defined for %.3f - %.3f GHz" % (self.min_freq_Hz * 1e-9, self.max_freq_Hz * 1e-9)
        return descr
    
    def __repr__(self):
        """Short human-friendly string representation of antenna object."""
        return "<katpoint.Target '%s' body=%s at 0x%x>" % (self.name, self.tags[0], id(self))
    
    def _set_timestamp_antenna_defaults(self, timestamp, antenna):
        """Set defaults for timestamp and antenna, if they are unspecified.
        
        If *timestamp* is None, it is replaced by the current time. If *antenna*
        is None, it is replaced by the default antenna for the target. 
        
        Parameters
        ----------
        timestamp : float or sequence, or None
            Local timestamp(s) in seconds since Unix epoch
        antenna : :class:`Antenna` object, or None
            Antenna which points at target
        
        Returns
        -------
        timestamp : float or sequence
            Local timestamp(s) in seconds since Unix epoch
        antenna : :class:`Antenna` object
            Antenna which points at target
        
        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either
        
        """
        if timestamp is None:
            timestamp = time.time()
        if antenna is None:
            antenna = self.antenna
        if antenna is None:
            raise ValueError('Antenna object needed to calculate target position')
        return timestamp, antenna
    
    def is_stationary(self):
        """Check if target is stationary, i.e. its (az, el) coordinates are fixed."""
        return self.tags[0].lower() == "azel"
    
    def get_description(self):
        """String representation of antenna object, sufficient to reconstruct it."""
        names = ' | '.join([self.name] + self.aliases)
        tags = ' '.join(self.tags)
        fluxinfo = None
        if self.min_freq_Hz and self.max_freq_Hz and self.coefs:
            fluxinfo = '(%s %s %s)' % (self.min_freq_Hz * 1e-6, self.max_freq_Hz * 1e-6,
                                       ' '.join([str(s) for s in self.coefs]))
        fields = [names, tags]
        body_type = self.tags[0].lower()
        if body_type == 'azel':
            # Check if it's an unnamed target with a default name
            if names.startswith('Az:'):
                fields = [tags]
            fields += [str(self.body.az), str(self.body.el)]
        
        elif body_type == 'radec':
            # Check if it's an unnamed target with a default name
            if names.startswith('Ra:'):
                fields = [tags]
            # pylint: disable-msg=W0212
            fields += [str(self.body._ra), str(self.body._dec)]
            if fluxinfo:
                fields += [fluxinfo]
        
        elif body_type == 'tle':
            # Switch body type to xephem, as XEphem only saves bodies in xephem edb format (no TLE output)
            tags = tags.replace(tags.partition(' ')[0], 'xephem')
            edb_string = self.body.writedb().replace(',', '~')
            # Suppress name if it's the same as in the xephem db string
            edb_name = edb_string[:edb_string.index('~')]
            if edb_name == names:
                fields = [tags, edb_string]
            else:
                fields = [names, tags, edb_string]
        
        elif body_type == 'xephem':
            edb_string = self.body.writedb().replace(',', '~')
            # Suppress name if it's the same as in the xephem db string
            edb_name = edb_string[:edb_string.index('~')]
            if edb_name == names:
                fields = [tags]
            fields += [edb_string]
        
        return ', '.join(fields)
    
    def add_tags(self, tags):
        """Add tags to target object.
        
        This is a convenience function to add extra tags to a target, while
        checking the sanity of the tags. It also prevents duplicate tags without
        resorting to a tag set, which would be problematic since the tag order
        is meaningful (tags[0] is the body type).
        
        Parameters
        ----------
        tags : string, list of strings, or None
            Tag or list of tags to add
        
        Returns
        -------
        target : :class:`Target` object
            Updated target object
        
        """
        if tags is None:
            tags = []
        if isinstance(tags, basestring):
            tags = [tags]
        self.tags.extend([tag for tag in tags if not tag in self.tags])
        return self
    
    def azel(self, timestamp=None, antenna=None):
        """Calculate target (az, el) coordinates as seen from antenna at time(s).
        
        Parameters
        ----------
        timestamp : float or sequence, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)
        
        Returns
        -------
        az : :class:`ephem.Angle` object, or sequence of objects
            Azimuth angle(s), in radians
        el : :class:`ephem.Angle` object, or sequence of objects
            Elevation angle(s), in radians
        
        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either
        
        """
        timestamp, antenna = self._set_timestamp_antenna_defaults(timestamp, antenna)
        def _scalar_azel(t):
            """Calculate (az, el) coordinates for a single time instant."""
            antenna.observer.date = unix_to_ephem_time(t)
            self.body.compute(antenna.observer)
            return self.body.az, self.body.alt
        if np.isscalar(timestamp):
            return _scalar_azel(timestamp)
        else:
            azel = np.array([_scalar_azel(t) for t in timestamp])
            return azel[:, 0], azel[:, 1]
    
    def radec(self, timestamp=None, antenna=None):
        """Calculate target's apparent (ra, dec) coordinates as seen from antenna at time(s).
        
        This calculates the *apparent topocentric position* of the target for
        the epoch-of-date in equatorial coordinates. Take note that this is
        *not* the "star-atlas" position of the target, but the position as seen
        from the antenna at the given times. The difference is on the order of a
        few arcminutes.
        
        Parameters
        ----------
        timestamp : float or sequence, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)
        
        Returns
        -------
        ra : :class:`ephem.Angle` object, or sequence of objects
            Right ascension, in radians
        dec : :class:`ephem.Angle` object, or sequence of objects
            Declination, in radians
        
        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either
        
        """
        timestamp, antenna = self._set_timestamp_antenna_defaults(timestamp, antenna)
        def _scalar_radec(t):
            """Calculate (ra, dec) coordinates for a single time instant."""
            antenna.observer.date = unix_to_ephem_time(t)
            self.body.compute(antenna.observer)
            return self.body.ra, self.body.dec
        if np.isscalar(timestamp):
            return _scalar_radec(timestamp)
        else:
            radec = np.array([_scalar_radec(t) for t in timestamp])
            return radec[:, 0], radec[:, 1]
    
    def astrometric_radec(self, timestamp=None, antenna=None):
        """Calculate target's astrometric (ra, dec) coordinates as seen from antenna at time(s).
        
        This calculates the *astrometric geocentric position* for the star atlas
        epoch contained in the target, in equatorial coordinates. Some targets
        are unable to provide this, notably stationary (*azel*) targets, which
        provide the *apparent topocentric position* instead. The difference is
        on the order of a few arcminutes.
        
        Parameters
        ----------
        timestamp : float or sequence, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)
        
        Returns
        -------
        ra : :class:`ephem.Angle` object, or sequence of objects
            Right ascension, in radians
        dec : :class:`ephem.Angle` object, or sequence of objects
            Declination, in radians
        
        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either
        
        """
        timestamp, antenna = self._set_timestamp_antenna_defaults(timestamp, antenna)
        def _scalar_radec(t):
            """Calculate (ra, dec) coordinates for a single time instant."""
            antenna.observer.date = unix_to_ephem_time(t)
            self.body.compute(antenna.observer)
            return self.body.a_ra, self.body.a_dec
        if np.isscalar(timestamp):
            return _scalar_radec(timestamp)
        else:
            radec = np.array([_scalar_radec(t) for t in timestamp])
            return radec[:, 0], radec[:, 1]
    
    def az_increases(self, timestamp=None, antenna=None):
        """Check if azimuth of target increases with time at given timestamp.
        
        This will be true if the antenna has to turn clockwise to keep tracking
        the target at the given timestamp.
        
        Parameters
        ----------
        timestamp : float or sequence, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at target (defaults to default antenna)
        
        Returns
        -------
        az_increases : bool
            True if azimuth increases with time at the given timestamp
        
        Raises
        ------
        ValueError
            If no antenna is specified, and no default antenna was set either
        
        """
        timestamp, antenna = self._set_timestamp_antenna_defaults(timestamp, antenna)
        return self.azel(timestamp + 1.0, antenna)[0] > self.azel(timestamp, antenna)[0]
        
    def flux_density(self, flux_freq_Hz=None):
        """Calculate flux density for given observation frequency.
        
        This uses a polynomial flux model of the form::
            
            log10 S[Jy] = a + b*log10(f[MHz]) + c*(log10(f[MHz]))^2
        
        as used in Baars 1977. If the flux frequency is unspecified, the default
        value supplied to the target object during construction is used.
        
        Parameters
        ----------
        flux_freq_Hz : float, optional
            Frequency at which to evaluate flux density, in Hz
        
        Returns
        -------
        flux_density : float
            Flux density in Jy, or None if frequency is out of range or target
            does not have flux info
        
        Raises
        ------
        ValueError
            If no frequency is specified, and no default frequency was set either
        
        """
        if flux_freq_Hz is None:
            flux_freq_Hz = self.flux_freq_Hz
        if flux_freq_Hz is None:
            raise ValueError('Please specify frequency at which to measure flux density')
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            # Target has no specified flux density
            return None
        if (flux_freq_Hz < self.min_freq_Hz) or (flux_freq_Hz > self.max_freq_Hz):
            # Frequency out of range for flux calculation of target
            return None
        log10_freq = np.log10(flux_freq_Hz * 1e-6)
        log10_flux = 0.0
        acc = 1.0
        for coeff in self.coefs:
            log10_flux += float(coeff) * acc
            acc *= log10_freq
        return 10.0 ** log10_flux

    def separation(self, other_target, timestamp=None, antenna=None):
        """Angular separation between this target and another one.
        
        Parameters
        ----------
        other_target : :class:`Target` object
            The other target
        timestamp : float, optional
            Timestamp when separation is measured, in seconds since Unix epoch
            (defaults to now)
        antenna : class:`Antenna` object, optional
            Antenna that observes both targets, from where separation is measured
            (defaults to default antenna of this target)
        
        Returns
        -------
        separation : :class:`ephem.Angle` object
            Angular separation between the targets, in radians
        
        """
        timestamp, antenna = self._set_timestamp_antenna_defaults(timestamp, antenna)
        return ephem.separation(self.radec(timestamp, antenna), other_target.radec(timestamp, antenna))

    def sphere_to_plane(self, az, el, timestamp=None, antenna=None, projection_type='ARC', coord_system='azel'):
        """Project spherical coordinates to plane with target position as reference.
    
        This is a convenience function that projects spherical coordinates to a 
        plane with the target position as the origin of the plane. The function is
        vectorised and can operate on single or multiple timestamps, as well as
        single or multiple coordinate vectors. The spherical coordinates may be
        (az, el) or (ra, dec), and the projection type can also be specified.
    
        Parameters
        ----------
        az : float or array
            Azimuth or right ascension, in radians
        el : float or array
            Elevation or declination, in radians
        timestamp : float or array, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna pointing at target (defaults to default antenna)
        projection_type : {'ARC', 'SIN', 'TAN', 'STG'}, optional
            Type of spherical projection
        coord_system : {'azel', 'radec'}, optional
            Spherical coordinate system
    
        Returns
        -------
        x : float or array
            Azimuth-like coordinate(s) on plane, in radians
        y : float or array
            Elevation-like coordinate(s) on plane, in radians
    
        """
        if coord_system == 'radec':
            # The target (ra, dec) coordinates will serve as reference point on the sphere
            ref_ra, ref_dec = self.radec(timestamp, antenna)
            return sphere_to_plane[projection_type](ref_ra, ref_dec, az, el)
        else:
            # The target (az, el) coordinates will serve as reference point on the sphere
            ref_az, ref_el = self.azel(timestamp, antenna)
            return sphere_to_plane[projection_type](ref_az, ref_el, az, el)

    def plane_to_sphere(self, x, y, timestamp=None, antenna=None, projection_type='ARC', coord_system='azel'):
        """Deproject plane coordinates to sphere with target position as reference.
    
        This is a convenience function that deprojects plane coordinates to a
        sphere with the target position as the origin of the plane. The function is
        vectorised and can operate on single or multiple timestamps, as well as
        single or multiple coordinate vectors. The spherical coordinates may be
        (az, el) or (ra, dec), and the projection type can also be specified.
    
        Parameters
        ----------
        x : float or array
            Azimuth-like coordinate(s) on plane, in radians
        y : float or array
            Elevation-like coordinate(s) on plane, in radians
        timestamp : float or array, optional
            Local timestamp(s) in seconds since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna pointing at target (defaults to default antenna)
        projection_type : {'ARC', 'SIN', 'TAN', 'STG'}, optional
            Type of spherical projection
        coord_system : {'azel', 'radec'}, optional
            Spherical coordinate system
    
        Returns
        -------
        az : float or array
            Azimuth or right ascension, in radians
        el : float or array
            Elevation or declination, in radians
    
        """
        if coord_system == 'radec':
            # The target (ra, dec) coordinates will serve as reference point on the sphere
            ref_ra, ref_dec = self.radec(timestamp, antenna)
            return plane_to_sphere[projection_type](ref_ra, ref_dec, x, y)
        else:
            # The target (az, el) coordinates will serve as reference point on the sphere
            ref_az, ref_el = self.azel(timestamp, antenna)
            return plane_to_sphere[projection_type](ref_az, ref_el, x, y)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  construct_target
#--------------------------------------------------------------------------------------------------

def construct_target(description):
    """Construct Target object from string representation.
    
    The description string contains up to five comma-separated fields, with the
    format::
        
        <name list>, <tags>, <longitudinal>, <latitudinal>, <flux info>
    
    The <name list> contains a pipe-separated list of alternate names for the
    target, with the preferred name either indicated by a prepended asterisk or
    assumed to be the first name in the list. The names may contain spaces, and
    the list may be empty. The <tags> field contains a space-separated list of
    descriptive tags for the target. The first tag is mandatory and indicates
    the body type of the target, which should be one of (*azel*, *radec*,
    *tle*, *special*, *star*, *xephem*). The longidutinal and latitudinal fields
    are only relevant to *azel* and *radec* targets, in which case they contain
    the relevant coordinates.
    
    The <flux info> is a space-separated list of numbers used to represent the
    flux density of the target. The first two numbers specify the frequency
    range for which the flux model is valid (in MHz), and the rest of the numbers
    are Baars polynomial coefficients. The <flux info> may be enclosed in
    parentheses to distinguish it from the other fields. An example string is::
        
        name1 | *name 2, radec cal, 12:34:56.7, -04:34:34.2, (1000.0 2000.0 1.0)
    
    For *special* and *star* body types, only the target name is required. The
    *special* body name is assumed to be a PyEphem class name, and is typically
    one of the major solar system objects. The *star* name is looked up in the
    PyEphem star database, which contains a modest list of bright stars.
    
    For *tle* bodies, the final field in the description string should contain
    the three lines of the TLE. If the name list is empty, the target name is
    taken from the TLE instead. The *xephem* body contains a string in XEphem
    EDB database format as the final field, with commas replaced by tildes. If
    the name list is empty, the target name is taken from the XEphem string
    instead.
    
    Parameters
    ----------
    description : string
        String containing target name(s), tags, location and flux info
    
    Returns
    -------
    target : :class:`Target` object
        Constructed Target object
    
    Raises
    ------
    ValueError
        If *description* has the wrong format
    
    """
    fields = [s.strip() for s in description.split(',')]
    if len(fields) < 2:
        raise ValueError("Target description '%s' must have at least two fields" % description)
    # Check if first name starts with body type tag, while the next field does not
    # This indicates a missing names field -> add an empty name list in front
    body_types = ['azel', 'radec', 'tle', 'special', 'star', 'xephem']
    if np.any([fields[0].startswith(s) for s in body_types]) and \
       not np.any([fields[1].startswith(s) for s in body_types]):
        fields = [''] + fields
    # Extract preferred name from name list (starred or first entry), and make the rest aliases
    names = [s.strip() for s in fields[0].split('|')]
    if len(names) == 0:
        preferred_name, aliases = '', []
    else:
        try:
            ind = [name.startswith('*') for name in names].index(True)
            preferred_name, aliases = names[ind][1:], names[:ind] + names[ind + 1:]
        except ValueError:
            preferred_name, aliases = names[0], names[1:]
    tags = [s.strip() for s in fields[1].split(' ')]
    if len(tags) == 0:
        raise ValueError("Target description '%s' needs at least one tag (body type)" % description)
    body_type = tags[0].lower()
    # Remove empty fields starting from the end (useful when parsing CSV files with fixed number of fields)
    while len(fields[-1]) == 0:
        fields.pop()
    
    # Create appropriate PyEphem body based on body type
    if body_type == 'azel':
        if len(fields) < 4:
            raise ValueError("Target description '%s' contains *azel* body with no (az, el) coordinates"
                             % description)
        body = StationaryBody(fields[2], fields[3], preferred_name)
    
    elif body_type == 'radec':
        if len(fields) < 4:
            raise ValueError("Target description '%s' contains *radec* body with no (ra, dec) coordinates"
                             % description)
        body = ephem.FixedBody()
        ra, dec = ephem.hours(fields[2]), ephem.degrees(fields[3])
        if preferred_name:
            body.name = preferred_name
        else:
            body.name = "Ra: %s Dec: %s" % (ra, dec)
        # Extract epoch info from tags
        if ('B1900' in tags) or ('b1900' in tags):
            body._epoch = ephem.B1900
        elif ('B1950' in tags) or ('b1950' in tags):
            body._epoch = ephem.B1950
        else:
            body._epoch = ephem.J2000
        body._ra = ra
        body._dec = dec
    
    elif body_type == 'tle':
        lines = fields[-1].split('\n')
        if len(lines) != 3:
            raise ValueError("Target description '%s' contains *tle* body without the expected three lines"
                             % description)
        if not preferred_name:
            preferred_name = lines[0].strip()
        try:
            body = ephem.readtle(preferred_name, lines[1], lines[2])
        except ValueError:
            raise ValueError("Target description '%s' contains malformed *tle* body" % description)
    
    elif body_type == 'special':
        try:
            body = eval('ephem.%s()' % preferred_name.capitalize())
        except AttributeError:
            raise ValueError("Target description '%s' contains unknown *special* body '%s'"
                             % (description, preferred_name))
    
    elif body_type == 'star':
        try:
            body = eval("ephem.star('%s')" % ' '.join([w.capitalize() for w in preferred_name.split()]))
        except KeyError:
            raise ValueError("Target description '%s' contains unknown *star* '%s'"
                             % (description, preferred_name))
    
    elif body_type == 'xephem':
        edb_string = fields[-1].replace('~', ',')
        if preferred_name:
            edb_string.replace(edb_string.partition(',')[0], preferred_name)
        try:
            body = eval("ephem.readdb('%s')" % edb_string)
        except ValueError:
            raise ValueError("Target description '%s' contains malformed *xephem* body" % description)
    
    else:
        raise ValueError("Target description '%s' contains unknown body type '%s'" % (description, body_type))
    
    # Extract flux info if it is available
    if (len(fields) > 4) and (len(fields[4].strip(' ()')) > 0):
        flux_info = [float(num) for num in fields[4].strip(' ()').split()]
        if len(flux_info) < 3:
            raise ValueError("Target description '%s' has invalid flux info" % description)
        min_freq_Hz, max_freq_Hz, coefs = 1e6 * flux_info[0], 1e6 * flux_info[1], tuple(flux_info[2:])
    else:
        min_freq_Hz = max_freq_Hz = coefs = None
    
    return Target(body, tags, aliases, min_freq_Hz, max_freq_Hz, coefs)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  construct_azel_target
#--------------------------------------------------------------------------------------------------

def construct_azel_target(az, el):
    """Convenience function to create unnamed stationary target (*azel* body type).
    
    The input parameters will also accept :class:`ephem.Angle` objects, as these
    are floats in radians internally.
    
    Parameters
    ----------
    az : string or float
        Azimuth, either in 'D:M:S' string format, or as a float in radians
    el : string or float
        Elevation, either in 'D:M:S' string format, or as a float in radians
    
    Returns
    -------
    target : :class:`Target` object
        Constructed target object
    
    """
    return Target(StationaryBody(az, el), ['azel'])

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  construct_radec_target
#--------------------------------------------------------------------------------------------------

def construct_radec_target(ra, dec):
    """Convenience function to create unnamed fixed target (*radec* body type).
    
    The input parameters will also accept :class:`ephem.Angle` objects, as these
    are floats in radians internally. The epoch is assumed to be J2000.
    
    Parameters
    ----------
    ra : string or float
        Right ascension, either in 'H:M:S' string format, or as a float in radians
    dec : string or float
        Declination, either in 'D:M:S' string format, or as a float in radians
    
    Returns
    -------
    target : :class:`Target` object
        Constructed target object
    
    """
    body = ephem.FixedBody()
    ra, dec = ephem.hours(ra), ephem.degrees(dec)
    body.name = "Ra: %s Dec: %s" % (ra, dec)
    body._epoch = ephem.J2000
    body._ra = ra
    body._dec = dec
    return Target(body, ['radec'])
