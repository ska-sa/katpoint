"""Target object used for pointing and flux density calculation."""

import numpy as np
import ephem

from .ephem_extra import unix_to_ephem_time, StationaryBody

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Target
#--------------------------------------------------------------------------------------------------

class Target(object):
    """A target which can be pointed at by an antenna.
    
    This is a wrapper around a PyEphem :class:`ephem.Body` that adds flux
    density and descriptive tags.
    
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
    
    Arguments
    ---------
    name : string
        Name of target
    
    """
    def __init__(self, body, tags, aliases=None, min_freq_Hz=None, max_freq_Hz=None, coefs=None):
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
    
    def __str__(self):
        """Human-friendly string representation of target object."""
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            return "%s: %s body, no flux info" % (self.name, self.tags[0])
        else:
            return "%s: %s body, flux defined for %.3f - %.3f GHz" % \
                   (self.name, self.tags[0], self.min_freq_Hz * 1e-9, self.max_freq_Hz * 1e-9)
    
    def get_description(self):
        """Machine-friendly string representation of target object."""
        names = ' | '.join([self.name] + self.aliases)
        tags = ' '.join(self.tags)
        fluxinfo = None
        if self.min_freq_Hz and self.max_freq_Hz and self.coefs:
            fluxinfo = '(%s %s %s)' % (self.min_freq_Hz * 1e-6, self.max_freq_Hz * 1e-6, 
                                       ' '.join([str(s) for s in self.coefs]))
        fields = [names, tags]
        if self.tags[0] == 'azel':
            if names.startswith('Az:'):
                fields = [tags]
            fields += [str(self.body.az), str(self.body.el)]
        
        elif self.tags[0] == 'radec':
            if names.startswith('Ra:'):
                fields = [tags]
            fields += [str(self.body._ra), str(self.body._dec)]
            if fluxinfo:
                fields += [fluxinfo]
                
        elif self.tags[0] == 'tle':
            # XEphem only saves bodies in xephem db format (TLE output will be a hassle)
            tags = tags.replace(tags.partition(' ')[0], 'xephem')
            edb_string = self.body.writedb().replace(',', '~')
            # Suppress name if it's the same as in the xephem db string
            edb_name = edb_string[:edb_string.index('~')]
            if edb_name == names:
                fields = [tags, edb_string]
            else:
                fields = [names, tags, edb_string]
        
        elif self.tags[0] == 'xephem':
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
        checking the sanity of the tags.
        
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
        self.tags += tags
        return self
        
    def radec(self, antenna, timestamps):
        """Calculate target (ra, dec) coordinates as seen from antenna at time(s).
        
        This calculates the *apparent topocentric position* for the epoch-of-date
        in equatorial coordinates. Take note that this is *not* the "star-atlas"
        position of the target, but the position as seen from the antenna at the
        given times. The difference is on the order of a few arcminutes.
        
        Parameters
        ----------
        antenna : :class:`katpoint.antenna.Antenna` object
            Antenna which points at target
        timestamps : float or sequence
            Local timestamp(s) in seconds since Unix epoch
        
        Returns
        -------
        ra : :class:`ephem.Angle` object, or sequence of objects
            Right ascension, in radians
        dec : :class:`ephem.Angle` object, or sequence of objects
            Declination, in radians
        
        """
        def _scalar_radec(t):
            """Calculate (ra, dec) coordinates for a single time instant."""
            antenna.observer.date = unix_to_ephem_time(t)
            self.body.compute(antenna.observer)
            return self.body.ra, self.body.dec
        if np.isscalar(timestamps):
            return _scalar_radec(timestamps)
        else:
            azel = np.array([_scalar_radec(t) for t in timestamps])
            return azel[:, 0], azel[:, 1]
    
    def flux_density(self, obs_freq_Hz):
        """Calculate flux density for given observation frequency.
        
        This uses a polynomial flux model of the form::
        
            log10 S[Jy] = a + b*log10(f[MHz]) + c*(log10(f[MHz]))^2
        
        as used in Baars 1977.
        
        Parameters
        ----------
        obs_freq_Hz : float
            Frequency at which to evaluate flux density
        
        Returns
        -------
        flux_density : float
            Flux density in Jy, or None if frequency is out of range or target
            does not have flux info
        
        """
        if None in [self.min_freq_Hz, self.max_freq_Hz, self.coefs]:
            # Target has no specified flux density
            return None
        if (obs_freq_Hz < self.min_freq_Hz) or (obs_freq_Hz > self.max_freq_Hz):
            # Frequency out of range for flux calculation of target
            return None
        log10_freq = np.log10(obs_freq_Hz * 1e-6)
        log10_flux = 0.0
        acc = 1.0
        for coeff in self.coefs:
            log10_flux += float(coeff) * acc
            acc *= log10_freq
        return 10.0 ** log10_flux

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
        raise ValueError('Target description string must have at least two fields')
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
    tags = [s.strip().lower() for s in fields[1].split(' ')]
    if len(tags) == 0:
        raise ValueError('Target description needs at least one tag (body type)')
    body_type = tags[0]
    # Remove empty fields starting from the end (useful when parsing CSV files with fixed number of fields)
    while len(fields[-1]) == 0:
        fields.pop()
    
    # Create appropriate PyEphem body based on body type
    if body_type == 'azel':
        if len(fields) < 4:
            raise ValueError('Target description contains *azel* body with no (az, el) coordinates')
        body = StationaryBody(fields[2], fields[3], preferred_name)
    
    elif body_type == 'radec':
        if len(fields) < 4:
            raise ValueError('Target description contains *radec* body with no (ra, dec) coordinates')
        body = ephem.FixedBody()
        ra, dec = ephem.hours(fields[2]), ephem.degrees(fields[3])
        if preferred_name:
            body.name = preferred_name
        else:
            body.name = "Ra: %s Dec: %s" % (ra, dec)
        body._epoch = ephem.J2000
        body._ra = ra
        body._dec = dec
    
    elif body_type == 'tle':
        lines = fields[-1].split('\n')
        if len(lines) != 3:
            raise ValueError('Target description contains *tle* body without the expected three lines')
        if not preferred_name:
            preferred_name = lines[0].strip()
        try:
            body = ephem.readtle(preferred_name, lines[1], lines[2])
        except ValueError:
            raise ValueError('Target description contains malformed *tle* body')
    
    elif body_type == 'special':
        try:
            body = eval('ephem.%s()' % preferred_name.capitalize())
        except AttributeError:
            raise ValueError("Target description contains unknown *special* body '%s'" % preferred_name)
    
    elif body_type == 'star':
        try:
            body = eval("ephem.star('%s')" % preferred_name.capitalize())
        except KeyError:
            raise ValueError("Target description contains unknown *star* '%s'" % preferred_name)
    
    elif body_type == 'xephem':
        edb_string = fields[-1].replace('~', ',')
        if preferred_name:
            edb_string.replace(edb_string.partition(',')[0], preferred_name)
        try:
            body = eval("ephem.readdb('%s')" % edb_string)
        except ValueError:
            raise ValueError("Target description contains malformed *xephem* body")
    
    else:
        raise ValueError("Target description contains unknown body type '%s'" % body_type)
    
    # Extract flux info if it is available
    if (len(fields) > 4) and (len(fields[4].strip(' ()')) > 0):
        flux_info = [float(num) for num in fields[4].strip(' ()').split()]
        if len(flux_info) < 3:
            raise ValueError('Target description has invalid flux info')
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
    are floats in radians internally.
    
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
