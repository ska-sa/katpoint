"""Target catalogue."""

import logging
import time
import re

import ephem.stars
import numpy as np
# This is needed for tab completion, but is ignored if no IPython is installed
try:
    import IPython.ipapi
except ImportError:
    pass
# The same goes for readline
try:
    import readline
    readline_found = True
except ImportError:
    readline_found = False

from .target import construct_target, Target
from .ephem_extra import rad2deg

logger = logging.getLogger("katpoint.catalogue")

specials = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

def _hash(name):
    """Normalise string to make name lookup more robust."""
    return name.strip().lower().replace(' ', '')

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Catalogue
#--------------------------------------------------------------------------------------------------

class Catalogue(object):
    """A searchable and filterable catalogue of targets.
    
    Parameters
    ----------
    targets : :class:`Target` object or string, or sequence of these, optional
        Target or list of targets to add to catalogue (may also be file object)
    tags : string or sequence of strings, optional
        Tag or list of tags to add to *targets*
    add_specials: {True, False}, optional
        True if *special* bodies specified in :data:`specials` should be added
    add_stars:  {False, True}, optional
        True if *star* bodies from PyEphem star catalogue should be added
    antenna : :class:`Antenna` object, optional
        Default antenna to use for position calculations for all targets
    flux_freq_MHz : float, optional
        Default frequency at which to evaluate flux density of all targets, in MHz
    
    """
    def __init__(self, targets=None, tags=None, add_specials=True, add_stars=False,
                 antenna=None, flux_freq_MHz=None):
        self.lookup = {}
        self.targets = []
        self.antenna = antenna
        self.flux_freq_MHz = flux_freq_MHz
        if add_specials:
            self.add(['%s, special' % (name,) for name in specials], tags)
            self.add('Zenith, azel, 0, 90', tags)
        if add_stars:
            self.add(['%s, star' % (name,) for name in ephem.stars.stars.iterkeys()], tags)
        if targets is None:
            targets = []
        self.add(targets, tags)
    
    def __str__(self):
        """Verbose human-friendly string representation of catalogue object."""
        return '\n'.join(['%s' % (target,) for target in self.targets])
    
    def __repr__(self):
        """Short human-friendly string representation of catalogue object."""
        return "<katpoint.Catalogue targets=%d names=%d at 0x%x>" % \
               (len(self.targets), len(self.lookup.keys()), id(self))
    
    def __getitem__(self, name):
        """Look up target name in catalogue and return target object.
        
        Parameters
        ----------
        name : string
            Target name to look up (can be alias as well)
        
        Returns
        -------
        target : :class:`Target` object, or None
            Associated target object, or None if no target was found
        
        """
        try:
            return self.lookup[_hash(name)]
        except KeyError:
            return None
    
    def __iter__(self):
        """Iterate over targets in catalogue."""
        return iter(self.targets)
    
    def iternames(self):
        """Iterator over known target names in catalogue which can be searched for.
        
        There are potentially more names than targets in the catalogue, as the
        same target can have many names.
        
        """
        for target in self.targets:
            yield target.name
            for alias in target.aliases:
                yield alias
    
    def add(self, targets, tags=None):
        """Add targets to catalogue.
        
        Parameters
        ----------
        targets : :class:`Target` object or string, or sequence of these
            Target or list of targets to add to catalogue (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to *targets*
                
        """
        if isinstance(targets, basestring) or isinstance(targets, Target):
            targets = [targets]
        for target in targets:
            if isinstance(target, basestring):
                target = construct_target(target)
            if not isinstance(target, Target):
                raise ValueError('List of targets should either contain Target objects or description strings')
            if self.lookup.has_key(_hash(target.name)):
                logger.warn("Skipped '%s' [%s] (already in catalogue)" % (target.name, target.tags[0]))
            else:
                target.add_tags(tags)
                target.antenna = self.antenna
                target.flux_freq_MHz = self.flux_freq_MHz
                self.targets.append(target)
                for name in [target.name] + target.aliases:
                    self.lookup[_hash(name)] = target
                logger.debug("Added '%s' [%s] (and %d aliases)" % 
                             (target.name, target.tags[0], len(target.aliases)))
    
    def add_tle(self, lines, tags=None):
        """Add Two-Line Element (TLE) targets to catalogue.
        
        Parameters
        ----------
        lines : sequence of strings
            List of lines containing one or more TLEs (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to targets
        
        """
        targets, tle = [], []
        for line in lines:
            tle += [line]
            if len(tle) == 3:
                targets.append('tle,' + ' '.join(tle))
                tle = []
        if len(tle) > 0:
            logger.warning('Did not receive a multiple of three lines when constructing TLEs')
        self.add(targets, tags)
    
    def remove(self, name):
        """Remove target from catalogue.
        
        Parameters
        ----------
        name : string
            Name of target to remove (may also be an alternate name of target)
        
        """
        if self.lookup.has_key(_hash(name)):
            target = self[name]
            self.lookup.pop(_hash(target.name))
            for alias in target.aliases:
                self.lookup.pop(_hash(alias))
            self.targets.remove(target)
    
    def iterfilter(self, tags=None, flux_limit_Jy=None, flux_freq_MHz=None, el_limit_deg=None,
                   dist_limit_deg=None, proximity_targets=None, timestamp=None, antenna=None):
        """Iterator which returns targets satisfying various criteria.
        
        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. If None
            or an empty list, all tags are accepted.
        flux_limit_Jy : float or sequence of 2 floats, optional
            Allowed flux density range, in Jy. If this is a single number, it is
            the lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        el_limit_deg : float or sequence of 2 floats, optional
            Allowed elevation range, in degrees. If this is a single number, it
            is the lower limit, otherwise it takes the form [lower, upper].
            If None, any elevation is accepted.
        dist_limit_deg : float or sequence of 2 floats, optional
            Allowed range of angular distance to proximity targets, in degrees.
            If this is a single number, it is the lower limit, otherwise it
            takes the form [lower, upper]. If None, any distance is accepted.
        proximity_targets : :class:`Target` object, or sequence of objects
            Target or list of targets used in proximity filter
        timestamp : float or string, optional
            UTC timestamp at which to evaluate target positions, in seconds
            since Unix epoch or string date/time. If None, the current time
            *at each iteration* is used.
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        
        Returns
        -------
        iter : iterator object
            The generated iterator object which will return filtered targets
        
        Raises
        ------
        ValueError
            If some required parameters are missing
        
        """
        tag_filter = not tags is None
        flux_filter = not flux_limit_Jy is None
        elevation_filter = not el_limit_deg is None
        proximity_filter = not dist_limit_deg is None
        # Copy targets to a new list which will be pruned by filters
        targets = list(self.targets)
        
        # First apply static criteria (tags, flux) which do not depend on timestamp
        if tag_filter:
            if isinstance(tags, basestring):
                tags = [tags]
            desired_tags = set([tag for tag in tags if tag[0] != '~'])
            undesired_tags = set([tag[1:] for tag in tags if tag[0] == '~'])
            if desired_tags:
                targets = [target for target in targets if set(target.tags) & desired_tags]
            if undesired_tags:
                targets = [target for target in targets if not (set(target.tags) & undesired_tags)]
        
        if flux_filter:
            if np.isscalar(flux_limit_Jy):
                flux_limit_Jy = [flux_limit_Jy, np.inf]
            flux = [target.flux_density(flux_freq_MHz) for target in targets]
            targets = [target for n, target in enumerate(targets)
                       if (flux[n] >= flux_limit_Jy[0]) & (flux[n] <= flux_limit_Jy[1])]
        
        # Now prepare for dynamic criteria (elevation, proximity) which depend on potentially changing timestamp
        if elevation_filter and np.isscalar(el_limit_deg):
            el_limit_deg = [el_limit_deg, 90.0]
        
        if proximity_filter:
            if proximity_targets is None:
                raise ValueError('Please specify proximity target(s) for proximity filter')
            if np.isscalar(dist_limit_deg):
                dist_limit_deg = [dist_limit_deg, 180.0]
            if isinstance(proximity_targets, Target):
                proximity_targets = [proximity_targets]
        
        # Keep checking targets while there are some in the list
        while targets:
            latest_timestamp = timestamp
            # Obtain current time if no timestamp is supplied - this will differ for each iteration
            if (elevation_filter or proximity_filter) and latest_timestamp is None:
                latest_timestamp = time.time()
            # Iterate over targets until one is found that satisfies dynamic criteria
            for n, target in enumerate(targets):
                if elevation_filter:
                    el_deg = rad2deg(target.azel(latest_timestamp, antenna)[1])
                    if (el_deg < el_limit_deg[0]) or (el_deg > el_limit_deg[1]):
                        continue
                if proximity_filter:
                    dist_deg = np.array([rad2deg(target.separation(prox_target, latest_timestamp, antenna))
                                         for prox_target in proximity_targets])
                    if (dist_deg < dist_limit_deg[0]).any() or (dist_deg > dist_limit_deg[1]).any():
                        continue
                # Break if target is found - popping the target inside the for-loop is a bad idea!
                found_one = n
                break
            else:
                # No targets in list satisfied dynamic criteria - iterator stops
                return
            # Return successful target and remove from list to ensure it is not picked again
            yield targets.pop(found_one)
    
    def filter(self, tags=None, flux_limit_Jy=None, flux_freq_MHz=None, el_limit_deg=None,
               dist_limit_deg=None, proximity_targets=None, timestamp=None, antenna=None):
        """Filter catalogue on various criteria.
        
        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. If None
            or an empty list, all tags are accepted.
        flux_limit_Jy : float or sequence of 2 floats, optional
            Allowed flux density range, in Jy. If this is a single number, it is
            the lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        el_limit_deg : float or sequence of 2 floats, optional
            Allowed elevation range, in degrees. If this is a single number, it
            is the lower limit, otherwise it takes the form [lower, upper].
            If None, any elevation is accepted.
        dist_limit_deg : float or sequence of 2 floats, optional
            Allowed range of angular distance to proximity targets, in degrees.
            If this is a single number, it is the lower limit, otherwise it
            takes the form [lower, upper]. If None, any distance is accepted.
        proximity_targets : :class:`Target` object, or sequence of objects
            Target or list of targets used in proximity filter
        timestamp : float or string, optional
            UTC timestamp at which to evaluate target positions, in seconds
            since Unix epoch or string date/time (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        
        Returns
        -------
        subset : :class:`Catalogue` object
            Filtered catalogue
        
        Raises
        ------
        ValueError
            If some required parameters are missing
        
        """
        return Catalogue([target for target in
                          self.iterfilter(tags, flux_limit_Jy, flux_freq_MHz, el_limit_deg,
                                          dist_limit_deg, proximity_targets, timestamp, antenna)],
                         add_specials=False)
        
    def sort(self, key='name', ascending=True, flux_freq_MHz=None, timestamp=None, antenna=None):
        """Sort targets in catalogue.
        
        Parameters
        ----------
        key : {'name', 'ra', 'dec', 'az', 'el', 'flux'}, optional
            Sort the targets according to this field
        ascending : {True, False}, optional
            True if key should be sorted in ascending order
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        timestamp : float or string, optional
            UTC timestamp at which to evaluate target positions, in seconds
            since Unix epoch or string date/time (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        
        Returns
        -------
        sorted : :class:`Catalogue` object
            Sorted catalogue
        
        Raises
        ------
        ValueError
            If some required parameters are missing or key is unknown
        
        """
        # Set up index list that will be sorted
        if key == 'name':
            index = [target.name for target in self.targets]
        elif key == 'ra':
            index = [target.radec(timestamp, antenna)[0] for target in self.targets]
        elif key == 'dec':
            index = [target.radec(timestamp, antenna)[1] for target in self.targets]
        elif key == 'az':
            index = [target.azel(timestamp, antenna)[0] for target in self.targets]
        elif key == 'el':
            index = [target.azel(timestamp, antenna)[1] for target in self.targets]
        elif key == 'flux':
            index = [target.flux_density(flux_freq_MHz) for target in self.targets]
        else:
            raise ValueError('Unknown key to sort on')
        # Sort index indirectly, either in ascending or descending order
        if ascending:
            self.targets = np.array(self.targets)[np.argsort(index)].tolist()
        else:
            self.targets = np.array(self.targets)[np.flipud(np.argsort(index))].tolist()
        return self
    
    def visibility_list(self, timestamp=None, antenna=None, flux_freq_MHz=None):
        """Print out list of targets in catalogue, sorted by decreasing elevation.
        
        This prints out the name, azimuth and elevation of each target in the
        catalogue, in order of decreasing elevation. It indicates the horizon
        itself by a line of dashes. It also displays the target flux density
        if a frequency is supplied. It is useful to quickly see which sources
        are visible.
        
        Parameters
        ----------
        timestamp : float or string, optional
            UTC timestamp at which to evaluate target positions, in seconds
            since Unix epoch or string date/time (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        flux_freq_MHz : float, optional
            Frequency at which to evaluate flux density, in MHz
        
        """
        above_horizon = True
        if timestamp is None:
            timestamp = time.time()
        if antenna is None:
            antenna = self.antenna
        if antenna is None:
            raise ValueError('Antenna object needed to calculate target position')
        title = "Targets visible from antenna '%s' at %s" % \
                (antenna.name, time.strftime('%Y/%m/%d %H:%M:%S %Z', time.localtime(timestamp)))
        if flux_freq_MHz is None:
            flux_freq_MHz = self.flux_freq_MHz
        if not flux_freq_MHz is None:
            title += ', with flux density evaluated at %f MHz' % (flux_freq_MHz,)
        print title
        print
        print 'Target                    Azimuth    Elevation    Flux'
        print '------                    -------    ---------    ----'
        for target in self.sort('el', timestamp=timestamp, antenna=antenna, ascending=False):
            az, el = target.azel(timestamp, antenna)
            # If no flux frequency is given, do not attempt to evaluate the flux, as it will fail
            if not flux_freq_MHz is None:
                flux = target.flux_density(flux_freq_MHz)
            else:
                flux = None
            if above_horizon and el < 0.0:
                # Draw horizon line
                print '------------------------------------------------------'
                above_horizon = False
            if not flux is None:
                print '%-20s %12s %12s %7.1f' % (target.name, az, el, flux)
            else:
                print '%-20s %12s %12s' % (target.name, az, el)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  _catalogue_completer
#--------------------------------------------------------------------------------------------------

cat_lookup_match = re.compile(r"""(?:.*\=)?(.*)\[(?P<quote>['|"])(?!.*(?P=quote))(.*)$""")

def _catalogue_completer(self, event):
    """Custom IPython completer for catalogue name lookups.
    
    This is inspired by Darren Dale's custom dict-like completer for h5py.
    
    """
    # Parse command line as (ignored = )base['start_of_name
    base, start_of_name = cat_lookup_match.split(event.line)[1:4:2]
    
    # Avoid calling any functions during eval()...
    if '(' in base:
        raise IPython.ipapi.TryNext
    
    # Obtain catalogue object from user namespace
    try:
        cat = eval(base, self.shell.user_ns)
    except:
        raise IPython.ipapi.TryNext
    
    # Only continue if this object is actually a Catalogue
    if not isinstance(cat, Catalogue):
        raise IPython.ipapi.TryNext
    
    if readline_found:
        # Remove space from delimiter list, so completion works past spaces in names
        readline.set_completer_delims('\t\n`!@#$^&*()=+[{]}\\|;:\'",<>?')
    
    return [name for name in cat.iternames() if name[:len(start_of_name)] == start_of_name]
