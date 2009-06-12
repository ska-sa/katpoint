"""Target catalogue."""

import logging
import time

import ephem
import numpy as np

from .target import construct_target, Target, separation
from .ephem_extra import rad2deg

logger = logging.getLogger("katpoint.catalogue")

specials = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

def _hash(name):
    """Normalise string to make name lookup more robust."""
    return name.lower().replace(' ', '')

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
    
    """
    def __init__(self, targets=None, tags=None, add_specials=True, add_stars=False):
        self.lookup = {}
        self.targets = []
        if add_specials:
            self.add(['%s, special' % (name,) for name in specials], tags)
        if add_stars:
            self.add(['%s, star' % (name,) for name in ephem.stars.stars.iterkeys()], tags)
        if targets is None:
            targets = []
        self.add(targets, tags)
    
    def __getitem__(self, name):
        """Look up target name in catalogue and return target object."""
        return self.lookup[_hash(name)]
    
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
    
    def names(self):
        """List of known target names in catalogue which can be searched for.
        
        There are potentially more names than targets in the catalogue, as the
        same target can have many names.
        
        """
        return [name for name in self.iternames()]
    
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
                logger.debug("Skipped '%s' [%s]" % (target.name, target.tags[0]))
            else:
                target.add_tags(tags)
                self.targets.append(target)
                for name in [target.name] + target.aliases:
                    self.lookup[_hash(name)] = target
                logger.debug("Added '%s' [%s] (and %d aliases)" % (target.name, target.tags[0], len(target.aliases)))
    
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
    
    def filter(self, tags=None, flux_Jy_limit=None, flux_freq_Hz=None, el_deg_limit=None,
               dist_deg_limit=None, proximity_targets=None, antenna=None, timestamp=None):
        """Filter catalogue on various criteria.
        
        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. If None
            or an empty list, all tags are accepted.
        flux_Jy_limit : float or sequence of 2 floats, optional
            Allowed flux density range, in Jy. If this is a single number, it is
            the lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_freq_Hz : float, optional
            Frequency at which to evaluate the flux density, in Hz (required for
            flux filter)
        el_deg_limit : float or sequence of 2 floats, optional
            Allowed elevation range, in degrees. If this is a single number, it
            is the lower limit, otherwise it takes the form [lower, upper].
            If None, any elevation is accepted.
        dist_deg_limit : float or sequence of 2 floats, optional
            Allowed range of angular distance to proximity targets, in degrees.
            If this is a single number, it is the lower limit, otherwise it
            takes the form [lower, upper]. If None, any distance is accepted.
        proximity_targets : :class:`Target` object, or sequence of objects
            Target or list of targets used in proximity filter
        antenna : :class:`katpoint.Antenna` object, optional
            Antenna which points at targets (needed for position-based filters)
        timestamp : float, optional
            Timestamp at which to evaluate target positions, in seconds since
            Unix epoch. If None, the current time is used.
        
        Returns
        -------
        subset : :class:`Catalogue` object
            Filtered catalogue
        
        """
        # Put targets in a numpy array to allow use of fancy indexing
        targets = np.array(self.targets)
        if not flux_Jy_limit is None:
            if np.isscalar(flux_Jy_limit):
                flux_Jy_limit = [flux_Jy_limit, np.inf]
            if not flux_freq_Hz:
                raise ValueError('Please specify frequency at which to measure flux density')
            flux = np.array([target.flux_density(flux_freq_Hz) for target in targets])
            targets = targets[(flux >= flux_Jy_limit[0]) & (flux <= flux_Jy_limit[1])]
        
        if not tags is None:
            if isinstance(tags, basestring):
                tags = [tags]
            desired_tags = set([tag for tag in tags if tag[0] != '~'])
            undesired_tags = set([tag[1:] for tag in tags if tag[0] == '~'])
            if desired_tags:
                targets = [target for target in targets if set(target.tags) & desired_tags]
            if undesired_tags:
                targets = np.array([target for target in targets if not (set(target.tags) & undesired_tags)])
        
        if not el_deg_limit is None:
            if np.isscalar(el_deg_limit):
                el_deg_limit = [el_deg_limit, 90.0]
            if antenna is None:
                raise ValueError('Antenna object needed to calculate target elevation')
            if timestamp is None:
                timestamp = time.time()
            el_deg = np.array([rad2deg(antenna.point(target, timestamp)[1]) for target in targets])
            targets = targets[(el_deg >= el_deg_limit[0]) & (el_deg <= el_deg_limit[1])]
        
        if (not dist_deg_limit is None) and (not proximity_targets is None):
            if np.isscalar(dist_deg_limit):
                dist_deg_limit = [dist_deg_limit, 180.0]
            if isinstance(proximity_targets, Target):
                proximity_targets = [proximity_targets]
            if antenna is None:
                raise ValueError('Antenna object needed to calculate angular separation of targets')
            if timestamp is None:
                timestamp = time.time()
            dist = np.array([[rad2deg(separation(target1, target2, antenna, timestamp))
                              for target2 in proximity_targets] for target1 in targets])
            print dist
            targets = targets[(dist >= dist_deg_limit[0]).all(axis=1) &
                              (dist <= dist_deg_limit[1]).all(axis=1)]
            
        return Catalogue(targets, add_specials=False)
