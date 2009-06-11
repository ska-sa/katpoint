"""Target catalogue."""

import logging

import ephem

from .target import construct_target, Target

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
    
    def names(self):
        """List of known target names in catalogue which can be searched for.
        
        There are potentially more names than targets in the catalogue, as the
        same target can have many names.
        
        """
        names = []
        for target in self.targets:
            names.append(target.name)
            names.extend(target.aliases)
        return names
    
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
    
    def filter(self, flux, az, el, tags):
        pass
        