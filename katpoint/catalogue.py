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

"""Target catalogue."""

import logging
import re

import ephem.stars
import numpy as np
# This is needed for tab completion, but is ignored if no IPython is installed
try:
    # IPython 0.11 and above
    from IPython.core.error import TryNext
except ImportError:
    try:
        # IPython 0.10 and below
        from IPython.ipapi import TryNext
    except ImportError:
        pass
# The same goes for readline
try:
    import readline
except ImportError:
    readline = None

from .target import Target
from .timestamp import Timestamp
from .ephem_extra import rad2deg

logger = logging.getLogger(__name__)

specials = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


def _hash(name):
    """Normalise string to make name lookup more robust."""
    return name.strip().lower().replace(' ', '').replace('_', '')

# --------------------------------------------------------------------------------------------------
# --- CLASS :  Catalogue
# --------------------------------------------------------------------------------------------------


class Catalogue(object):
    """A searchable and filterable catalogue of targets.

    Overview
    --------

    A :class:`Catalogue` object combines two concepts:

    - A list of targets, which can be filtered, sorted, pretty-printed and
      iterated over. The list is accessible as :meth:`Catalogue.targets`, and
      the catalogue itself is iterable, returning the next target on each
      iteration. An example is::

        cat = katpoint.Catalogue()
        t = cat.targets[0]
        for t in cat:
            # Do something with target t

    - Lookup by name, by using the catalogue as if it were a dictionary. This is
      simpler for the user, who does not have to remember all the target details.
      The named lookup supports tab completion in IPython, which further
      simplifies finding a target in the catalogue. An example is::

        cat = katpoint.Catalogue()
        t = cat['Sun']

    Construction
    ------------

    A catalogue can be constructed in many ways. The simplest way is::

        cat = katpoint.Catalogue()

    which produces an empty catalogue. The standard *special* targets, which are
    the Sun, Moon, planets and Zenith, can be added as follows::

        cat = katpoint.Catalogue(add_specials=True)

    Another built-in set of targets is the small star catalogue included with
    PyEphem. These *star* targets are added as follows::

        cat = katpoint.Catalogue(add_stars=True)

    Additional targets may be loaded during initialisation of the catalogue by
    providing a list of :class:`Target` objects (or a single object by itself),
    as in the following example::

        t1 = katpoint.Target('Ganymede, special')
        t2 = katpoint.Target('Takreem, azel, 20, 30')
        cat1 = katpoint.Catalogue(t1)
        cat2 = katpoint.Catalogue([t1, t2])

    Alternatively, the list of targets may be replaced by a list of target
    description strings (or a single description string). The target objects are
    then constructed before being added, as in::

        cat1 = katpoint.Catalogue('Takreem, azel, 20, 30')
        cat2 = katpoint.Catalogue(['Ganymede, special', 'Takreem, azel, 20, 30'])

    Taking this one step further, the list may be replaced by any iterable object
    that returns strings. A very useful example of such an object is the Python
    :class:`file` object, which iterates over the lines of a text file. If the
    catalogue file contains one target description string per line (with comments
    and blank lines allowed too), it may be loaded as::

        cat = katpoint.Catalogue(file('catalogue.csv'))

    Once a catalogue is initialised, more targets may be added to it. The
    :meth:`Catalogue.add` method is the most direct way. It accepts a single
    target object, a list of target objects, a single string, a list of strings
    or a string iterable. This is illustrated below::

        t1 = katpoint.Target('Ganymede, special')
        t2 = katpoint.Target('Takreem, azel, 20, 30')
        cat = katpoint.Catalogue(add_specials=False)
        cat.add(t1)
        cat.add([t1, t2])
        cat.add('Ganymede, special')
        cat.add(['Ganymede, special', 'Takreem, azel, 20, 30'])
        cat.add(file('catalogue.csv'))

    The only functionality that :meth:`Catalogue.add` lacks is the ability to
    add all *special* and *star* targets in one go. They may still be added
    individually, although this is less convenient (and the reason for the
    existence of ``add_specials`` and ``add_stars`` in the :class:`Catalogue`
    initialiser in the first place).

    Some target types are typically found in files with standard formats.
    Notably, *tle* targets are found in TLE files with three lines per target,
    and many *xephem* targets are stored in EDB database files. Editing these
    files to make each line a valid :class:`Target` description string is
    cumbersome, especially in the case of TLE files which are regularly updated.
    Two special methods simplify the loading of targets from these files::

        cat = katpoint.Catalogue(add_specials=False)
        cat.add_tle(file('gps-ops.txt'))
        cat.add_edb(file('hipparcos.edb'))

    Whenever targets are added to the catalogue, a tag or list of tags may be
    specified. The tags can also be given as a single string of
    whitespace-delimited tags, since tags may not contain whitespace. These tags
    are added to the targets currently being added. This makes it easy to tag
    groups of related targets in the catalogue, as shown below::

        cat = katpoint.Catalogue(tags='default')
        cat.add_tle(file('gps-ops.txt'), tags='gps satellite')
        cat.add_tle(file('glo-ops.txt'), tags=['glonass', 'satellite'])
        cat.add(file('source_list.csv'), tags='calibrator')
        cat.add_edb(file('hipparcos.edb'), tags='star')

    Finally, targets may be removed from the catalogue. This is useful when a
    target is to be updated, as adding a target which is already in the catalogue
    will silently fail. The reasoning behind this is that a target object is
    added once to the target list, but may have multiple references in the lookup
    dictionary, one per alias. When updating a target, it is not clear what to do
    with all the alternate names. For now, the user has to explicitly remove a
    target by name before loading a new version of the target into the catalogue.
    The target may be removed via any of its names::

        cat = katpoint.Catalogue()
        cat.remove('Sun')

    Filtering and sorting
    ---------------------

    A :class:`Catalogue` object may be filtered based on various criteria. The
    following filters are available:

    - *Tag filter*. Returns all targets that have a specified set of tags, and
      *not* another set of tags. Tags prepended with a tilde (~) indicate tags
      which targets should not have. All tags have to be present (or absent) for
      a target to be selected. Remember that the body type is also a tag. An
      example is::

        cat = katpoint.Catalogue(tags='default')
        cat1 = cat.filter(tags=['special', '~radec'])
        cat1 = cat.filter(tags='special ~radec')

    - *Flux filter*. Returns all targets with a flux density between the
      specified limits, at a given frequency. If only one limit is given, it is
      a lower limit. To simplify filtering, a default flux frequency may be
      supplied to the catalogue during initialisation. This is stored in each
      target in the catalogue. An example is::

        cat = katpoint.Catalogue(file('source_list.csv'))
        cat1 = cat.filter(flux_limit_Jy=[1, 100], flux_freq_MHz=1500)
        cat = katpoint.Catalogue(file('source_list.csv'), flux_freq_MHz=1500)
        cat1 = cat.filter(flux_limit_Jy=1)

    - *Azimuth filter*. Returns all targets with an azimuth angle in the given
      range. The range is specified in degrees as [left, right], where *left* is
      the leftmost or starting azimuth, and *right* is the rightmost or ending
      azimuth. The azimuth angle increases clockwise from *left* to *right* to
      form the range. If *right* is less than *left*, the azimuth angles range
      around +-180 degrees. Since the target azimuth is dependent on time and
      observer position, a timestamp and :class:`katpoint.Antenna` object has to
      be provided. The timestamp defaults to now, and the antenna object may be
      associated with the catalogue during initialisation, from where it is
      stored in each target. An example is::

        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        cat = katpoint.Catalogue()
        cat1 = cat.filter(az_limit_deg=[0, 90], timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(az_limit_deg=[90, 0])

    - *Elevation filter*. Returns all targets with an elevation angle within the
      given limits, in degrees. If only one limit is given, it is assumed to be
      a lower limit. As with the azimuth filter, a timestamp and antenna object
      is required (or defaults will be used). An example is::

        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        cat = katpoint.Catalogue()
        cat1 = cat.filter(el_limit_deg=[10, 30], timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(el_limit_deg=10)

    - *Proximity filter*. Returns all targets with angular separation from a
      given set of targets within a specified range. The range is given as a
      lower and upper limit, in degrees, and a single number is taken as the
      lower limit. The typical use of this filter is to return all targets more
      than a specified number of degrees away from a known set of interfering
      targets. As with the azimuth filter, a timestamp and antenna object is
      required (or defaults will be used). An example is::

        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        cat = katpoint.Catalogue()
        cat.add_tle(file('geo.txt'))
        sun = cat['Sun']
        afristar = cat['AFRISTAR']
        cat1 = cat.filter(dist_limit_deg=5, proximity_targets=[sun, afristar],
                          timestamp='2009-10-10', antenna=ant)
        cat = katpoint.Catalogue(antenna=ant)
        cat1 = cat.filter(dist_limit_deg=[0, 5], proximity_targets=sun)

    The criteria may be divided into *static* criteria which are independent of
    time (tags and flux) and *dynamic* criteria which do depend on time
    (azimuth, elevation and proximity). There are two filtering mechanisms that
    both support the same criteria, but differ on their handling of dynamic
    criteria:

    - A direct filter, implemented by the :meth:`Catalogue.filter` method. This
      returns the filtered catalogue as a new catalogue which contains the
      subset of targets that satisfy the criteria. All criteria are evaluated at
      the same time instant. A typical use-case is::

        cat = katpoint.Catalogue(file('source_list.csv'))
        strong_sources = cat.filter(flux_limit_Jy=10.0, flux_freq_MHz=1500)

    - An iterator filter, implemented by the :meth:`Catalogue.iterfilter` method.
      This is a Python *generator function*, which returns a *generator iterator*,
      to be more precise. Each time the returned iterator's .next() method is
      invoked, the next suitable :class:`Target` object is returned. If no
      timestamp is provided, the criteria are re-evaluated at the time instant
      of the .next() call, which makes it easy to cycle through a list of
      targets over an extended period of time (as during observation). The
      iterator filter is typically used in a for-loop::

        cat = katpoint.Catalogue(file('source_list.csv'))
        ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        for t in cat.iterfilter(el_limit_deg=10, antenna=ant):
            # < observe target t >

    When a catalogue is sorted, the order of the target list is changed. The
    catalogue may be sorted according to name (the default), right ascension,
    declination, azimuth, elevation and flux. Any position-based key requires a
    timestamp and :class:`katpoint.Antenna` object to evaluate the position of
    each target, and the flux key requires a frequency at which to evaluate the
    flux.

    Parameters
    ----------
    targets : :class:`Target` object or string, or sequence of these, optional
        Target or list of targets to add to catalogue (may also be file object)
    tags : string or sequence of strings, optional
        Tag or list of tags to add to *targets* (strings will be split on
        whitespace)
    add_specials: {False, True}, optional
        True if *special* bodies specified in :data:`specials` (and 'Zenith')
        should be added
    add_stars:  {False, True}, optional
        True if *star* bodies from PyEphem star catalogue should be added
    antenna : :class:`Antenna` object, optional
        Default antenna to use for position calculations for all targets
    flux_freq_MHz : float, optional
        Default frequency at which to evaluate flux density of all targets, in MHz

    """
    def __init__(self, targets=None, tags=None, add_specials=False, add_stars=False,
                 antenna=None, flux_freq_MHz=None):
        self.lookup = {}
        self.targets = []
        self._antenna = antenna
        self._flux_freq_MHz = flux_freq_MHz
        if add_specials:
            self.add(['%s, special' % (name,) for name in specials], tags)
            self.add('Zenith, azel, 0, 90', tags)
        if add_stars:
            self.add(['%s, star' % (name,) for name in ephem.stars.stars.iterkeys()], tags)
        if targets is None:
            targets = []
        self.add(targets, tags)

    # Provide properties so that default antenna or flux frequency changes are passed on to targets
    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0212
    def antenna():
        """Class method which creates antenna property."""
        doc = 'Default antenna used to calculate target positions.'

        def fget(self):
            return self._antenna

        def fset(self, value):
            self._antenna = value
            for target in self.targets:
                target.antenna = self._antenna
        return locals()
    antenna = property(**antenna())

    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0212,C0103
    def flux_freq_MHz():
        """Class method which creates flux_freq_MHz property."""
        doc = 'Default frequency at which to evaluate flux density, in MHz.'

        def fget(self):
            return self._flux_freq_MHz

        def fset(self, value):
            self._flux_freq_MHz = value
            for target in self.targets:
                target.flux_freq_MHz = self._flux_freq_MHz
        return locals()
    flux_freq_MHz = property(**flux_freq_MHz())

    def __str__(self):
        """Verbose human-friendly string representation of catalogue object."""
        return '\n'.join(['%s' % (target,) for target in self.targets])

    def __repr__(self):
        """Short human-friendly string representation of catalogue object."""
        return "<katpoint.Catalogue targets=%d names=%d at 0x%x>" % \
               (len(self.targets), len(self.lookup.keys()), id(self))

    def __len__(self):
        """Number of targets in catalogue."""
        return len(self.targets)

    def __getitem__(self, name):
        """Look up target name in catalogue and return target object.

        The name string may be tab-completed in IPython to simplify finding a
        target.

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

    def __contains__(self, obj):
        """Test whether catalogue contains exact target, or target with given name."""
        name = obj.name if isinstance(obj, Target) else obj
        target = self[name]
        return target is not None and (not isinstance(obj, Target) or target == obj)

    def __eq__(self, other):
        """Equality comparison operator."""
        # Use lookup dict instead of targets list, as it has a fixed order based on target name
        return isinstance(other, Catalogue) and (self.lookup.values() == other.lookup.values())

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __hash__(self):
        """Hash value matches behaviour of equality comparison operator."""
        return hash(tuple(self.lookup.values()))

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

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        targets : :class:`Target` object or string, or sequence of these
            Target or list of targets to add to catalogue (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to *targets* (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add(file('source_list.csv'), tags='cal')
        >>> cat.add('Sun, special')
        >>> cat2 = Catalogue(add_specials=False)
        >>> cat2.add(cat.targets)

        """
        if isinstance(targets, basestring) or isinstance(targets, Target):
            targets = [targets]
        for target in targets:
            if isinstance(target, basestring):
                # Ignore strings starting with a hash (assumed to be comments) or only containing whitespace
                if (target[0] == '#') or (len(target.strip()) == 0):
                    continue
                target = Target(target)
            if not isinstance(target, Target):
                raise ValueError('List of targets should either contain Target objects or description strings')
            if _hash(target.name) in self.lookup:
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
        """Add NORAD Two-Line Element (TLE) targets to catalogue.

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        lines : sequence of strings
            List of lines containing one or more TLEs (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to targets (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add TLE targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add_tle(file('gps-ops.txt'), tags='gps')
        >>> lines = ['ISS DEB [TOOL BAG]\n',
                     '1 33442U 98067BL  09195.86837279  .00241454  37518-4  34022-3 0  3424\n',
                     '2 33442  51.6315 144.2681 0003376 120.1747 240.0135 16.05240536 37575\n']
        >>> cat2.add_tle(lines)

        """
        targets, tle = [], []
        for line in lines:
            if (line[0] == '#') or (len(line.strip()) == 0):
                continue
            tle += [line]
            if len(tle) == 3:
                targets.append('tle,' + ' '.join(tle))
                tle = []
        if len(tle) > 0:
            logger.warning('Did not receive a multiple of three lines when constructing TLEs')

        # Check TLE epochs and warn if some are too far in past or future, which would make TLE inaccurate right now
        max_epoch_diff_days, num_outdated, worst = 0, 0, None
        for target in targets:
            # Extract name, epoch and mean motion (revolutions per day)
            name = target.split('\n')[0][4:].strip()
            epoch_year, epoch_day = float(target.split('\n')[1][19:21]), float(target.split('\n')[1][21:33])
            epoch_year = epoch_year + 1900 if epoch_year >= 57 else epoch_year + 2000
            epoch = Timestamp('%d' % (epoch_year,)) + (epoch_day - 1.0) * 24. * 3600.
            revs_per_day = float(target.split('\n')[2][53:64])
            # Use orbital period to distinguish near-earth and deep-space objects (which have different accuracies)
            orbital_period_mins = 24. / revs_per_day * 60.
            now = Timestamp()
            epoch_diff_days = np.abs(now - epoch) / 3600. / 24.
            direction = 'past' if epoch < now else 'future'
            # Near-earth models should be good for about a week (conservative estimate)
            if orbital_period_mins < 225 and epoch_diff_days > 7:
                num_outdated += 1
                if epoch_diff_days > max_epoch_diff_days:
                    worst = "Worst case: TLE epoch for '%s' is %d days in %s, should be <= 7 for near-earth model" % \
                            (name, epoch_diff_days, direction)
                    max_epoch_diff_days = epoch_diff_days
            # Deep-space models are more accurate (three weeks for a conservative estimate)
            if orbital_period_mins >= 225 and epoch_diff_days > 21:
                num_outdated += 1
                if epoch_diff_days > max_epoch_diff_days:
                    worst = "Worst case: TLE epoch for '%s' is %d days in %s, should be <= 21 for deep-space model" % \
                            (name, epoch_diff_days, direction)
                    max_epoch_diff_days = epoch_diff_days
        if num_outdated > 0:
            logger.warning('%d of %d TLE set(s) are outdated, probably making them inaccurate for use right now' %
                           (num_outdated, len(targets)))
            logger.warning(worst)
        self.add(targets, tags)

    def add_edb(self, lines, tags=None):
        """Add XEphem database format (EDB) targets to catalogue.

        Examples of catalogue construction can be found in the :class:`Catalogue`
        documentation.

        Parameters
        ----------
        lines : sequence of strings
            List of lines containing a target per line (may also be file object)
        tags : string or sequence of strings, optional
            Tag or list of tags to add to targets (strings will be split on
            whitespace)

        Examples
        --------
        Here are some ways to add EDB targets to a catalogue:

        >>> from katpoint import Catalogue
        >>> cat = Catalogue()
        >>> cat.add_edb(file('hipparcos.edb'), tags='star')
        >>> lines = ['HYP71683,f|S|G2,14:39:35.88 ,-60:50:7.4 ,-0.010,2000,\n',
                     'HYP113368,f|S|A3,22:57:39.055,-29:37:20.10,1.166,2000,\n']
        >>> cat2.add_edb(lines)

        """
        targets = []
        for line in lines:
            if (line[0] == '#') or (len(line.strip()) == 0):
                continue
            targets.append('xephem,' + line.replace(',', '~'))
        self.add(targets, tags)

    def remove(self, name):
        """Remove target from catalogue.

        Parameters
        ----------
        name : string
            Name of target to remove (may also be an alternate name of target)

        """
        if _hash(name) in self.lookup:
            target = self[name]
            self.lookup.pop(_hash(target.name))
            for alias in target.aliases:
                self.lookup.pop(_hash(alias))
            self.targets.remove(target)

    def save(self, filename):
        """Save catalogue to file in CSV format.

        Parameters
        ----------
        filename : string
            Name of file to write catalogue to (overwriting existing contents)

        """
        file(filename, 'w').writelines([t.description + '\n' for t in self.targets])

    def closest_to(self, target, timestamp=None, antenna=None):
        """Determine target in catalogue that is closest to given target.

        The comparison is based on the apparent angular separation between the
        targets, as seen from the specified antenna and at the given time instant.

        Parameters
        ----------
        target : :class:`Target` object
            Target with which catalogue targets are compared
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp at which to evaluate target positions, in UTC seconds
            since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)

        Returns
        -------
        closest_target : :class:`Target` object or None
            Target in catalogue that is closest to given *target*, or None if
            catalogue is empty
        min_dist : float
            Angular separation between *target* and *closest_target*, in degrees

        """
        if len(self.targets) == 0:
            return None, 180.0
        dist = rad2deg(np.array([target.separation(tgt, timestamp, antenna) for tgt in self.targets]))
        closest = dist.argmin()
        return self.targets[closest], dist[closest]

    def iterfilter(self, tags=None, flux_limit_Jy=None, flux_freq_MHz=None, az_limit_deg=None, el_limit_deg=None,
                   dist_limit_deg=None, proximity_targets=None, timestamp=None, antenna=None):
        """Generator function which returns targets satisfying various criteria.

        This returns a (generator-)iterator which returns targets satisfying
        various criteria, one at a time. The standard use of this method is in a
        for-loop (i.e. ``for target in cat.iterfilter(...):``). This differs from
        the :meth:`filter` method in that all time-dependent criteria (such as
        elevation) may be evaluated at the time of the specific iteration, and
        not in advance as with :meth:`filter`. This simplifies finding the next
        suitable target during an extended observation of several targets.

        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. The string
            may contain multiple tags separated by whitespace. If None or an
            empty list, all tags are accepted. Remember that the body type is
            also a tag.
        flux_limit_Jy : float or sequence of 2 floats, optional
            Allowed flux density range, in Jy. If this is a single number, it is
            the lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        az_limit_deg : sequence of 2 floats, optional
            Allowed azimuth range, in degrees. It takes the form [left, right],
            where *left* is the leftmost or starting azimuth, and *right* is the
            rightmost or ending azimuth. If *right* is less than *left*, the
            azimuth angles range around +-180. If None, any azimuth is accepted.
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
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp at which to evaluate target positions, in UTC seconds since
            Unix epoch. If None, the current time *at each iteration* is used.
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)

        Returns
        -------
        iter : iterator object
            The generator-iterator object which will return filtered targets

        Raises
        ------
        ValueError
            If some required parameters are missing

        Examples
        --------
        Here are some ways to filter a catalogue iteratively:

        >>> from katpoint import Catalogue, Antenna
        >>> ant = Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        >>> cat = Catalogue(antenna=ant)
        >>> for t in cat.iterfilter(el_limit_deg=10):
                # Observe target t
                pass

        """
        tag_filter = tags is not None
        flux_filter = flux_limit_Jy is not None
        azimuth_filter = az_limit_deg is not None
        elevation_filter = el_limit_deg is not None
        proximity_filter = dist_limit_deg is not None
        # Copy targets to a new list which will be pruned by filters
        targets = list(self.targets)

        # First apply static criteria (tags, flux) which do not depend on timestamp
        if tag_filter:
            if isinstance(tags, basestring):
                tags = tags.split()
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
        # Quick fix to accommodate negative azimuth values (assumes target az is in range 0 to 360 degrees)
        if azimuth_filter:
            az_limit_deg = [az_limit_deg[0] % 360., az_limit_deg[1] % 360.]
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
            if (azimuth_filter or elevation_filter or proximity_filter) and latest_timestamp is None:
                latest_timestamp = Timestamp()
            # Iterate over targets until one is found that satisfies dynamic criteria
            for n, target in enumerate(targets):
                if azimuth_filter:
                    az_deg = rad2deg(target.azel(latest_timestamp, antenna)[0])
                    if az_limit_deg[1] > az_limit_deg[0]:
                        if (az_deg < az_limit_deg[0]) or (az_deg > az_limit_deg[1]):
                            continue
                    else:
                        if (az_deg > az_limit_deg[1]) and (az_deg < az_limit_deg[0]):
                            continue
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

    def filter(self, tags=None, flux_limit_Jy=None, flux_freq_MHz=None, az_limit_deg=None, el_limit_deg=None,
               dist_limit_deg=None, proximity_targets=None, timestamp=None, antenna=None):
        """Filter catalogue on various criteria.

        This returns a new catalogue containing the subset of targets that
        satisfy the given criteria. All criteria are evaluated at the same time
        instant. For real-time continuous filtering, consider using
        :meth:`iterfilter` instead.

        Parameters
        ----------
        tags : string, or sequence of strings, optional
            Tag or list of tags which targets should have. Tags prepended with
            a tilde (~) indicate tags which targets should *not* have. The string
            may contain multiple tags separated by whitespace. If None or an
            empty list, all tags are accepted. Remember that the body type is
            also a tag.
        flux_limit_Jy : float or sequence of 2 floats, optional
            Allowed flux density range, in Jy. If this is a single number, it is
            the lower limit, otherwise it takes the form [lower, upper]. If None,
            any flux density is accepted.
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        az_limit_deg : sequence of 2 floats, optional
            Allowed azimuth range, in degrees. It takes the form [left, right],
            where *left* is the leftmost or starting azimuth, and *right* is the
            rightmost or ending azimuth. If *right* is less than *left*, the
            azimuth angles range around +-180. If None, any azimuth is accepted.
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
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp at which to evaluate target positions, in UTC seconds
            since Unix epoch (defaults to now)
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

        Examples
        --------
        Here are some ways to filter a catalogue:

        >>> from katpoint import Catalogue, Antenna
        >>> ant = Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
        >>> cat = Catalogue(antenna=ant, flux_freq_MHz=1500)
        >>> cat1 = cat.filter(el_limit_deg=10)
        >>> cat2 = cat.filter(az_limit_deg=[150, -150])
        >>> cat3 = cat.filter(flux_limit_Jy=10)
        >>> cat4 = cat.filter(tags='special ~radec')
        >>> cat5 = cat.filter(dist_limit_deg=5, proximity_targets=cat['Sun'])

        """
        return Catalogue([target for target in
                          self.iterfilter(tags, flux_limit_Jy, flux_freq_MHz, az_limit_deg, el_limit_deg,
                                          dist_limit_deg, proximity_targets, timestamp, antenna)],
                         add_specials=False, antenna=self.antenna, flux_freq_MHz=self.flux_freq_MHz)

    def sort(self, key='name', ascending=True, flux_freq_MHz=None, timestamp=None, antenna=None):
        """Sort targets in catalogue.

        This returns a new catalogue with the target list sorted according to
        the given key.

        Parameters
        ----------
        key : {'name', 'ra', 'dec', 'az', 'el', 'flux'}, optional
            Sort the targets according to this field
        ascending : {True, False}, optional
            True if key should be sorted in ascending order
        flux_freq_MHz : float, optional
            Frequency at which to evaluate the flux density, in MHz
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp at which to evaluate target positions, in UTC seconds
            since Unix epoch (defaults to now)
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

    def visibility_list(self, timestamp=None, antenna=None, flux_freq_MHz=None, antenna2=None):
        """Print out list of targets in catalogue, sorted by decreasing elevation.

        This prints out the name, azimuth and elevation of each target in the
        catalogue, in order of decreasing elevation. The motion of the target at
        the given timestamp is indicated by a character code, which is '/' if
        the target is rising, '\' if it is setting, and '-' if it is stationary
        (i.e. if the elevation angle changes by less than 1 arcminute during the
        one-minute interval surrounding the timestamp).

        The method indicates the horizon itself by a line of dashes. It also
        displays the target flux density if a frequency is supplied, and the
        delay and fringe period if a second antenna is supplied. It is useful
        to quickly see which targets are visible (or will be soon).

        Parameters
        ----------
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp at which to evaluate target positions, in UTC seconds
            since Unix epoch (defaults to now)
        antenna : :class:`Antenna` object, optional
            Antenna which points at targets (defaults to default antenna)
        flux_freq_MHz : float, optional
            Frequency at which to evaluate flux density, in MHz
        antenna2 : :class:`Antenna` object, optional
            Second antenna of baseline pair (baseline vector points from
            *antenna* to *antenna2*), used to calculate delays and fringe rates
            per target

        """
        above_horizon = True
        timestamp = Timestamp(timestamp)
        if antenna is None:
            antenna = self.antenna
        if antenna is None:
            raise ValueError('Antenna object needed to calculate target position')
        title = "Targets visible from antenna '%s' at %s" % (antenna.name, timestamp.local())
        if flux_freq_MHz is None:
            flux_freq_MHz = self.flux_freq_MHz
        if flux_freq_MHz is not None:
            title += ', with flux density (Jy) evaluated at %g MHz' % (flux_freq_MHz,)
        if antenna2 is not None:
            title += " and fringe period (s) toward antenna '%s' at same frequency" % (antenna2.name)
        print title
        print
        print 'Target                        Azimuth    Elevation <    Flux Fringe period'
        print '------                        -------    --------- -    ---- -------------'
        for target in self.sort('el', timestamp=timestamp, antenna=antenna, ascending=False):
            az, el = target.azel(timestamp, antenna)
            delta_el = rad2deg(target.azel(timestamp + 30.0, antenna)[1] - target.azel(timestamp - 30.0, antenna)[1])
            el_code = '-' if (np.abs(delta_el) < 1.0 / 60.0) else ('/' if delta_el > 0.0 else '\\')
            # If no flux frequency is given, do not attempt to evaluate the flux, as it will fail
            flux = target.flux_density(flux_freq_MHz) if flux_freq_MHz is not None else np.nan
            if antenna2 is not None and flux_freq_MHz is not None:
                delay, delay_rate = target.geometric_delay(antenna2, timestamp, antenna)
                fringe_period = 1. / (delay_rate * flux_freq_MHz * 1e6)
            else:
                fringe_period = None
            if above_horizon and el < 0.0:
                # Draw horizon line
                print '--------------------------------------------------------------------------'
                above_horizon = False
            line = '%-24s %12s %12s %c' % (target.name, az.znorm, el, el_code)
            line = line + ' %7.1f' % (flux,) if not np.isnan(flux) else line + '        '
            if fringe_period is not None:
                line += '    %10.2f' % (fringe_period,)
            print line

# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  _catalogue_completer
# --------------------------------------------------------------------------------------------------

dict_lookup_match = re.compile(r"""(?:.*\=)?(.*)\[(?P<quote>['|"])(?!.*(?P=quote))(.*)$""")


def _catalogue_completer(context, event):
    """Custom IPython completer for catalogue name lookups.

    This is inspired by Darren Dale's custom dict-like completer for h5py.

    """
    # Parse command line as (ignored = )base['start_of_name
    base, start_of_name = dict_lookup_match.split(event.line)[1:4:2]

    # Avoid calling any functions during eval()...
    if '(' in base:
        raise TryNext

    # Obtain catalogue object from user namespace
    try:
        cat = eval(base, context.user_ns)
    except (NameError, AttributeError):
        try:
            # IPython version < 1.0
            cat = eval(base, context.shell.user_ns)
        except (NameError, AttributeError):
            raise TryNext

    # Only continue if this object is actually a Catalogue
    if not isinstance(cat, Catalogue):
        raise TryNext

    if readline:
        # Remove space and plus from delimiter list, so completion works past spaces and pluses in names
        readline.set_completer_delims(readline.get_completer_delims().replace(' ', '').replace('+', ''))

    return [name for name in cat.iternames() if name[:len(start_of_name)] == start_of_name]
