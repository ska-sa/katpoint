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

"""Delay model and correction.

This implements the basic delay model used to calculate the delay
contribution from each antenna, as well as a class that performs
delay correction for a correlator.

"""

import logging

import numpy as np

from .model import Parameter, Model
from .conversion import azel_to_enu
from .ephem_extra import lightspeed, is_iterable

# Speed of EM wave in fixed path (typically due to cables / clock distribution).
# This number is not critical - only meant to convert delays to "nice" lengths.
# Typical factors are: fibre = 0.7, coax = 0.84.
FIXEDSPEED = 0.7 * lightspeed

logger = logging.getLogger(__name__)


class DelayModel(Model):
    """Model of the delay contribution from a single antenna.

    This object is purely used as a repository for model parameters, allowing
    easy construction, inspection and saving of the delay model. The actual
    calculations happen in :class:`DelayCorrection`, which is more efficient
    as it handles multiple antenna delays simultaneously.

    Parameters
    ----------
    model : file-like or model object, sequence of floats, or string, optional
        Model specification. If this is a file-like or model object, load the
        model from it. If this is a sequence of floats, accept it directly as
        the model parameters (defaults to sequence of zeroes). If it is a
        string, interpret it as a comma-separated (or whitespace-separated)
        sequence of parameters in their string form (i.e. a description
        string). The default is an empty model.

    """
    def __init__(self, model=None):
        # Instantiate the relevant model parameters and register with base class
        params = []
        params.append(Parameter('POS_E', 'm', 'antenna position: offset East of reference position'))
        params.append(Parameter('POS_N', 'm', 'antenna position: offset North of reference position'))
        params.append(Parameter('POS_U', 'm', 'antenna position: offset above reference position'))
        params.append(Parameter('FIX_H', 'm', 'fixed additional path length for H feed due to electronics / cables'))
        params.append(Parameter('FIX_V', 'm', 'fixed additional path length for V feed due to electronics / cables'))
        params.append(Parameter('NIAO', 'm', 'non-intersecting axis offset - distance between az and el axes'))
        Model.__init__(self, params)
        self.set(model)
        # The EM wave velocity associated with each parameter
        self._speeds = np.array([lightspeed] * 3 + [FIXEDSPEED] * 2 + [lightspeed])

    @property
    def delay_params(self):
        """The model parameters converted to delays in seconds."""
        return np.array(self.values()) / self._speeds

    def fromdelays(self, delays):
        """Update model from a sequence of delay parameters.

        Parameters
        ----------
        delays : sequence of floats
            Model parameters in delay form (i.e. in seconds)

        """
        self.fromlist(delays * self._speeds)


class DelayCorrection(object):
    """Calculate delay corrections for a set of correlator inputs / antennas.

    This uses delay models from multiple antennas connected to a correlator to
    produce delay and phase corrections for a given target and timestamp, for
    all correlator inputs at once. The delay corrections are guaranteed to be
    strictly positive. Each antenna is assumed to have two polarisations (H
    and V), resulting in two correlator inputs per antenna.

    For now, the reference antenna position must match the reference positions
    of each antenna in the array, so that the ENU offset in each antenna's
    delay model directly represent the baseline between that antenna and the
    reference antenna. This should be fine as this is the standard case, but
    may cause problems for e.g. VLBI with a geocentric reference antenna.

    Parameters
    ----------
    ants : sequence of *M* :class:`Antenna` objects
        Sequence of antennas forming an array and connected to correlator
    ref_ant : :class:`Antenna` object
        Reference antenna for the array
    sky_centre_freq : float, optional
        RF centre frequency that serves as reference for fringe phase

    Attributes
    ----------
    inputs : list of *2M* strings
        List of correlator input names (typically 2 per antenna), aligned with
        calculated delays
    max_delay : float
        Maximum absolute delay achievable in array, in seconds, used to ensure
        strictly positive delay corrections

    Raises
    ------
    ValueError
        If all antennas do not share the same reference position as ref_ant

    """

    # Maximum size for delay cache
    CACHE_SIZE = 1000

    def __init__(self, ants, ref_ant, sky_centre_freq=0.0):
        self.ants = list(ants)
        self.ref_ant = ref_ant
        if any([ant.ref_position_wgs84 != ref_ant.position_wgs84
                for ant in self.ants + [ref_ant]]):
            msg = "Antennas '%s' do not all share the same reference " \
                  "position of the reference antenna %r" % \
                  ("', '".join(ant.description for ant in self.ants),
                   self.ref_ant.description)
            raise ValueError(msg)
        self.sky_centre_freq = sky_centre_freq
        self.inputs = [ant.name + pol for ant in ants for pol in ('h', 'v')]
        self._params = np.array([ant.delay_model.delay_params for ant in ants])
        # With no antennas, let params still have correct shape
        if not self.ants:
            self._params = np.empty((0, len(DelayModel())))
        self._cache = {}
        self.max_delay = self._calculate_max_delay()

    def _calculate_max_delay(self):
        """The maximum (absolute) delay achievable in the array, in seconds."""
        # Worst case is wavefront moving along baseline connecting ant to ref
        max_delay_per_ant = np.sqrt((self._params[:, :3] ** 2).sum(axis=1))
        # Pick largest fixed delay
        max_delay_per_ant += self._params[:, 3:5].max(axis=1)
        # Worst case for NIAO is looking at the horizon
        max_delay_per_ant += self._params[:, 5]
        # Add a 1% safety margin to guarantee positive delay corrections
        return 1.01 * max(max_delay_per_ant) if self.ants else 0.0

    def _calculate_delays(self, target, timestamp):
        """Calculate delays for all inputs / antennas for a given target.

        Parameters
        ----------
        target : :class:`Target` object
            Target providing direction for geometric delays
        timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp in UTC seconds since Unix epoch

        Returns
        -------
        delays : sequence of *2M* floats
            Delays (one per correlator input) in seconds

        """
        az, el = target.azel(timestamp, self.ref_ant)
        targetdir = np.array(azel_to_enu(az, el))
        cos_el = np.cos(el)
        design_mat = np.array([np.r_[-targetdir, 1.0, 0.0, cos_el],
                               np.r_[-targetdir, 0.0, 1.0, cos_el]])
        return np.dot(self._params, design_mat.T).ravel()

    def _cached_delays(self, target, timestamp):
        """Try to load delays from cache, else calculate it.

        This uses the timestamp to look up previously calculated delays in
        a cache. If not found, calculate the delays and store it in the
        cache instead. Each cache value is used only once. Clean out the
        oldest timestamp if cache is full.

        See :meth:`_calculate_delays` for parameter and return lists,
        as these two methods can be used interchangeably.

        """
        delays = self._cache.pop(timestamp, None)
        if delays is None:
            delays = self._calculate_delays(target, timestamp)
            # Clean out the oldest timestamp if cache is full
            while len(self._cache) >= DelayCorrection.CACHE_SIZE:
                self._cache.pop(min(self._cache.keys()))
            self._cache[timestamp] = delays
        return delays

    def corrections(self, target, timestamp=None, next_timestamp=None):
        """Delay and phase corrections for a given target and timestamp(s).

        Calculate delay and phase corrections for the direction towards
        *target* at *timestamp*. If the timestamp of the next delay
        calculation is provided, it is used to calculate a delay rate that can
        be used for linear interpolation in the period up to the next update.
        This process is repeated if a sequence of timestamps is given. Both
        delay (aka phase slope) and phase (aka phase offset or fringe phase)
        corrections are provided, and optionally their derivatives with
        respect to time (delay rate and fringe rate, respectively).

        Parameters
        ----------
        target : :class:`Target` object
            Target providing direction for geometric delays
        timestamp : :class:`Timestamp` object or equivalent, or sequence, optional
            Timestamp(s) in UTC seconds since Unix epoch when delays are
            evaluated (default is now). If more than one timestamp is given,
            the corrections will include slopes to be used for linear
            interpolation between the times
        next_timestamp : :class:`Timestamp` object or equivalent, optional
            Timestamp when next delay will be evaluated, used to determine
            a slope for linear interpolation (default is no slope). This is
            ignored if *timestamp* is a sequence.

        Returns
        -------
        delays : dict mapping string to float or array of floats
            Dict mapping correlator input name to delay correction,
            which consists of a delay value (in seconds) and optionally
            a delay rate value (in seconds per second). If a sequence
            of *T* timestamps are provided, each input maps to an array
            of shape (*T*, 2).
        phases : dict mapping string to float or array of floats
            Dict mapping correlator input name to phase correction, which
            consists of a fringe phase value (in radians) and optionally a
            fringe rate value (in radians per second). If a sequence of *T*
            timestamps are provided, each input maps to an array of shape
            (*T*, 2).

        """
        if is_iterable(timestamp):
            # Append one more timestamp to get a slope for the last timestamp
            last_step = timestamp[-1] - timestamp[-2]
            all_times = np.r_[timestamp, [timestamp[-1] + last_step]]
            next_timestamp = all_times[1:]
            # Don't use cache, as the next_times are included in all_delays
            all_delays = np.array([self._calculate_delays(target, t)
                                   for t in all_times]).T
            delays, next_delays = all_delays[:, :-1], all_delays[:, 1:]
        else:
            # Use cache for a single timestamp
            delays = self._cached_delays(target, timestamp)

        def phase(t0):
            """The phase associated with delay t0 at the centre frequency."""
            return - 2.0 * np.pi * self.sky_centre_freq * t0
        delay_corrections = self.max_delay - delays
        phase_corrections = - phase(delays)
        if next_timestamp is None:
            return (dict(zip(self.inputs, delay_corrections)),
                    dict(zip(self.inputs, phase_corrections)))
        step = next_timestamp - timestamp
        # We still have to get next_delays in the single timestamp case
        if not is_iterable(next_timestamp):
            next_delays = self._cached_delays(target, next_timestamp)
        next_delay_corrections = self.max_delay - next_delays
        next_phase_corrections = - phase(next_delays)
        delay_slopes = (next_delay_corrections - delay_corrections) / step
        phase_slopes = (next_phase_corrections - phase_corrections) / step
        # This construction works for both the scalar and vector cases.
        # The squeeze() gets rid of an extra singleton in the scalar case.
        # It is safe to squeeze as the other two dimensions involved will
        # never be singletons (number of inputs >= 2 even for 1 antenna, and
        # number of polynomial terms is 2 by design).
        delay_polys = np.dstack((delay_corrections, delay_slopes)).squeeze()
        phase_polys = np.dstack((phase_corrections, phase_slopes)).squeeze()
        return (dict(zip(self.inputs, delay_polys)),
                dict(zip(self.inputs, phase_polys)))
