"""Delay model.

This implements the basic delay model used to calculate the delay contribution
from each antenna.

"""

import logging

import numpy as np

from .model import Parameter, Model
from .conversion import azel_to_enu
from .ephem_extra import lightspeed

# Speed of EM wave in fixed path (typically due to cables / electronics).
# This number is not critical - only meant to convert delays to "nice" lengths.
# E.g. typical factors are: RF over fibre = 0.7 (KAT-7), coax = 0.84 (MeerKAT).
# Since MeerKAT has a mix of free space and coax anyway, pick something between
# KAT-7 and full speed.
FIXEDSPEED = 0.85 * lightspeed

logger = logging.getLogger(__name__)


class DelayModel(Model):
    """Model of the delay contribution from a single antenna.

    Parameters
    ----------
    model : file-like object, sequence of floats, or string, optional
        Model specification. If this is a file-like object, load the model
        from it. If this is a sequence of floats, accept it directly as the
        model parameters (defaults to sequence of zeroes). If it is a string,
        interpret it as a comma-separated (or whitespace-separated) sequence
        of parameters in their string form (i.e. a description string).

    """
    def __init__(self, model=None):
        # Instantiate the relevant model parameters and register with base class
        params = []
        params.append(Parameter('POS_E', 'm', 'antenna position: offset East of reference position'))
        params.append(Parameter('POS_N', 'm', 'antenna position: offset North of reference position'))
        params.append(Parameter('POS_U', 'm', 'antenna position: offset above reference position'))
        params.append(Parameter('NIAO',  'm', 'non-intersecting axis offset - distance between az and el axes'))
        params.append(Parameter('FIX_H', 'm', 'fixed additional path length for H feed due to electronics / cables'))
        params.append(Parameter('FIX_V', 'm', 'fixed additional path length for V feed due to electronics / cables'))
        Model.__init__(self, params)
        self.set(model)
        # The EM wave velocity associated with each parameter
        self._speeds = np.array([lightspeed] * 4 + [FIXEDSPEED] * 2)

    @property
    def delay_params(self):
        """The model parameters converted to delays in seconds."""
        return np.array(self.values()) / self._speeds

    def fromdelays(self, delays):
        """Update model from a sequence of delay parameters."""
        self.fromlist(delays * self._speeds)


class CorrelatorDelays(object):
    """Calculate delay corrections for a set of correlator inputs / antennas."""
    def __init__(self, ants, ref_ant, sky_centre_freq):
        self.ants = ants
        self.ref_ant = ref_ant
        self.sky_centre_freq = sky_centre_freq
        self.inputs = [ant.name + pol for ant in ants for pol in ('h', 'v')]
        self._params = np.array([ant.delay_model.delay_params for ant in ants])
        # With no antennas, let params still have correct shape
        if not self._params:
            self._params = np.empty((0, len(DelayModel())))
        self._cache = {}
        self.max_delay = self._calculate_max_delay()

    def _calculate_max_delay(self):
        """The maximum (absolute) delay achievable in the array in seconds."""
        max_delay_per_ant = np.linalg.norm(self._params[:, :3], axis=1)
        max_delay_per_ant += self._params[:, 3]
        max_delay_per_ant += self._params[:, 4:6].max(axis=1)
        # Add a 1% safety margin to guarantee positive delay corrections
        return 1.01 * max(max_delay_per_ant) if max_delay_per_ant else 0.0

    def _calculate_delays(self, target, timestamp):
        """Calculate delays for all inputs / antennas for a given target."""
        az, el = target.azel(timestamp, self.ref_ant)
        targetdir = np.array(azel_to_enu(az, el))
        cos_el = np.cos(el)
        design_mat = np.array([np.r_[-targetdir, cos_el, 1.0, 0.0],
                               np.r_[-targetdir, cos_el, 0.0, 1.0]])
        return np.dot(self._params, design_mat.T).ravel()

    def _cached_delays(self, target, timestamp):
        """Try to load delays from cache, else calculate it."""
        delays = self._cache.pop(timestamp, None)
        if delays is None:
            delays = self._calculate_delays(target, timestamp)
            self._cache[timestamp] = delays
        return delays

    def corrections(self, target, timestamp, next_timestamp=None):
        """Delay and phase corrections for a given target and timestamp."""
        delays = self._cached_delays(target, timestamp)
        omega_centre = 2.0 * np.pi * self.sky_centre_freq
        delay_corrections = self.max_delay - delays
        phase_corrections = - omega_centre * delays
        if not next_timestamp:
            return dict(zip(self.inputs, delay_corrections)), \
                   dict(zip(self.inputs, phase_corrections))
        step = next_timestamp - timestamp
        next_delays = self._cached_delays(target, next_timestamp)
        next_delay_corrections = self.max_delay - next_delays
        delay_slopes = (next_delay_corrections - delay_corrections) / step
        next_phase_corrections = - omega_centre * next_delays
        phase_slopes = (next_phase_corrections - phase_corrections) / step
        return dict(zip(self.inputs, np.c_[delay_corrections, delay_slopes])), \
               dict(zip(self.inputs, np.c_[phase_corrections, phase_slopes]))
