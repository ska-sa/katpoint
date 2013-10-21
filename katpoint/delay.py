"""Delay model.

This implements the basic delay model used to calculate the delay contribution
from each antenna.

"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DelayModel(object):
    # Number of parameters in full antenna delay model
    num_params = 6

    __doc__ = """Model of the delay contribution from a single antenna.

    Parameters
    ----------
    params : sequence of floats, or string, optional
        Parameters of full model, in metres (defaults to sequence of zeroes).
        If it is a string, it is interpreted as a whitespace-separated sequence
        of parameters as produced by the :attr:`description` property
        (ignoring commas). The parameters are assigned in Procrustean fashion:
        select the first %d parameters of *params* or the entire *params*,
        whichever is smallest, and set the unused parameters to zero
        (useful to load old versions of the model).

    """ % (num_params, num_params, num_params)
    def __init__(self, params=None):
        if params is None:
            params = np.zeros(self.num_params)
        elif isinstance(params, basestring):
            params = np.array([float(p.strip(', ')) for p in params.split()])
        params = np.asarray(params)
        if len(params) < self.num_params:
            padded = np.zeros(self.num_params)
            padded[:len(params)] = params
            params = padded
        else:
            discarded_actives = len(params[self.num_params:].nonzero()[0])
            if discarded_actives:
                logger.warning('Delay model received too many parameters (%d instead of %d), '
                               'and %d non-zero parameters will be discarded' %
                               (len(params), self.num_params, discarded_actives))
            params = params[:self.num_params]
        self.params = params

    def __repr__(self):
        """Short human-friendly string representation of delay model object."""
        return "<katpoint.DelayModel active_params=%d/%d at 0x%x>" % \
               (len(self.params.nonzero()[0]), self.num_params, id(self))

    def __str__(self):
        """Verbose human-friendly string representation of delay model object."""
        num_active = len(self.params.nonzero()[0])
        summary = "Delay model has %d parameters with %d active (non-zero)" % (self.num_params, num_active)
        if num_active == 0:
            return summary
        descr = ['POS_E = %7.3f m (antenna position: offset East of reference position)',
                 'POS_N = %7.3f m (antenna position: offset North of reference position)',
                 'POS_U = %7.3f m (antenna position: offset above reference position)',
                 'NIAO  = %7.3f m (non-intersecting axis offset - distance between az and el axes)',
                 'CAB_H = %7.3f m (electronic path / cable length for H feed)',
                 'CAB_V = %7.3f m (electronic path / cable length for V feed)']
        param_strs = [(descr[p] % self.params[p]) for p in range(self.num_params) if self.params[p] != 0.0]
        return summary + ':\n' + '\n'.join(param_strs)

    def __eq__(self, other):
        """Equality comparison operator."""
        return self.description == (other.description if isinstance(other, DelayModel) else other)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    @property
    def description(self):
        """String representation of delay model, sufficient to reconstruct it."""
        return ', '.join([('%.4f' % self.params[p]) for p in xrange(self.num_params)])


class CorrelatorDelays(object):
    """Correlator delay model."""
    def __init__(self, ants, ref_ant, freq):
        self.ants = ants
        self.ref_ant = ref_ant
        self.freq = freq
        self.inputs = None

    def delays(self, t_start, t_next):
        """"""
        pass

