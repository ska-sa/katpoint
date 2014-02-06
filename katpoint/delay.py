"""Delay model.

This implements the basic delay model used to calculate the delay contribution
from each antenna.

"""

import logging

import numpy as np

from .model import Parameter, Model


logger = logging.getLogger(__name__)


class DelayModel(Model):
    """Model of the delay contribution from a single antenna.

    Parameters
    ----------
    model : file-like object, sequence of %d floats, or string, optional
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
        params.append(Parameter('CAB_H', 'm', 'electronic path / cable length for H feed'))
        params.append(Parameter('CAB_V', 'm', 'electronic path / cable length for V feed'))
        Model.__init__(self, params)
        self.set(model)
        # Fix docstrings to contain the number of parameters
        if '%d' in self.__class__.__doc__:
            self.__class__.__doc__ = self.__class__.__doc__ % (len(self),)
        # if '%d' in self.__class__.fit.im_func.__doc__:
        #     self.__class__.fit.im_func.__doc__ = self.__class__.fit.im_func.__doc__ % \
        #                                          (len(self), len(self))


class CorrelatorDelays(object):
    """Correlator delay model."""
    def __init__(self, ants, ref_ant, sky_centre_freq):
        self.ants = ants
        self.ref_ant = ref_ant
        self.sky_centre_freq = sky_centre_freq
        self.inputs = None

    def delays(self, target, t_start, t_next):
        """"""
        pass
