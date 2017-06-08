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

"""Refraction correction.

This implements correction for refractive bending in the atmosphere.

"""

import logging

import numpy as np

from .ephem_extra import rad2deg, deg2rad, is_iterable

logger = logging.getLogger(__name__)


def refraction_offset_vlbi(el, temperature_C, pressure_hPa, humidity_percent):
    """Calculate refraction correction using model in VLBI Field System.

    This uses the refraction model in the VLBI Field System to calculate a
    correction to a given elevation angle to account for refractive bending in
    the atmosphere, based on surface weather measurements. Each input parameter
    can either be a scalar value or an array of values, as long as all arrays
    are of the same shape.

    Parameters
    ----------
    el : float or array
        Requested elevation angle(s), in radians
    temperature_C : float or array
        Ambient air temperature at surface, in degrees Celsius
    pressure_hPa : float or array
        Total barometric pressure at surface, in hectopascal (hPa) or millibars
    humidity_percent : float or array
        Relative humidity at surface, as a percentage in range [0, 100]

    Returns
    -------
    el_offset : float or array
        Refraction offset(s) in radians, which needs to be *added* to
        elevation angle(s) to correct it

    Notes
    -----
    The code is based on poclb/refrwn.c in Field System version 9.9.2, which
    was added on 2006-11-15. This is a C version (with typos fixed) of the
    Fortran version in polb/refr.f. As noted in the Field System
    documentation [1]_, the refraction model originated with the Haystack
    pointing system, but the documentation for this algorithm seems to have
    been lost. It agrees well with the DSN refraction model, though.

    References
    ----------
    .. [1] Himwich, "Station Programs," Mark IV Field System Reference Manual,
       Version 8.2, 1 September 1993, available at
       `<ftp://gemini.gsfc.nasa.gov/pub/fsdocs/stprog.pdf>`_

    """
    p = (0.458675e1, 0.322009e0, 0.103452e-1, 0.274777e-3, 0.157115e-5)
    cvt = 1.33289
    a = 40.
    b = 2.7
    c = 4.
    d = 42.5
    e = 0.4
    f = 2.64
    g = 0.57295787e-4

    # Compute SN (surface refractivity) (via dewpoint and water vapor partial pressure? [LS])
    rhumi = (100. - humidity_percent) * 0.9
    dewpt = temperature_C - rhumi * (0.136667 + rhumi * 1.33333e-3 + temperature_C * 1.5e-3)
    pp = p[0] + p[1] * dewpt + p[2] * dewpt ** 2 + p[3] * dewpt ** 3 + p[4] * dewpt ** 4
    temperature_K = temperature_C + 273.
    # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
    sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

    # Compute refraction at elevation (clipped at 1 degree to avoid cot(el) blow-up at horizon)
    el_deg = np.clip(rad2deg(el), 1.0, 90.0)
    aphi = a / ((el_deg + b) ** c)
    dele = -d / ((el_deg + e) ** f)
    zenith_angle = deg2rad(90. - el_deg)
    bphi = g * (np.tan(zenith_angle) + dele)
    # Threw out an (el < 0.01) check here, which will never succeed because el is clipped to be above 1.0 [LS]

    return deg2rad(bphi * sn - aphi)


class RefractionCorrection(object):
    """Correct pointing for refractive bending in atmosphere.

    This uses the specified refraction model to calculate a correction to a
    given elevation angle to account for refractive bending in the atmosphere,
    based on surface weather measurements. The refraction correction can also
    be undone, usually to refer the actual antenna position to the coordinate
    frame before corrections were applied.

    Parameters
    ----------
    model : string, optional
        Name of refraction model to use

    Raises
    ------
    ValueError
        If the specified refraction model is unknown

    """
    def __init__(self, model='VLBI Field System'):
        self.models = {'VLBI Field System': refraction_offset_vlbi}
        try:
            self.offset = self.models[model]
        except KeyError:
            raise ValueError("Unknown refraction correction model '%s' - should be one of %s" %
                             (model, self.models.keys()))
        self.model = model

    def __repr__(self):
        """Short human-friendly string representation of refraction correction object."""
        return "<katpoint.RefractionCorrection model='%s' at 0x%x>" % (self.model, id(self))

    def __eq__(self, other):
        """Equality comparison operator."""
        return isinstance(other, RefractionCorrection) and (self.model == other.model)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not (self == other)

    def __hash__(self):
        """Base hash on underlying model name, just like equality operator."""
        return hash((self.__class__, self.model))

    def apply(self, el, temperature_C, pressure_hPa, humidity_percent):
        """Apply refraction correction to elevation angle.

        Each input parameter can either be a scalar value or an array of values,
        as long as all arrays are of the same shape.

        Parameters
        ----------
        el : float or array
            Requested elevation angle(s), in radians
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal (hPa) or millibars
        humidity_percent : float or array
            Relative humidity at surface, as a percentage in range [0, 100]

        Returns
        -------
        refracted_el : float or array
            Elevation angle(s), corrected for refraction, in radians

        """
        return el + self.offset(el, temperature_C, pressure_hPa, humidity_percent)

    def reverse(self, refracted_el, temperature_C, pressure_hPa, humidity_percent):
        """Remove refraction correction from elevation angle.

        This undoes a refraction correction that resulted in the given elevation
        angle. It is the inverse of :meth:`apply`.

        Parameters
        ----------
        refracted_el : float or array
            Elevation angle(s), corrected for refraction, in radians
        temperature_C : float or array
            Ambient air temperature at surface, in degrees Celsius
        pressure_hPa : float or array
            Total barometric pressure at surface, in hectopascal (hPa) or millibars
        humidity_percent : float or array
            Relative humidity at surface, as a percentage in range [0, 100]

        Returns
        -------
        el : float or array
            Elevation angle(s) before refraction correction, in radians

        """
        # Maximum difference between input elevation and refraction-corrected version of final output elevation
        tolerance = deg2rad(0.01 / 3600)
        # Assume offset from corrected el is similar to offset from uncorrected el -> get lower bound on desired el
        close_offset = self.offset(refracted_el, temperature_C, pressure_hPa, humidity_percent)
        lower = refracted_el - 4 * np.abs(close_offset)
        # We know that corrected el > uncorrected el (mostly) -> this becomes upper bound on desired el
        upper = refracted_el + deg2rad(1. / 3600.)
        # Do binary search for desired el within this range (but cap iterations in case of a mishap)
        # This assumes that refraction-corrected elevation is monotone function of uncorrected elevation
        for iteration in xrange(40):
            el = 0.5 * (lower + upper)
            test_el = self.apply(el, temperature_C, pressure_hPa, humidity_percent)
            if np.all(np.abs(test_el - refracted_el) < tolerance):
                break
            # Handle both scalars and arrays (and lists) as cleanly as possible
            if not is_iterable(refracted_el):
                if test_el < refracted_el:
                    lower = el
                else:
                    upper = el
            else:
                lower = np.where(test_el < refracted_el, el, lower)
                upper = np.where(test_el > refracted_el, el, upper)
        else:
            logger.warning('Reverse refraction correction did not converge' +
                           ' in %d iterations - elevation differs by at most %f arcsecs' %
                           (iteration + 1, rad2deg(np.abs(test_el - refracted_el).max()) * 3600.))
        return el
