"""Pointing corrections.

This implements correction for refractive bending in the atmosphere, and a
pointing model for a non-ideal antenna mount.

"""

import logging

import numpy as np
import ephem

from .ephem_extra import rad2deg, deg2rad, is_iterable

logger = logging.getLogger("katpoint.correction")

#--------------------------------------------------------------------------------------------------
#--- Refraction correction
#--------------------------------------------------------------------------------------------------

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
    pp = p[0] + p[1]*dewpt + p[2]*(dewpt**2) + p[3]*(dewpt**3) + p[4]*(dewpt**4)
    temperature_K = temperature_C + 273.
    # This looks like Smith & Weintraub (1953) or Crane (1976) [LS]
    sn = 77.6 * (pressure_hPa + (4810.0 * cvt * pp) / temperature_K) / temperature_K

    # Compute refraction at elevation (clipped at 1 degree to avoid cot(el) blow-up at horizon)
    el_deg = np.clip(rad2deg(el), 1.0, 90.0)
    aphi =  a / ((el_deg + b) ** c)
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
        self.models = {'VLBI Field System' : refraction_offset_vlbi}
        try:
            self.offset = self.models[model]
        except KeyError:
            raise ValueError("Unknown refraction correction model '%s' - should be one of %s" %
                             (model, self.models.keys()))
        self.model = model

    def __repr__(self):
        """Short human-friendly string representation of refraction correction object."""
        return "<katpoint.RefractionCorrection model='%s' at 0x%x>" % (self.model, id(self))

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

#--------------------------------------------------------------------------------------------------
#--- Pointing model
#--------------------------------------------------------------------------------------------------

def dynamic_doc(*args):
    """Decorator that updates a function docstring to allow string formatting."""
    def doc_updater(func):
        func.__doc__ = func.__doc__ % args
        return func
    return doc_updater

class PointingModel(object):
    # Number of parameters in full pointing model
    num_params = 22

    __doc__ = """Correct pointing using model of non-ideal antenna mount.

    The pointing model is the one found in the VLBI Field System and has the
    standard terms found in most pointing models, including the DSN and TPOINT
    models. These terms are numbered P1 to P%d. The first 8 have a standard
    physical interpretation related to misalignment of the mount coordinate
    system and gravitational deformation, while the rest are ad hoc parameters
    that model remaining systematic effects in the pointing error residuals.
    Gravitational deformation may be considered ad hoc, too. The pointing model
    is specialised for an alt-az mount.

    Parameters
    ----------
    params : sequence of %d floats, or string, optional
        Parameters of full model, in radians (defaults to sequence of zeroes).
        If it is a string, it is interpreted as a comma-separated (or whitespace-
        separated) sequence of parameters, in *degrees*, as produced by the
        :attr:`description` property. The string form is in degrees, as this
        will be stored in configuration files and is therefore considered to be
        user-facing.
    strict : {True, False}, optional
        If True, only accept exactly %d parameters in *params*. If False, do a
        Procrustean assignment: select the first %d parameters of *params* or
        the entire *params*, whichever is smallest, and set the unused parameters
        to zero (useful to load old versions of the model).

    Raises
    ------
    ValueError
        If the *params* vector has the wrong length and *strict* is True

    """ % (num_params, num_params, num_params, num_params)
    def __init__(self, params=None, strict=True):
        if params is None:
            params = np.zeros(self.num_params)
        elif isinstance(params, basestring):
            params = np.array([ephem.degrees(p.strip(', ')) for p in params.split()])
            # Fix P9 and P12, which are scale factors (not angles) and therefore should not be converted to rads
            if len(params) >= 9:
                params[8] = rad2deg(params[8])
            if len(params) >= 12:
                params[11] = rad2deg(params[11])
        params = np.asarray(params)
        if len(params) != self.num_params:
            if strict:
                raise ValueError(("Pointing model expects exactly %d parameters, but received %d" +
                                  " (use 'strict=False' to override)") % (self.num_params, len(params)))
            else:
                if len(params) < self.num_params:
                    padded = np.zeros(self.num_params)
                    padded[:len(params)] = params
                    params = padded
                else:
                    discarded_actives = len(params[self.num_params:].nonzero()[0])
                    if discarded_actives > 0:
                        logger.warning('Pointing model received too many parameters ' +
                                       '(%d instead of %d), and %d non-zero parameters will be discarded' %
                                       (len(params), self.num_params, discarded_actives))
                    params = params[:self.num_params]
        self.params = params

    def param_str(self, param, scale_format='%.9g'):
        """Human-friendly string representation of a specific parameter value.

        Parameters
        ----------
        param : integer
            Index of parameter (starts at **1** and corresponds to P-number)
        scale_format : string, optional
            Format string for P9 and P12, which are scale factors and not angles

        Returns
        -------
        param_str : string
            String representation of parameter

        """
        if self.params[param - 1] == 0.0:
            return '0'
        elif param in [9, 12]:
            return scale_format % self.params[param - 1]
        else:
            return str(ephem.degrees(self.params[param - 1]).znorm)

    def __repr__(self):
        """Short human-friendly string representation of pointing model object."""
        return "<katpoint.PointingModel active_params=%d/%d at 0x%x>" % \
               (len(self.params.nonzero()[0]), self.num_params, id(self))

    def __str__(self):
        """Verbose human-friendly string representation of pointing model object."""
        num_active = len(self.params.nonzero()[0])
        summary = "Pointing model has %d parameters with %d active (non-zero)" % (self.num_params, num_active)
        if num_active == 0:
            return summary
        descr = ['P1  = %12s deg [-IA] (az offset = encoder bias - tilt around)',
                 'P2  = %12s deg (az gravitational sag, should be 0.0)',
                 'P3  = %12s deg [-NPAE] (left-right axis skew = non-perpendicularity of az/el axes)',
                 'P4  = %12s deg [CA] (az box offset / collimation error = RF-axis misalignment)',
                 'P5  = %12s deg [AN] (tilt out = az ring tilted towards north)',
                 'P6  = %12s deg [-AW] (tilt over = az ring tilted towards east)',
                 'P7  = %12s deg [IE] (el offset = encoder bias - forward axis skew - el box offset)',
                 'P8  = %12s deg [ECEC/-TF] (gravity sag / Hooke law flexure / el centering error)',
                 'P9  = %12s     [PEE1] (el excess scale factor)',
                 'P10 = %12s deg (ad hoc cos(el) term in delta_el, redundant with P8)',
                 'P11 = %12s deg [ECES] (asymmetric sag / el centering error)',
                 'P12 = %12s     [-PAA1] (az excess scale factor)',
                 'P13 = %12s deg [ACEC] (az centering error)',
                 'P14 = %12s deg [-ACES] (az centering error)',
                 'P15 = %12s deg [HECA2] (elevation nod twice per az revolution)',
                 'P16 = %12s deg [-HESA2] (elevation nod twice per az revolution)',
                 'P17 = %12s deg [-HACA2] (az encoder tilt)',
                 'P18 = %12s deg [HASA2] (az encoder tilt)',
                 'P19 = %12s deg [HECE8] (high-order distortions in el encoder scale)',
                 'P20 = %12s deg [HESE8] (high-order distortions in el encoder scale)',
                 'P21 = %12s deg [-HECA] (elevation nod once per az revolution)',
                 'P22 = %12s deg [HESA] (elevation nod once per az revolution)']
        param_strs = [descr[p] % self.param_str(p + 1) for p in xrange(self.num_params) if self.params[p] != 0.0]
        return summary + ':\n' + '\n'.join(param_strs)

    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0212
    def description():
        """Class method which creates description property."""
        doc = 'String representation of pointing model, sufficient to reconstruct it.'
        def fget(self):
            return ', '.join([self.param_str(p + 1) for p in xrange(self.num_params)])
        return locals()
    description = property(**description())

    # pylint: disable-msg=R0914,C0103,W0612
    def offset(self, az, el):
        """Obtain pointing offset at requested (az, el) position(s).

        Parameters
        ----------
        az : float or sequence
            Requested azimuth angle(s), in radians
        el : float or sequence
            Requested elevation angle(s), in radians

        Returns
        -------
        delta_az : float or array
            Offset(s) that has to be *added* to azimuth to correct it, in radians
        delta_el : float or array
            Offset(s) that has to be *added* to elevation to correct it, in radians

        Notes
        -----
        The model is based on poclb/fln.c and poclb/flt.c in Field System version
        9.9.0. The C implementation differs from the official description in
        [1]_, introducing minor changes to the ad hoc parameters. In this
        implementation, the angle *phi* is fixed at 90 degrees, which hard-codes
        the model for a standard alt-az mount.

        The model breaks down at the pole of the alt-az mount, which is at zenith
        (an elevation angle of 90 degrees). At zenith, the azimuth of the antenna
        is undefined, and azimuth offsets produced by the pointing model may
        become arbitrarily large close to zenith. To avoid this singularity, the
        azimuth offset is capped by adjusting the elevation away from 90 degrees
        specifically in its calculation. This adjustment occurs within 6
        arcminutes of zenith.

        References
        ----------
        .. [1] Himwich, "Pointing Model Derivation," Mark IV Field System Reference
           Manual, Version 8.2, 1 September 1993, available at
           `<ftp://gemini.gsfc.nasa.gov/pub/fsdocs/model.pdf>`_

        """
        # Unpack parameters to make the code correspond to the maths
        P1, P2, P3, P4, P5, P6, P7, P8, \
        P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22 = self.params
        # Compute each trig term only once and store it
        sin_az, cos_az, sin_2az, cos_2az = np.sin(az), np.cos(az), np.sin(2 * az), np.cos(2 * az)
        sin_el, cos_el, sin_8el, cos_8el = np.sin(el), np.cos(el), np.sin(8 * el), np.cos(8 * el)
        # Avoid singularity at zenith by keeping cos(el) away from zero - this only affects az offset
        # Preserve the sign of cos(el), as this will allow for correct antenna plunging
        sec_el = np.sign(cos_el) / np.clip(np.abs(cos_el), deg2rad(6. / 60.), 1.0)
        tan_el = sin_el * sec_el

        # Obtain pointing correction using full VLBI model for alt-az mount (no P2 or P10 allowed!)
        delta_az = P1 + P3*tan_el - P4*sec_el + P5*sin_az*tan_el - P6*cos_az*tan_el + \
                   P12*az + P13*cos_az + P14*sin_az + P17*cos_2az + P18*sin_2az
        delta_el = P5*cos_az + P6*sin_az + P7 + P8*cos_el + \
                   P9*el + P11*sin_el + P15*cos_2az + P16*sin_2az + P19*cos_8el + P20*sin_8el + P21*cos_az + P22*sin_az

        return delta_az, delta_el

    def apply(self, az, el):
        """Apply pointing correction to requested (az, el) position(s).

        Parameters
        ----------
        az : float or sequence
            Requested azimuth angle(s), in radians
        el : float or sequence
            Requested elevation angle(s), in radians

        Returns
        -------
        pointed_az : float or array
            Azimuth angle(s), corrected for pointing errors, in radians
        pointed_el : float or array
            Elevation angle(s), corrected for pointing errors, in radians

        """
        delta_az, delta_el = self.offset(az, el)
        return az + delta_az, el + delta_el

    def _jacobian(self, az, el):
        """Jacobian matrix of pointing correction function.

        This evaluates the Jacobian matrix of the pointing correction function
        ``corraz, correl = f(az, el)`` (as implemented by the :meth:`apply`
        method) at the requested (az, el) coordinates. This is used by the
        :meth:`reverse` method to invert the correction function.

        Parameters
        ----------
        az, el : float or sequence
            Requested azimuth and elevation angle(s), in radians

        Returns
        -------
        d_corraz_d_az, d_corraz_d_el, d_correl_d_az, d_correl_d_el : float or array
            Elements of Jacobian matrix (or matrices)

        """
        # Unpack parameters to make the code correspond to the maths
        P1, P2, P3, P4, P5, P6, P7, P8, \
        P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22 = self.params
        # Compute each trig term only once and store it
        sin_az, cos_az, sin_2az, cos_2az = np.sin(az), np.cos(az), np.sin(2 * az), np.cos(2 * az)
        sin_el, cos_el, sin_8el, cos_8el = np.sin(el), np.cos(el), np.sin(8 * el), np.cos(8 * el)
        # Avoid singularity at zenith by keeping cos(el) away from zero - this only affects az offset
        # Preserve the sign of cos(el), as this will allow for correct antenna plunging
        sec_el = np.sign(cos_el) / np.clip(np.abs(cos_el), deg2rad(6. / 60.), 1.0)
        tan_el = sin_el * sec_el

        d_corraz_d_az = 1.0 + P5*cos_az*tan_el + P6*sin_az*tan_el + \
                        P12 - P13*sin_az + P14*cos_az - P17*2*sin_2az + P18*2*cos_2az
        d_corraz_d_el = sec_el * (P3*sec_el - P4*tan_el + P5*sin_az*sec_el - P6*cos_az*sec_el)
        d_correl_d_az = -P5*sin_az + P6*cos_az - P15*2*sin_2az + P16*2*cos_2az - P21*sin_az + P22*cos_az
        d_correl_d_el = 1.0 - P8*sin_el + P9 + P11*cos_el - P19*8*sin_8el + P20*8*cos_8el

        return d_corraz_d_az, d_corraz_d_el, d_correl_d_az, d_correl_d_el

    def reverse(self, pointed_az, pointed_el):
        """Remove pointing correction from (az, el) coordinate(s).

        This undoes a pointing correction that resulted in the given (az, el)
        coordinates. It is the inverse of :meth:`apply`.

        Parameters
        ----------
        pointed_az : float or sequence
            Azimuth angle(s), corrected for pointing errors, in radians
        pointed_el : float or sequence
            Elevation angle(s), corrected for pointing errors, in radians

        Returns
        -------
        az : float or array
            Azimuth angle(s) before pointing correction, in radians
        el : float or array
            Elevation angle(s) before pointing correction, in radians

        """
        # Maximum difference between input az/el and pointing-corrected version of final output az/el
        tolerance = deg2rad(0.01 / 3600)
        # Initial guess of uncorrected az/el is the corrected az/el minus fixed offsets
        az, el = pointed_az - self.params[0], pointed_el - self.params[6]
        # Solve F(az, el) = apply(az, el) - (pointed_az, pointed_el) = 0 via Newton's method, should converge quickly
        for iteration in xrange(30):
            # Set up linear system J dx = -F (or A x = b), where J is Jacobian matrix of apply()
            a11, a12, a21, a22 = self._jacobian(az, el)
            test_az, test_el = self.apply(az, el)
            b1, b2 = pointed_az - test_az, pointed_el - test_el
            sky_error = np.sqrt((np.cos(el) * b1) ** 2 + b2 ** 2)
            if np.all(sky_error < tolerance):
                break
            # Newton step: Solve linear system via crappy Cramer rule... 3 reasons why this is OK:
            # (1) J is nearly an identity matrix, as long as model parameters are all small
            # (2) It allows parallel solution of many 2x2 systems, one per (az, el) input
            # (3) It's part of an iterative process, so it does not have to be perfect, just helpful
            det_J = a11 * a22 - a21 * a12
            az = az + (a22 * b1 - a12 * b2) / det_J
            el = el + (a11 * b2 - a21 * b1) / det_J
        else:
            max_error, max_az, max_el = np.vstack((sky_error, pointed_az, pointed_el))[:, np.argmax(sky_error)]
            logger.warning('Reverse pointing correction did not converge in ' +
                           '%d iterations - maximum error is %f arcsecs at (az, el) = (%f, %f) radians' %
                           (iteration + 1, rad2deg(max_error) * 3600., max_az, max_el))
        return az, el

    @dynamic_doc(num_params, num_params)
    def fit(self, az, el, delta_az, delta_el, sigma_daz=None, sigma_del=None, enabled_params=None):
        """Fit pointing model parameters to observed offsets.

        This fits the pointing model to a sequence of observed (az, el) offsets.
        A subset of the parameters can be fit, while the rest will be zeroed.
        This is generally a good idea, as most of the parameters (P9 and above)
        are ad hoc and should only be enabled if there are sufficient evidence
        for them in the pointing error residuals. Standard errors can be
        specified for the input offsets, and will be reflected in the returned
        standard errors on the fitted parameters.

        Parameters
        ----------
        az, el : sequence of floats, length *N*
            Requested azimuth and elevation angles, in radians
        delta_az, delta_el : sequence of floats, length *N*
            Corresponding observed azimuth and elevation offsets, in radians
        sigma_daz, sigma_del : sequence of floats, length *N*, optional
            Standard deviation of azimuth and elevation offsets, in radians
        enabled_params : sequence of ints or bools, optional
            List of model parameters that will be enabled during fitting,
            specified by a list of integer indices or boolean flags. The
            integers start at **1** and correspond to the P-number. The default
            is to select the 6 main parameters modelling coordinate misalignment,
            which are P1, P3, P4, P5, P6 and P7.

        Returns
        -------
        params : float array, shape (%d,)
            Fitted model parameters (full model), in radians
        sigma_params : float array, shape (%d,)
            Standard errors on fitted parameters, in radians

        Notes
        -----
        Since the standard pointing model is linear in the model parameters, it
        is fit with linear least-squares techniques. This is done by creating a
        design matrix and solving the linear system via singular value
        decomposition (SVD), as explained in [1]_.

        References
        ----------
        .. [1] Press, Teukolsky, Vetterling, Flannery, "Numerical Recipes in C,"
           2nd Ed., pp. 671-681, 1992. Section 15.4: "General Linear Least
           Squares", available at `<http://www.nrbook.com/a/bookcpdf/c15-4.pdf>`_

        """
        # Set default inputs
        if sigma_daz is None:
            sigma_daz = np.ones(np.shape(az))
        if sigma_del is None:
            sigma_del = np.ones(np.shape(el))
        # Ensure all inputs are numpy arrays of the same shape
        az, el, delta_az, delta_el = np.asarray(az), np.asarray(el), np.asarray(delta_az), np.asarray(delta_el)
        sigma_daz, sigma_del = np.asarray(sigma_daz), np.asarray(sigma_del)
        assert az.shape == el.shape == delta_az.shape == delta_el.shape == sigma_daz.shape == sigma_del.shape, \
               'Input parameters should all have the same shape'

        # Blank out the existing model
        self.params[:] = 0.0
        sigma_params = np.zeros(len(self.params))

        # Handle parameter enabling
        if enabled_params is None:
            enabled_params = [1, 3, 4, 5, 6, 7]
        enabled_params = np.asarray(enabled_params)
        # Convert boolean selection to integer indices
        if enabled_params.dtype == np.bool:
            enabled_params = enabled_params.nonzero()[0] + 1
        enabled_params = set(enabled_params)
        # Remove troublesome parameters if enabled
        if 2 in enabled_params:
            logger.warning('Pointing model parameter P2 is meaningless for alt-az mount - disabled P2')
            enabled_params.remove(2)
        if 10 in enabled_params:
            logger.warning('Pointing model parameter P10 is redundant for alt-az mount (same as P8) - disabled P10')
            enabled_params.remove(10)
        enabled_params = np.array(list(enabled_params))
        # If no parameters are enabled, a zero model is returned
        if len(enabled_params) == 0:
            return self.params, sigma_params

        # Number of active parameters
        M = len(enabled_params)
        cos_el = np.cos(el)
        # Number of data points (az and el measurements count as separate data points)
        N = 2 * len(az)
        # Construct design matrix, containing weighted basis functions
        A = np.zeros((N, M))
        for m, param in enumerate(enabled_params):
            # Create parameter vector that will select a single column of design matrix
            self.params[:] = 0.0
            self.params[param - 1] = 1.0
            basis_az, basis_el = self.offset(az, el)
            A[:, m] = np.hstack((basis_az * cos_el / sigma_daz, basis_el / sigma_del))
        # Measurement vector, containing weighted observed offsets
        b = np.hstack((delta_az * cos_el / sigma_daz, delta_el / sigma_del))
        # Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        self.params[enabled_params - 1] = np.dot(Vt.T, np.dot(U.T, b) / s)
        # Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
        sigma_params[enabled_params - 1] = np.sqrt(np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1))
#        logger.info('Fit pointing model using %dx%d design matrix with condition number %.2f' % (N, M, s[0] / s[-1]))

        return self.params, sigma_params
