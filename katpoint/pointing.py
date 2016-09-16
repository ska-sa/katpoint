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

"""Pointing model.

This implements a pointing model for a non-ideal antenna mount.

"""

import logging

import numpy as np

from .model import Parameter, Model
from .ephem_extra import rad2deg, deg2rad, angle_from_degrees

logger = logging.getLogger(__name__)


class PointingModel(Model):
    """Correct pointing using model of non-ideal antenna mount.

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
    model : file-like object, sequence of %d floats, or string, optional
        Model specification. If this is a file-like object, load the model
        from it. If this is a sequence of floats, accept it directly as the
        model parameters (defaults to sequence of zeroes). If it is a string,
        interpret it as a comma-separated (or whitespace-separated) sequence
        of parameters in their string form (i.e. a description string).

    """
    def __init__(self, model=None):
        # There are two main types of parameter: angles and scale factors
        def angle_to_string(a):
            return str(angle_from_degrees(a).znorm) if a else '0'

        def angle_param(name, doc):
            """Create angle-valued parameter."""
            return Parameter(name, 'deg', doc, from_str=angle_from_degrees,
                             to_str=angle_to_string)

        def scale_param(name, doc):
            """Create scale-valued parameter."""
            return Parameter(name, '', doc,
                             to_str=lambda s: ('%.9g' % (s,)) if s else '0')
        # Instantiate the relevant model parameters and register with base class
        params = []
        params.append(angle_param('P1', 'az offset = encoder bias - tilt around [tpoint -IA]'))
        params.append(angle_param('P2', 'az gravitational sag, should be 0.0'))
        params.append(angle_param('P3', 'left-right axis skew = non-perpendicularity of az/el axes [tpoint -NPAE]'))
        params.append(angle_param('P4', 'az box offset / collimation error = RF-axis misalignment [tpoint CA]'))
        params.append(angle_param('P5', 'tilt out = az ring tilted towards north [tpoint AN]'))
        params.append(angle_param('P6', 'tilt over = az ring tilted towards east [tpoint -AW]'))
        params.append(angle_param('P7', 'el offset = encoder bias - forward axis skew - el box offset [tpoint IE]'))
        params.append(angle_param('P8', 'gravity sag / Hooke law flexure / el centering error [tpoint ECEC/-TF]'))
        params.append(scale_param('P9', 'el excess scale factor [tpoint PEE1]'))
        params.append(angle_param('P10', 'ad hoc cos(el) term in delta_el, redundant with P8'))
        params.append(angle_param('P11', 'asymmetric sag / el centering error [tpoint ECES]'))
        params.append(scale_param('P12', 'az excess scale factor [tpoint -PAA1]'))
        params.append(angle_param('P13', 'az centering error [tpoint ACEC]'))
        params.append(angle_param('P14', 'az centering error [tpoint -ACES]'))
        params.append(angle_param('P15', 'elevation nod twice per az revolution [tpoint HECA2]'))
        params.append(angle_param('P16', 'elevation nod twice per az revolution [tpoint -HESA2]'))
        params.append(angle_param('P17', 'az encoder tilt [tpoint -HACA2]'))
        params.append(angle_param('P18', 'az encoder tilt [tpoint HASA2]'))
        params.append(angle_param('P19', 'high-order distortions in el encoder scale [tpoint HECE8]'))
        params.append(angle_param('P20', 'high-order distortions in el encoder scale [tpoint HESE8]'))
        params.append(angle_param('P21', 'elevation nod once per az revolution [tpoint -HECA]'))
        params.append(angle_param('P22', 'elevation nod once per az revolution [tpoint HESA]'))
        Model.__init__(self, params)
        self.set(model)
        # Fix docstrings to contain the number of parameters
        if '%d' in self.__class__.__doc__:
            self.__class__.__doc__ = self.__class__.__doc__ % (len(self), len(self))
        if '%d' in self.__class__.fit.im_func.__doc__:
            self.__class__.fit.im_func.__doc__ = self.__class__.fit.im_func.__doc__ % \
                (len(self), len(self))

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
            P9, P10, P11, P12, P13, P14, P15, \
            P16, P17, P18, P19, P20, P21, P22 = self.values()
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
            P9, P10, P11, P12, P13, P14, P15, \
            P16, P17, P18, P19, P20, P21, P22 = self.values()
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
        az, el = pointed_az - self['P1'], pointed_el - self['P7']
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
        self.set()
        sigma_params = np.zeros(len(self))

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
            return np.array(self.values()), sigma_params

        # Number of active parameters
        M = len(enabled_params)
        cos_el = np.cos(el)
        # Number of data points (az and el measurements count as separate data points)
        N = 2 * len(az)
        # Construct design matrix, containing weighted basis functions
        A = np.zeros((N, M))
        param_vector = np.zeros(len(self))
        for m, param in enumerate(enabled_params):
            # Create parameter vector that will select a single column of design matrix
            param_vector[:] = 0.0
            param_vector[param - 1] = 1.0
            self.fromlist(param_vector)
            basis_az, basis_el = self.offset(az, el)
            A[:, m] = np.hstack((basis_az * cos_el / sigma_daz, basis_el / sigma_del))
        # Measurement vector, containing weighted observed offsets
        b = np.hstack((delta_az * cos_el / sigma_daz, delta_el / sigma_del))
        # Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        param_vector[enabled_params - 1] = np.dot(Vt.T, np.dot(U.T, b) / s)
        self.fromlist(param_vector)
        # Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
        sigma_params[enabled_params - 1] = np.sqrt(np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1))
#        logger.info('Fit pointing model using %dx%d design matrix with condition number %.2f' % (N, M, s[0] / s[-1]))

        return param_vector, sigma_params
