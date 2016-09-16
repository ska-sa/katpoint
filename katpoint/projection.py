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

"""Spherical projections.

This module provides a basic set of routines that projects spherical coordinates
onto a plane and deprojects the plane coordinates back to the sphere. It
complements the ephem module, which focuses on transformations between various
spherical coordinate systems instead. The routines are derived from AIPS, as
documented in [1]_ and [2]_ and implemented in the DIRCOS and NEWPOS routines in
the 31DEC08 release, with minor improvements. The projections are
referred to by their AIPS (and FITS) codes, as also described in [3]_ and
implemented in Calabretta's WCSLIB. The (x, y) coordinates in this module
correspond to the (L, M) direction cosines calculated in [1]_ and [2]_.

Any spherical coordinate system can be used in the projections, as long as the
target and reference points are expressed in the same system of longitude and
latitude. The latitudinal coordinate is referred to as *elevation*, but could
also be geodetic latitude or declination. It ranges between -pi/2 and pi/2
radians, with zero representing the equator, pi/2 the north pole and -pi/2 the
south pole.

The longitudinal coordinate is referred to as *azimuth*, but could also be
geodetic longitude or right ascension. It can be any value in radians. The fact
that azimuth increases clockwise while right ascension and geodetic longitude
increase anti-clockwise is not a concern, as it simply changes the direction
of the *x*-axis on the plane (which is defined to point in the direction of
increasing longitudinal coordinate).

The projection plane is tangent to the sphere at the reference point, which also
coincides with the origin of the plane. All projections in this module (except
the plate carree projection) are *zenithal* or *azimuthal* projections that map
the sphere directly onto this plane. The *y* coordinate axis in the plane points
along the reference meridian of longitude towards the north pole of the sphere
(in the direction of increasing elevation). The *x* coordinate axis is
perpendicular to it and points in the direction of increasing azimuth (which may
be towards the right or left, depending on whether the azimuth coordinate
increases clockwise or anti-clockwise).

If the reference point is at a pole, its azimuth angle is undefined and the
reference meridian is therefore arbitrary. Nevertheless, the (x, y) axes are
still aligned to this meridian, with the *y* axis pointing away from the
intersection of the meridian with the equator for the north pole, and towards
the intersection for the south pole. The axes at the poles can therefore be seen
as a continuation of the axes obtained while moving along the reference meridian
from the equator to the pole.

The following projections are implemented:

- Orthographic (**SIN**): This is the standard projection in aperture synthesis
  radio astronomy, as it ties in closely with the 2-D Fourier imaging equation
  and the resultant (l, m) coordinate system. It is the simple orthographic
  projection of AIPS and [1]_, not the generalised slant orthographic projection
  of [3]_.

- Gnomonic (**TAN**): This is commonly used in optical astronomy. Great circles
  are projected as straight lines, so that the shortest distance between two
  points on the sphere is represented as a straight line interval (non-uniformly
  divided though).

- Zenithal equidistant (**ARC**): This is commonly used for single-dish maps,
  and is obtained if relative (cross-el, el) coordinates are directly plotted
  (cross-elevation is azimuth scaled by the cosine of elevation). It preserves
  angular distances from the reference point.

- Stereographic (**STG**): This is useful to represent polar regions and large
  fields. It preserves angles and circles.

- Plate carree (**CAR**): This is a very simple cylindrical projection that
  directly maps azimuth and elevation to a rectangular (*x*, *y*) grid, and
  returns offsets from the reference point on this plane. The *x* offset is
  therefore equal to the azimuth offset, while the *y* offset is equal to the
  elevation offset. It does not preserve angles, distances or circles.

- Swapped orthographic (**SSN**): This is the standard SIN projection with the
  roles of reference and target points reversed. It is useful for holography
  and other beam pattern measurements where a dish moves relative to a fixed
  beacon but the beam pattern is referenced to the boresight of the moving dish.

Each projection typically has restrictions on the input domain and output range
of values, which are highlighted in the docstrings of the individual functions.
Each function in this module is also vectorised, and will operate on single
floating-point values as well as :mod:`numpy` arrays of floats. The standard
:mod:`numpy` broadcasting rules apply. It is therefore possible to have an
array of target points and a single reference point, or vice versa.

All coordinates in this module are in radians.

These projections are normally accessed via the :class:`katpoint.Target` object
by calling its :meth:`katpoint.Target.sphere_to_plane` and
:meth:`katpoint.Target.plane_to_sphere` methods, e.g.::

  tgt = katpoint.Target('Sun, special')
  ant = katpoint.Antenna('XDM, -25:53:23, 27:41:03, 1406, 15.0')
  tgt.antenna = ant
  # Map from (ra, dec) coordinates to (l, m) plane with target as phase centre
  l, m = tgt.sphere_to_plane(ra, dec, projection_type='SIN', coord_system='radec')
  # Find (az, el) coordinates that scans dish relative to target position
  az, el = tgt.plane_to_sphere(x, y, projection_type='ARC', coord_system='azel')

Alternatively they can be called directly::

  x, y = katpoint.sphere_to_plane['ARC'](az0, el0, az, el)
  az, el = katpoint.plane_to_sphere['ARC'](az0, el0, x, y)

.. [1] Greisen, "Non-linear Coordinate Systems in AIPS," AIPS Memo 27, 1993.
.. [2] Greisen, "Additional Non-linear Coordinates in AIPS," AIPS Memo 46, 1993.
.. [3] Calabretta, Greisen, "Representations of celestial coordinates in
   FITS. II," Astronomy & Astrophysics, vol. 395, pp. 1077-1122, 2002.

"""

import numpy as np

# --------------------------------------------------------------------------------------------------
# --- Common
# --------------------------------------------------------------------------------------------------


def sphere_to_ortho(az0, el0, az, el):
    """Do calculations common to all zenithal/azimuthal projections."""
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    # Keep azimuth delta between -pi and pi - probably unnecessary, but the only normalisation of az inputs
    delta_az = (az - az0 + np.pi) % (2.0 * np.pi) - np.pi
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)
    # Theta is the native latitude (0 at reference point, increases radially outwards)
    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    # Do basic orthographic projection: x = sin(theta) * sin(phi), y = sin(theta) * cos(phi)
    ortho_x = cos_el * sin_daz
    ortho_y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return ortho_x, ortho_y, cos_theta

# --------------------------------------------------------------------------------------------------
# --- Orthographic projection (SIN)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_sin(az0, el0, az, el):
    """Project sphere to plane using orthographic (SIN) projection.

    The orthographic projection requires the target point to be within the
    hemisphere centred on the reference point. The angular separation between
    the target and reference points should be less than or equal to pi/2
    radians. The output (x, y) coordinates are constrained to lie within or on
    the unit circle in the plane.

    This is the standard projection in aperture synthesis radio astronomy as
    found in the 2-D Fourier imaging equation. The (x, y) coordinates are
    equivalent to the (l, m) coordinates found in the image plane when the
    reference point is treated as the phase centre and the celestial longitude
    and latitude are picked to be right ascension and declination, respectively.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane (equivalent to l), in radians
    y : float or array
        Elevation-like coordinate(s) on plane (equivalent to m), in radians

    Raises
    ------
    ValueError
        If input values are out of range, or target is too far from reference

    Notes
    -----
    This implements the original SIN projection as in AIPS, not the generalised
    'slant orthographic' projection as in WCSLIB.

    """
    ortho_x, ortho_y, cos_theta = sphere_to_ortho(az0, el0, az, el)
    if np.any(cos_theta < 0.0):
        raise ValueError('Target point more than pi/2 radians away from reference point')
    # x = sin(theta) * sin(phi), y = sin(theta) * cos(phi)
    return ortho_x, ortho_y


def plane_to_sphere_sin(az0, el0, x, y):
    """Deproject plane to sphere using orthographic (SIN) projection.

    The orthographic projection requires the (x, y) coordinates to lie within
    or on the unit circle. The target point is constrained to lie within the
    hemisphere centred on the reference point.

    This is the standard deprojection in aperture synthesis radio astronomy as
    found in the 2-D Fourier imaging equation. The (x, y) coordinates are
    equivalent to the (l, m) coordinates found in the image plane when the
    reference point is treated as the phase centre and the celestial longitude
    and latitude are picked to be right ascension and declination, respectively.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane (equivalent to l), in radians
    y : float or array
        Elevation-like coordinate(s) on plane (equivalent to m), in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Raises
    ------
    ValueError
        If input values are out of range, or the radius of (x, y) > 1.0

    Notes
    -----
    This implements the original SIN projection as in AIPS, not the generalised
    'slant orthographic' projection as in WCSLIB.

    """
    if np.any(np.abs(el0) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin2_theta = x * x + y * y
    if np.any(sin2_theta > 1.0):
        raise ValueError('Length of (x, y) vector bigger than 1.0')
    cos_theta = np.sqrt(1.0 - sin2_theta)
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_el = sin_el0 * cos_theta + cos_el0 * y
    # Safeguard the arcsin - in AIPS, clipping triggered "answer undefined", but that seems too harsh
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    cos_el_cos_daz = cos_el0 * cos_theta - sin_el0 * y
    az = az0 + np.arctan2(x, cos_el_cos_daz)
    return az, el

# --------------------------------------------------------------------------------------------------
# --- Gnomonic projection (TAN)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_tan(az0, el0, az, el):
    """Project sphere to plane using gnomonic (TAN) projection.

    The gnomonic projection requires the target point to be within the
    hemisphere centred on the reference point. The angular separation between
    the target and reference points should be less than pi/2 radians.
    The output (x, y) coordinates are unrestricted.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Raises
    ------
    ValueError
        If input values are out of range, or target is too far from reference

    """
    ortho_x, ortho_y, cos_theta = sphere_to_ortho(az0, el0, az, el)
    if np.any(cos_theta <= 0.0):
        raise ValueError('Target point pi/2 radians or more away from reference point')
    # x = tan(theta) * sin(phi), y = tan(theta) * cos(phi)
    return ortho_x / cos_theta, ortho_y / cos_theta


def plane_to_sphere_tan(az0, el0, x, y):
    """Deproject plane to sphere using gnomonic (TAN) projection.

    The input (x, y) coordinates are unrestricted. The returned target point is
    constrained to lie within the hemisphere centred on the reference point.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Raises
    ------
    ValueError
        If input values are out of range

    """
    if np.any(np.abs(el0) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    # This term is cos(el) * cos(daz) / cos(theta)
    den = cos_el0 - y * sin_el0
    az = az0 + np.arctan2(x, den)
    el = np.arctan(np.cos(az - az0) * (sin_el0 + y * cos_el0) / den)
    return az, el

# --------------------------------------------------------------------------------------------------
# --- Zenithal equidistant projection (ARC)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_arc(az0, el0, az, el):
    """Project sphere to plane using zenithal equidistant (ARC) projection.

    The target point can be anywhere on the sphere. The output (x, y)
    coordinates are constrained to lie within or on a circle of radius pi
    radians centred on the origin in the plane.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Raises
    ------
    ValueError
        If input values are out of range

    """
    ortho_x, ortho_y, cos_theta = sphere_to_ortho(az0, el0, az, el)
    # Safeguard the arccos, as over-ranging happens occasionally due to round-off error
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if np.isscalar(theta):
        if theta == 0.0:
            scale = 1.0
        else:
            scale = theta / np.sin(theta)
    else:
        scale = np.ones(theta.shape)
        nonzero = (theta != 0.0)
        scale[nonzero] = theta[nonzero] / np.sin(theta[nonzero])
    # x = theta * sin(phi), y = theta * cos(phi)
    return scale * ortho_x, scale * ortho_y


def plane_to_sphere_arc(az0, el0, x, y):
    """Deproject plane to sphere using zenithal equidistant (ARC) projection.

    The input (x, y) coordinates should lie within or on a circle of radius pi
    radians centred on the origin in the plane. The target point can be anywhere
    on the sphere.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Raises
    ------
    ValueError
        If input values are out of range, or the radius of (x, y) > pi

    """
    if np.any(np.abs(el0) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    theta = np.sqrt(x * x + y * y)
    if np.any(theta > np.pi):
        raise ValueError('Length of (x, y) vector bigger than pi')
    cos_theta = np.cos(theta)
    if np.isscalar(theta):
        if theta == 0.0:
            scale = 1.0
        else:
            scale = np.sin(theta) / theta
    else:
        scale = np.ones(theta.shape)
        nonzero = (theta != 0.0)
        scale[nonzero] = np.sin(theta[nonzero]) / theta[nonzero]
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_el = cos_el0 * scale * y + sin_el0 * cos_theta
    # Safeguard the arcsin - in AIPS, clipping triggered "answer undefined", but that seems too harsh
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    az = az0 + np.arctan2(num, den)
    return az, el

# --------------------------------------------------------------------------------------------------
# --- Stereographic projection (STG)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_stg(az0, el0, az, el):
    """Project sphere to plane using stereographic (STG) projection.

    The target point can be anywhere on the sphere except in a small region
    diametrically opposite the reference point, which get mapped to infinity.
    The output (x, y) coordinates are unrestricted.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Raises
    ------
    ValueError
        If input values are out of range, or target point opposite to reference

    """
    ortho_x, ortho_y, cos_theta = sphere_to_ortho(az0, el0, az, el)
    den = 1.0 + cos_theta
    if np.any(den < 1e-5):
        raise ValueError('Target point too close to pi radians away from reference point')
    # x = 2 sin(theta) sin(phi) / (1 + cos(theta)), y = 2 sin(theta) cos(phi) / (1 + cos(theta))
    return 2.0 * ortho_x / den, 2.0 * ortho_y / den


def plane_to_sphere_stg(az0, el0, x, y):
    """Deproject plane to sphere using stereographic (STG) projection.

    The input (x, y) coordinates are unrestricted. The target point can be
    anywhere on the sphere.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Raises
    ------
    ValueError
        If input values are out of range

    """
    if np.any(np.abs(el0) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    # This is the square of 2 sin(theta) / (1 + cos(theta))
    r2 = x * x + y * y
    cos_theta = (4.0 - r2) / (4.0 + r2)
    scale = (1.0 + cos_theta) / 2.0
    sin_el = cos_el0 * scale * y + sin_el0 * cos_theta
    # Safeguard the arcsin - in AIPS, clipping triggered "answer undefined", but that seems too harsh
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    # The M-check in AIPS NEWPOS can be avoided by using arctan2 instead of arcsin.
    # This follows the same approach as in the AIPS code for ARC, and improves
    # azimuth accuracy substantially for large (x, y) values.
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    az = az0 + np.arctan2(num, den)
    return az, el

# --------------------------------------------------------------------------------------------------
# --- Plate carree projection (CAR)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_car(az0, el0, az, el):
    """Project sphere to plane using plate carree (CAR) projection.

    The target point can be anywhere on the sphere. The output (x, y)
    coordinates are likewise unrestricted.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    """
    return az - az0, el - el0


def plane_to_sphere_car(az0, el0, x, y):
    """Deproject plane to sphere using plate carree (CAR) projection.

    The input (x, y) coordinates are unrestricted. The target point can likewise
    be anywhere on the sphere.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    """
    return az0 + x, el0 + y

# --------------------------------------------------------------------------------------------------
# --- Swapped orthographic projection (SSN)
# --------------------------------------------------------------------------------------------------


def sphere_to_plane_ssn(az0, el0, az, el):
    """Project sphere to plane using swapped orthographic (SSN) projection.

    This is identical to the usual orthographic (SIN) projection, but with the
    roles of the reference point (az0, el0) and target point (az, el) swapped.
    It has the same restrictions as the orthographic projection, i.e. the
    angular separation between the target and reference points should be less
    than or equal to pi/2 radians. The output (x, y) coordinates are also
    constrained to lie within or on the unit circle in the plane.

    This projection is useful for holography and other beam pattern measurements
    where a dish moves relative to a fixed beacon but the beam pattern is
    referenced to the boresight of the moving dish. In this scenario the fixed
    beacon / source would be the reference point (as observed by the tracking
    antenna in holography) while the scanning antenna follows the target point.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane (similar to l), in radians
    y : float or array
        Elevation-like coordinate(s) on plane (similar to m), in radians

    Raises
    ------
    ValueError
        If input values are out of range, or target is too far from reference

    Notes
    -----
    This projection was originally introduced by Mattieu de Villiers for use
    in holography experiments.

    """
    return sphere_to_plane_sin(az, el, az0, el0)


def plane_to_sphere_ssn(az0, el0, x, y):
    """Deproject plane to sphere using swapped orthographic (SSN) projection.

    The swapped orthographic deprojection has more restrictions than the
    corresponding orthographic (SIN) deprojection:

    - The (x, y) coordinates should lie within or on the unit circle
    - The magnitude of the x coordinate should be less than cos(el0) radians
    - The y coordinate should satisfy
        y >= -sqrt(cos(el0) ** 2 - x ** 2), el0 >= 0, and
        y <=  sqrt(cos(el0) ** 2 - x ** 2), el0 < 0,
      to ensure that the target elevation is within pi/2 radians of reference
      elevation - the y domain is therefore bounded by two semicircles with
      radii 1 and cos(el0), respectively
    - The target azimuth will be within pi/2 radians of the reference azimuth

    This deprojection is useful for holography and other beam measurements
    where a dish moves relative to a fixed beacon but the beam pattern is
    referenced to the boresight of the moving dish. In this scenario the fixed
    beacon / source would be the reference point (as observed by the tracking
    antenna in holography) while the scanning antenna follows the target point.

    Please read the module documentation for the interpretation of the input
    parameters and return values.

    Parameters
    ----------
    az0 : float or array
        Azimuth / right ascension / longitude of reference point(s), in radians
    el0 : float or array
        Elevation / declination / latitude of reference point(s), in radians
    x : float or array
        Azimuth-like coordinate(s) on plane (similar to l), in radians
    y : float or array
        Elevation-like coordinate(s) on plane (similar to m), in radians

    Returns
    -------
    az : float or array
        Azimuth / right ascension / longitude of target point(s), in radians
    el : float or array
        Elevation / declination / latitude of target point(s), in radians

    Raises
    ------
    ValueError
        If input values are out of range, or (x, y) is outside valid domain

    Notes
    -----
    This projection was originally introduced by Mattieu de Villiers for use
    in holography experiments.

    """
    if np.any(np.abs(el0) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin2_theta = x * x + y * y
    if np.any(sin2_theta > 1.0):
        raise ValueError('Length of (x, y) vector bigger than 1.0')
    cos_theta = np.sqrt(1.0 - sin2_theta)
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_daz = -x / cos_el0
    if np.any(np.abs(sin_daz) > 1.0):
        raise ValueError('The x coordinate is outside range of +- cos(el0) radians')
    # Since delta_az = az - az0 is the azimuth angle of final (x, cos(theta), y)
    # unit vector and cos(theta) >= 0, delta_az is restricted to +-90 degrees,
    # making the use of arcsin OK here
    az = az0 + np.arcsin(sin_daz)
    # Because of restrictions of el0 and delta_az, cos(el0) cos(delta_az) >= 0
    cos_el0_cos_daz = cos_el0 * np.cos(az - az0)
    num = sin_el0 * cos_theta - cos_el0_cos_daz * y
    den = sin_el0 * y + cos_theta * cos_el0_cos_daz
    # Ensure that cos(el) denominator term is positive to have abs(el) <= 90 degrees
    if np.any(den < -1e-12):
        raise ValueError('The y coordinate causes el to be outside range of +- pi/2 radians')
    el = np.arctan2(num, den)
    return az, el

# --------------------------------------------------------------------------------------------------
# --- Top-level projection routines
# --------------------------------------------------------------------------------------------------

# Maps projection code to appropriate function
sphere_to_plane = {'SIN': sphere_to_plane_sin,
                   'TAN': sphere_to_plane_tan,
                   'ARC': sphere_to_plane_arc,
                   'STG': sphere_to_plane_stg,
                   'CAR': sphere_to_plane_car,
                   'SSN': sphere_to_plane_ssn}

plane_to_sphere = {'SIN': plane_to_sphere_sin,
                   'TAN': plane_to_sphere_tan,
                   'ARC': plane_to_sphere_arc,
                   'STG': plane_to_sphere_stg,
                   'CAR': plane_to_sphere_car,
                   'SSN': plane_to_sphere_ssn}
