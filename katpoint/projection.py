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
coincides with the origin of the plane. All projections in this module are
*zenithal* or *azimuthal* projections that map the sphere directly onto this
plane. The *y* coordinate axis in the plane points along the reference meridian
of longitude towards the north pole of the sphere (in the direction of
increasing elevation). The *x* coordinate axis is perpendicular to it and points
in the direction of increasing azimuth (which may be towards the right or left,
depending on whether the azimuth coordinate increases clockwise or
anti-clockwise).

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
  
- Gnomonic (**TAN**): This is commonly used in optical astronomy.

- Zenithal equidistant (**ARC**): This is commonly used for single-dish maps,
  and is obtained if relative (az, el) coordinates are directly plotted. It
  preserves angular distances.

- Stereographic (**STG**): This is useful to represent polar regions and large
  fields. It preserves circles.

Each projection typically has restrictions on the input domain and output range
of values, which are highlighted in the docstrings of the individual functions.
Each function in this module is also vectorised, and will operate on single
floating-point values as well as :mod:`numpy` arrays of floats. The standard
:mod:`numpy` broadcasting rules apply. It is therefore possible to have an
array of target points and a single reference point, or vice versa.

All coordinates in this module are in radians.

.. [1] Greisen, "Non-linear Coordinate Systems in AIPS," AIPS Memo 27, 1993.
.. [2] Greisen, "Additional Non-linear Coordinates in AIPS," AIPS Memo 46, 1993.
.. [3] Calabretta, Greisen, "Representations of celestial coordinates in
   FITS. II," Astronomy & Astrophysics, vol. 395, pp. 1077-1122, 2002.

"""

import numpy as np

#--------------------------------------------------------------------------------------------------
#--- Common
#--------------------------------------------------------------------------------------------------

def _sphere_to_plane_common(az0, el0, az, el):
    """Do calculations common to all zenithal/azimuthal projections."""
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    delta_az = az - az0
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)
    # Theta is the native latitude (0 at reference point, increases radially outwards)
    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    # Do basic orthographic projection: x = sin(theta) * sin(phi), y = sin(theta) * cos(phi)
    ortho_x = cos_el * sin_daz
    ortho_y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return ortho_x, ortho_y, cos_theta
    
#--------------------------------------------------------------------------------------------------
#--- Orthographic projection (SIN)
#--------------------------------------------------------------------------------------------------

def sphere_to_plane_sin(az0, el0, az, el):
    """Project sphere to plane using orthographic (SIN) projection.
    
    The orthographic projection requires the target point to be within the
    hemisphere centred on the reference point. The angular separation between
    the target and reference points should be less than or equal to pi/2
    radians. The output (x, y) coordinates are constrained to lie within or on
    the unit circle in the plane.
    
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
    
    Notes
    -----
    This implements the original SIN projection as in AIPS, not the generalised
    'slant orthographic' projection as in WCSLIB.
    
    """
    ortho_x, ortho_y, cos_theta = _sphere_to_plane_common(az0, el0, az, el)
    if np.any(cos_theta < 0.0):
        raise ValueError('Target point more than pi/2 radians away from reference point')
    # x = sin(theta) * sin(phi), y = sin(theta) * cos(phi)
    return ortho_x, ortho_y

def plane_to_sphere_sin(az0, el0, x, y):
    """Deproject plane to sphere using orthographic (SIN) projection.
    
    The orthographic projection requires the (x, y) coordinates to lie within
    or on the unit circle. The target point is constrained to lie within the
    hemisphere centred on the reference point.
    
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
    el = np.arcsin(sin_el)
    cos_el_cos_daz = cos_el0 * cos_theta - sin_el0 * y
    az = az0 + np.arctan2(x, cos_el_cos_daz)    
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Gnomonic projection (TAN)
#--------------------------------------------------------------------------------------------------

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
    ortho_x, ortho_y, cos_theta = _sphere_to_plane_common(az0, el0, az, el)
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

#--------------------------------------------------------------------------------------------------
#--- Zenithal equidistant projection (ARC)
#--------------------------------------------------------------------------------------------------

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
    ortho_x, ortho_y, cos_theta = _sphere_to_plane_common(az0, el0, az, el)
    theta = np.arccos(cos_theta)
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
    el = np.arcsin(sin_el)
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    az = az0 + np.arctan2(num, den)
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Stereographic projection (STG)
#--------------------------------------------------------------------------------------------------

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
    ortho_x, ortho_y, cos_theta = _sphere_to_plane_common(az0, el0, az, el)
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
    el = np.arcsin(sin_el)
    # The M-check in AIPS NEWPOS can be avoided by using arctan2 instead of arcsin.
    # This follows the same approach as in the AIPS code for ARC, and improves
    # azimuth accuracy substantially for large (x, y) values.
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    az = az0 + np.arctan2(num, den)
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Selector
#--------------------------------------------------------------------------------------------------

# Maps projection code to appropriate function
sphere_to_plane = {'SIN' : sphere_to_plane_sin, 
                   'TAN' : sphere_to_plane_tan, 
                   'ARC' : sphere_to_plane_arc,
                   'STG' : sphere_to_plane_stg}

plane_to_sphere = {'SIN' : plane_to_sphere_sin,
                   'TAN' : plane_to_sphere_tan, 
                   'ARC' : plane_to_sphere_arc,
                   'STG' : plane_to_sphere_stg}
