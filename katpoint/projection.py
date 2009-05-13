"""Spherical projections."""

import numpy as np

#--------------------------------------------------------------------------------------------------
#--- Orthographic projection (SIN)
#--------------------------------------------------------------------------------------------------

def sphere_to_plane_SIN(az0, el0, az, el):
    """Project sphere to plane using orthographic (SIN) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions. The orthographic projection requires the target point to be
    within the hemisphere centred on the reference point. The angular separation
    between the target and reference points should be less than or equal to
    pi/2 radians. The output (x, y) coordinates are constrained to lie within 
    the unit circle in the plane.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    delta_az = az - az0
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)
    
    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    if np.any(cos_theta < 0.0):
        raise ValueError('Target point more than 90 degrees away from reference point')
    x = cos_el * sin_daz
    y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return x, y

def plane_to_sphere_SIN(az0, el0, x, y):
    """Deproject plane to sphere using orthographic (SIN) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions. The orthographic projection requires the (x, y) coordinates
    to lie within or on the unit circle. The target point is constrained to
    lie within the hemisphere centred on the reference point.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
        If the radius of (x, y) > 1.0
    
    Notes
    -----
    This implements the original SIN projection as in AIPS, not the generalised
    'slant orthographic' projection as in WCSLIB.
        
    """
    sin2_theta = x * x + y * y
    if sin2_theta > 1.0:
        raise ValueError('Length of (x, y) vector bigger than 1.0')
    cos_theta = np.sqrt(1.0 - sin2_theta)
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_el = sin_el0 * cos_theta + cos_el0 * y
    # This check in AIPS should never be triggered, as y and cos_theta are
    # between -1 and 1, and their convex combination will be too
    if np.any(np.abs(sin_el) > 1.0):
        raise ValueError('(x, y) coordinates out of range')
    el = np.arcsin(sin_el)
    cos_el_cos_daz = cos_el0 * cos_theta - sin_el0 * y
    # This check in AIPS throws out the case when cos(el) == 0, i.e. when
    # the target point lands on one of the poles of the sphere. (This is also
    # triggered when the reference point is at the pole and only an azimuthal
    # x-offset is given.) This is not such a big disaster - just use the
    # reference point azimuth value as guidance.
    #if (cos_el_cos_daz == 0.0) and (x == 0.0):
    #    raise ValueError('Target point too close to pole of sphere - azimuth undefined')
    az = az0 + np.arctan2(x, cos_el_cos_daz)    
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Gnomonic projection (TAN)
#--------------------------------------------------------------------------------------------------

def sphere_to_plane_TAN(az0, el0, az, el):
    """Project sphere to plane using gnomonic (TAN) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions. The gnomonic projection requires the target point to be
    within the hemisphere centred on the reference point. The angular separation
    between the target and reference points should be less than pi/2 radians.
    The output (x, y) coordinates are unrestricted.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    delta_az = az - az0
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)
    
    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    if np.any(cos_theta <= 0.0):
        raise ValueError('Target point 90 degrees or more away from reference point')
    x = cos_el * sin_daz
    y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return x / cos_theta, y / cos_theta

def plane_to_sphere_TAN(az0, el0, x, y):
    """Deproject plane to sphere using gnomonic (TAN) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions. The target point is constrained to lie within the hemisphere
    centred on the reference point.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
        If the radius of (x, y) > 1.0
    
    """
    tan2_theta = x * x + y * y
    # This is an unnecessarily strict check in AIPS, which restricts target
    # points to within pi/4 radians of the reference point
    if tan2_theta > 1.0:
        raise ValueError('Length of (x, y) vector bigger than 1.0')
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    # This term is cos(el) * cos(daz) / cos(theta)
    den = cos_el0 - y * sin_el0
    # This check in AIPS throws out the case when cos(el) == 0, i.e. when
    # the target point lands on one of the poles of the sphere. (This is also
    # triggered when the reference point is at the pole and only an azimuthal
    # x-offset is given.) This is not such a big disaster - just use the
    # reference point azimuth value as guidance.
    #if den == 0:
    #    raise ValueError('Target point too close to pole of sphere - azimuth undefined')
    az = az0 + np.arctan2(x, den)
    el = np.arctan(np.cos(az - az0) * (sin_el0 + y * cos_el0) / den)
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Zenithal equidistant projection (ARC)
#--------------------------------------------------------------------------------------------------

def sphere_to_plane_ARC(az0, el0, az, el):
    """Project sphere to plane using zenithal equidistant (ARC) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions.  The target point can be anywhere on the sphere except in a
    small region diametrically opposite the reference point, which get mapped
    to infinity. The output (x, y) coordinates are constrained to lie within a
    circle of radius pi radians centred on the origin in the plane.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    delta_az = az - az0
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)

    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if theta == 0.0:
        scale = 1.0
    else:
        scale = theta / np.sin(theta)
    x = cos_el * sin_daz
    y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return scale * x, scale * y

def plane_to_sphere_ARC(az0, el0, x, y):
    """Deproject plane to sphere using zenithal equidistant (ARC) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions.  The input (x, y) coordinates should lie within a circle of
    radius pi radians centred on the origin in the plane.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
        If the radius of (x, y) > pi
    
    """
    theta = np.sqrt(x * x + y * y)
    if theta > np.pi:
        raise ValueError('Length of (x, y) vector bigger than pi')
    cos_theta = np.cos(theta)
    if theta == 0.0:
        scale = 1.0
    else:
        scale = np.sin(theta) / theta
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    sin_el = cos_el0 * scale * y + sin_el0 * cos_theta
    # This check in AIPS should never be triggered, as scale * y and cos_theta are
    # between -1 and 1, and their convex combination will be too
    if np.any(np.abs(sin_el) > 1.0):
        raise ValueError('(x, y) coordinates out of range')
    el = np.arcsin(sin_el)
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    # This check in AIPS throws out the case when cos(el) == 0 or cos(el0) == 0,
    # i.e. when the target or reference point lands on one of the poles of the
    # sphere. This is not such a big disaster - just use the reference point
    # azimuth value as guidance.
    #if (num == 0) and (den == 0):
    #    raise ValueError('Target or reference point too close to pole of sphere - azimuth undefined')
    az = az0 + np.arctan2(num, den)
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Stereographic projection (STG)
#--------------------------------------------------------------------------------------------------

def sphere_to_plane_STG(az0, el0, az, el):
    """Project sphere to plane using stereographic (STG) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions. The target point can be anywhere on the sphere except in a
    small region diametrically opposite the reference point, which get mapped
    to infinity.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
    if np.any(np.abs(el0) > np.pi / 2.0) or np.any(np.abs(el) > np.pi / 2.0):
        raise ValueError('Elevation angle outside range of +- pi/2 radians')
    sin_el, cos_el, sin_el0, cos_el0 = np.sin(el), np.cos(el), np.sin(el0), np.cos(el0)
    delta_az = az - az0
    sin_daz, cos_daz = np.sin(delta_az), np.cos(delta_az)

    cos_theta = sin_el * sin_el0 + cos_el * cos_el0 * cos_daz
    den = 1.0 + cos_theta
    if np.any(den < 1e-5):
        raise ValueError('Target point ~180 degrees away from reference point')
    x = cos_el * sin_daz
    y = sin_el * cos_el0 - cos_el * sin_el0 * cos_daz
    return 2.0 * x / den, 2.0 * y / den

def plane_to_sphere_STG(az0, el0, x, y):
    """Deproject plane to sphere using stereographic (STG) projection.
    
    The elevation ranges between -pi/2 and +pi/2 radians, while azimuth has no
    restrictions.
    
    The y coordinate axis points along the reference meridian of longitude
    towards the north pole of the sphere (in the direction of increasing
    elevation). The x coordinate axis is perpendicular to it and points in the
    direction of increasing azimuth (which may be on the right or left of y).
    If the target or reference point is at one of the poles of the sphere, the
    axes are still aligned to the reference meridian.
    
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
        If the radius of (x, y) > pi
    
    """
    sin_el0, cos_el0 = np.sin(el0), np.cos(el0)
    # This is the square of 2 sin(theta) / (1 + cos(theta))
    r2 = x * x + y * y
    cos_theta = (4.0 - r2) / (4.0 + r2)
    # This check in AIPS will never get triggered, since r2 >= 0 implies
    # abs(cos_theta) <= 1
    #if np.any(np.abs(cos_theta) > 1.0):
    #    raise ValueError('(x, y) coordinates out of range')
    scale = (1.0 + cos_theta) / 2.0
    sin_el = cos_el0 * scale * y + sin_el0 * cos_theta
    # This check in AIPS will also never be triggered, as scale = 4 / (4 + r2)
    # and scale * y reaches a maximum of 1 for (x,y) = (0,2), so both cos_theta
    # and scale * y lie between -1 and 1, as does their convex combination
    if np.any(np.abs(sin_el) > 1.0):
        raise ValueError('(x, y) coordinates out of range')
    el = np.arcsin(sin_el)
    # cos_el = np.cos(el)
    # This check in AIPS throws out the case when cos(el) ~ 0, i.e. when
    # the target point lands on one of the poles of the sphere. This is not
    # such a big disaster - just use the reference point azimuth value as guidance.
    #if np.any(np.abs(cos_el) < 1e-5):
    #    raise ValueError('Target point too close to pole of sphere - azimuth undefined')
    # The M-check in AIPS can be avoided by using arctan2 instead of arcsin
    # This follows the same approach as in the AIPS code for ARC
    # This term is cos(el) * cos(el0) * sin(delta_az)
    num = x * scale * cos_el0
    # This term is cos(el) * cos(el0) * cos(delta_az)
    den = cos_theta - sin_el * sin_el0
    az = az0 + np.arctan2(num, den)
    return az, el

#--------------------------------------------------------------------------------------------------
#--- Selector
#--------------------------------------------------------------------------------------------------

sphere_to_plane = {'SIN' : sphere_to_plane_SIN, 
                   'TAN' : sphere_to_plane_TAN, 
                   'ARC' : sphere_to_plane_ARC,
                   'STG' : sphere_to_plane_STG}

plane_to_sphere = {'SIN' : plane_to_sphere_SIN,
                   'TAN' : plane_to_sphere_TAN, 
                   'ARC' : plane_to_sphere_ARC,
                   'STG' : plane_to_sphere_STG}
