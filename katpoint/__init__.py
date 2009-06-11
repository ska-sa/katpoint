"""Module that abstracts pointing and related coordinate transformations.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Antenna class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

from .antenna import Antenna, construct_antenna
from .target import Target, construct_target, construct_azel_target, construct_radec_target
from .catalogue import Catalogue
from .ephem_extra import lightspeed, rad2deg, deg2rad

from .projection import sphere_to_plane as _sphere_to_plane
from .projection import plane_to_sphere as _plane_to_sphere

#--------------------------------------------------------------------------------------------------
#--- Projections
#--------------------------------------------------------------------------------------------------

def sphere_to_plane(antenna, target, az, el, timestamps, projection_type='ARC', coord_system='azel'):
    """Project spherical coordinates to plane with target position as reference.
    
    This is a convenience function that projects spherical coordinates to a 
    plane with the target position as the origin of the plane. The function is
    vectorised and can operate on single or multiple timestamps, as well as
    single or multiple coordinate vectors. The spherical coordinates may be
    (az, el) or (ra, dec), and the projection type can also be specified.
    
    Parameters
    ----------
    antenna : :class:`antenna.Antenna` object
        Antenna pointing at target
    target : :class:`target.Target` object
        Target object serving as origin for projected coordinates
    az : float or array
        Azimuth or right ascension, in radians
    el : float or array
        Elevation or declination, in radians
    timestamps : float or array
        Local timestamp(s) in seconds since Unix epoch
    projection_type : {'ARC', 'SIN', 'TAN', 'STG'}, optional
        Type of spherical projection
    coord_system : {'azel', 'radec'}, optional
        Spherical coordinate system
    
    Returns
    -------
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians
    
    """
    if coord_system == 'radec':
        # The target (ra, dec) coordinates will serve as reference point on the sphere
        ref_ra, ref_dec = target.radec(antenna, timestamps)
        return _sphere_to_plane[projection_type](ref_ra, ref_dec, az, el)
    else:
        # The target (az, el) coordinates will serve as reference point on the sphere
        ref_az, ref_el = antenna.point(target, timestamps)
        return _sphere_to_plane[projection_type](ref_az, ref_el, az, el)

def plane_to_sphere(antenna, target, x, y, timestamps, projection_type='ARC', coord_system='azel'):
    """Deproject plane coordinates to sphere with target position as reference.
    
    This is a convenience function that deprojects plane coordinates to a
    sphere with the target position as the origin of the plane. The function is
    vectorised and can operate on single or multiple timestamps, as well as
    single or multiple coordinate vectors. The spherical coordinates may be
    (az, el) or (ra, dec), and the projection type can also be specified.
    
    Parameters
    ----------
    antenna : :class:`antenna.Antenna` object
        Antenna pointing at target
    target : :class:`target.Target` object
        Target object serving as origin for projected coordinates
    x : float or array
        Azimuth-like coordinate(s) on plane, in radians
    y : float or array
        Elevation-like coordinate(s) on plane, in radians
    timestamps : float or array
        Local timestamp(s) in seconds since Unix epoch
    projection_type : {'ARC', 'SIN', 'TAN', 'STG'}, optional
        Type of spherical projection
    coord_system : {'azel', 'radec'}, optional
        Spherical coordinate system
    
    Returns
    -------
    az : float or array
        Azimuth or right ascension, in radians
    el : float or array
        Elevation or declination, in radians
    
    """
    if coord_system == 'radec':
        # The target (ra, dec) coordinates will serve as reference point on the sphere
        ref_ra, ref_dec = target.radec(antenna, timestamps)
        return _plane_to_sphere[projection_type](ref_az, ref_el, x, y)
    else:
        # The target (az, el) coordinates will serve as reference point on the sphere
        ref_az, ref_el = antenna.point(target, timestamps)
        return _plane_to_sphere[projection_type](ref_az, ref_el, x, y)
