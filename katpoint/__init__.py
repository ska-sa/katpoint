"""Module that abstracts pointing and related coordinate transformations.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Antenna class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

from .antenna import Antenna, construct_antenna

from .target import Target, construct_target, construct_azel_target, construct_radec_target, separation

from .catalogue import Catalogue, specials

from .ephem_extra import lightspeed, rad2deg, deg2rad

from .projection import sphere_to_plane, plane_to_sphere
