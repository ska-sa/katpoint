"""Module that abstracts pointing and related coordinate transformations.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Antenna class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

import logging as _logging

from .antenna import Antenna

from .target import FluxDensityModel, Target, construct_azel_target, construct_radec_target

from .catalogue import Catalogue, specials, _catalogue_completer

from .ephem_extra import Timestamp, lightspeed, rad2deg, deg2rad, is_iterable

from .conversion import lla_to_ecef, ecef_to_lla, enu_to_ecef, ecef_to_enu, \
                        azel_to_enu, hadec_to_enu, enu_to_xyz

from .projection import sphere_to_plane, plane_to_sphere

from .correction import RefractionCorrection, PointingModel

# Hide submodules in module namespace, to avoid confusion with corresponding class names
# If the module is reloaded, this will fail - ignore the resulting NameError
# pylint: disable-msg=E0601
try:
    _antenna, _target, _catalogue, _ephem_extra, _conversion, _projection, _correction = \
        antenna, target, catalogue, ephem_extra, conversion, projection, correction
    del antenna, target, catalogue, ephem_extra, conversion, projection, correction
except NameError:
    pass

# Attempt to register custom IPython tab completer for catalogue name lookups
try:
    import IPython.ipapi as _ipapi
    _ip = _ipapi.get()
except ImportError:
    _ip = None
if not _ip is None:
    _ip.set_hook('complete_command', _catalogue_completer, re_key = r"(?:.*\=)?(.+?)\[")

# Setup library logger, and suppress spurious logger messages via a null handler
class _NullHandler(_logging.Handler):
    def emit(self, record):
        pass
logger = _logging.getLogger("katpoint")
logger.addHandler(_NullHandler())
