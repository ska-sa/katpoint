"""Module that abstracts pointing and related coordinate transformations.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Antenna class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

import logging as _logging

from .antenna import Antenna, construct_antenna

from .target import Target, construct_target, construct_azel_target, construct_radec_target

from .catalogue import Catalogue, specials, _catalogue_completer

from .ephem_extra import Timestamp, lightspeed, rad2deg, deg2rad

from .projection import sphere_to_plane, plane_to_sphere

# Hide submodules in module namespace, to avoid confusion with corresponding class names
# pylint: disable-msg=E0601
_antenna, _target, _catalogue, _ephem_extra, _projection = antenna, target, catalogue, ephem_extra, projection
del antenna, target, catalogue, ephem_extra, projection

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
