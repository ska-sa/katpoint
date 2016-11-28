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

"""Module that abstracts pointing and related coordinate transformations.

This module provides a simplified interface to the underlying coordinate
library, and provides functionality lacking in it. It defines a Target and
Antenna class, analogous to the Body and Observer classes in PyEphem, and
provides spherical coordinate transformations and spherical projections.

Currently it only caters for PyEphem, but it could be extended to include ACSM
and CASA.

"""

import logging as _logging

from .target import Target, construct_azel_target, construct_radec_target
from .antenna import Antenna
from .timestamp import Timestamp
from .flux import FluxDensityModel
from .catalogue import Catalogue, specials, _catalogue_completer
from .ephem_extra import lightspeed, rad2deg, deg2rad, wrap_angle, is_iterable
from .conversion import (lla_to_ecef, ecef_to_lla, enu_to_ecef, ecef_to_enu,
                         azel_to_enu, enu_to_azel, hadec_to_enu, enu_to_xyz)
from .projection import sphere_to_plane, plane_to_sphere
from .model import Parameter, Model, BadModelFile
from .pointing import PointingModel
from .refraction import RefractionCorrection
from .delay import DelayModel, DelayCorrection

# Hide submodules in module namespace, to avoid confusion with corresponding class names
# If the module is reloaded, this will fail - ignore the resulting NameError
# pylint: disable-msg=E0601
try:
    _target, _antenna, _timestamp, _flux, _catalogue, _ephem_extra, \
        _conversion, _projection, _pointing, _refraction, _delay = \
        target, antenna, timestamp, flux, catalogue, ephem_extra, \
        conversion, projection, pointing, refraction, delay
    del target, antenna, timestamp, flux, catalogue, ephem_extra, \
        conversion, projection, pointing, refraction, delay
except NameError:
    pass

# Attempt to register custom IPython tab completer for catalogue name lookups (only when run from IPython shell)
try:
    # IPython 0.11 and above
    _ip = get_ipython()
except NameError:
    # IPython 0.10 and below (or normal Python shell)
    _ip = __builtins__.get('__IPYTHON__')
if hasattr(_ip, 'set_hook'):
    _ip.set_hook('complete_command', _catalogue_completer, re_key=r"(?:.*\=)?(.+?)\[")


# Setup library logger and add a print-like handler used when no logging is configured
class _NoConfigFilter(_logging.Filter):
    """Filter which only allows event if top-level logging is not configured."""
    def filter(self, record):
        return 1 if not _logging.root.handlers else 0
_no_config_handler = _logging.StreamHandler()
_no_config_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT))
_no_config_handler.addFilter(_NoConfigFilter())
logger = _logging.getLogger(__name__)
logger.addHandler(_no_config_handler)

# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])
# END VERSION CHECK
