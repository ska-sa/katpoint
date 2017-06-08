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

"""A Timestamp object."""

import time
import math

import numpy as np
import ephem


class Timestamp(object):
    """Basic representation of time, in UTC seconds since Unix epoch.

    This is loosely based on :class:`ephem.Date`. Its base representation
    of time is UTC seconds since the Unix epoch, i.e. the standard Posix
    timestamp (:class:`ephem.Date` uses UTC days since noon on 1899/12/31, or
    the *Dublin Julian Day*). Fractional seconds are allowed, as the basic data
    type is a Python (double-precision) float.

    The following input formats are accepted for a timestamp:

    - None, which uses the current time (the default).

    - A floating-point number, directly representing the number of UTC seconds
      since the Unix epoch. Fractional seconds are allowed.

    - A string with format 'YYYY-MM-DD HH:MM:SS.SSS' or 'YYYY/MM/DD HH:MM:SS.SSS',
      or any prefix thereof. Examples are '1999-12-31 12:34:56.789', '1999-12-31',
      '1999-12-31 12:34:56' and even '1999'. The input string is always in UTC.

    - A :class:`ephem.Date` object, which is the standard time representation
      in PyEphem.

    Parameters
    ----------
    timestamp : float, string, :class:`ephem.Date` object or None
        Timestamp, in various formats (if None, defaults to now)

    Arguments
    ---------
    secs : float
        Timestamp as UTC seconds since Unix epoch

    """
    def __init__(self, timestamp=None):
        if isinstance(timestamp, basestring):
            try:
                timestamp = ephem.Date(timestamp.strip().replace('-', '/'))
            except ValueError:
                raise ValueError("Timestamp string '%s' not in correct format - " % (timestamp,) +
                                 "should be 'YYYY-MM-DD HH:MM:SS' or 'YYYY/MM/DD HH:MM:SS' or prefix thereof " +
                                 "(all UTC, fractional seconds allowed)")
        if timestamp is None:
            self.secs = time.time()
        elif isinstance(timestamp, ephem.Date):
            timestamp = list(timestamp.tuple()) + [0, 0, 0]
            int_secs = math.floor(timestamp[5])
            frac_secs = timestamp[5] - int_secs
            timestamp[5] = int(int_secs)
            self.secs = time.mktime(timestamp) - time.timezone + frac_secs
        else:
            self.secs = float(timestamp)

    # Keep object small by using __slots__ instead of __dict__
    __slots__ = 'secs'

    def __repr__(self):
        """Short machine-friendly string representation of timestamp object."""
        return 'Timestamp(%s)' % repr(self.secs)

    def __str__(self):
        """Verbose human-friendly string representation of timestamp object."""
        return self.to_string()

    def __cmp__(self, other):
        """Compare timestamps based on chronological order."""
        return np.sign(self.secs - other.secs)

    def __add__(self, other):
        """Add seconds (as floating-point number) to timestamp and return result."""
        return Timestamp(self.secs + other)

    def __sub__(self, other):
        """Subtract seconds (or another timestamp) from timestamp and return result."""
        if isinstance(other, Timestamp):
            return self.secs - other.secs
        else:
            return Timestamp(self.secs - other)

    def __mul__(self, other):
        """Multiply timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs * other)

    def __div__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __truediv__(self, other):
        """Divide timestamp by numerical factor (useful for processing timestamps)."""
        return Timestamp(self.secs / other)

    def __radd__(self, other):
        """Add timestamp to seconds (as floating-point number) and return result."""
        return Timestamp(other + self.secs)

    def __iadd__(self, other):
        """Add seconds (as floating-point number) to timestamp in-place."""
        self.secs += other
        return self

    def __isub__(self, other):
        """Subtract seconds (as floating-point number) from timestamp in-place."""
        self.secs -= other
        return self

    def __float__(self):
        """Convert to floating-point UTC seconds."""
        return self.secs

    def __hash__(self):
        """Base hash on internal timestamp, just like equality operator."""
        return hash(self.secs)

    def local(self):
        """Convert timestamp to local time string representation (for display only)."""
        int_secs = math.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int_secs))
        timezone = time.strftime('%Z', time.localtime(int_secs))
        if frac_secs == 0.0:
            return '%s %s' % (datetime, timezone)
        else:
            return '%s%5.3f %s' % (datetime[:-1], float(datetime[-1]) + frac_secs, timezone)

    def to_string(self):
        """Convert timestamp to UTC string representation."""
        int_secs = math.floor(self.secs)
        frac_secs = np.round(1000.0 * (self.secs - int_secs)) / 1000.0
        if frac_secs >= 1.0:
            int_secs += 1.0
            frac_secs -= 1.0
        datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(int_secs))
        if frac_secs == 0.0:
            return datetime
        else:
            return '%s%5.3f' % (datetime[:-1], float(datetime[-1]) + frac_secs)

    def to_ephem_date(self):
        """Convert timestamp to :class:`ephem.Date` object."""
        int_secs = math.floor(self.secs)
        timetuple = list(time.gmtime(int_secs)[:6])
        timetuple[5] += self.secs - int_secs
        return ephem.Date(tuple(timetuple))

    def to_mjd(self):
        """Convert timestamp to Modified Julian Day (MJD)."""
        # Ephem dates are in Dublin Julian Days
        djd = self.to_ephem_date()
        return djd + 2415020 - 2400000.5
