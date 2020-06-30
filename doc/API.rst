.. katpoint documentation master file, created by
   sphinx-quickstart on Mon Jun 10 15:18:10 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Katpoint API and astropy integration options
============================================

katpoint.Antenna
----------------
These are the ways in which the MeerKAT telescope interacts with katpoint.Antenna (extracted from CAM / TM and SDP code):

Construction
^^^^^^^^^^^^
* Antenna(string) - by far the most common way
* Antenna(string, float, float, float, float) - corner case in katcbfsim
* Antenna(string, string, string, string, float, tuple) - corner case in katcbfsim
* Antenna(string, float, float, float) - corner case in katsdpimager - replace by ref ant code?

Popular methods and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* name
* description
* delay_model
* position_wgs84
* local_sidereal_time

katpoint.Target
---------------
These are the ways in which the MeerKAT telescope interacts with katpoint.Target (extracted from CAM / TM and SDP code):

Construction
^^^^^^^^^^^^
* Target(string) - by far the most common way
* Target(body=string, antenna=Antenna) - corner case in katportalclient - easily fixed

Popular methods and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* name
* description
* tags
* body_type
* separation
* radec
* azel
* parallactic_angle
* uvw_basis

Less popular methods
^^^^^^^^^^^^^^^^^^^^
* astrometric_radec
* apparent_radec
* galactic

katpoint.Timestamp
------------------

These are the ways in which the MeerKAT telescope interacts with katpoint.Timestamp (extracted from CAM / TM and SDP code):

Construction
^^^^^^^^^^^^

* Timestamp()
* Timestamp(float)

Popular methods and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* secs
* local
* to_mjd
* to_string

Helper functions
----------------

These are popular helper functions called by MeerKAT code:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* deg2rad - obsolete if using astropy units
* rad2deg - obsolete if using astropy units
* lightspeed - obsolete if using astropy constants
* plane_to_sphere[] - astropy wcs
* construct_azel_target - replace with class method, e.g. Target.from_azel
* construct_radec_target - replace with class method, e.g. Target.from_radec
* wrap_angle

Less popular ones:
^^^^^^^^^^^^^^^^^^
* ephem_extra.angle_from_degrees - obsolete if using astropy units
* lla_to_ecef - EarthLocation functionality
* ecef_to_lla - EarthLocation functionality
* ecef_to_enu - EarthLocation functionality
* azel_to_enu - EarthLocation functionality
* is_iterable


Problematic use cases
---------------------

Problematic use cases of Target.body and Antenna.observer (assumes PyEphem features):
* body.compute
* observer.next_rising(body, time)
* observer.next_setting(body, time)
* ephem.separation - use Target.separation instead
* ephem.degrees
* ephem.hours
