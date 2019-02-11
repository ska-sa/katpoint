#! /usr/bin/python

################################################################################
# Copyright (c) 2009-2019, National Research Foundation (Square Kilometre Array)
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

#
# Tool that creates BAE optical pointing star catalogue from following files:
#
# - hipparcos.edb [Hipparcos Input Catalogue in xephem EDB format]
# - bae_stars.txt [Mapping of star names in BAE list to Hipparcos HIC numbers]
# - bae_stars2.txt [Similar mapping for second BAE list]
#
# The names in the BAE list are a mixture of traditional (proper) names, Hubble
# Space Telescope Guide Star Catalogue (GSC) numbers (which uses the Tycho
# Catalogue numbering) and Smithsonian Astrophysical Observatory Star Catalogue
# (SAO) numbers. Additionally, each star is provided with its Greek-letter name
# to avoid ambiguity and improve user experience. Each HIC number was checked
# against the SIMBAD entry for the star.
#
# Ludwig Schwardt
# 11 July 2009
#

import numpy as np
import matplotlib.pyplot as plt

import katpoint


# Create lookup that returns names for a given HIC number
names = open('bae_stars2.txt').readlines()
names = [[part.strip() for part in name.split(',')] for name in names]
lookup = {}
for name, num in names:
    lookup['HYP' + num] = name

inlines = open('hipparcos.edb').readlines()

# Start with Solar System bodies, and add stars found in list as xephem bodies
outlines = ['Jupiter, special\n', 'Mars, special\n', 'Moon, special\n']
for line in inlines:
    line = '~'.join([edb_field.strip() for edb_field in line.split(',')])
    try:
        outlines.append('%s, xephem, %s\n' % (lookup[line.partition('~')[0]], line.replace('HYP', 'HIC ')))
    except KeyError:
        continue

# Save results
f = open('bae_optical_pointing_sources.csv', 'w')
f.writelines("""# These are the sources to be used by BAE for optical pointing tests of the
# KAT-7 dishes, in response to Mantis ticket 460 (second BAE list).
# Compiled by Ludwig Schwardt from various sources on 6 November 2009.
#
# Stars from Hipparcos Input Catalogue, Version 2, originally from
# ftp://cdsarc.u-strasbg.fr/cats/I/196, downloaded from
# http://www.yvonnet.org/xephem/hipparcos.edb.gz in XEphem edb format.
# Created by bae_optical_pointing_sources.py, based on bae_stars2.txt.
#
""")
f.writelines(np.sort(outlines))
f.close()

# Test the catalogue
ant = katpoint.Antenna('KAT7, -30:43:16.71, 21:24:35.86, 1055, 12.0')
cat = katpoint.Catalogue(open('bae_optical_pointing_sources.csv'),
                         add_specials=False, antenna=ant)
timestamp = katpoint.Timestamp()
ra, dec = np.array([t.radec(timestamp) for t in cat]).transpose()
constellation = [t.aliases[0].partition(' ')[2][:3] if t.aliases else 'SOL' for t in cat]
ra, dec = katpoint.rad2deg(ra), katpoint.rad2deg(dec)
az, el = np.hstack([targ.azel([katpoint.Timestamp(timestamp + t)
                               for t in range(0, 24 * 3600, 30 * 60)]) for targ in cat])
az, el = katpoint.rad2deg(az), katpoint.rad2deg(el)

plt.figure(1)
plt.clf()
for n, c in enumerate(constellation):
    plt.text(ra[n], dec[n], c, ha='left', va='center', size='xx-small')
plt.axis([0, 360, -90, 90])
plt.xlabel('Right Ascension (degrees)')
plt.ylabel('Declination (degrees)')
plt.title("Catalogue seen from '%s' on %s UTC" % (ant.name, timestamp))

plt.figure(2)
plt.clf()
plt.plot(az, el, '*')
plt.axis([0, 360, 0, 90])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Elevation (degrees)')
plt.title("Catalogue seen from '%s'\nfor 24-hour period starting %s UT" % (ant.name, timestamp))

plt.show()
