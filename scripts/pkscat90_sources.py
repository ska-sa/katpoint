#! /usr/bin/python

################################################################################
# Copyright (c) 2009-2021, National Research Foundation (SARAO)
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
# Tool that extracts bright sources from Parkes Radio Sources Catalogue (PKSCAT90).
#
# This builds a katpoint catalogue from the included pkscat90_S1410_min_10Jy.vot file.
# This file is obtained as follows:
#
# - Visit the VizieR web site: http://vizier.u-strasbg.fr/
# - In the leftmost text entry field for the catalogue name, enter "pkscat90"
# - Click on "VIII/15/pkscat90" ("The Catalogue")
# - Select at least the following fields:
#     Jname Alias Ident RA2000 DE2000
#     all flux densities (S80 S178 S408 S1410 S2700 S5000 S8400 S22000)
# - Add a constraint to S1410 of ">=10", to select sources with L-Band flux above 10 Jy
# - Select "VOTable" output layout and click on "Submit query"
# - This downloads a file named vizier_votable.vot
# - Save file as pkscat90_S1410_min_10Jy.vot
#
# Thereafter, install the vo Python package from https://www.stsci.edu/trac/ssb/astrolib/
# (also referred to as votable2recarray). I used vo-0.5.tar.gz. Then this script can be
# run for the rest.
#
# Ludwig Schwardt
# 12 March 2010
#

import numpy as np
import matplotlib.pyplot as plt

import katpoint
from astropy.table import Table


# For each flux field in table, specify the name, centre frequency and start frequency (in MHz)
flux_bins = ['S80', 'S178', 'S408', 'S635', 'S1410', 'S2700', 'S5000', 'S8400', 'S22000']
freq = np.array([80.0, 178.0, 408.0, 635.0, 1410.0, 2700.0, 5000.0, 8400.0, 22000.0])
start = [20.0, 100.0, 200.0, 400.0, 750.0, 1500.0, 3000.0, 6000.0, 12000.0, 30000.0]
# List of anomalous flux fields that will be edited out for the purpose of fitting
anomalies = {'J0108+1320': 6, 'J0541-0154': 2, 'J2253+1608': 6}

# Load main table in one shot (don't verify, as the VizieR VOTables contain a deprecated DEFINITIONS element)
table = Table.read('pkscat90_S1410_min_10Jy.vot')

# Fit all sources onto one figure
plt.figure(1)
plt.clf()
plot_rows = int(np.ceil(np.sqrt(len(table))))
src_strings = []

# Iterate through sources
for n, src in enumerate(table):
    names = src['Jname']
    if len(src['Alias']) > 0:
        names += ' | *' + src['Alias']
    tags = 'radec J2000'
    ra = src['RA2000'].strip().replace(' ', ':')
    dec = src['DE2000'].strip().replace(' ', ':')
    flux = np.array([src[bin] for bin in flux_bins])
    # Edit out anomalous flux bins for specific sources to improve fit (these are outside KAT7 frequency band anyway)
    anomalous_flux, anomalous_bin = np.nan, -1
    if src['Jname'] in anomalies:
        anomalous_bin = anomalies[src['Jname']]
        anomalous_flux = flux[anomalous_bin]
        flux[anomalous_bin] = np.nan
    # Fit Baars 1977 polynomial flux model: log10 S[Jy] = a + b*log10(f[MHz]) + c*(log10(f[MHz]))^2
    flux_defined = ~np.isnan(flux)
    log_freq = np.log10(freq[flux_defined])
    log_flux = np.log10(flux[flux_defined])
    flux_poly = np.polyfit(log_freq, log_flux, 2 if len(log_flux) > 3 else 1 if len(log_flux) > 1 else 0)
    # Determine widest possible frequency range where flux is defined (ignore internal gaps in this range)
    defined_bins = flux_defined.nonzero()[0]
    freq_range = [start[defined_bins[0]], start[defined_bins[-1] + 1]]
    # For better or worse, extend range to at least KAT7 frequency band
    freq_range = [min(freq_range[0], 1000.0), max(freq_range[1], 2000.0)]
    flux_str = katpoint.FluxDensityModel(freq_range[0], freq_range[1], flux_poly[::-1]).description
    src_strings.append(', '.join((names, tags, ra, dec, flux_str)) + '\n')
    print(src_strings[-1].strip())

    # Display flux polynomial fits
    test_log_freq = np.linspace(np.log10(start[0]), np.log10(start[-1]), 200)
    test_log_flux = np.polyval(flux_poly, test_log_freq)
    plt.subplot(plot_rows, plot_rows, n + 1)
    plt.plot(log_freq, log_flux, 'ob')
    plt.plot(test_log_freq, test_log_flux, 'r')
    if not np.isnan(anomalous_flux):
        plt.plot(np.log10(freq[anomalous_bin]), np.log10(anomalous_flux), '*b')
    plt.xticks([])
    plt.yticks([])
    plt.axvspan(np.log10(freq_range[0]), np.log10(freq_range[1]), facecolor='g', alpha=0.5)

with open('parkes_source_list.csv', 'w') as f:
    f.writelines(src_strings)

plt.figtext(0.5, 0.93, 'Spectra (log S vs. log v) for %d sources' % (len(src_strings)), ha='center', va='center')
plt.show()
