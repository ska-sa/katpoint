#!/usr/bin/env python
#
# Tool that converts the ATNF PSRCAT database into a katpoint Catalogue.
#
# This needs the psrcat.db file included with the PSRCAT package which can be
# downloaded from http://www.atnf.csiro.au/people/pulsar/psrcat/download.html.
#
# The default psrcat.db contains both (ra, dec) and ecliptic (lon, lat)
# coordinates while this script only handles the former. You therefore need
# to run PSRCAT on the basic file to produce a 'long' ephemeris file:
#
#   psrcat -db_file psrcat.db -e2 > psrcat_full.db
#
# The output catalogue is printed to stdout, which can be redirected to a file.
#
# Ludwig Schwardt
# 31 October 2014
#

import argparse
import re

import numpy as np

import katpoint


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--db-file',
                    help='Path to long version of psrcat.db file')
args = parser.parse_args()

if not args.db_file:
    raise RuntimeError("Please obtain a long ephemeris file from the PSRCAT package")

# Regexp that finds key-value pair associated with pulsar
key_val = re.compile(r'^([A-Z0-9_]+)\s+(\S+)')
# Regexp that finds SNR name associated with pulsar
snr = re.compile(r'SNR:(PWN:)*(.+?)[[(,;$]')
# Regexp that finds single flux measurement associated with pulsar
flux_bin = re.compile(r'^S(\d{3,4})$')

# Turn database file into list of dicts, one per pulsar
pulsars = []
psr = {}
for line in open(args.db_file):
    if line.startswith('#'):
        continue
    if line.startswith('@'):
        pulsars.append(psr)
        psr = {}
        continue
    kv_match = key_val.match(line)
    if not kv_match:
        continue
    key, val = kv_match.groups()
    psr[key] = val
if psr:
    pulsars.append(psr)

for psr in pulsars:
    names = psr['PSRJ']
    if 'PSRB' in psr:
        names += ' | *' + psr['PSRB']
    if 'ASSOC' in psr:
        snr_match = snr.match(psr['ASSOC'])
        if snr_match:
            names += ' | ' + snr_match.group(2)
    if 'RAJ' in psr:
        ra, dec = psr['RAJ'], psr['DECJ']
    else:
        raise RuntimeError("Please run 'psrcat -db_file psrcat.db -e2 > psrcat_full.db' to get radecs for each pulsar")

    # Extract all flux measurements of pulsar
    flux_matches = [flux_bin.match(k) for k in psr]
    freq = np.array([float(m.group(1)) for m in flux_matches if m])
    flux_mJy = np.array([float(psr[m.group(0)]) for m in flux_matches if m])
    if len(freq) == 0:
        flux_model = None
    else:
        # Fit Baars 1977 polynomial flux model: log10 S[Jy] = a + b*log10(f[MHz]) + c*(log10(f[MHz]))^2
        log_freq = np.log10(freq)
        log_flux = np.log10(flux_mJy / 1000.)
        order = 2 if len(log_flux) > 3 else 1 if len(log_flux) > 1 else 0
        flux_poly = np.polyfit(log_freq, log_flux, order)
        freq_range = [1000., 2000.]
        flux_model = katpoint.FluxDensityModel(freq_range[0], freq_range[1],
                                               flux_poly[::-1])
    description = '%s, radec psr, %s, %s' % (names, ra, dec)
    if flux_model:
        description += ', ' + flux_model.description
    print(description)
