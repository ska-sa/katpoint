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
# Tool that downloads ATCA calibrator list via their web interface.
#
# Ludwig Schwardt
# 10 October 2011
#

import urllib2
import json
import time


# Download catalogue in JSON format and fix non-JSON string quotes
url = urllib2.urlopen('http://www.narrabri.atnf.csiro.au/cgi-bin/Calibrators/new'
                      '/calinfo_json.pl?action=caltable&rarange=0,24&decrange=-90,90')
table_str = url.read().replace("'", '"')
# Parse JSON and extract calibrator list
calibrators = json.loads(table_str)['calibrators']
# Convert to katpoint target description strings (ignoring planets)
descriptions = [('%(name)s, radec atca_cal, %(ra)s, %(dec)s\n' % cal).encode('utf-8') for cal in calibrators
                if cal['name'] not in ('uranus', 'jupiter')]
# Save catalogue to CSV file
with open('atca_calibrators.csv', 'w') as f:
    f.write('# ATCA calibrator list retrieved on %s\n' % (time.strftime('%Y-%m-%d', time.localtime()),))
    f.writelines(descriptions)
