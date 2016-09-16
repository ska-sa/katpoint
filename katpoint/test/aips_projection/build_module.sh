#!/bin/bash

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

#
# Build aips_projection Python module using f2py.
# Requires rsync, patch, numpy and gfortran.
#
# On Mac OS 10.7 (Lion), f2py of the system numpy can be found at
# /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/bin/f2py
#

# Obtain AIPS source files (keep URL up to date!)
aips_src=ftp.aoc.nrao.edu::31DEC16
rsync -auvz --timeout=120 --files-from=aips_files.lst --no-relative $aips_src .
for f in *.FOR; do mv $f ${f/.FOR/.F}; done
# Add f2py icing and comment out troublesome AIPS calls
patch -p0 < aips_files.patch

# On some systems the Python version is appended to f2py executable name (probably to avoid clashes)
if which f2py; then
  f2py_exe='f2py'
else
  pyver=`python -c "import sys; print '%d.%d' % sys.version_info[:2]"`
  f2py_exe='f2py'$pyver
fi
echo "Using f2py compiler '$f2py_exe'"

$f2py_exe -c -m aips_projection DIRCOS.F NEWPOS.F CELNAT.F NATCEL.F MOLGAM.F
if [ -f aips_projection.so ]; then
  mv -f aips_projection.so ..
fi
