#!/bin/bash
#
# Build aips_projection Python module using f2py. Requires numpy and gfortran.
#
# On Mac OS 10.7 (Lion), f2py of the system numpy can be found at
# /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/bin/f2py
#

# On some systems the Python version is appended to f2py executable name (probably to avoid clashes)
if which f2py; then
  f2py_exe='f2py'
else
  pyver=`python -c "import sys; print '%d.%d' % sys.version_info[:2]"`
  f2py_exe='f2py'$pyver
fi
echo "Using f2py compiler '$f2py_exe'"

$f2py_exe -c -m aips_projection DIRCOS.F NEWPOS.F
if [ -f aips_projection.so ]; then
  mv -f aips_projection.so ..
fi
