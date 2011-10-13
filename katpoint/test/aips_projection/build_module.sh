#!/bin/bash
#
# Build aips_projection Python module using f2py. Requires numpy and gfortran.
#
# On Mac OS 10.7 (Lion), f2py of the system numpy can be found at
# /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/bin/f2py
#
f2py -c -m aips_projection DIRCOS.F NEWPOS.F
mv -f aips_projection.so ..
