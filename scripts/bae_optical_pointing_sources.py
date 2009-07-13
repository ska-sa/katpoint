#! /usr/bin/python
#
# Tool to help create BAE optical pointing source catalogue from following files:
# - hipparcos.edb [Hipparcos catalogue]
# - bae_stars.txt [Mapping of star names in BAE list to Hipparcos HIC numbers]
#
# Ludwig Schwardt
# 11 July 2009
#

import numpy as np

names = file('bae_stars.txt').readlines()
names = [[part.strip() for part in name.split(',')] for name in names]
lookup = {}
for name, num in names:
    lookup['HYP' + num] = name

inlines = file('hipparcos.edb').readlines()

outlines = []
for line in inlines:
    line = '~'.join([edb_field.strip() for edb_field in line.split(',')])
    try:
        outlines.append('%s, xephem, %s\n' % (lookup[line.partition('~')[0]], line.replace('HYP', 'HIC ')))
    except KeyError:
        continue

f = file('hic.txt','w')
f.writelines(np.sort(outlines))
f.close()
