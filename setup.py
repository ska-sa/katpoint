#!/usr/bin/env python

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

from setuptools import setup, find_packages


with open('README.rst') as readme:
    long_description = readme.read()

setup(name="katpoint",
      description="Karoo Array Telescope pointing coordinate library",
      long_description=long_description,
      author="Ludwig Schwardt",
      author_email="ludwig@ska.ac.za",
      packages=find_packages(),
      url='https://github.com/ska-sa/katpoint',
      license="BSD",
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Astronomy"],
      platforms=["OS Independent"],
      keywords="meerkat ska",
      zip_safe=False,
      setup_requires=['katversion'],
      use_katversion=True,
      test_suite="nose.collector",
      install_requires=[
          "numpy",
          "pyephem"
      ],
      tests_require=[
          "nose",
          "coverage",
          "nosexcover",
          "unittest2"
      ])
