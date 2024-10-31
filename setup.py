#!/usr/bin/env python

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

import os.path

from setuptools import setup, find_packages


here = os.path.dirname(__file__)
readme = open(os.path.join(here, 'README.md')).read()
news = open(os.path.join(here, 'NEWS.md')).read()
long_description = readme + '\n\n' + news

setup(name="katpoint",
      description="Karoo Array Telescope pointing coordinate library",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author="Ludwig Schwardt",
      author_email="ludwig@ska.ac.za",
      packages=find_packages(),
      url='https://github.com/ska-sa/katpoint',
      license="Modified BSD",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",
          "Programming Language :: Python :: 3.13",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Astronomy"],
      platforms=["OS Independent"],
      keywords="meerkat ska",
      zip_safe=False,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      setup_requires=['katversion'],
      use_katversion=True,
      test_suite="nose.collector",
      install_requires=[
          "future",
          "numpy",
          "ephem",
      ],
      tests_require=[
          "nose",
          "coverage",
          "nosexcover",
          "unittest2",
      ])
