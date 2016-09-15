#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name="katpoint",
      description="Karoo Array Telescope pointing coordinate library'",
      author="Ludwig Schwardt",
      author_email="ludwig@ska.ac.za",
      packages=find_packages(),
      url='https://github.com/ska-sa/katpoint',
      license="BSD",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
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
