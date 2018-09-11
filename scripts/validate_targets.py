#!/usr/bin/env python
# coding: utf-8
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

from __future__ import print_function
import argparse
import os
import katpoint


def find_markers(body, marker):
    """
    Find all the markers in body.

    Parameters
    ----------
    body : string
        Input string to parse.
    marker : character
        Search parameter

    Returns
    -------
    list
        Input position where the marker was found.
    """
    return [i for i, ch in enumerate(body) if ch == marker]


def print_markers(marker_indices):
    """
    Show the markers graphically with '▲', all other positions are shown with '.'.

    Parameters
    ----------
    marker_indices : list
        Marker positions.
    """
    output = [u'.'] * (max(marker_indices) + 1)
    for pos in marker_indices:
        output[pos] = u'▲'
    output = u''.join(output)

    print(output.encode('utf-8'))


def show_separators(body, separator, exception):
    """
    Show the separators graphically.

    Parameters
    ----------
    body : string
        Input string to parse.
    separator : character
        Search parameter
    exception: string
        Exception error.
    """
    print('\nPotential invalid separator - {}!'.format(exception))
    print(body)
    print_markers(find_markers(body, separator))


def show_non_ascii(body):
    """
    Show the non-ascii graphically.

    Parameters
    ----------
    body : string
        Input string to parse.
    """
    # Force body string to unicode
    target_uni = body.decode('utf-8')
    # Now convert to ascii and mark all unicode with '?'
    target_uni = target_uni.encode('ascii', errors='replace')
    print ('\nNon ASCII characters found!')
    print(target_uni)
    print_markers(find_markers(target_uni, '?'))


def validate_target(target_file):
    """
    Validate all target strings from the supplied csv file using katPoint.Target.

    Non-Ascii characters are shown as '?' and their positions are shown graphically.
    All separators are shown graphically if the maximum field count is exceeded.

    Parameters
    ----------
    target_file : string
        A csv file with multiple target strings

    Returns
    -------
    target_validation_pass: bool
        Target validation status.
    """
    target_validation_pass = True
    with open(target_file, 'r') as csv_file:
        for line in csv_file:
            if not line.strip().startswith('#'):
                try:
                    katpoint.Target(line)
                except katpoint.NonAsciiError:
                    show_non_ascii(line)
                    target_validation_pass = False
                except katpoint.FluxError as exception:
                    show_separators(line, ',', exception)
                    target_validation_pass = False
                except ValueError as exception:
                    if line.strip() in str(exception):
                        print("\nParsing Error!\n{}".format(exception))
                    else:
                        print("\nParsing Error!!\n{}\n{}".format(line, exception))
                    target_validation_pass = False

    return target_validation_pass


def parse_cmd_line():
    """Parse the script command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
            Validate all target strings from the supplied csv file using
            katPoint.Target.
        """)
    parser.add_argument(
        'filename', help="csv file with list of target strings")
    config = parser.parse_args()

    if not os.path.exists(config.filename):
        print("\nFile {} does not exist!\n".format(
            config.filename))
        parser.print_help()
        exit(-1)

    return vars(config)


def main():
    config = parse_cmd_line()
    if validate_target(config['filename']):
        print("\nNo errors found in {}".format(config['filename']))


if __name__ == "__main__":
    main()
