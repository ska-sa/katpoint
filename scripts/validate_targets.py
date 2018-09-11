#!/usr/bin/env python
# coding: utf-8
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from __future__ import print_function
import argparse
import os
import katpoint


def find_markers(input, marker):
    """
    Find all the markers in input.

    Parameters
    ----------
    input : string
        Input string to parse.
    marker : character
        Search parameter

    Returns
    -------
    list
        Input position where the marker was found.
    """
    return [i for i, ch in enumerate(input) if ch == marker]


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


def show_separators(input, separator, exception):
    """
    Show the separators graphically.

    Parameters
    ----------
    input : string
        Input string to parse.
    separator : character
        Search parameter
    exception: string
        Exception error.
    """
    print('\nPotential invalid separator - {}!'.format(exception))
    print(input)
    print_markers(find_markers(input, separator))


def show_non_ascii(input):
    """
    Show the non-ascii graphically.

    Parameters
    ----------
    input : string
        Input string to parse.
    """
    # Force input string to unicode
    target_uni = input.decode('utf-8')
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
        description="""Validate all target strings from the supplied csv file using katPoint.Target.\n
            Non-Ascii characters are shown as '?' and their positions are shown graphically.\n
            All separators are shown graphically if the maximum field count is exceeded.\n
            \tUse it like this: \n
            \tpython validate_targets.py --use-file example.csv""")
    parser.add_argument(
        '--use-file',
        required=True,
        help="target csv input file")
    config = parser.parse_args()

    if not os.path.exists(config.use_file):
        print("\nFile {} does not exist!\n".format(
            config.use_file))
        parser.print_help()
        exit(-1)

    return vars(config)


def main():
    config = parse_cmd_line()
    validate_target(config['use_file'])


if __name__ == "__main__":
    main()
