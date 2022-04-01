#!/usr/bin/env python3

"""
File:         check_results.py
Created:      2022/04/01
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 M.Vochteloo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the LICENSE file in the
root directory of this source tree. If not, see <https://www.gnu.org/licenses/>.
"""

# Standard imports.
from __future__ import print_function
import glob
import argparse
import re
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Check Results"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "GPLv3"
__version__ = 1.0
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)

"""
Syntax:
./check_results.py -h

./check_results.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output \
    -p 2022-03-24-

./check_results.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output \
    -p 2022-03-24-
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.prefix = getattr(arguments, 'prefix')

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to input directory.")
        parser.add_argument("-p",
                            "--prefix",
                            type=str,
                            required=True,
                            help="The input prefix.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        input_directories = glob.glob(os.path.join(self.indir, self.prefix + "*"))
        for input_directory in input_directories:
            for i in range(1, 100):
                inpath = os.path.join(input_directory, "PIC{}".format(i), "n_ieqtls_per_sample.txt.gz")
                if os.path.exists(inpath):
                    df = pd.read_csv(inpath, sep="\t", header=0, index_col=0)
                    lowest_value = df.min(axis=1).min()
                    if lowest_value == 0:
                        print("Output directory: {} PIC{} has samples with 0 ieQTLs".format(os.path.basename(input_directory), i))

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Prefix: {}".format(self.prefix))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
