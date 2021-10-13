#!/usr/bin/env python3

"""
File:         encode_matrix.py
Created:      2021/10/13
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
from pathlib import Path
import argparse
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Encode Matrix"
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
./encode_matrix.py -filepath /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-02-03-phenotype-table/2020-03-09.brain.phenotypes.txt -header 0 -low_memory
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.filepath = getattr(arguments, 'filepath')
        self.header = getattr(arguments, 'header')
        self.index_col = getattr(arguments, 'index_col')
        self.low_memory = getattr(arguments, 'low_memory')

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'encode_matrix')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, os.path.basename(self.filepath))

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__,
                                         )

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit.")
        parser.add_argument("-filepath",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-header",
                            type=int,
                            default=None,
                            help="Row number(s) to use as the column names, "
                                 "and the start of the data.")
        parser.add_argument("-index_col",
                            type=int,
                            default=None,
                            help="Column(s) to use as the row labels of the "
                                 "DataFrame, either given as string name or "
                                 "column index.")
        parser.add_argument("-low_memory",
                            action='store_false',
                            help="Internally process the file in chunks, "
                                 "resulting in lower memory use while "
                                 "parsing, but possibly mixed type inference.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        df = pd.read_csv(self.filepath,
                         sep="\t",
                         header=self.header,
                         index_col=self.index_col,
                         low_memory=self.low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(self.filepath),
                                      df.shape))

        print(df)

        print("Encoding data")
        encoded_data = []
        translate_dict = {}
        for index, row in df.T.iterrows():
            encoded_row = row
            if row.dtype == object:
                codes, uniques = pd.factorize(row)
                translate_dict[index] = {i: unique for i, unique in enumerate(uniques)}
                encoded_row = codes
            encoded_data.append(encoded_row)

        encoded_df = pd.DataFrame(encoded_data, index=df.columns, columns=df['rnaseq_id']).T
        encoded_df.replace(-1, np.nan, inplace=True)
        print(encoded_df)

        print("Saving data.")
        compression = 'infer'
        if self.outpath.endswith('.gz'):
            compression = 'gzip'

        encoded_df.to_csv(self.outpath,
                          sep='\t',
                          header=True,
                          index=True,
                          compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(self.outpath),
                                      df.shape))

    def print_arguments(self):
        print("Arguments:")
        print("  > Filepath: {}".format(self.filepath))
        print("  > Header: {}".format(self.header))
        print("  > Index_col: {}".format(self.index_col))
        print("  > Low_memory: {}".format(self.low_memory))
        print("  > Output file: {}".format(self.outpath))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
