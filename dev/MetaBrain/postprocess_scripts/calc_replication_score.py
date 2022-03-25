#!/usr/bin/env python3

"""
File:         repication.py
Created:      2021/06/04
Last Changed: 2021/07/07
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
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Calc Replication Score"
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
./calc_replication_score.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.pec_path = getattr(arguments, 'pic_expr_corr')
        self.expr_path = getattr(arguments, 'expression')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'calc_replication_score')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-pec",
                            "--pic_expr_corr",
                            type=str,
                            required=True,
                            help="The path to the component ~ gene correlation"
                                 " matrix")
        parser.add_argument("-e",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading data")
        pec_df = self.load_file(self.pec_path, header=0, index_col=None)
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=None)

        print("### Step2 ###")
        print("Pre-processing data")
        pec_df.index = pec_df["ProbeName"]
        pec_df.drop(["index", "ProbeName", "HGNCName"], axis=1, inplace=True)

        print("### Step3 ###")
        print("Subsetting overlap and converting data to numpy matrix")
        overlap = set(pec_df.index.tolist()).intersection(set(expr_df.index.tolist()))
        pec_df = pec_df.loc[overlap, :]
        expr_df = expr_df.loc[overlap, :]
        print(pec_df)
        print(expr_df)

        print("### Step3 ###")
        print("Converting data to numpy matrix")
        pec_m = pec_df.to_numpy()
        expr_m = expr_df.to_numpy()
        components = pec_df.columns.tolist()
        samples = expr_df.columns.tolist()
        del pec_df, expr_df

        print("### Step4 ###")
        print("Calculating score")
        score_m = np.empty((len(components), len(samples)), dtype=np.float64)
        for i, component in enumerate(components):
            comp_score = np.empty(len(samples), dtype=np.float64)
            for j, sample in enumerate(samples):
                comp_score[j] = np.dot(pec_m[:, i], expr_m[:, j])
            score_m[i, :] = comp_score
        score_df = pd.DataFrame(score_m, index=components, columns=samples)
        print(score_df)

        print("### Step5 ###")
        print("Saving results")
        self.save_file(df=score_df, outpath=os.path.join(self.outdir, "{}.txt.gz".format(self.out_filename)))

    @staticmethod
    def load_file(inpath, header=None, index_col=None, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith(".pkl"):
            df = pd.read_pickle(inpath)
        else:
            df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                             low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def print_arguments(self):
        print("Arguments:")
        print("  > PIC ~ gene expression correlation path: {}".format(self.pec_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
