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
from pathlib import Path
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
        self.cgc_path = getattr(arguments, 'comp_gene_corr')
        self.expr_path = getattr(arguments, 'expression')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'calc_replication_score')
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
        parser.add_argument("-cgc",
                            "--comp_gene_corr",
                            type=str,
                            required=True,
                            help="The path to the component ~ gene correlation"
                                 " matrix")
        parser.add_argument("-e",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading data")
        cgc_df_m = self.load_file(self.cgc_path, header=0, index_col=None)
        expr_df = self.load_file(self.expr_path, header=0, index_col=0)

        print("### Step2 ###")
        print("Pre-processing data")
        cgc_df_m.drop(['abs correlation'], axis=1, inplace=True)
        cgc_df = cgc_df_m.pivot_table(index='gene', columns='component')
        cgc_df.columns = cgc_df.columns.droplevel().rename(None)
        cgc_df.index.name = "-"
        del cgc_df_m
        print(cgc_df)
        print(expr_df)

        print("### Step4 ###")
        print("Subsetting overlap and converting data to numpy matrix")
        overlap = set(cgc_df.index.tolist()).intersection(set(expr_df.index.tolist()))
        cgc_m = cgc_df.loc[overlap, :].to_numpy()
        expr_m = expr_df.loc[overlap, :].to_numpy()
        components = cgc_df.columns.tolist()
        samples = expr_df.columns.tolist()
        del cgc_df, expr_df

        print("### Step5 ###")
        print("Calculating score")
        score_m = np.empty((len(components), len(samples)), dtype=np.float64)
        for i, component in enumerate(components):
            comp_score = np.empty(len(samples), dtype=np.float64)
            for j, sample in enumerate(samples):
                comp_score[j] = np.dot(cgc_m[:, i], expr_m[:, j])
            score_m[i, :] = comp_score

        score_df = pd.DataFrame(score_m, index=components, columns=samples)
        print(score_df)
        score_df.to_csv("replication_scores.txt.gz", sep="\t", header=True, index=True, compression="gzip")

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

    def print_arguments(self):
        print("Arguments:")
        print("  > Component ~ gene correlation path: {}".format(self.cgc_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
