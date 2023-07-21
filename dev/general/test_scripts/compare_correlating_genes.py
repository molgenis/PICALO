#!/usr/bin/env python3

"""
File:         compare_correlating_genes.py
Created:      2022/04/28
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats

# Local application imports.


# Metadata
__program__ = "Compare Correlating Genes"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "BSD (3-Clause)"
__version__ = 1.0
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)

"""
Syntax:
./compare_correlating_genes.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data1_path = getattr(arguments, 'data1')
        self.name1 = getattr(arguments, 'name1')
        self.data2_path = getattr(arguments, 'data2')
        self.name2 = getattr(arguments, 'name2')

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
        parser.add_argument("-d1",
                            "--data1",
                            type=str,
                            required=True,
                            help="The path to the first PIC - gene "
                                 "correlation matrix.")
        parser.add_argument("-n1",
                            "--name1",
                            type=str,
                            default="data1",
                            help="The name for the first deconvolution matrix")
        parser.add_argument("-d2",
                            "--data2",
                            type=str,
                            required=True,
                            help="The path to the second PIC - gene "
                                 "correlation matrix.")
        parser.add_argument("-n2",
                            "--name2",
                            type=str,
                            default="data2",
                            help="The name for the second deconvolution matrix")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        df1 = self.load_file(self.data1_path, header=0, index_col=0)
        df2 = self.load_file(self.data2_path, header=0, index_col=0)

        df1.index = [probename.split(".")[0] for probename in df1["ProbeName"]]
        df2.index = [probename.split(".")[0] for probename in df2["ProbeName"]]

        print("Calculating")
        for i in range(100):
            pic = "PIC{}".format(i)
            if not pic in df1.columns or not pic in df2.columns:
                continue
            print("\t{}".format(pic))

            df1_subset = df1[["avgExpression", pic]].copy()
            df1_subset.sort_values(by=pic, ascending=False, inplace=True)
            df1_top_200 = set(df1_subset.head(200).index)
            df1_bottom_200 = set(df1_subset.tail(200).index)
            df1_subset.columns = ["{} avgExpression".format(self.name1), "{} correlation".format(self.name1)]

            df2_subset = df2[["avgExpression", pic]].copy()
            df2_subset.sort_values(by=pic, ascending=False, inplace=True)
            df2_top_200 = set(df2_subset.head(200).index)
            df2_bottom_200 = set(df2_subset.tail(200).index)
            df2_subset.columns = ["{} avgExpression".format(self.name2), "{} correlation".format(self.name2)]

            df = df1_subset.merge(df2_subset, how="outer", left_index=True, right_index=True)
            print(df)
            del df1_subset, df2_subset

            nas = np.logical_or(df["{} correlation".format(self.name1)].isnull(),
                                df["{} correlation".format(self.name2)].isnull())

            coef, _ = stats.pearsonr(df.loc[~nas, "{} correlation".format(self.name1)],
                                     df.loc[~nas, "{} correlation".format(self.name2)])
            print("\t  Correlation: {:.2f}".format(coef))

            if coef < 0:
                df["{} correlation".format(self.name2)] = df["{} correlation".format(self.name2)] * -1

                df2_bottom_200_tmp = df2_bottom_200.copy()
                df2_bottom_200 = df2_top_200
                df2_top_200 = df2_bottom_200_tmp
                del df2_bottom_200_tmp

            print("\t  Overlapping hits:")
            for prefix, set1, set2 in (["top", df1_top_200, df2_top_200],
                                       ["bottom", df1_bottom_200, df2_bottom_200]):
                print("\t\t{} 200 genes".format(prefix))
                overlap = set1.intersection(set2)
                set1_unique = set1.difference(set2)
                set2_unique = set2.difference(set1)
                for name, genes in (["overlap", overlap],
                                    ["{} unique".format(self.name1), set1_unique],
                                    ["{} unique".format(self.name2), set2_unique]):
                    if len(genes) == 0:
                        continue
                    overlap_mean = df.loc[genes, :].mean(axis=0)
                    print("\t\t  {} [N={:,}]:".format(name, len(genes)))
                    print("\t\t\t{}:\tavgExpression: {:.2f}\tcorrelation: {:.2f}".format(self.name1,
                                                                                         overlap_mean["{} avgExpression".format(self.name1)],
                                                                                         overlap_mean["{} correlation".format(self.name1)]))
                    print("\t\t\t{}:\tavgExpression: {:.2f}\tcorrelation: {:.2f}".format(self.name2,
                                                                                         overlap_mean["{} avgExpression".format(self.name2)],
                                                                                         overlap_mean["{} correlation".format(self.name2)]))
                    print("")

            exit()

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > {}".format(self.name1))
        print("    > Data path: {}".format(self.data1_path))
        print("  > {}".format(self.name2))
        print("    > Data path: {}".format(self.data2_path))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
