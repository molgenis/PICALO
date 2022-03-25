#!/usr/bin/env python3

"""
File:         count_n_ieqtls.py
Created:      2021/12/20
Last Changed: 2022/02/10
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
__program__ = "Count N-ieQTLs"
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
./count_n_ieqtls.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.n_files = getattr(arguments, 'n_files')
        self.conditional = getattr(arguments, 'conditional')

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
        parser.add_argument("-n",
                            "--n_files",
                            type=int,
                            default=None,
                            help="The number of files to load. "
                                 "Default: all.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading PICALO results")
        ieqtl_fdr_df_list = []
        inpaths = glob.glob(os.path.join(self.indir, "*.txt.gz"))
        if self.conditional:
            inpaths = [inpath for inpath in inpaths if inpath.endswith("_conditional")]
        else:
            inpaths = [inpath for inpath in inpaths if not inpath.endswith("_conditional")]
        inpaths.sort(key=self.natural_keys)
        count = 0
        for inpath in inpaths:
            if self.n_files is not None and count == self.n_files:
                continue

            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats", "PIC1", "PIC4", "Comp1", "Comp2"]:
                continue
            # if filename in ["call_rate", "genotype_stats"]:
            #     continue

            df = self.load_file(inpath, header=0, index_col=None)
            signif_col = "FDR"
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df[signif_col] <= 0.05, :].index
            ieqtl_fdr_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_fdr_df.loc[ieqtls, filename] = 1
            ieqtl_fdr_df_list.append(ieqtl_fdr_df)

            del ieqtl_fdr_df
            count += 1

        ieqtl_fdr_df = pd.concat(ieqtl_fdr_df_list, axis=1)
        ieqtl_fdr_df.to_csv("b.txt.gz", sep="\t", header=True, index=True, compression="gzip")
        cov_sum = ieqtl_fdr_df.sum(axis=0)
        print(cov_sum)

        print("Stats per covariate:")
        print("\tSum: {:,}".format(cov_sum.sum()))
        print("\tMean: {:.1f}".format(cov_sum.mean()))
        print("\tSD: {:.2f}".format(cov_sum.std()))
        print("\tMax: {:.2f}".format(cov_sum.max()))

        print("Stats per eQTL")
        counts = dict(zip(*np.unique(ieqtl_fdr_df.sum(axis=1), return_counts=True)))
        eqtls_w_inter = ieqtl_fdr_df.loc[ieqtl_fdr_df.sum(axis=1) > 0, :].shape[0]
        total_eqtls = ieqtl_fdr_df.shape[0]
        for value, n in counts.items():
            if value != 0:
                print("\tN-eQTLs with {} interaction: {:,} [{:.2f}%]".format(value, n, (100 / eqtls_w_inter) * n))
        print("\tUnique: {:,} / {:,} [{:.2f}%]".format(eqtls_w_inter, total_eqtls, (100 / total_eqtls) * eqtls_w_inter))

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        if self.n_files is None:
            print("  > N-files: all")
        else:
            print("  > N-files: {:,}".format(self.n_files))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
