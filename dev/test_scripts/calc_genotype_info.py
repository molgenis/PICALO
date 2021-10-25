#!/usr/bin/env python3

"""
File:         calc_genotype_info.py
Created:      2021/06/14
Last Changed: 2021/10/21
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
__program__ = "Calc Genotype Info"
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
./calc_genotype_info.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.geno_path = getattr(arguments, 'genotype')

        self.outdir = os.path.join(self.indir, 'genotype_info')
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to input directory.")
        parser.add_argument("-g",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading genotype data")
        geno_df = self.load_file(self.geno_path, header=0, index_col=0)
        geno_df.drop_duplicates(inplace=True)

        badly_imputated_indices = [np.round(x, 2) for x in np.hstack((np.arange(0.26, 0.75, 0.01), np.arange(1.26, 1.75, 0.01)))]

        print("### Step2 ###")
        print("Loading PICALO results")
        for i in range(10):
            component = "PIC{}".format(i)
            for j in range(100):
                iteration = "iteration{}".format(j)
                fpath = os.path.join(self.indir, component, "results_{}.txt.gz".format(iteration))
                if os.path.exists(fpath):
                    df = self.load_file(fpath, header=0, index_col=None)
                    snps = list(df.loc[df["FDR"] < 0.05, "SNP"].values)
                    del df

                    geno_subset_df = geno_df.loc[snps, :]

                    geno_counts = geno_subset_df.apply(lambda x: x.value_counts())
                    geno_counts.loc["sum", :] = geno_counts.sum(axis=0)
                    geno_counts.loc["missing", :] = geno_counts.loc[-1, :]
                    geno_counts.loc["not imputed", :] = geno_counts.loc[0, :] + geno_counts.loc[1, :] + geno_counts.loc[2, :]
                    geno_counts.loc["imputed", :] = geno_counts.loc["sum", :] - geno_counts.loc["not imputed", :] - geno_counts.loc["missing", :]

                    subset_badly_imputated_indices = []
                    for index in badly_imputated_indices:
                        if index in geno_counts.index:
                            subset_badly_imputated_indices.append(index)
                    geno_counts.loc["badly imputed", :] = geno_counts.loc[subset_badly_imputated_indices, :].sum(axis=0)

                    save_df = geno_counts.loc[["sum", "missing", "not imputed", "imputed", "badly imputed"], :].T
                    save_df.insert(0, "rnaseq_id", save_df.index)
                    print(save_df)

                    outpath = os.path.join(self.outdir, "{}_genotype_info_{}.txt.gz".format(component, iteration))
                    self.save_file(save_df,
                                   outpath=outpath)

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
    def save_file(df, outpath, header=True, index=False, sep="\t"):
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
        print("  > Input directory: {}".format(self.indir))
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
