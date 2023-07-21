#!/usr/bin/env python3

"""
File:         count_n_ieqtls.py
Created:      2021/12/20
Last Changed: 2022/02/10
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import itertools
import glob
import argparse
import re
import os

# Third party imports.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import upsetplot as up

# Local application imports.

# Metadata
__program__ = "Count N-ieQTLs"
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
./count_n_ieqtls.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.exclude = getattr(arguments, 'exclude')
        self.skip_files = getattr(arguments, 'skip_files')
        self.n_files = getattr(arguments, 'n_files')
        self.conditional = getattr(arguments, 'conditional')
        self.plot = getattr(arguments, 'plot')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
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
        parser.add_argument("-e",
                            "--exclude",
                            nargs="*",
                            type=str,
                            default=None,
                            help="The covariates to exclude.")
        parser.add_argument("-s",
                            "--skip_files",
                            type=int,
                            default=0,
                            help="The number of files to load. "
                                 "Default: 0.")
        parser.add_argument("-n",
                            "--n_files",
                            type=int,
                            default=None,
                            help="The number of files to load. "
                                 "Default: all.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
        parser.add_argument("-plot",
                            action='store_true',
                            help="Create upsetplot. Default: False.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading PICALO results")
        ieqtl_fdr_df_list = []
        inpaths = glob.glob(os.path.join(self.indir, "*.txt.gz"))
        if self.conditional:
            inpaths = [inpath for inpath in inpaths if inpath.endswith("_conditional.txt.gz")]
        else:
            inpaths = [inpath for inpath in inpaths if not inpath.endswith("_conditional.txt.gz")]
        inpaths.sort(key=self.natural_keys)
        for i, inpath in enumerate(inpaths):
            if (self.skip_files is not None and i < self.skip_files) or \
                    (self.n_files is not None and len(ieqtl_fdr_df_list) == self.n_files):
                continue

            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"] or (self.exclude is not None and filename in self.exclude):
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            signif_col = "FDR"
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df[signif_col] <= 0.05, :].index
            ieqtl_fdr_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_fdr_df.loc[ieqtls, filename] = 1
            ieqtl_fdr_df_list.append(ieqtl_fdr_df)

            del ieqtl_fdr_df

        ieqtl_fdr_df = pd.concat(ieqtl_fdr_df_list, axis=1)
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

        if self.plot:
            pic_data = {}
            for col in ieqtl_fdr_df.columns:
                pic_data[col] = set(ieqtl_fdr_df.loc[ieqtl_fdr_df[col] == 1, :].index.tolist())

            # Plot upsetplot of all PICS combined.
            counts = self.count(pic_data)
            counts = counts[counts > 0]
            print(counts)

            print("Creating plot.")
            up.plot(counts, sort_by='cardinality', show_counts=True)
            plt.savefig(os.path.join(self.outdir, "{}_PICS_upsetplot.png".format(os.path.basename(self.indir))))
            plt.close()

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


    @staticmethod
    def count(input_data):
        combinations = []
        cols = list(input_data.keys())
        for i in range(1, len(cols) + 1):
            combinations.extend(list(itertools.combinations(cols, i)))

        indices = []
        data = []
        for combination in combinations:
            index = []
            for col in cols:
                if col in combination:
                    index.append(True)
                else:
                    index.append(False)

            background = set()
            for key in cols:
                if key not in combination:
                    work_set = input_data[key].copy()
                    background.update(work_set)

            overlap = None
            for key in combination:
                work_set = input_data[key].copy()
                if overlap is None:
                    overlap = work_set
                else:
                    overlap = overlap.intersection(work_set)

            duplicate_set = overlap.intersection(background)
            length = len(overlap) - len(duplicate_set)

            indices.append(index)
            data.append(length)

        s = pd.Series(data, index=pd.MultiIndex.from_tuples(indices, names=cols))
        s.name = "value"
        return s

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Exclude: {}".format(self.exclude))
        print("  > Skip-files: {:,}".format(self.skip_files))
        if self.n_files is None:
            print("  > N-files: all")
        else:
            print("  > N-files: {:,}".format(self.n_files))
        print("  > Plot: {:,}".format(self.plot))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
