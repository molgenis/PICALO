#!/usr/bin/env python3

"""
File:         plot_ieqtl_barplot.py
Created:      2022/05/05
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import glob
import re
import os

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot ieQTLs Barplot"
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
./plot_ieqtl_barplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir1 = getattr(arguments, 'indir1')
        self.label1 = getattr(arguments, 'label1')
        self.exclude1 = getattr(arguments, 'exclude1')
        self.n_files1 = getattr(arguments, 'n_files1')
        self.indir2 = getattr(arguments, 'indir2')
        self.label2 = getattr(arguments, 'label2')
        self.exclude2 = getattr(arguments, 'exclude2')
        self.n_files2 = getattr(arguments, 'n_files2')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "blood": "#D55E00",
            "brain": "#0072B2"
        }

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
        parser.add_argument("-i1",
                            "--indir1",
                            type=str,
                            required=True,
                            help="The path to the first input directory.")
        parser.add_argument("-l1",
                            "--label1",
                            type=str,
                            default="data1",
                            help="The label for the first deconvolution matrix")
        parser.add_argument("-e1",
                            "--exclude1",
                            nargs="*",
                            type=str,
                            default=[],
                            help="The covariates to exclude from the first"
                                 "indir.")
        parser.add_argument("-n1",
                            "--n_files1",
                            type=int,
                            default=None,
                            help="The number of files to load from the first"
                                 "indir. Default: all.")
        parser.add_argument("-i2",
                            "--indir2",
                            type=str,
                            required=True,
                            help="The path to the second input directory.")
        parser.add_argument("-l2",
                            "--label2",
                            type=str,
                            default="data2",
                            help="The label for the second deconvolution matrix")
        parser.add_argument("-e2",
                            "--exclude2",
                            nargs="*",
                            type=str,
                            default=[],
                            help="The covariates to exclude from the second"
                                 "indir.")
        parser.add_argument("-n2",
                            "--n_files2",
                            type=int,
                            default=None,
                            help="The number of files to load from the second"
                                 "indir. Default: all.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")
        parser.add_argument("-e",
                            "--extension",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        df1 = self.load_data(
            indir=self.indir1,
            exclude=self.exclude1,
            n_files=self.n_files1
        )

        df2 = self.load_data(
            indir=self.indir2,
            exclude=self.exclude2,
            n_files=self.n_files2
        )

        data = []
        for index, value in df1.sum(axis=0).iteritems():
            data.append([self.label1, index, value])
        for index, value in df2.sum(axis=0).iteritems():
            data.append([self.label2, index, value])
        df = pd.DataFrame(data, columns=["hue", "x", "y"])
        print(df)

        print("Merging data.")
        print("Plotting.")
        self.barplot(
            df=df,
            hue="hue",
            palette=self.palette,
            ylabel="#ieQTLs",
            filename=self.out_filename
        )

    def load_data(self, indir, exclude, n_files):
        ieqtl_fdr_df_list = []
        inpaths = glob.glob(os.path.join(indir, "*.txt.gz"))
        inpaths.sort(key=self.natural_keys)
        for i, inpath in enumerate(inpaths):
            if (n_files is not None and len(ieqtl_fdr_df_list) == n_files):
                continue

            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"] or (exclude is not None and filename in exclude):
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df["FDR"] <= 0.05, :].index
            ieqtl_fdr_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_fdr_df.loc[ieqtls, filename] = 1
            ieqtl_fdr_df_list.append(ieqtl_fdr_df)

            del ieqtl_fdr_df

        ieqtl_fdr_df = pd.concat(ieqtl_fdr_df_list, axis=1)
        return ieqtl_fdr_df

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def barplot(self, df, x="x", y="y", hue=None, palette=None, xlabel="",
                 ylabel="", title="", filename="plot"):
        sns.set(rc={'figure.figsize': (24, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.barplot(x=x,
                        y=y,
                        hue=hue,
                        data=df,
                        palette=palette,
                        ax=ax)

        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')

        fig.suptitle(title,
                     fontsize=14,
                     fontweight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        n_files1 = self.n_files1
        if n_files1 is None:
            n_files1 = "all"

        n_files2 = self.n_files2
        if n_files2 is None:
            n_files2 = "all"

        print("Arguments:")
        print("  > {}".format(self.label1))
        print("    > Input directory: {}".format(self.indir1))
        print("    > N-files: {}".format(n_files1))
        print("    > Exclude: {}".format(", ".join(self.exclude1)))
        print("  > {}".format(self.label2))
        print("    > Input directory: {}".format(self.indir2))
        print("    > N-files: {}".format(n_files2))
        print("    > Exclude: {}".format(", ".join(self.exclude2)))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
