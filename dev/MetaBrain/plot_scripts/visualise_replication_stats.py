#!/usr/bin/env python3

"""
File:         visualise_replication_stats.py
Created:      2022/07/18
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
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Visualise Replication Stats"
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
./visualise_replication_stats.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'visualise_replication_stats')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the input data.")
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

        print("Load data.")
        df = self.load_file(self.data_path, header=0, index_col=0)

        print("Pre-processing data.")
        df = df.loc[df["label"] == "discovery significant", :]

        print("Creating plots.")
        for variable in ["N", "N concordant", "concordance", "pearsonr", "pi1", "Rb"]:
            self.barplot(
                df=df.loc[df["variable"] == variable, :],
                x="col",
                y="value",
                ylabel=variable,
                filename=self.out_filename + "_" + variable
            )

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
        color = None
        if hue is None:
            color = "#000000"

        sns.set(rc={'figure.figsize': (24, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.barplot(x=x,
                        y=y,
                        hue=hue,
                        data=df,
                        color=color,
                        palette=palette,
                        ax=ax)

        y_adjust = ax.get_ylim()[1] * 0.01
        for i, (_, row) in enumerate(df.iterrows()):
            g.text(i, row[y] + y_adjust,
                   round(row[y], 2),
                   color="#000000",
                   ha="center")

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
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
