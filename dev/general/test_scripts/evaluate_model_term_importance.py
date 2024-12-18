#!/usr/bin/env python3

"""
File:         evaluate_model_term_importance.py
Created:      2022/05/09
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
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Evaluate Model Term Importance"
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
./evaluate_model_term_importance.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_directory = getattr(arguments, 'picalo_directory')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'evaluate_model_term_importance')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.terms = ["intercept", "genotype", "covariate", "interaction"]

        self.palette = {
            "intercept": "#404040",
            "genotype": "#0072B2",
            "covariate": "#D55E00",
            "interaction": "#009E73"
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
        parser.add_argument("-pd",
                            "--picalo_directory",
                            type=str,
                            required=True,
                            help="The path to the PICALO output directory.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")
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

        print("Analyzing EM data")
        em_df, em_panels = self.load_em_data()
        self.barplot(
            df=em_df,
            panels=em_panels,
            panel_col="variable",
            x="term",
            y="chi_sum",
            palette=self.palette,
            order=self.terms,
            ylabel="sum(t-value ^ 2)",
            appendix="_EM_data"
        )

        print("Analyzing ieQTL data")
        ieqtl_df, ieqtl_panels = self.load_ieqtl_data()
        self.barplot(
            df=ieqtl_df,
            panels=ieqtl_panels,
            panel_col="variable",
            x="term",
            y="chi_sum",
            palette=self.palette,
            order=self.terms,
            ylabel="sum(t-value ^ 2)",
            appendix="_ieqtl_data"
        )

    def load_em_data(self):
        data = []
        panels = []
        for i in range(1, 100):
            pic = "PIC{}".format(i)
            picalo_pic_dir = os.path.join(self.picalo_directory, pic)
            if not os.path.exists(picalo_pic_dir):
                break

            fpaths = glob.glob(os.path.join(picalo_pic_dir, "results_*.txt.gz"))
            fpaths.sort()
            df = self.load_file(fpaths[-1], header=0, index_col=None)
            df = df.loc[df["FDR"] < 0.05, :]
            label = "{} [n={:,}]".format(pic, df.shape[0])
            panels.append(label)

            for term in self.terms:
                df["tvalue-{}".format(term)] = df["beta-{}".format(term)] / df["std-{}".format(term)]
                chi_sum = (df["tvalue-{}".format(term)] ** 2).sum()
                data.append([pic, label, term, chi_sum])

        df = pd.DataFrame(data, columns=["pic", "variable", "term", "chi_sum"])
        return df, panels

    def load_ieqtl_data(self):
        data = []
        panels = []
        for i in range(1, 100):
            pic = "PIC{}".format(i)
            fpath = os.path.join(self.picalo_directory, "PIC_interactions", "PIC{}.txt.gz".format(i))
            if not os.path.exists(fpath):
                break

            df = self.load_file(fpath, header=0, index_col=None)
            df = df.loc[df["FDR"] < 0.05, :]
            label = "{} [n={:,}]".format(pic, df.shape[0])
            panels.append(label)

            for term in self.terms:
                df["tvalue-{}".format(term)] = df["beta-{}".format(term)] / df["std-{}".format(term)]
                chi_sum = (df["tvalue-{}".format(term)] ** 2).sum()
                data.append([pic, label, term, chi_sum])

        df = pd.DataFrame(data, columns=["pic", "variable", "term", "chi_sum"])
        return df, panels

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith("pkl"):
            df = pd.read_pickle(inpath)
        else:
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
        print("  > PICALO directory: {}".format(self.picalo_directory))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
