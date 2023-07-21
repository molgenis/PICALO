#!/usr/bin/env python3

"""
File:         plot_cf_per_pic2.py
Created:      2022/05/10
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
from pathlib import Path
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
from colour import Color
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Cell Fraction per PIC 2"
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
./plot_cf_per_pic2.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.cf_path = getattr(arguments, 'cell_fractions')
        self.cell_type = getattr(arguments, 'cell_type').replace("_", " ")
        self.pics_path = getattr(arguments, 'pic_matrix')
        self.pic = getattr(arguments, 'pic_index')
        self.outfile = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')
        self.n_bins = 10

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            "Myeloid Progenitor": "#D55E00",
            "Neuron": "#0072B2"
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
        parser.add_argument("-cf",
                            "--cell_fractions",
                            type=str,
                            required=True,
                            help="The path to the cell fractions matrix.")
        parser.add_argument("-ct",
                            "--cell_type",
                            type=str,
                            required=True,
                            help="The cell type to color.")
        parser.add_argument("-pm",
                            "--pic_matrix",
                            type=str,
                            required=True,
                            help="The path to the PICS matrix.")
        parser.add_argument("-pi",
                            "--pic_index",
                            type=str,
                            default=None,
                            help="The PIC to plot.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        cf_df = self.load_file(self.cf_path)
        pics_df = self.load_file(self.pics_path).T

        print("Preprocessing data.")
        df = cf_df[[self.cell_type]].merge(pics_df[[self.pic]], left_index=True, right_index=True)
        df.columns = ["proportion", "loading"]
        df["proportion"] = df["proportion"] * 100

        # Normalize loadings between bins.
        df["normalized"] = self.normalize(values=df["loading"],
                                          actual_bounds=(df["loading"].min(), df["loading"].max()),
                                          desired_bounds=(0, self.n_bins - 1))
        df["bin"] = df["normalized"].round(0) + 1
        print(df)

        # Add labels
        counts = df["bin"].value_counts()
        df["label"] = ["{}\n[n={:,}]".format(bin, counts[bin]) for bin in df["bin"]]
        label_dict = dict(zip(df["bin"], df["label"]))
        order = [label_dict[i] for i in range(1, self.n_bins+1)]

        self.plot_boxplot(
            df=df,
            x="label",
            y="proportion",
            reg_x="normalized",
            color=self.palette[self.cell_type],
            order=order,
            xlabel="",
            ylabel="cell fraction %",
            title="{} - {}".format(self.pic, self.cell_type),
            filename="{}_{}_{}".format(self.outfile, self.pic, self.cell_type)
        )

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None,
                  low_memory=True):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows, low_memory=low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    @staticmethod
    def normalize(values, actual_bounds, desired_bounds):
        return [desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]) for x in values]

    def plot_boxplot(self, df, x="variable", y="value", reg_x=None, color="b22222",
                     order=None, xlabel="", ylabel="", title="", filename=""):

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": [0.1, 0.9]})

        ax1.set_axis_off()
        ax1.annotate(
            'N = {:,}'.format(df.shape[0]),
            xy=(0.045, 0.2),
            xycoords=ax1.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold'
        )
        handles = [mpatches.Patch(color=color, label=self.cell_type),
                   mpatches.Patch(color="#808080", label="Other")]
        ax1.legend(handles=handles, loc=8, ncol=2)

        sns.despine(fig=fig, ax=ax2)

        sns.barplot(x=x,
                    y=y,
                    order=order,
                    color="#808080",
                    data=pd.DataFrame([[i, 100] for i in df[x].unique()], columns=[x, y]),
                    ax=ax2)

        sns.barplot(x=x,
                    y=y,
                    order=order,
                    data=df,
                    color=color,
                    ax=ax2)

        if reg_x is not None:
            coef, p = stats.pearsonr(df[reg_x], df[y])
            ax1.annotate(
                'r = {:.2f}'.format(coef),
                xy=(0.2, 0.2),
                xycoords=ax1.transAxes,
                color="#000000",
                fontsize=14,
                fontweight='bold')

            sns.regplot(x=reg_x,
                        y=y,
                        data=df,
                        scatter=False,
                        ci=None,
                        line_kws={"color": "#000000"},
                        ax=ax2)

        ax2.set_ylim(0, 100)

        ax1.set_title(title,
                      fontsize=20,
                      fontweight='bold')
        ax2.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax2.set_ylabel(ylabel,
                       fontsize=14,
                       fontweight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Cell fraction path: {}".format(self.cf_path))
        print("  > Cell type: {}".format(self.cell_type))
        print("  > Pics path: {}".format(self.pics_path))
        print("  > PIC: {}".format(self.pic))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
