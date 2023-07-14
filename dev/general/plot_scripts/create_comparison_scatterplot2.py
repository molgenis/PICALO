#!/usr/bin/env python3

"""
File:         create_comparison_scatterplot2.py
Created:      2023/06/22
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import json
import os

# Third party imports.
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Comparison Scatterplot 2"
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
./create_comparison_scatterplot2.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.x_data_path = getattr(arguments, 'x_data')
        self.x_transpose = getattr(arguments, 'x_transpose')
        self.y_data_path = getattr(arguments, 'y_data')
        self.y_transpose = getattr(arguments, 'y_transpose')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        outdir = getattr(arguments, 'outdir')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-xd",
                            "--x_data",
                            type=str,
                            required=True,
                            help="The path to the x-axis data matrix.")
        parser.add_argument("-x_transpose",
                            action='store_true',
                            help="Transpose X.")
        parser.add_argument("-yd",
                            "--y_data",
                            type=str,
                            required=True,
                            help="The path to the y-axis data matrix.")
        parser.add_argument("-y_transpose",
                            action='store_true',
                            help="Transpose Y.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfile",
                            type=str,
                            required=False,
                            default="comparison_scatterplot",
                            help="The name of the outfile.")
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
        x_df = self.load_file(self.x_data_path, header=0, index_col=0)
        y_df = self.load_file(self.y_data_path, header=0, index_col=0)
        # y_df = y_df.iloc[:5, :]

        print("Pre-process")
        if self.x_transpose:
            x_df = x_df.T
        if self.y_transpose:
            y_df = y_df.T

        print(x_df)
        print(y_df)

        sa_df = None
        palette = None
        if self.std_path is not None:
            print("Loading color data.")
            sa_df = self.load_file(self.std_path, header=0, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["hue"]
            palette = self.palette

        print("Plotting.")
        self.plot(x_df=x_df, y_df=y_df, sa_df=sa_df, palette=palette)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, x_df, y_df, sa_df=None, palette=None, color="#000000",
             ci=95, lines=True):
        ncols = x_df.shape[1]
        nrows = y_df.shape[1]
        if sa_df is not None and palette is not None:
            ncols += 1

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(12 * ncols, 9 * nrows))
        sns.set(color_codes=True)

        for i, y_col in enumerate(y_df.columns):
            for j, x_col in enumerate(x_df.columns):
                print(i, j)

                df = x_df[[x_col]].merge(y_df[[y_col]], left_index=True, right_index=True)
                df.columns = ["x", "y"]
                facecolors = "#808080"
                if sa_df is not None and palette is not None:
                    df = df.merge(sa_df, left_index=True, right_index=True)
                    df["facecolors"] = df["hue"].map(palette)
                    facecolors = df["facecolors"]

                if nrows == 1 and ncols == 1:
                    ax = axes
                elif nrows == 1:
                    ax = axes[j]
                elif ncols == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                sns.despine(fig=fig, ax=ax)

                n = df.shape[0]
                coef = np.nan
                concordance = np.nan
                rss = np.nan

                if n > 0:
                    lower_quadrant = df.loc[(df["x"] < 0) & (df["y"] < 0), :]
                    upper_quadrant = df.loc[(df["x"] > 0) & (df["y"] > 0), :]
                    n_concordant = lower_quadrant.shape[0] + upper_quadrant.shape[0]
                    concordance = (100 / n) * n_concordant

                    res = (df["x"] - df["y"]).to_numpy()
                    rss = np.sum(res * res)

                    if n > 1:
                        coef, p = stats.pearsonr(df["x"], df["y"])

                    sns.regplot(x="x", y="y", data=df, ci=ci,
                                scatter_kws={'facecolors': facecolors,
                                             'edgecolors': "#808080",
                                             'alpha': 0.60},
                                line_kws={"color": color},
                                ax=ax
                                )

                if lines:
                    ax.axhline(0, ls='--', color="#D7191C", alpha=0.3,
                               zorder=-1)
                    ax.axvline(0, ls='--', color="#D7191C", alpha=0.3,
                               zorder=-1)
                    ax.axline((0, 0), slope=1 if coef > 0 else -1, ls='--', color="#D7191C",
                              alpha=0.3,
                              zorder=1)

                x_pos = 0.03
                if coef < 1:
                    x_pos = 0.8

                y_pos = 0.9
                if n > 0:
                    ax.annotate(
                        'N = {:,}'.format(n),
                        xy=(x_pos, y_pos),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold'
                    )
                    y_pos -= 0.05

                if not np.isnan(coef):
                    ax.annotate(
                        'r = {:.2f}'.format(coef),
                        xy=(x_pos, y_pos),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold'
                    )
                    y_pos -= 0.05

                if not np.isnan(concordance):
                    ax.annotate(
                        'concordance = {:.0f}%'.format(concordance),
                        xy=(x_pos, y_pos),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold'
                    )
                    y_pos -= 0.05

                if not np.isnan(rss):
                    ax.annotate(
                        'rss = {:.2f}'.format(rss),
                        xy=(x_pos, y_pos),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold'
                    )
                    y_pos -= 0.05

                ax.set_title("",
                             fontsize=22,
                             color=color,
                             weight='bold')
                ax.set_ylabel(y_col,
                              fontsize=14,
                              fontweight='bold')
                ax.set_xlabel(x_col,
                              fontsize=14,
                              fontweight='bold')

        if sa_df is not None and palette is not None:
            for i in range(nrows):
                ax = axes[i, ncols - 1]
                ax.set_axis_off()
                if i == 0:
                    groups_present = df["hue"].unique()
                    handles = []
                    for key, value in palette.items():
                        if key in groups_present:
                            handles.append(
                                mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=4, fontsize=25)


        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_comparison_scatterplot2.{}".format(self.out_filename, extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > X-axis:")
        print("    > Data: {}".format(self.x_data_path))
        print("    > Transpose: {}".format(self.x_transpose))
        print("  > Y-axis:")
        print("    > Data: {}".format(self.y_data_path))
        print("    > Transpose: {}".format(self.y_transpose))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
