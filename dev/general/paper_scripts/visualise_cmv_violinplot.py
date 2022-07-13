#!/usr/bin/env python3

"""
File:         visualise_cmv_violinplot.py
Created:      2020/11/24
Last Changed:
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
import math
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Visualise CMV Violinplot"
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

./visualise_cmv_violinplot.py \
    -xd /groups/umcg-bios/tmp01/projects/PICALO/antibodyPc2VsPic3.txt \
    -x_transpose \
    -xi antiBodyPc2 \
    -xl Anti-Body_PC2 \
    -yd /groups/umcg-bios/tmp01/projects/PICALO/antibodyPc2VsPic3.txt \
    -y_transpose \
    -yi pic3 \
    -yl blood_PIC3 \
    -o 2022-03-24-BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICGeneExpressionCorrelation_PIC3_vs_ABPC2 \
    -e png pdf

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.x_data_path = getattr(arguments, 'x_data')
        self.x_transpose = getattr(arguments, 'x_transpose')
        self.x_index = " ".join(getattr(arguments, 'x_index'))
        x_label = getattr(arguments, 'x_label')
        if x_label is None:
            x_label = self.x_index
        self.x_label = x_label
        self.y_data_path = getattr(arguments, 'y_data')
        self.y_transpose = getattr(arguments, 'y_transpose')
        self.y_index = " ".join(getattr(arguments, 'y_index'))
        y_label = getattr(arguments, 'y_label')
        if y_label is None:
            y_label = self.y_index
        self.y_label = y_label
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(
            str(os.path.dirname(os.path.abspath(__file__))), 'plot')
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
        parser.add_argument("-xd",
                            "--x_data",
                            type=str,
                            required=True,
                            help="The path to the x-axis data matrix.")
        parser.add_argument("-x_transpose",
                            action='store_true',
                            help="Transpose X.")
        parser.add_argument("-xi",
                            "--x_index",
                            nargs="*",
                            type=str,
                            required=True,
                            help="The index name.")
        parser.add_argument("-xl",
                            "--x_label",
                            type=str,
                            required=False,
                            default=None,
                            help="The x-axis label.")
        parser.add_argument("-yd",
                            "--y_data",
                            type=str,
                            required=True,
                            help="The path to the y-axis data matrix.")
        parser.add_argument("-y_transpose",
                            action='store_true',
                            help="Transpose Y.")
        parser.add_argument("-yi",
                            "--y_index",
                            nargs="*",
                            type=str,
                            required=True,
                            help="The index name.")
        parser.add_argument("-yl",
                            "--y_label",
                            type=str,
                            required=False,
                            default=None,
                            help="The y-axis label.")
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
        x_df = self.load_file(self.x_data_path, header=0, index_col=0)
        y_df = self.load_file(self.y_data_path, header=0, index_col=0)

        print("Pre-process")
        if self.x_transpose:
            x_df = x_df.T
        if self.y_transpose:
            y_df = y_df.T

        if self.x_index not in x_df.index:
            for index in x_df.index:
                if index.startswith(self.x_index):
                    self.x_index = index
                    break
        if self.y_index not in y_df.index:
            for index in y_df.index:
                if index.startswith(self.y_index):
                    self.y_index = index
                    break

        x_subset_df = x_df.loc[[self.x_index], :].T
        y_subset_df = y_df.loc[[self.y_index], :].T

        print("Merging.")
        plot_df = x_subset_df.merge(y_subset_df, left_index=True, right_index=True)
        plot_df.columns = ["x", "y"]
        plot_df.dropna(inplace=True)
        plot_df = plot_df.astype(float)

        plot_df.loc[plot_df["x"] < 0, "group"] = "negative"
        plot_df.loc[plot_df["x"] > 0, "group"] = "positive"
        print(plot_df)

        print(stats.mannwhitneyu(plot_df.loc[plot_df["group"] == "negative", "y"], plot_df.loc[plot_df["group"] == "positive", "y"]))

        print("Plotting.")
        self.plot(df=plot_df,
                  x="group",
                  y="y",
                  xlabel=self.x_label,
                  ylabel=self.y_label,
                  name=self.out_filename)

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None,
                  low_memory=True):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows, low_memory=low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def plot(self, df, x="variable", y="value", hue=None, xlabel="",
             ylabel="", name=""):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       data=df,
                       color="#808080",
                       cut=0,
                       dodge=True,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    hue=hue,
                    data=df,
                    whis=np.inf,
                    color="white",
                    dodge=True,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

        # Annotate the mean.
        for i, x_value in enumerate(df[x].unique()):
            subset = df.loc[df[x] == x_value, y].copy()
            print(subset)
            print(i, x_value, subset.max())
            ax.annotate(
                'N={:,}\n{:.2f}%'.format(subset.shape[0], subset.median()),
                xy=(i, subset.max() + 0.3),
                ha="center",
                va="center",
                color="#000000",
                fontsize=15,
                fontweight='bold')

        ax.set_ylim(ax.get_ylim()[0], df[y].max() + 0.5)

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_boxplot.{}".format(name, extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > X-axis:")
        print("    > Data: {}".format(self.x_data_path))
        print("    > Transpose: {}".format(self.x_transpose))
        print("    > Index: {}".format(self.x_index))
        print("    > Label: {}".format(self.x_label))
        print("  > Y-axis:")
        print("    > Data: {}".format(self.y_data_path))
        print("    > Transpose: {}".format(self.y_transpose))
        print("    > Index: {}".format(self.y_index))
        print("    > Label: {}".format(self.y_label))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
