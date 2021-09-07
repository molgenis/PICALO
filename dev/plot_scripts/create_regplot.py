#!/usr/bin/env python3

"""
File:         create_regplot.py
Created:      2021/04/29
Last Changed: 2021/07/07
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
import sys
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Regplot"
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
./create_regplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.x_path = getattr(arguments, 'x_data')
        self.y_path = getattr(arguments, 'y_data')
        self.header_row = getattr(arguments, 'header_row')
        self.index_col = getattr(arguments, 'index_col')
        self.axis = getattr(arguments, 'axis')
        self.log_transform = getattr(arguments, 'log_transform')
        self.plot_index = getattr(arguments, 'plot_index')
        self.sa_path = getattr(arguments, 'sample_annotation')
        self.sample_id = getattr(arguments, 'sample_id')
        self.color_id = getattr(arguments, 'color_id')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "MAYO": "#9c9fa0",
            "CMC HBCC": "#0877b4",
            "GTEx": "#0fa67d",
            "ROSMAP": "#6950a1",
            "Brain GVEx": "#48b2e5",
            "Target ALS": "#d5c77a",
            "MSBB": "#5cc5bf",
            "NABEC": "#6d743a",
            "LIBD": "#e49d26",
            "ENA": "#d46727",
            "GVEX": "#000000",
            "UCLA ASD": "#f36d2a",
            "CMC": "#eae453"
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
        parser.add_argument("-xd",
                            "--x_data",
                            type=str,
                            required=True,
                            help="The path to the x-axis data matrix.")
        parser.add_argument("-yd",
                            "--y_data",
                            type=str,
                            required=True,
                            help="The path to the y-axis data matrix.")
        parser.add_argument("-hr",
                            "--header_row",
                            type=int,
                            default=None,
                            help="Position of the data header. "
                                 "Default: None.")
        parser.add_argument("-ic",
                            "--index_col",
                            type=int,
                            default=None,
                            help="Position of the index column. "
                                 "Default: None.")
        parser.add_argument("-a",
                            "--axis",
                            type=int,
                            default=1,
                            choices=[0, 1],
                            help="The axis of the index to plot. "
                                 "Default: 0")
        parser.add_argument("-log_transform",
                            action='store_true',
                            help="-log10 transform the values."
                                 " Default: False.")
        parser.add_argument("-p",
                            "--plot_index",
                            nargs="*",
                            type=str,
                            required=False,
                            default=None,
                            help="Index label to plot. Default: all overlapping"
                                 "indices.")
        parser.add_argument("-cid",
                            "--color_id",
                            type=str,
                            required=False,
                            default=None,
                            choices=["MetaBrain_cohort"],
                            help="The color column(s) name in the -sa / "
                                 "--sample_annotation file.")

        required = False
        if "-cid" in sys.argv or "--color_id" in sys.argv:
            required = True

        parser.add_argument("-sa",
                            "--sample_annotation",
                            type=str,
                            required=required,
                            default=None,
                            help="The path to the sample annotation file.")
        parser.add_argument("-sid",
                            "--sample_id",
                            type=str,
                            required=required,
                            default=None,
                            help="The sample column name in the -sa / "
                                 "--sample_annotation file.")

        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        x_df = self.load_file(self.x_path, header=self.header_row, index_col=self.index_col)
        y_df = self.load_file(self.y_path, header=self.header_row, index_col=self.index_col)
        # y_df.index = ["component{}".format(i) for i in range(5)]
        print(x_df)
        print(y_df)

        if self.axis == 1:
            x_df = x_df.T
            y_df = y_df.T

        print("Loading color data.")
        sa_df = None
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)
            sa_df = sa_df.loc[:, [self.sample_id, self.color_id]]
            sa_df.set_index(self.sample_id, inplace=True)
            sa_df.columns = ["label"]

        print("Getting overlap.")
        row_overlap = [x for x in x_df.index if x in y_df.index]
        col_overlap = [x for x in x_df.columns if x in y_df.columns]
        print("\trows = {}".format(len(col_overlap)))
        print("\tcolumns = {}".format(len(row_overlap)))
        if len(row_overlap) == 0 or len(col_overlap) == 0:
            print("No data overlapping.")
            exit()
        x_df = x_df.loc[row_overlap, col_overlap]
        y_df = y_df.loc[row_overlap, col_overlap]

        print(x_df)
        print(y_df)

        # Get the name.
        split_index = 0
        for index, letter in enumerate(self.x_path):
            if self.y_path[index] == letter:
                if letter == "/":
                    split_index = index
                continue
            else:
                break
        x_name = self.x_path[split_index:]
        y_name = self.y_path[split_index:]

        # Plotting.
        columns = self.plot_index
        if self.plot_index is None:
            columns = col_overlap

        self.plot(x_df=x_df, y_df=y_df, sa_df=sa_df, columns=columns,
                  x_name=x_name, y_name=y_name)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, x_df, y_df, sa_df, columns, x_name, y_name):
        nplots = len(columns) + 1
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            print(i)
            if nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(columns):
                sns.despine(fig=fig, ax=ax)

                pi = columns[i]

                # Get the columns.
                x_data = None
                if self.axis == 0:
                    x_data = x_df.loc[[pi], :].copy()
                    x_data = x_data.T
                if self.axis == 1:
                    x_data = x_df.loc[:, [pi]].copy()
                x_data.columns = ["x"]

                y_data = None
                if self.axis == 0:
                    y_data = y_df.loc[[pi], :].copy()
                    y_data = y_data.T
                if self.axis == 1:
                    y_data = y_df.loc[:, [pi]].copy()
                y_data.columns = ["y"]

                # Merge.
                plot_df = x_data.merge(y_data, left_index=True, right_index=True)
                plot_df.dropna(inplace=True)

                # Log 10 transform.
                if self.log_transform:
                    plot_df = -np.log10(plot_df)

                # Add color
                color = None
                if sa_df is not None:
                    color = self.color_id
                    plot_df = plot_df.merge(sa_df, left_index=True, right_index=True)
                    plot_df[self.color_id] = plot_df["label"].map(self.palette, na_action="#000000")

                # Set labels.
                xlabel = ""
                if row_index == (nrows - 1):
                    xlabel = x_name
                ylabel = ""
                if col_index == 0:
                    ylabel = y_name

                # Plot.
                self.plot_regplot(ax=ax, df=plot_df, xlabel=xlabel,
                                  ylabel=ylabel, color=color, title=pi)
            else:
                ax.set_axis_off()

                if sa_df is not None and i == (nplots - 1):
                    handles = []
                    for key, value in self.palette.items():
                        handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=4, fontsize=25)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "{}_regplot.png".format(self.out_filename)))
        plt.close()

    @staticmethod
    def plot_regplot(ax, df, x="x", y="y", color=None, xlabel="", ylabel="",
                     title=""):
        # Set color
        point_color = "#000000"
        reg_color = "#b22222"
        if color is not None:
            point_color = df[color]
            reg_color = "#000000"

        # Plot.
        sns.regplot(x=x, y=y, data=df, ci=95,
                    scatter_kws={'facecolors': point_color,
                                 'linewidth': 0,
                                 'alpha': 0.75},
                    line_kws={"color": reg_color},
                    ax=ax
                    )

        # Regression.
        coef, _ = stats.spearmanr(df[y], df[x])

        # Add the text.
        ax.annotate(
            'r = {:.2f}'.format(coef),
            xy=(0.03, 0.9),
            xycoords=ax.transAxes,
            color=reg_color,
            alpha=0.75,
            fontsize=40,
            fontweight='bold')

        ax.set_title(title,
                     fontsize=40,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=20,
                      fontweight='bold')

    def print_arguments(self):
        print("Arguments:")
        print("  > X-axis data path: {}".format(self.x_path))
        print("  > Y-axis data path: {}".format(self.y_path))
        print("  > Header: {}".format(self.header_row))
        print("  > Index col: {}".format(self.index_col))
        print("  > Axis: {}".format(self.axis))
        print("  > -log10 transform: {}".format(self.log_transform))
        print("  > Plot index: {}".format(self.plot_index))
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
