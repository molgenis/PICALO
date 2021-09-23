#!/usr/bin/env python3

"""
File:         create_scatterplot.py
Created:      2021/04/29
Last Changed: 2021/09/23
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

# Local application imports.

# Metadata
__program__ = "Create Scatterplot"
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
./create_scatterplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.header_row = getattr(arguments, 'header_row')
        self.index_col = getattr(arguments, 'index_col')
        self.axis = getattr(arguments, 'axis')
        self.log_transform = getattr(arguments, 'log_transform')
        self.plot_index = getattr(arguments, 'plot_index')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
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
        parser.add_argument("-pi",
                            "--plot_index",
                            nargs="*",
                            type=str,
                            required=False,
                            default=None,
                            help="Index label to plot. Default: all overlapping"
                                 "indices.")
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
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        df = self.load_file(self.data_path, header=self.header_row, index_col=self.index_col)

        if self.axis == 1:
            df = df.T

        print("Loading color data.")
        sa_df = None
        if self.std_path is not None:
            sa_df = self.load_file(self.std_path, header=None, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["dataset"]

        # Plotting.
        columns = self.plot_index
        if self.plot_index is None:
            columns = df.columns.tolist()

        self.plot(df=df, sa_df=sa_df, columns=columns)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, df, sa_df, columns):
        nplots = len(columns) + 1
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 # sharex='col',
                                 # sharey='row',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        groups_present = set()
        for i in range(ncols * nrows):
            if nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(columns):
                sns.despine(fig=fig, ax=ax)

                pi = columns[i]

                # Merge.
                plot_df = df[[pi]].copy()
                plot_df.columns = ["y"]
                plot_df.dropna(inplace=True)

                # Log 10 transform.
                if self.log_transform:
                    plot_df = -np.log10(plot_df)

                # Add color
                hue = None
                palette = None
                if sa_df is not None:
                    hue = sa_df.columns[0]
                    palette = self.palette
                    plot_df = plot_df.merge(sa_df, left_index=True, right_index=True)

                # set order.
                #plot_df["x"] = df[pi].argsort()
                counter = 0
                for group in plot_df[sa_df.columns[0]].unique():
                    mask = plot_df[sa_df.columns[0]] == group
                    subset = plot_df.loc[mask, :]
                    plot_df.loc[mask, "x"] = subset["y"].argsort() + counter
                    counter += np.sum(mask)
                    groups_present.add(group)

                # Plot.
                self.plot_scatterplot(ax=ax, df=plot_df, hue=hue,
                                      palette=palette, title=pi)
            else:
                ax.set_axis_off()

                if sa_df is not None and self.palette is not None and i == (nplots - 1):
                    handles = []
                    for key, value in self.palette.items():
                        if key in groups_present:
                            handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=4, fontsize=25)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "{}_scatterplot.png".format(self.out_filename)))
        plt.close()

    @staticmethod
    def plot_scatterplot(ax, df, x="x", y="y", hue=None, palette=None,
                     xlabel="", ylabel="", title=""):

        # Plot.
        sns.scatterplot(x=x,
                        y=y,
                        hue=hue,
                        data=df,
                        palette=palette,
                        linewidth=0,
                        legend=False,
                        ax=ax)

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
        print("  > Data path: {}".format(self.data_path))
        print("  > Header: {}".format(self.header_row))
        print("  > Index col: {}".format(self.index_col))
        print("  > Axis: {}".format(self.axis))
        print("  > -log10 transform: {}".format(self.log_transform))
        print("  > Plot index: {}".format(self.plot_index))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
