#!/usr/bin/env python3

"""
File:         no_ieqtls_per_sample_plot.py
Created:      2021/10/22
Last Changed: 2021/10/25
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
import json
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Local application imports.

# Metadata
__program__ = "Number of ieQTLs per Sample Plot"
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
./no_ieqtls_per_sample_plot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
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

        print("Loading data")
        pics = []
        pic_df_m_collection = []
        for i in range(1, 50):
            pic = "PIC{}".format(i)
            data_path = os.path.join(self.input_directory, pic, "n_ieqtls_per_sample.txt.gz")

            if not os.path.exists(data_path):
                continue

            df = self.load_file(data_path, header=0, index_col=0)
            df.index = [i+1 for i in range(df.shape[0])]
            df_m = df.T.melt()
            df_m["group"] = pic

            pics.append(pic)
            pic_df_m_collection.append(df_m)

        combined_df_m = pd.concat(pic_df_m_collection, axis=0)

        print("Plotting.")
        self.plot_boxplot(df_m=combined_df_m,
                          xlabel="iteration",
                          ylabel="#ieQTLs per sample",
                          order=pics,
                          palette=self.palette)

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_boxplot(self, df_m, order, x="variable", y="value", panel="group",
                     xlabel="", ylabel="", palette=None):
        nplots = len(order) + 1
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
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

            if i < len(order):
                sns.despine(fig=fig, ax=ax)

                subset = df_m.loc[df_m[panel] == order[i], :]

                color = "#808080"
                if palette is not None and order[i] in palette:
                    color = palette[order[i]]

                sns.violinplot(x=x,
                               y=y,
                               data=subset,
                               color=color,
                               ax=ax)

                plt.setp(ax.collections, alpha=.75)

                sns.boxplot(x=x,
                            y=y,
                            data=subset,
                            color="white",
                            ax=ax)

                plt.setp(ax.artists, edgecolor='k', facecolor='w')
                plt.setp(ax.lines, color='k')

                ax.set_title(order[i],
                             fontsize=25,
                             fontweight='bold')
                ax.set_ylabel(ylabel,
                              fontsize=20,
                              fontweight='bold')
                ax.set_xlabel(xlabel,
                              fontsize=20,
                              fontweight='bold')

                start, end = ax.get_xlim()
                ax.xaxis.set_ticks(np.arange(start + 0.5, end + 0.5, 10))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

                ax.tick_params(axis='both', which='major', labelsize=14)
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "{}_no_ieqtls_per_sample_plot.png".format(self.out_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.input_directory))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
