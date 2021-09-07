#!/usr/bin/env python3

"""
File:         create_comparison_scatterplot.py
Created:      2021/06/30
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
import sys
import os

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Comparison Scatterplot"
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
./create_comparison_scatterplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
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
        df = self.load_file(self.data_path, header=0, index_col=0)
        df = df.T
        columns = df.columns.tolist()

        print("Loading color data.")
        hue = None
        palette = None
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)
            sa_df = sa_df.loc[:, [self.sample_id, self.color_id]]
            sa_df.set_index(self.sample_id, inplace=True)
            sa_df.columns = ["hue"]
            df = df.merge(sa_df, left_index=True, right_index=True)

            hue = "hue"
            palette = self.palette

        # Plotting.
        self.plot(df=df, columns=columns, hue=hue, palette=palette)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, df, columns, hue, palette):
        ncols = len(columns)
        nrows = len(columns)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(10 * ncols, 10 * nrows))
        sns.set(color_codes=True)

        for i, y_col in enumerate(columns):
            for j, x_col in enumerate(columns):
                print(i, j)

                ax = axes[i, j]
                if i == 0 and j == (ncols - 1):
                    ax.set_axis_off()
                    if hue is not None and palette is not None:
                        groups_present = df[hue].unique()
                        handles = []
                        for key, value in self.palette.items():
                            if key in groups_present:
                                handles.append(mpatches.Patch(color=value, label=key))
                        ax.legend(handles=handles, loc=4, fontsize=25)

                elif i < j:
                    ax.set_axis_off()
                    continue
                elif i == j:
                    ax.set_axis_off()

                    ax.annotate(y_col,
                                xy=(0.5, 0.5),
                                ha='center',
                                xycoords=ax.transAxes,
                                color="#000000",
                                fontsize=40,
                                fontweight='bold')
                else:
                    sns.despine(fig=fig, ax=ax)

                    sns.scatterplot(x=x_col,
                                    y=y_col,
                                    hue=hue,
                                    data=df,
                                    s=100,
                                    palette=palette,
                                    linewidth=0,
                                    legend=False,
                                    ax=ax)

                    ax.set_ylabel("",
                                  fontsize=20,
                                  fontweight='bold')
                    ax.set_xlabel("",
                                  fontsize=20,
                                  fontweight='bold')

        fig.savefig(os.path.join(self.outdir, "{}_comparison_scatterplot.png".format(self.out_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
