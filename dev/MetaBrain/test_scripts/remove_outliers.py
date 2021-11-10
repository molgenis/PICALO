#!/usr/bin/env python3

"""
File:         remove_outliers.py
Created:      2021/05/21
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
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Local application imports.

# Metadata
__program__ = "Remove Outliers"
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
./remove_outliers.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.iteration = getattr(arguments, 'iteration')
        self.cutoff = getattr(arguments, 'cutoff')
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
        parser.add_argument("-i",
                            "--iteration",
                            type=str,
                            required=True,
                            help="The iteration on which to select.")
        parser.add_argument("-c",
                            "--cutoff",
                            type=int,
                            default=3,
                            help="The z-score cut-off. Default = 4.")
        parser.add_argument("-sa",
                            "--sample_annotation",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample annotation file.")
        parser.add_argument("-sid",
                            "--sample_id",
                            type=str,
                            required=False,
                            default="rnaseq_id",
                            help="The sample column name in the -sa / "
                                 "--sample_annotation file.")
        parser.add_argument("-cid",
                            "--color_id",
                            type=str,
                            required=False,
                            default="MetaBrain_cohort",
                            help="The color column(s) name in the -sa / "
                                 "--sample_annotation file.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        full_df = self.load_file(self.data_path, header=0, index_col=0)
        df = full_df.loc[[self.iteration], :].T
        del full_df

        # Convert to z-score
        df["z-score"] = (df - df.mean(axis=0)) / df.std(axis=0)
        df["abs z-score"] = df["z-score"].abs()
        df["include"] = df["z-score"].abs() < self.cutoff
        df["x"] = np.arange(1, df.shape[0] + 1)
        df.sort_values(by="abs z-score", ascending=False, inplace=True)
        print(df)

        print("Loading color data.")
        hue = None
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)
            cohort_dict = dict(zip(sa_df[self.sample_id], sa_df[self.color_id]))
            df["cohort"] = df.index.map(cohort_dict)
            hue = "cohort"

        exclude_samples = df.loc[df["include"] == False, :].copy()
        print(exclude_samples.index.tolist())
        counts = exclude_samples["cohort"].value_counts()
        print("Removing {} samples with >{}SD:".format(exclude_samples.shape[0], self.cutoff))
        for index, value in counts.iteritems():
            print("\t{}:\t{} samples".format(index, value))

        for z_score in range(1, 6):
            subset = df.loc[df["z-score"].abs() > z_score, :]
            print(z_score, subset.shape)

        self.plot(df=df, x="x", y="z-score", hue=hue, palette=self.palette,
                  hlines=[-self.cutoff, self.cutoff],
                  outpath=os.path.join(self.outdir, "{}_cutoff_plot.png".format(self.out_filename)))

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def plot(df, x="x", y="y", hue="hue", palette=None, hlines=None, outpath="plot.png"):
        z_scores = [x for x in range(1, 6)]
        nplots = len(z_scores) + 1
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

            if i < len(z_scores):
                sns.despine(fig=fig, ax=ax)

                cut_off = z_scores[i]

                sns.scatterplot(x=x,
                                y=y,
                                hue=hue,
                                data=df,
                                palette=palette,
                                linewidth=0,
                                legend=None,
                                ax=ax)

                ax.set_xlabel("",
                              fontsize=10,
                              fontweight='bold')
                ax.set_ylabel("z-score",
                              fontsize=10,
                              fontweight='bold')
                ax.set_title("{} SD".format(cut_off),
                             fontsize=20,
                             fontweight='bold')

                for hline_pos in [-cut_off, cut_off]:
                    ax.axhline(hline_pos, ls='--', color="#b22222", zorder=-1)
            else:
                ax.set_axis_off()


            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data: {}".format(self.data_path))
        print("  > Iteration: {}".format(self.iteration))
        print("  > Cut-off: {}".format(self.cutoff))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
