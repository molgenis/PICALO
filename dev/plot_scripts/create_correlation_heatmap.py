#!/usr/bin/env python3

"""
File:         create_correlation_heatmap.py
Created:      2021/04/26
Last Changed: 2021/10/13
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
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hcl

# Local application imports.

# Metadata
__program__ = "Create Correlation Matrix"
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
./create_correlation_heatmap.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.row_data_path = getattr(arguments, 'row_data')
        self.row_name = " ".join(getattr(arguments, 'row_name'))
        self.col_data_path = getattr(arguments, 'col_data')
        self.col_name = " ".join(getattr(arguments, 'col_name'))
        self.method = getattr(arguments, 'method')
        self.call_rate = getattr(arguments, 'call_rate')
        self.row_cluster = getattr(arguments, 'row_cluster')
        self.col_cluster = getattr(arguments, 'col_cluster')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-rd",
                            "--row_data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-rn",
                            "--row_name",
                            nargs="*",
                            type=str,
                            required=False,
                            default="",
                            help="The name of -r / --row_data.")
        parser.add_argument("-cd",
                            "--col_data",
                            type=str,
                            required=False,
                            help="The path to the data matrix.")
        parser.add_argument("-cn",
                            "--col_name",
                            nargs="*",
                            type=str,
                            required=False,
                            default="",
                            help="The name of -c / --col_data.")
        parser.add_argument("-a",
                            "--axis",
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help="The axis that denotes the samples. "
                                 "Default: 0")
        parser.add_argument("-m",
                            "--method",
                            type=str,
                            choices=["Pearson", "Spearman"],
                            default="Spearman",
                            help="The correlation method. Default: Spearman.")
        parser.add_argument("-cr",
                            "--call_rate",
                            type=float,
                            default=0.01,
                            help="The minimal call rate (not NaN values) for "
                                 "calculating a correlation.")
        parser.add_argument("-row_cluster",
                            action='store_true',
                            help="Cluster the rows."
                                 " Default: False.")
        parser.add_argument("-col_cluster",
                            action='store_true',
                            help="Cluster the rows."
                                 " Default: False.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading row data.")
        row_df = self.load_file(self.row_data_path, header=0, index_col=0)

        col_df = row_df
        triangle = True
        if self.col_data_path is not None:
            print("Loading column data.")
            col_df = self.load_file(self.col_data_path, header=0, index_col=0)
            triangle = False

        if row_df.shape[1] > row_df.shape[0]:
            row_df = row_df.T

        if col_df.shape[1] > col_df.shape[0]:
            col_df = col_df.T

        print("Removing columns without variance.")
        row_df = row_df.loc[:, row_df.std(axis=0) != 0]
        col_df = col_df.loc[:, col_df.std(axis=0) != 0]

        print("Getting overlap.")
        overlap = set(row_df.index).intersection(set(col_df.index))
        print("\tN = {}".format(len(overlap)))
        if len(overlap) == 0:
            print("No data overlapping.")
            exit()
        row_df = row_df.loc[overlap, :]
        col_df = col_df.loc[overlap, :]

        print("\tCorrelating.")
        corr_df, sample_size_df = self.correlate(index_df=row_df, columns_df=col_df, triangle=triangle)
        print(corr_df)
        print(sample_size_df)

        print("\tClustering.")
        if self.row_cluster:
            corr_df = self.cluster(corr_df, axis=0)
        if self.col_cluster:
            corr_df = self.cluster(corr_df, axis=1)

        print("\tPlotting.")
        self.plot_heatmap(corr_df=corr_df, xlabel=self.col_name, ylabel=self.row_name, appendix="correlations")
        self.plot_heatmap(corr_df=sample_size_df, xlabel=self.col_name, ylabel=self.row_name, appendix="sample_sizes", vmax=len(overlap), as_int=True)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def correlate(self, index_df, columns_df, triangle=False):
        corr_df = pd.DataFrame(np.nan, index=index_df.columns, columns=columns_df.columns)
        sample_size_df = pd.DataFrame(0, index=index_df.columns, columns=columns_df.columns)

        for i, index_column in enumerate(index_df.columns):
            for j, column_column in enumerate(columns_df.columns):
                if triangle and i < j:
                    continue
                corr_data = pd.concat([index_df[index_column], columns_df[column_column]], axis=1)
                pre_filter_shape = corr_data.shape[0]
                corr_data.dropna(inplace=True)
                post_filter_shape = corr_data.shape[0]

                coef = np.nan
                if (post_filter_shape / pre_filter_shape) > self.call_rate and np.min(corr_data.std(axis=0)) > 0:
                    if self.method == "Pearson":
                        coef, _ = stats.pearsonr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])
                    elif self.method == "Spearman":
                        coef, _ = stats.spearmanr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])

                corr_df.loc[index_column, column_column] = coef
                sample_size_df.loc[index_column, column_column] = post_filter_shape

        return corr_df, sample_size_df

    @staticmethod
    def cluster(df, axis):
        tmp = df.copy()
        if axis == 1:
            tmp = tmp.T

        distances = pdist(tmp.values, metric='euclidean')
        dist_matrix = squareform(distances)
        linkage = hcl.linkage(squareform(dist_matrix), method="average")
        Z = hcl.dendrogram(linkage, orientation='right')
        index = Z['leaves']

        if axis == 1:
            tmp = df.iloc[:, index]
        else:
            tmp = tmp.iloc[index, :]

        return tmp

    def plot_heatmap(self, corr_df, xlabel="", ylabel="", appendix="", vmin=-1, vmax=1, as_int=False):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(1 * corr_df.shape[1] + 5, 1 * corr_df.shape[0] + 5),
                                 gridspec_kw={"width_ratios": [0.2, 0.8],
                                              "height_ratios": [0.8, 0.2]})
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for _ in range(4):
            ax = axes[row_index, col_index]
            if row_index == 0 and col_index == 1:

                annot_df = corr_df.round(2)
                if as_int:
                    annot_df = corr_df.astype(int)

                sns.heatmap(corr_df, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
                            square=True, annot=annot_df, fmt='',
                            cbar=False, annot_kws={"size": 14, "color": "#000000"},
                            ax=ax)

                plt.setp(ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20, rotation=0))
                plt.setp(ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20, rotation=90))

                ax.set_xlabel(xlabel, fontsize=14)
                ax.xaxis.set_label_position('top')

                ax.set_ylabel(ylabel, fontsize=14)
                ax.yaxis.set_label_position('right')
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > 1:
                col_index = 0
                row_index += 1

        # plt.tight_layout()
        fig.savefig(os.path.join(self.outdir, "{}_corr_heatmap_{}_{}.png".format(self.out_filename, self.method, appendix)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Row data path: {}".format(self.row_data_path))
        print("  > Row name: {}".format(self.row_name))
        print("  > Col data path: {}".format(self.col_data_path))
        print("  > Col name: {}".format(self.col_name))
        print("  > Correlation method: {}".format(self.method))
        print("  > Call rate: {}".format(self.call_rate))
        print("  > Row cluster: {}".format(self.row_cluster))
        print("  > Col cluster: {}".format(self.col_cluster))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
