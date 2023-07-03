#!/usr/bin/env python3

"""
File:         pic_regplot.py
Created:      2021/04/29
Last Changed: 2021/10/21
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
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Create Regplot"
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
./pic_regplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_data_path = getattr(arguments, 'input_data')
        self.palette_path = getattr(arguments, 'palette')
        self.comparison_data_path = getattr(arguments, 'comparison_data')
        self.columns = getattr(arguments, 'columns')
        self.axis = getattr(arguments, 'axis')
        self.corr_threshold = getattr(arguments, 'correlation_coefficient')
        self.n_samples = getattr(arguments, 'n_samples')
        self.std_path = getattr(arguments, 'sample_to_dataset')
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
        parser.add_argument("-id",
                            "--input_data",
                            type=str,
                            required=True,
                            help="The path to the PICALO results.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-cd",
                            "--comparison_data",
                            type=str,
                            required=True,
                            help="The path to comparison matrix.")
        parser.add_argument("-c",
                            "--columns",
                            nargs="*",
                            type=str,
                            required=False,
                            default=None,
                            help="The names of the comparison data to plot"
                                 "regardless of correlation.")
        parser.add_argument("-a",
                            "--axis",
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help="The axis that denotes the samples. "
                                 "Default: 0")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-cc",
                            "--correlation_coefficient",
                            type=float,
                            default=0.4,
                            help="The minimal correlation for plotting.")
        parser.add_argument("-n",
                            "--n_samples",
                            type=int,
                            default=30,
                            help="The minimum number of samples.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        components_df = self.load_file(os.path.join(self.input_data_path, "components.txt.gz"), header=0, index_col=0)
        comparison_df = self.load_file(self.comparison_data_path, header=0, index_col=0)

        print("Pre-process")
        components_df = components_df.T
        if self.axis == 1:
            comparison_df = comparison_df.T

        print("Loading color data.")
        sa_df = None
        hue = None
        if self.palette is not None and self.std_path is not None:
            sa_df = self.load_file(self.std_path, header=None, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["dataset"]
            sa_df["hue"] = sa_df["dataset"].map(self.palette)
            hue = "hue"

        print("Getting overlap.")
        overlap = set(components_df.index).intersection(set(comparison_df.index))
        print("\tN = {}".format(len(overlap)))
        if len(overlap) == 0:
            print("No data overlapping.")
            exit()
        components_df = components_df.loc[overlap, :]
        comparison_df = comparison_df.loc[overlap, :]

        print("Plotting")
        for i, pic_column in enumerate(components_df.columns):
            for j, comparison_column in enumerate(comparison_df.columns):
                corr_data = pd.concat([components_df[pic_column], comparison_df[comparison_column]], axis=1).dropna()
                if corr_data.shape[0] < self.n_samples:
                    continue

                comparison_name = ''.join(e for e in comparison_column if e.isalnum())

                if np.min(corr_data.std(axis=0)) > 0:
                    coef, _ = stats.spearmanr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])

                    if np.abs(coef) > self.corr_threshold or comparison_column in self.columns:
                        if sa_df is not None:
                            corr_data = corr_data.merge(sa_df, left_index=True, right_index=True)

                        self.single_regplot(df=corr_data,
                                            x=pic_column,
                                            y=comparison_column,
                                            hue=hue,
                                            filename="{}_vs_{}".format(pic_column.lower(), comparison_name.lower()))

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def single_regplot(self, df, x, y, hue=None, xlabel=None, ylabel=None,
                       title="", filename="plot"):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        facecolors = "#000000"
        if hue is not None:
            facecolors = df[hue]

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.regplot(x=x, y=y, data=df, ci=None,
                    scatter_kws={'facecolors': facecolors,
                                 'linewidth': 0},
                    line_kws={"color": "#b22222"},
                    ax=ax)

        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_title(title,
                     fontsize=18,
                     fontweight='bold')

        # Change margins.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xmargin = (xlim[1] - xlim[0]) * 0.05
        ymargin = (ylim[1] - ylim[0]) * 0.05

        ax.set_xlim(xlim[0] - xmargin, xlim[1] + xmargin)
        ax.set_ylim(ylim[0] - ymargin, ylim[1] + ymargin)

        # Set annotation.
        coef, _ = stats.pearsonr(df[y], df[x])
        ax.annotate(
            'r = {:.2f}'.format(coef),
            xy=(0.03, 0.94),
            xycoords=ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')
        ax.annotate(
            'total N = {:,}'.format(df.shape[0]),
            xy=(0.03, 0.90),
            xycoords=ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')

        outpath = os.path.join(self.outdir, "{}.png".format(filename))
        fig.savefig(outpath)
        plt.close()
        print("\tSaved figure: {} ".format(os.path.basename(outpath)))

    def print_arguments(self):
        print("Arguments:")
        print("  > Input data path: {}".format(self.input_data_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Comparison data path: {}".format(self.comparison_data_path))
        print("  > Correlation threshold: {}".format(self.corr_threshold))
        print("  > N-samples: {}".format(self.n_samples))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
