#!/usr/bin/env python3

"""
File:         plot_double_triangle_heatmap.py
Created:      2022/05/21
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
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

# Local application imports.

# Metadata
__program__ = "Plot Double Triangle Heatmap"
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
./plot_double_triangle_heatmap.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data1_path = getattr(arguments, 'data1')
        self.data1_transpose = getattr(arguments, 'data1_transpose')
        self.data1_name = getattr(arguments, 'data1_name').replace("_", " ")

        self.data2_path = getattr(arguments, 'data2')
        self.data2_transpose = getattr(arguments, 'data2_transpose')
        self.data2_name = getattr(arguments, 'data2_name').replace("_", " ")

        self.extensions = getattr(arguments, 'extensions')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'plot')
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
        parser.add_argument("-d1",
                            "--data1",
                            type=str,
                            required=True,
                            help="The path to first file.")
        parser.add_argument("-data1_transpose",
                            action='store_true',
                            help="Transpose the first data file.")
        parser.add_argument("-n1",
                            "--data1_name",
                            type=str,
                            default="metabrain1",
                            help="The name of the first file.")

        parser.add_argument("-d2",
                            "--data2",
                            type=str,
                            required=True,
                            help="The path to second file.")
        parser.add_argument("-data2_transpose",
                            action='store_true',
                            help="Transpose the second data file.")
        parser.add_argument("-n2",
                            "--data2_name",
                            type=str,
                            default="metabrain1",
                            help="The name of the second file.")

        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        df1 = self.load_file(self.data1_path, header=0, index_col=0)
        df2 = self.load_file(self.data2_path, header=0, index_col=0)

        if self.data1_transpose:
            df1 = df1.T

        if self.data2_transpose:
            df2 = df2.T

        print("Correlating")
        pics = df1.columns if df1.shape[1] > df2.shape[1] else df2.columns
        corr_df = pd.DataFrame(np.nan, index=pics, columns=pics)
        pvalue_df = pd.DataFrame(np.nan, index=pics, columns=pics)
        for i, pic1 in enumerate(pics):
            for j, pic2 in enumerate(pics):

                corr = np.nan
                p = np.nan
                if i < j and pic1 in df1.columns and pic2 in df1.columns:
                    corr, p = stats.pearsonr(df1[pic1], df1[pic2])
                elif i > j and pic1 in df2.columns and pic2 in df2.columns:
                    corr, p = stats.pearsonr(df2[pic1], df2[pic2])
                else:
                    pass

                corr_df.loc[pic1, pic2] = corr
                pvalue_df.loc[pic1, pic2] = p

        print("Plotting heatmap")
        self.plot_heatmap(df=corr_df,
                          xlabel=self.data1_name,
                          ylabel=self.data2_name)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_heatmap(self, df, xlabel="", ylabel="", title=""):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(1 * df.shape[1] + 10, 1 * df.shape[0] + 10),
                                 gridspec_kw={"width_ratios": [0.2, 0.8],
                                              "height_ratios": [0.8, 0.2]})
        sns.set_style("ticks")
        sns.set(color_codes=True)

        annot_df = df.copy()
        annot_df = annot_df.round(2)
        annot_df.fillna("", inplace=True)

        row_index = 0
        col_index = 0
        for _ in range(4):
            ax = axes[row_index, col_index]
            if row_index == 0 and col_index == 1:
                sns.heatmap(df, cmap=cmap, vmin=-1, vmax=1, center=0,
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

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}.{}".format(self.outfile, extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data:")
        print("  >     (1) {}: {} {}".format(self.data1_name, self.data1_path, "[T]" if self.data1_transpose else ""))
        print("  >     (2) {}: {} {}".format(self.data1_name, self.data1_path, "[T]" if self.data2_transpose else ""))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
