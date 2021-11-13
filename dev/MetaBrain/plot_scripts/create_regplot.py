#!/usr/bin/env python3

"""
File:         create_regplot.py
Created:      2021/11/09
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

./create_regplot.py -xd ../../preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier/expression_table_CovariatesRemovedOLS.txt.gz -xi ENSG00000166900 -yd /groups/umcg-bios/tmp01/projects/decon_optimizer/data/BIOS_RNA_pheno.txt.gz -yi Neut_Perc -y_transpose -o Neut_Perc_vs_STX3

./create_regplot.py -xd ../../preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier/expression_table.txt.gz -xi ENSG00000166900 -yd /groups/umcg-bios/tmp01/projects/decon_optimizer/data/BIOS_RNA_pheno.txt.gz -yi Neut_Perc -y_transpose -o Neut_Perc_vs_STX3

./create_regplot.py -xd  ../../output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-MAF5-PIC1/PIC_interactions/PIC1.txt.gz -x_transpose -xi t-value -xl all_samples -yd ../../output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMDSOutlier-NoPIC1Outliers-MAF5-PIC1/PIC_interactions/PIC1.txt.gz -y_transpose -yi t-value -yl no_outliers -o PIC1_all_samples_vs_no_outliers
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.x_data_path = getattr(arguments, 'x_data')
        self.x_transpose = getattr(arguments, 'x_transpose')
        self.x_index = getattr(arguments, 'x_index')
        x_label = getattr(arguments, 'x_label')
        if x_label is None:
            x_label = self.x_index
        self.x_label = x_label
        self.y_data_path = getattr(arguments, 'y_data')
        self.y_transpose = getattr(arguments, 'y_transpose')
        self.y_index = getattr(arguments, 'y_index')
        y_label = getattr(arguments, 'y_label')
        if y_label is None:
            y_label = self.y_index
        self.y_label = y_label
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
        parser.add_argument("-xd",
                            "--x_data",
                            type=str,
                            required=True,
                            help="The path to the x-axis data matrix.")
        parser.add_argument("-x_transpose",
                            action='store_true',
                            help="Combine the created files with force."
                                 " Default: False.")
        parser.add_argument("-xi",
                            "--x_index",
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
                            help="Combine the created files with force."
                                 " Default: False.")
        parser.add_argument("-yi",
                            "--y_index",
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

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        x_df = self.load_file(self.x_data_path, header=0, index_col=None)
        y_df = self.load_file(self.y_data_path, header=0, index_col=None)

        # x_df["t-value"] = x_df["beta-interaction"].astype(float) / x_df["std-interaction"].astype(float)
        # y_df["t-value"] = y_df["beta-interaction"].astype(float) / y_df["std-interaction"].astype(float)

        print("Pre-process")
        if self.x_transpose:
            x_df = x_df.T
        if self.y_transpose:
            y_df = y_df.T

        x_subset_df = x_df.loc[[self.x_index], :].T.astype(float)
        y_subset_df = y_df.loc[[self.y_index], :].T.astype(float)

        print(x_subset_df)
        print(y_subset_df)

        print("Merging.")
        plot_df = x_subset_df.merge(y_subset_df, left_index=True, right_index=True)
        plot_df.columns = ["x", "y"]
        plot_df.dropna(inplace=True)
        plot_df = plot_df.astype(float)
        print(plot_df)

        print("Plotting.")
        self.single_regplot(df=plot_df,
                            xlabel=self.x_label,
                            ylabel=self.y_label,
                            filename=self.out_filename)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def single_regplot(self, df, x="x", y="y", xlabel=None, ylabel=None,
                       title="", filename="plot"):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.regplot(x=x, y=y, data=df, ci=None,
                    scatter_kws={'facecolors': "#000000",
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

        new_xlim = (xlim[0] - xmargin, xlim[1] + xmargin)
        new_ylim = (ylim[0] - ymargin, ylim[1] + ymargin)

        ax.set_xlim(new_xlim[0], new_xlim[1])
        ax.set_ylim(new_ylim[0], new_ylim[1])

        min_pos = min(new_xlim[0], new_ylim[0])
        max_pos = max(new_xlim[1], new_ylim[1])
        ax.plot([min_pos, max_pos], [min_pos, max_pos], ls='--', color="#000000", zorder=-1)

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
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
