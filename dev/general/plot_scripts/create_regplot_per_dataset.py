#!/usr/bin/env python3

"""
File:         create_regplot_per_dataset.py
Created:      2022/03/21
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import json
import math
import os

# Third party imports.
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Create Regplot Per Dataset"
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
./create_regplot_per_dataset.py -h

./create_regplot.py \
    -xd /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/components.txt.gz \
    -xi PIC1 \
    -yd /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/components.txt.gz \
    -yi PIC4 \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PIC1_vs_PIC4
    
./create_regplot.py \
    -xd /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first5ExpressionPCs.txt.gz \
    -xi Comp2 \
    -yd /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first5ExpressionPCs.txt.gz \
    -yi Comp4 \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_Comp2_vs_Comp4
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
        self.std_path = getattr(arguments, 'sample_to_dataset')
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
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=True,
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
        x_df = self.load_file(self.x_data_path, header=0, index_col=0)
        y_df = self.load_file(self.y_data_path, header=0, index_col=0)

        print("Pre-process")
        if self.x_transpose:
            x_df = x_df.T
        if self.y_transpose:
            y_df = y_df.T

        print(x_df)
        print(y_df)

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

        print(x_subset_df)
        print(y_subset_df)

        print("Merging.")
        plot_df = x_subset_df.merge(y_subset_df, left_index=True, right_index=True)
        plot_df.columns = ["x", "y"]
        plot_df.dropna(inplace=True)
        plot_df = plot_df.astype(float)
        print(plot_df)

        print("Loading color data.")
        sa_df = self.load_file(self.std_path, header=0, index_col=None)
        sa_df.set_index(sa_df.columns[0], inplace=True)
        sa_df.columns = ["dataset"]
        plot_df = plot_df.merge(sa_df, left_index=True, right_index=True)

        dataset_sample_counts = list(zip(*np.unique(sa_df["dataset"], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [x[0] for x in dataset_sample_counts]

        print("Plotting.")
        self.plot(df=plot_df,
                  panels=datasets,
                  palette=self.palette,
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

    def plot(self, df, x="x", y="y", group="dataset", panels=None, palette=None,
             xlabel=None, ylabel=None, title="", filename="plot"):
        if panels is None:
            panels = list(df[group].unique())
            panels.sort()

        nplots = len(panels)
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
                                 figsize=(9 * ncols, 9 * nrows))
        sns.set(color_codes=True)


        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1 and ncols > 1:
                ax = axes[col_index]
            elif nrows > 1 and ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            sns.despine(fig=fig, ax=ax)

            if i < nplots:
                plot_df = df.loc[df[group] == panels[i], [x, y]]
                if plot_df.shape[0] <= 2:
                    continue

                accent_color = palette[panels[i]]

                coef, _ = stats.spearmanr(plot_df[y], plot_df[x])

                sns.regplot(x=x,
                            y=y,
                            data=plot_df,
                            scatter_kws={'facecolors': "#000000",
                                         'linewidth': 0,
                                         's': 60,
                                         'alpha': 0.75},
                            line_kws={"color": accent_color,
                                      'linewidth': 5},
                            ax=ax)

                ax.axhline(0, ls='--', color="#000000", zorder=-1)
                ax.axvline(0, ls='--', color="#000000", zorder=-1)

                ax.annotate(
                    'N = {}'.format(plot_df.shape[0]),
                    xy=(0.03, 0.94),
                    xycoords=ax.transAxes,
                    color=accent_color,
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'r = {:.2f}'.format(coef),
                    xy=(0.03, 0.90),
                    xycoords=ax.transAxes,
                    color=accent_color,
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')

                tmp_xlabel = ""
                if row_index == (nrows - 1):
                    tmp_xlabel = xlabel
                ax.set_xlabel(tmp_xlabel,
                              color="#000000",
                              fontsize=20,
                              fontweight='bold')
                tmp_ylabel = ""
                if col_index == 0:
                    tmp_ylabel = ylabel
                ax.set_ylabel(tmp_ylabel,
                              color="#000000",
                              fontsize=20,
                              fontweight='bold')

                ax.set_title(panels[i],
                             color=accent_color,
                             fontsize=25,
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
            else:
                tmp_xlabel = ""
                if row_index == (nrows - 1):
                    tmp_xlabel = xlabel
                ax.set_xlabel(tmp_xlabel,
                              color="#000000",
                              fontsize=20,
                              fontweight='bold')

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

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
