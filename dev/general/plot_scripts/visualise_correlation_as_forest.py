#!/usr/bin/env python3

"""
File:         visualise_correlation_as_forest.py
Created:      2022/07/14
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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Visualise Correlation as Forest"
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
./visualise_correlation_as_forest.py -h

./visualise_correlation_as_forest.py \
    -yd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz \
    -xd1 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -xi1 PIC3 \
    -xd2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -xi2 Comp5 \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_CellFractionPercentages_vs_PIC3_Comp5 \
    -e png pdf
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.y_data_path = getattr(arguments, 'y_data')
        self.x_data_path1 = getattr(arguments, 'x_data1')
        self.x_data_index1 = getattr(arguments, 'x_index1')
        self.x_data_path2 = getattr(arguments, 'x_data2')
        self.x_data_index2 = getattr(arguments, 'x_index2')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Loading palette.
        self.palette = {
            "PIC3": "#D55E00",
            "Comp5": "#808080"
        }
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
        parser.add_argument("-yd",
                            "--y_data",
                            type=str,
                            required=True,
                            help="The path to the y data matrix.")
        parser.add_argument("-xd1",
                            "--x_data1",
                            type=str,
                            required=False,
                            help="The path to the x data matrix 2.")
        parser.add_argument("-xi1",
                            "--x_index1",
                            type=str,
                            required=False,
                            help="The index to plot.")
        parser.add_argument("-xd2",
                            "--x_data2",
                            type=str,
                            required=False,
                            help="The path to the x data matrix 2.")
        parser.add_argument("-xi2",
                            "--x_index2",
                            type=str,
                            required=False,
                            help="The index to plot.")
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
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Loading data ###")
        y_df = self.load_file(self.y_data_path, header=0, index_col=0)
        x_df1 = self.load_file(self.x_data_path1, header=0, index_col=0).T
        x_df2 = self.load_file(self.x_data_path2, header=0, index_col=0).T

        print("Calculating")
        data = []
        for x_index, x_df in ((self.x_data_index1, x_df1), (self.x_data_index2, x_df2)):
            for y_index in y_df.columns:
                df = y_df[[y_index]].merge(x_df[[x_index]], left_index=True, right_index=True)
                df.columns = ["y", "x"]
                df.dropna(inplace=True)

                coef, pvalue = stats.pearsonr(df["y"], df["x"])
                coef = np.abs(coef)
                lower, upper = self.calc_pearson_ci(coef, df.shape[0])
                if lower < 0:
                    lower = 0

                data.append([y_index, x_index, coef, lower, upper])

        df = pd.DataFrame(data, columns=["y_index", "x_index", "mean", "lower", "upper"])
        print(df)

        print("Plotting forest plot")
        self.plot_stripplot(df)

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=None, nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    @staticmethod
    def calc_pearson_ci(coef, n):
        stderr = 1.0 / math.sqrt(n - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(coef) - delta)
        upper = math.tanh(math.atanh(coef) + delta)
        return lower, upper

    def plot_stripplot(self, df):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        df_m = df.melt(id_vars=["y_index", "x_index"], value_vars=["lower", "upper"])
        sns.pointplot(x="value",
                      y="y_index",
                      data=df_m,
                      hue="x_index",
                      join=False,
                      palette=self.palette,
                      ax=ax)

        sns.stripplot(x="mean",
                      y="y_index",
                      data=df,
                      hue="x_index",
                      size=25,
                      dodge=False,
                      orient="h",
                      palette=self.palette,
                      linewidth=1,
                      edgecolor="w",
                      jitter=0,
                      ax=ax)

        ax.set_ylabel('',
                      fontsize=12,
                      fontweight='bold')
        ax.set_xlabel('absolute Pearson r',
                      fontsize=12,
                      fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=10)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

        for extension in self.extensions:
            filename = "{}_stripplot.{}".format(self.out_filename, extension)
            print("\t\tSaving plot: {}".format(filename))
            fig.savefig(os.path.join(self.outdir, filename), dpi=300)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Y-axis data path: {}".format(self.y_data_path))
        print("  > X-axis data path 1: {}".format(self.x_data_path1))
        print("  > X-axis data index 1: {}".format(self.x_data_index1))
        print("  > X-axis data path 2: {}".format(self.x_data_path2))
        print("  > X-axis data index 2: {}".format(self.x_data_index2))
        print("  > Palette {}".format(self.palette_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
