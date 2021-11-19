#!/usr/bin/env python3

"""
File:         compare_tvalues.py
Created:      2021/11/19
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
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Compare t-values"
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
./compare_tvalues.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz  \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-20RNAseqAlignemntMetrics-MAF5/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz  \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5-NoSTARMetricsCorrection/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz  \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5-NoCorrection/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz  \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC4/results_iteration049.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC4.txt.gz \
    -n AllSTARCorrected_Median5Pt3PBias 20STARCorrected_Median5Pt3PBias NoSTARCorrected_Median5Pt3PBias NoneCorrected_Median5Pt3PBias AllSTARCorrected_PIC4Iterative AllSTARCorrected_PIC4 -o plot1 
    
./compare_tvalues.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5-NoSTARMetricsCorrection/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC4/results_iteration049.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC4.txt.gz \
    -n NoSTARCorrected_Median5Pt3PBias AllSTARCorrected_Median5Pt3PBias AllSTARCorrected_PIC4Iterative AllSTARCorrected_PIC4 -o plot2 
    
./compare_tvalues.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC1.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC2.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC3.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC4.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC_interactions/PIC5.txt.gz \
    -n PIC1 PIC2 PIC3 PIC4 PIC5 -o plot3
    
./compare_tvalues.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration000.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration010.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration020.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration030.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration040.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration049.txt.gz \
    -n iteration000 iteration010 iteration020 iteration030 iteration040 iteration049 -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5-PIC1 
    
./compare_tvalues.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration000.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration001.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration002.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration003.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration004.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration005.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration006.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration007.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration008.txt.gz \
    /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5/PIC1/results_iteration009.txt.gz \
    -n iteration000 iteration001 iteration002 iteration003 iteration004 iteration005 iteration006 iteration007 iteration008 iteration009 -o BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-AllRNAseqAlignmentMetricsNoFiltering-MAF5-PIC1-First10Iterations 
    
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_paths = getattr(arguments, 'data')
        self.names = getattr(arguments, 'names')
        self.output_filename = getattr(arguments, 'output')

        if len(self.data_paths) != len(self.names):
            print("Inputs are not same length.")
            exit()

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "no signif": "#808080",
            "x signif": "#0072B2",
            "y signif": "#D55E00",
            "both signif": "#009E73"
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
                            nargs="*",
                            type=str,
                            required=True,
                            help="The paths to the input data.")
        parser.add_argument("-n",
                            "--names",
                            nargs="*",
                            type=str,
                            required=True,
                            help="The names of the data files.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            default="PlotPerColumn_ColorByCohort",
                            help="The name of the output file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        data = []
        for name, fpath in zip(self.names, self.data_paths):
            df = self.load_file(fpath, header=0, index_col=None)

            # Set index.
            if "snp" in df.columns and "gene" in df.columns:
                df.index = df["snp"] + ":" + df["gene"]
            elif "SNP" in df.columns and "gene" in df.columns:
                df.index = df["SNP"] + ":" + df["gene"]
            else:
                print(df.columns.tolist())
                exit()

            # Set t-value.
            if "ieQTL beta-interaction.1" in df.columns and "ieQTL std-interaction.1" in df.columns:
                df["t-value"] = df["ieQTL beta-interaction.1"] / df["ieQTL std-interaction.1"]
            elif "beta-interaction" in df.columns and "std-interaction" in df.columns:
                df["t-value"] = df["beta-interaction"] / df["std-interaction"]
            else:
                print(df.columns.tolist())
                exit()

            # Set significance.
            if "ieQTL FDR" in df.columns:
                df["significant"] = "False"
                df.loc[df["ieQTL FDR"] < 0.05, "significant"] = "True"
            elif "FDR" in df.columns:
                df["significant"] = "False"
                df.loc[df["FDR"] < 0.05, "significant"] = "True"
            else:
                print(df.columns.tolist())
                exit()

            # Subset.
            subset_df = df.loc[:, ["t-value", "significant"]]
            subset_df.columns = [name, "{}_significant".format(name)]

            data.append(subset_df)
            del subset_df
        df = pd.concat(data, axis=1)
        print(df)

        print("Plot")
        self.plot(df=df, columns=self.names)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, df, columns, title=""):
        ncols = len(columns)
        nrows = len(columns)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='all',
                                 sharey='all',
                                 figsize=(10 * ncols, 10 * nrows))
        sns.set(color_codes=True)
        sns.set_style("ticks")

        for i, y_col in enumerate(columns):
            for j, x_col in enumerate(columns):
                print(i, j)
                ax = axes[i, j]

                # Construct plot df.
                plot_df = df.loc[:, [x_col, y_col, "{}_significant".format(x_col), "{}_significant".format(y_col)]]
                plot_df.columns = ["x", "y", "x_significant", "y_significant"]

                plot_df["hue"] = self.palette["no signif"]
                plot_df.loc[(plot_df["x_significant"] == "True") & (plot_df["y_significant"] == "False"), "hue"] = self.palette["x signif"]
                plot_df.loc[(plot_df["x_significant"] == "False") & (plot_df["y_significant"] == "True"), "hue"] = self.palette["y signif"]
                plot_df.loc[(plot_df["x_significant"] == "True") & (plot_df["y_significant"] == "True"), "hue"] = self.palette["both signif"]

                if i == 0 and j == (ncols - 1):
                    ax.set_axis_off()
                    handles = []
                    for label, value in self.palette.items():
                        handles.append(mpatches.Patch(color=value, label=label))
                    ax.legend(handles=handles, loc=4, fontsize=25)
                elif i < j:
                    ax.set_axis_off()
                    continue
                elif i == j:
                    ax.set_axis_off()

                    ax.annotate(y_col.replace("_", "\n"),
                                xy=(0.5, 0.5),
                                ha='center',
                                xycoords=ax.transAxes,
                                color="#000000",
                                fontsize=40,
                                fontweight='bold')
                else:
                    sns.despine(fig=fig, ax=ax)

                    sns.regplot(x="x",
                                y="y",
                                data=plot_df,
                                ci=None,
                                scatter_kws={'facecolors': plot_df["hue"],
                                             'linewidth': 0},
                                line_kws={"color": "#000000"},
                                ax=ax)

                    ax.axvline(0, ls='--', color="#000000", alpha=0.15, zorder=-1)
                    ax.axhline(0, ls='--', color="#000000", alpha=0.15, zorder=-1)
                    ax.axline((0, 0), slope=1, ls='--', color="#000000", alpha=0.15, zorder=-1)
                    ax.axline((0, 0), slope=-1, ls='--', color="#000000", alpha=0.15, zorder=-1)

                    ax.set_ylabel("",
                                  fontsize=20,
                                  fontweight='bold')
                    ax.set_xlabel("",
                                  fontsize=20,
                                  fontweight='bold')

                    coef, _ = stats.pearsonr(plot_df["x"], plot_df["y"])
                    ax.annotate(
                        'r = {:.2f}'.format(coef),
                        xy=(0.03, 0.94),
                        xycoords=ax.transAxes,
                        color="#404040",
                        fontsize=14,
                        fontweight='bold')

                    counts = dict(zip(*np.unique(plot_df["hue"], return_counts=True)))
                    for color in self.palette.values():
                        if color not in counts:
                            counts[color] = 0
                    for annot_index, (color, n) in enumerate(counts.items()):
                        ax.annotate(
                            'N = {:,}'.format(n),
                            xy=(0.03, 0.90 - (annot_index * 0.04)),
                            xycoords=ax.transAxes,
                            color=color,
                            fontsize=14,
                            fontweight='bold')

        fig.suptitle(title,
                     fontsize=40,
                     fontweight='bold')

        fig.savefig(os.path.join(self.outdir, "{}.png".format(self.output_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data:")
        for i, (name, fpath) in enumerate(zip(self.names, self.data_paths)):
            print("  > {} = {}".format(name, fpath))
        print("  > Output filename: {}".format(self.output_filename))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
