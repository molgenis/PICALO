#!/usr/bin/env python3

"""
File:         create_correlation_scatterplot.py
Created:      2022/05/05
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
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Correlation Scatterplot"
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
./create_correlation_scatterplot.py -h

### MetaBrain ###

### BIOS ###

./create_correlation_scatterplot.py \
    -rd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -rl RNAseq alignment metrics \
    -cd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first100ExpressionPCs.txt.gz \
    -cn 10 \
    -cl PCs \
    -on 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_RNAseqMetrics_PCs
    
./create_correlation_scatterplot.py \
    -rd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -rl RNAseq alignment metrics \
    -cd /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -cn 10 \
    -cl PICs \
    -on 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_RNAseqMetrics_PICs
    
./create_correlation_scatterplot.py \
    -rd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz \
    -rl RNAseq alignment metrics \
    -cd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first100ExpressionPCs.txt.gz \
    -cn 10 \
    -cl PCs \
    -on 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_CFNormalRes_PCs
    
./create_correlation_scatterplot.py \
    -rd /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz \
    -rl RNAseq alignment metrics \
    -cd /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -cn 10 \
    -cl PICs \
    -on 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_CFNormalRes_PICs
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.row_data_path = getattr(arguments, 'row_data')
        self.row_n = getattr(arguments, 'row_n')
        self.row_label = " ".join(getattr(arguments, 'row_label'))
        self.col_data_path = getattr(arguments, 'col_data')
        self.col_n = getattr(arguments, 'col_n')
        self.col_label = " ".join(getattr(arguments, 'col_label'))
        self.method = getattr(arguments, 'method')
        self.outname = getattr(arguments, 'outname')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.ct_trans_dict ={
            "Baso": "Basophil",
            "Neut": "Neutrophil",
            "Eos": "Eosinophil",
            "Granulocyte": "Granulocyte",
            "Mono": "Monocyte",
            "LUC": "LUC",
            "Lymph": "Lymphocyte",
        }

        self.palette = {
            "star": "#0072B2",
            "bam": "#009E73",
            "fastqc_raw": "#CC79A7",
            "fastqc_clean": "#E69F00",
            "prime_bias": "#D55E00",
            "Basophil": "#009E73",
            "Neutrophil": "#D55E00",
            "Eosinophil": "#0072B2",
            "Granulocyte": "#808080",
            "Monocyte": "#E69F00",
            "LUC": "#F0E442",
            "Lymphocyte": "#CC79A7"
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
        parser.add_argument("-rd",
                            "--row_data",
                            type=str,
                            required=True,
                            help="The path to the row data matrix.")
        parser.add_argument("-rn",
                            "--row_n",
                            type=int,
                            required=False,
                            default=None,
                            help="The number row lines to use.")
        parser.add_argument("-rl",
                            "--row_label",
                            nargs="*",
                            type=str,
                            required=False,
                            default="",
                            help="The label of -r / --row_data.")
        parser.add_argument("-cd",
                            "--col_data",
                            type=str,
                            required=False,
                            help="The path to the col data matrix.")
        parser.add_argument("-cn",
                            "--col_n",
                            type=int,
                            required=False,
                            default=None,
                            help="The number col lines to use.")
        parser.add_argument("-cl",
                            "--col_label",
                            nargs="*",
                            type=str,
                            required=False,
                            default="",
                            help="The label of -c / --col_data.")
        parser.add_argument("-m",
                            "--method",
                            type=str,
                            choices=["Pearson", "Spearman"],
                            default="Spearman",
                            help="The correlation method. Default: Spearman.")
        parser.add_argument("-on",
                            "--outname",
                            type=str,
                            required=True,
                            help="The name of the output files.")

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

        print("Getting overlap.")
        overlap = list(set(row_df.index).intersection(set(col_df.index)))
        print("\tN = {}".format(len(overlap)))
        if len(overlap) == 0:
            print("No data overlapping.")
            exit()
        row_df = row_df.loc[overlap, :]
        if self.row_n is not None:
            row_df = row_df.iloc[:, :self.row_n]
        col_df = col_df.loc[overlap, :]
        if self.col_n is not None:
            col_df = col_df.iloc[:, :self.col_n]

        print("Correlating.")
        corr_df = self.correlate(index_df=row_df,
                                 columns_df=col_df,
                                 triangle=triangle)
        corr_df["abs coef"] = corr_df["coef"].abs()
        # corr_df["group"] = corr_df["variable1"].str.split(".", n=1, expand=True)[0]
        corr_df["group"] = [self.ct_trans_dict[ct] for ct in corr_df["variable1"].str.split("_", n=1, expand=True)[0]]
        corr_df = corr_df.loc[corr_df["pvalue"] < (0.05 / corr_df.shape[0]), :]
        print(corr_df)

        print("Plotting")
        self.plot(df=corr_df,
                  x="variable2",
                  y="abs coef",
                  group="group",
                  order=[col for col in col_df.columns if col in corr_df["variable2"].unique()],
                  annot=None,
                  palette=self.palette,
                  xlabel=self.col_label,
                  ylabel="absolute {} correlation".format(self.method),
                  title="{} - {} correlations".format(self.col_label, self.row_label),
                  filename=self.outname)

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
        data = []

        for i, index_column in enumerate(index_df.columns):
            for j, column_column in enumerate(columns_df.columns):
                if triangle and i < j:
                    continue
                corr_data = pd.concat([index_df[index_column], columns_df[column_column]], axis=1)
                corr_data.dropna(inplace=True)

                coef = np.nan
                pvalue = np.nan
                if np.min(corr_data.std(axis=0)) > 0:
                    if self.method == "Pearson":
                        coef, pvalue = stats.pearsonr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])
                    elif self.method == "Spearman":
                        coef, pvalue = stats.spearmanr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])

                data.append([index_column, column_column, coef, pvalue])

        return pd.DataFrame(data, columns=["variable1", "variable2", "coef", "pvalue"])

    def plot(self, df, x="x", y="y", group="group", order=None, palette=None,
             annot=None, xlabel="", ylabel="", title="",
             filename="ieqtl_plot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.boxplot(x=x,
                    y=y,
                    hue=group,
                    data=df,
                    order=order,
                    palette=palette,
                    ax=ax)
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .5))
        sns.swarmplot(x=x,
                      y=y,
                      hue=group,
                      data=df,
                      order=order,
                      palette=palette,
                      dodge=True,
                      ax=ax)
        ax.get_legend().remove()

        if palette is not None:
            handles = []
            for label, color in palette.items():
                if label in df[group].unique():
                    handles.append(mpatches.Patch(color=color, label=label))
            ax.legend(handles=handles, loc=1)

        if annot is not None:
            for i, annot_label in enumerate(annot):
                ax.annotate(annot_label,
                            xy=(0.03, 0.94 - (i * 0.04)),
                            xycoords=ax.transAxes,
                            color="#000000",
                            alpha=0.75,
                            fontsize=12,
                            fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        fig.suptitle(title,
                     fontsize=18,
                     fontweight='bold')

        ax.set_ylim(0, 1)

        outpath = os.path.join(self.outdir, "{}.png".format(filename))
        print("\t\tSaving plot: {}".format(os.path.basename(outpath)))
        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Row data path: {}".format(self.row_data_path))
        print("  > Row label: {}".format(self.row_label))
        print("  > Col data path: {}".format(self.col_data_path))
        print("  > Col label: {}".format(self.col_label))
        print("  > Correlation method: {}".format(self.method))
        print("  > Output name: {}".format(self.outname))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
