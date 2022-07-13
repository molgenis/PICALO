#!/usr/bin/env python3

"""
File:         plot_avg_expression.py
Created:      2022/07/07
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
import itertools
import argparse
import re
import glob
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import upsetplot as up
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Average Expression"
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
./plot_avg_expression.py -h

./plot_avg_expression.py \
    -mpic /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -mpc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PCsAsCov-Conditional \
    -me /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -bpic /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -bpc /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PCsAsCov-Conditional \
    -be /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -conditional \
    -o 20220706_MetaBrain_BIOS_gene_expression \
    -e png pdf
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta_pic_indir = getattr(arguments, 'metabrain_pic_indir')
        self.meta_pc_indir = getattr(arguments, 'metabrain_pc_indir')
        self.meta_expr_path = getattr(arguments, 'metabrain_expression')
        self.bios_pic_indir = getattr(arguments, 'bios_pic_indir')
        self.bios_pc_indir = getattr(arguments, 'bios_pc_indir')
        self.bios_expr_path = getattr(arguments, 'bios_expression')
        self.conditional = getattr(arguments, 'conditional')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot_avg_expression')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            "no interaction": "#808080",
            "yes interaction": "#0072B2",
            "biological": "#009E73",
            "technical": "#CC79A7"
        }

        self.bios_technical = ["PIC1", "PIC4", "PIC8"]
        self.meta_technical = ["PIC1", "PIC4", "PIC7"]

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
        parser.add_argument("-mpic",
                            "--metabrain_pic_indir",
                            type=str,
                            required=True,
                            help="The path to MetaBrain PIC interactions.")
        parser.add_argument("-mpc",
                            "--metabrain_pc_indir",
                            type=str,
                            required=True,
                            help="The path to MetaBrain PC interactions.")
        parser.add_argument("-me",
                            "--metabrain_expression",
                            type=str,
                            required=True,
                            help="The path to MetaBrain expression matrix.")
        parser.add_argument("-bpic",
                            "--bios_pic_indir",
                            type=str,
                            required=True,
                            help="The path to BIOS PIC interactions.")
        parser.add_argument("-bpc",
                            "--bios_pc_indir",
                            type=str,
                            required=True,
                            help="The path to BIOS PC interactions.")
        parser.add_argument("-be",
                            "--bios_expression",
                            type=str,
                            required=True,
                            help="The path to BIOS expression matrix.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")
        parser.add_argument("-e",
                            "--extension",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Load ieQTL data.")
        meta_pic_data = self.load_data(indir=self.meta_pic_indir, conditional=self.conditional)
        meta_pc_data = self.load_data(indir=self.meta_pc_indir, conditional=self.conditional)
        meta_expression = self.load_file(self.meta_expr_path, header=0, index_col=0)
        bios_pic_data = self.load_data(indir=self.bios_pic_indir, conditional=self.conditional)
        bios_pc_data = self.load_data(indir=self.bios_pc_indir, conditional=self.conditional)
        bios_expression = self.load_file(self.bios_expr_path, header=0, index_col=0)

        print("Pre-processing.")
        mean_meta_expression_df = meta_expression.mean(axis=1).to_frame()
        mean_meta_expression_df.columns = ["expression"]

        meta_pic_df = self.merge_data_frames(interaction_df=meta_pic_data, expression_df=mean_meta_expression_df)
        meta_pc_df = self.merge_data_frames(interaction_df=meta_pc_data, expression_df=mean_meta_expression_df)

        mean_bios_expression_df = bios_expression.mean(axis=1).to_frame()
        mean_bios_expression_df.columns = ["expression"]

        bios_pic_df = self.merge_data_frames(interaction_df=bios_pic_data, expression_df=mean_bios_expression_df)
        bios_pc_df = self.merge_data_frames(interaction_df=bios_pc_data, expression_df=mean_bios_expression_df)

        plot_data = []
        for (tissue, data, df, tech_cols) in (("blood", "PIC", bios_pic_df, self.bios_technical),
                                              ("blood", "PC", bios_pc_df, self.bios_technical),
                                              ("brain", "PIC", meta_pic_df, self.meta_technical),
                                              ("brain", "PC", meta_pc_df, self.meta_technical)):
            bio_cols = [col for col in df.columns if col not in tech_cols and col != "gene"]

            for variable in ["no interaction", "yes interaction", "biological", "technical"]:
                expression = []
                if variable == "no interaction":
                    expression = df.loc[df.iloc[:, :-2].sum(axis=1) == 0, "expression"].tolist()
                elif variable == "yes interaction":
                    expression = df.loc[df.iloc[:, :-2].sum(axis=1) > 0, "expression"].tolist()
                elif data == "PIC" and variable == "biological":
                    expression = df.loc[(df[tech_cols].sum(axis=1) == 0) & (df[bio_cols].sum(axis=1) > 0), "expression"].tolist()
                elif data == "PIC" and variable == "technical":
                    expression = df.loc[(df[bio_cols].sum(axis=1) == 0) & (df[tech_cols].sum(axis=1) > 0), "expression"].tolist()
                else:
                    pass

                if len(expression) > 0:
                    for value in expression:
                        plot_data.append([tissue + data, variable, value])

        df = pd.DataFrame(plot_data, columns=["tissue + data", "variable", "value"])
        print(df)
        print(df["variable"].unique())

        print("Creating plot.")
        self.create_boxplot(
            df=df,
            x="tissue + data",
            y="value",
            hue="variable",
            palette=self.palette,
            ylabel="TMM log2 expression",
            filename=self.out_filename
        )

    def load_data(self, indir, conditional=False, signif_col="FDR"):
        inpaths = glob.glob(os.path.join(indir, "*.txt.gz"))
        if conditional:
            inpaths = [inpath for inpath in inpaths if inpath.endswith("_conditional.txt.gz")]
        else:
            inpaths = [inpath for inpath in inpaths if not inpath.endswith("_conditional.txt.gz")]
        inpaths.sort(key=self.natural_keys)

        ieqtl_df_list = []
        for i, inpath in enumerate(inpaths):
            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df[signif_col] <= 0.05, :].index
            ieqtl_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_df.loc[ieqtls, filename] = 1
            ieqtl_df_list.append(ieqtl_df)

            del ieqtl_df

        ieqtl_df = pd.concat(ieqtl_df_list, axis=1)
        return ieqtl_df

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
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    @staticmethod
    def merge_data_frames(interaction_df, expression_df):
        genes = []
        for index in interaction_df.index:
            genes.append(index.split("_")[-1])

        interaction_df["gene"] = genes
        interaction_df = interaction_df.merge(expression_df, left_on="gene", right_index=True, how="left")
        return interaction_df

    def create_boxplot(self, df, x="variable", y="value", hue=None,
                       xlabel="", ylabel="", palette=None,
                       filename="plot"):
        sns.set(rc={'figure.figsize': (24, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        print(x, y, hue)

        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       data=df,
                       palette=palette,
                       cut=0,
                       dodge=True,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    hue=hue,
                    data=df,
                    whis=np.inf,
                    color="white",
                    dodge=True,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain input directories:")
        print("    > PIC: {}".format(self.meta_pic_indir))
        print("    > PC: {}".format(self.meta_pc_indir))
        print("    > Expression: {}".format(self.meta_expr_path))
        print("  > BIOS input directories:")
        print("    > PIC: {}".format(self.bios_pic_indir))
        print("    > PC: {}".format(self.bios_pc_indir))
        print("    > Expression: {}".format(self.bios_expr_path))
        print("  > Conditional: {}".format(self.conditional))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
