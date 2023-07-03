#!/usr/bin/env python3

"""
File:         plot_interferon_signaling.py
Created:      2022/07/12
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 M.Vochteloo
Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import upsetplot as up
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Interferon Signaling"
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
./plot_interferon_signaling.py -h
    
./plot_interferon_signaling.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/data/interferome_experiment_data_bios_eqtl_genes.txt.gz \
    -d /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/gene_set_enrichment/2022-04-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_ZscoreFiltering/info.txt.gz \
    -p PIC2 PIC9 PIC10 \
    -o 2022-04-13-BIOS_InterferonSignaling

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.experiment_data_path = getattr(arguments, 'interferome')
        self.data_path = getattr(arguments, 'data')
        self.pics = getattr(arguments, 'pics')
        self.fold_change = getattr(arguments, 'fold_change')
        self.outfile = getattr(arguments, 'outfile')
        self.alpha = getattr(arguments, 'alpha')
        self.top_n = getattr(arguments, 'top_n')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot_interferon_signaling')
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
        parser.add_argument("-i",
                            "--interferome",
                            type=str,
                            required=True,
                            help="The path to the in interferome experiment data.")
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data.")
        parser.add_argument("-p",
                            "--pics",
                            nargs="+",
                            type=str,
                            default=all,
                            help="The PICs to analyse."
                                 "Default: 'png'.")
        parser.add_argument("-fc",
                            "--fold_change",
                            type=float,
                            default=2.,
                            help="The fold change cut-off. Default: 2.0.")
        parser.add_argument("-a",
                            "--alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The significance cut-off. Default: 0.05.")
        parser.add_argument("-tn",
                            "--top_n",
                            type=int,
                            default=200,
                            help="The top n genes to include in the "
                                 "enrichment analysis. Default: 200.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        experiment_data_df = self.load_file(self.experiment_data_path, header=0, index_col=None, skiprows=18)
        print(experiment_data_df)

        picalo_data_df = self.load_file(self.data_path, header=0, index_col=0)
        print(picalo_data_df)

        print("Preprocessing data.")
        experiment_data_df = experiment_data_df.loc[experiment_data_df["Fold Change"].abs() > self.fold_change, :]
        interferon_annotation_df = experiment_data_df.groupby("Ensembl ID")["Inteferome Type"].apply(lambda x: ','.join([value for value in set(x)])).reset_index()
        print(interferon_annotation_df)

        print("Merging data.")
        df = picalo_data_df.merge(interferon_annotation_df, left_on="ProbeName", right_on="Ensembl ID", how="left")
        df["Inteferome Type"] = df["Inteferome Type"].fillna("NA")
        print(df)
        del experiment_data_df, picalo_data_df, interferon_annotation_df

        is_total_counts = df["Inteferome Type"].value_counts().to_frame()
        is_total_counts.columns = ["total"]
        print(is_total_counts)

        if self.pics is None:
            self.pics = [col.replace(" FDR", "") for col in df.columns if col.endswith(" FDR")]

        for pic in self.pics:
            print("### Analyzing {} ###".format(pic))

            for correlation_direction in ["positive", "negative"]:
                subset_df = df.loc[:, ["avgExpression",
                                       "OfficialSymbol",
                                       "Entrez",
                                       "{} r".format(pic),
                                       "{} pvalue".format(pic),
                                       "{} FDR".format(pic),
                                       "{} zscore".format(pic),
                                       "{} ieQTL FDR".format(pic),
                                       "{} ieQTL direction".format(pic),
                                       "Inteferome Type"]].copy()
                subset_df.columns = ["avg expression", "symbol", "entrez", "correlation coefficient", "correlation p-value", "correlation FDR", "correlation zscore", "ieQTL FDR", "ieQTL direction", "Inteferome Type"]
                subset_df = subset_df.loc[~subset_df["entrez"].isna(), :]
                subset_df["abs correlation zscore"] = subset_df["correlation zscore"].abs()
                subset_df.sort_values(by="abs correlation zscore", ascending=False, inplace=True)

                if correlation_direction == "positive":
                    subset_df = subset_df.loc[subset_df["correlation zscore"] > 0, :]
                elif correlation_direction == "negative":
                    subset_df = subset_df.loc[subset_df["correlation zscore"] < 0, :]
                else:
                    print("huh")
                    exit()

                print(subset_df)

                ################################################################

                corr_subset_df = subset_df.iloc[:self.top_n, :].copy()
                corr_subset_df = corr_subset_df.loc[corr_subset_df["correlation FDR"] < self.alpha, :]
                if corr_subset_df.shape[0] > 0:
                    print("\t{} correlation".format(correlation_direction))
                    print("\tAnalyzing {:,} genes".format(corr_subset_df.shape[0]))
                    print(corr_subset_df)
                    is_subset_counts = corr_subset_df["Inteferome Type"].value_counts().to_frame()
                    is_subset_counts.columns = ["subset"]
                    is_counts = is_total_counts.merge(is_subset_counts, left_index=True, right_index=True)
                    print(is_counts)
                    print(is_counts / is_counts.sum(axis=0))
                    for unique in corr_subset_df["Inteferome Type"].unique():
                        print(" ".join(corr_subset_df.loc[corr_subset_df["Inteferome Type"] == unique, "symbol"].values.tolist()))
                    print("")

                ################################################################

                # Filter on eQTL genes.
                signif_subset_df = subset_df.loc[subset_df["ieQTL FDR"] < self.alpha, :].iloc[:self.top_n, :].copy()
                if signif_subset_df.shape[0] > 0:
                    print("\t{} correlation (ieQTL FDR < {})".format(correlation_direction, self.alpha))
                    print("\tAnalyzing {:,} genes".format(signif_subset_df.shape[0]))
                    is_subset_counts = signif_subset_df["Inteferome Type"].value_counts().to_frame()
                    is_subset_counts.columns = ["subset"]
                    is_counts = is_total_counts.merge(is_subset_counts, left_index=True, right_index=True)
                    print(is_counts)
                    print(is_counts / is_counts.sum(axis=0))
                    for unique in signif_subset_df["Inteferome Type"].unique():
                        print(unique)
                        print(" ".join(signif_subset_df.loc[signif_subset_df["Inteferome Type"] == unique, "symbol"].values.tolist()))
                        print("")
                    print("")




        #
        # df = pic10_df.merge(interferon_annotation_df, on="Ensembl ID")
        # print(df)
        # print(df.loc[df["Direction"] == "negative", :])
        #
        # for direction in ["positive", "negative"]:
        #     print(df.loc[df["Direction"] == direction, "Inteferome Type"].value_counts())

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > Interfome experiment data path: {}".format(self.experiment_data_path))
        print("  > PICALO data path: {}".format(self.data_path))
        if self.pics is None:
            print("  > PICs: all")
        else:
            print("  > PICs: {}".format(", ".join(self.pics)))
        print("  > Fold change: {}".format(self.fold_change))
        print("  > Alpha: {}".format(self.alpha))
        print("  > Top-N: {}".format(self.top_n))
        print("  > Output filename: {}".format(self.outfile))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
