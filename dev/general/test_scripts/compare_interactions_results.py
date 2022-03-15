#!/usr/bin/env python3

"""
File:         compare_interactions_results.py
Created:      2021/12/21
Last Changed: 2022/02/10
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
import glob
import argparse
import re
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.


# Metadata
__program__ = "Compare Interactions Results"
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
./compare_interactions_results.py -h

./compare_interactions_results.py \
    -pic /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICsAsCov \
    -pc /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-First33ExprPCsAsCov \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs
    
./compare_interactions_results.py \
    -indir1 /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/PIC_interactions \
    -n1 ExprPCs \
    -indir2 /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-SP140AsCov/PIC_interactions \
    -n2 SP140 \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs_ExprPCsAsCov_vs_SP140AsCov
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir1 = getattr(arguments, 'input_directory1')
        self.name1 = getattr(arguments, 'name1')
        self.indir2 = getattr(arguments, 'input_directory2')
        self.name2 = getattr(arguments, 'name2')
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
        parser.add_argument("-indir1",
                            "--input_directory1",
                            type=str,
                            required=True,
                            help="The path to the first interaction directory.")
        parser.add_argument("-n1",
                            "--name1",
                            type=str,
                            required=True,
                            help="The name of the first interaction directory.")
        parser.add_argument("-indir2",
                            "--input_directory2",
                            type=str,
                            required=True,
                            help="The path to the second interaction directory.")
        parser.add_argument("-n2",
                            "--name2",
                            type=str,
                            required=True,
                            help="The name of the second interaction directory.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        gene_info_df = pd.read_csv("/groups/umcg-bios/tmp01/projects/PICALO/data/ArrayAddressToSymbol.txt.gz", sep="\t", header=0, index_col=None)
        gene_dict = dict(zip(gene_info_df["ArrayAddress"], gene_info_df["Symbol"]))
        del gene_info_df

        print("### Step1 ###")
        print("Loading data")
        df1 = self.load_interaction_results(indir=self.indir1)
        df2 = self.load_interaction_results(indir=self.indir2)
        #df1 = df1.iloc[:, 4:]
        #df2 = df2.iloc[:, [3, 4]]
        print(df1)
        print(df2)


        # print("Comparing result")
        # for covariate2 in df2.columns:
        #     ieqtls = df2.loc[df2[covariate2] < 0.05, :].index.tolist()
        #     print("\t{}: N={:,}".format(covariate2, len(ieqtls)))
        #     for covariate1 in df1.columns:
        #         df1_subset = df1.loc[ieqtls, [covariate1]]
        #         repl = df1_subset.loc[df1_subset[covariate1] < 0.05, :]
        #         print("\t  {} N={:,} [{:.2f}%]\t{}".format(covariate1, repl.shape[0], (100 / df1_subset.shape[0]) * repl.shape[0], ", ".join([gene_dict[gene] if gene in gene_dict else gene for gene in repl.index])))
        #
        # exit()

        print("### Step2 ###")
        print("Compare")
        df1_n_ieqtls = (df1 <= 0.05).sum().sum()
        df2_n_ieqtls = (df2 <= 0.05).sum().sum()
        df1_interaction_df = df1.loc[df1.min(axis=1) <= 0.05, :]
        df2_interaction_df = df2.loc[df2.min(axis=1) <= 0.05, :]
        print("\t{} = {} PICs: n-ieQTLs: {:,}    {:.2f}% of eQTLs has an interaction".format(self.name1, df1.shape[1], df1_n_ieqtls, (100 / df1.shape[0]) * df1_interaction_df.shape[0]))
        print("\t{} = {} PCs:  n-ieQTLs: {:,}    {:.2f}% of eQTLs has an interaction".format(self.name2, df2.shape[1], df2_n_ieqtls, (100 / df2.shape[0]) * df2_interaction_df.shape[0]))

        n_ieqtls_df1_to_df2 = len([x for x in df1_interaction_df.index if x in df2_interaction_df.index])
        n_ieqtls_df2_to_df1 = len([x for x in df2_interaction_df.index if x in df1_interaction_df.index])
        overlap_eqtls_with_interaction = set(df1_interaction_df.index).intersection(set(df2_interaction_df.index))
        print("\t{:,}/{:,} of the {} ieQTLs also interacts with one or more {}".format(n_ieqtls_df1_to_df2, df1_interaction_df.shape[0], self.name1, self.name2))
        print("\t{:,}/{:,} of the {} ieQTLs also interacts with one or more {}".format(n_ieqtls_df2_to_df1, df2_interaction_df.shape[0], self.name2, self.name1))
        print("\tOverlap in eQTLs with an interaction: {:,}".format(len(overlap_eqtls_with_interaction)))

        print("Plot")
        df1_df2_replication_df = self.create_replication_df(df1=df1, df2=df2)
        self.plot_heatmap(df=df1_df2_replication_df,
                          xlabel=self.name2,
                          ylabel=self.name1,
                          filename="{}_replicating_in_{}".format(self.name1, self.name2))

        df2_df1_replication_df = self.create_replication_df(df1=df2, df2=df1)
        self.plot_heatmap(df=df2_df1_replication_df,
                          xlabel=self.name1,
                          ylabel=self.name2,
                          filename="{}_replicating_in_{}".format(self.name2, self.name1))

    def load_interaction_results(self, indir):
        fdr_data = []
        ieqtl_result_paths = glob.glob(os.path.join(indir, "*.txt.gz"))
        ieqtl_result_paths.sort(key=self.natural_keys)
        for ieqtl_result_path in ieqtl_result_paths:
            covariate = os.path.basename(ieqtl_result_path).split(".")[0]
            if covariate in ["call_rate", "genotype_stats"]:
                continue

            fpath = os.path.join(indir, "{}.txt.gz".format(covariate))
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)

                df.index = df["gene"]
                if "SNP" in df.columns:
                    df.index = df["gene"] + df["SNP"]
                elif "snp" in df.columns:
                    df.index = df["gene"] + df["snp"]
                else:
                    pass

                fdr_df = None
                if "ieQTL FDR" in df.columns:
                    fdr_df = df[["ieQTL FDR"]]
                elif "FDR" in df.columns:
                    fdr_df = df[["FDR"]]
                else:
                    pass
                fdr_df.columns = [covariate]

                fdr_data.append(fdr_df)
        fdr_df = pd.concat(fdr_data, axis=1)
        return fdr_df

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    @staticmethod
    def create_replication_df(df1, df2):
        row_overlap = set(df1.index).intersection(set(df2.index))
        df1_subset = df1.loc[row_overlap, :]
        df2_subset = df2.loc[row_overlap, :]

        replication_df = pd.DataFrame(np.nan,
                                      index=df1_subset.columns,
                                      columns=df2_subset.columns)
        for cov1 in df1_subset.columns:
            cov1_signif_df = df2_subset.loc[df1_subset.loc[:, cov1] <= 0.05, :]
            for cov2 in df2_subset.columns:
                replication_df.loc[cov1, cov2] = (cov1_signif_df.loc[:, cov2] <= 0.05).sum() / cov1_signif_df.shape[0]

        df1_n_signif = (df1_subset <= 0.05).sum(axis=0)
        df2_n_signif = (df2_subset <= 0.05).sum(axis=0)

        replication_df.index = ["{} [n={}]".format(ct, df1_n_signif.loc[ct]) for ct in df1_subset.columns]
        replication_df.columns = ["{} [n={}]".format(ct, df2_n_signif.loc[ct]) for ct in df2_subset.columns]

        return replication_df

    def plot_heatmap(self, df, xlabel="", ylabel="", filename=""):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(1 * df.shape[1] + 5, 1 * df.shape[0] + 5),
                                 gridspec_kw={"width_ratios": [0.2, 0.8],
                                              "height_ratios": [0.8, 0.2]})
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for _ in range(4):
            ax = axes[row_index, col_index]
            if row_index == 0 and col_index == 1:

                sns.heatmap(df, cmap=cmap, vmin=-1, vmax=1, center=0,
                            square=True, annot=df.round(2), fmt='',
                            cbar=False, annot_kws={"size": 16, "color": "#000000"},
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
        fig.savefig(os.path.join(self.outdir, "{}_{}_heatmap.png".format(self.out_filename, filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data 1: ")
        print("    > Input directory: {}".format(self.indir1))
        print("    > Name: {}".format(self.name1))
        print("  > Data 2: ")
        print("    > Input directory: {}".format(self.indir2))
        print("    > Name: {}".format(self.name2))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
