#!/usr/bin/env python3

"""
File:         compare_PIC_interactions_to_PC_interactions.py
Created:      2021/12/21
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.


# Metadata
__program__ = "Compare PIC Interactions vs PC Interactions"
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
./compare_PIC_interactions_to_PC_interactions.py -h

./compare_PIC_interactions_to_PC_interactions.py -pic /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICsAsCov -pc /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-First33ExprPCsAsCov -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.pic_indir = getattr(arguments, 'pic_indir')
        self.pc_indir = getattr(arguments, 'pc_indir')
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
        parser.add_argument("-pic",
                            "--pic_indir",
                            type=str,
                            required=True,
                            help="The path to the PIC interaction directory.")
        parser.add_argument("-pc",
                            "--pc_indir",
                            type=str,
                            required=True,
                            help="The path to the PC interaction directory.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading data")
        pic_df = self.load_interaction_results(indir=self.pic_indir, cov_name="PIC")
        pc_df = self.load_interaction_results(indir=self.pc_indir, cov_name="Comp")
        print(pic_df)
        print(pc_df)

        print("### Step2 ###")
        print("Compare")
        pic_n_ieqtls = (pic_df < 0.05).sum().sum()
        pc_n_ieqtls = (pc_df < 0.05).sum().sum()
        pic_eqtl_interaction_df = pic_df.loc[pic_df.min(axis=1) < 0.05, :]
        pc_eqtl_interaction_df = pc_df.loc[pc_df.min(axis=1) < 0.05, :]
        print("\t{} PICs: n-ieQTLs: {:,}    {:.2f}% of eQTLs has an interaction".format(pic_df.shape[1], pic_n_ieqtls, (100 / pic_df.shape[0]) * pic_eqtl_interaction_df.shape[0]))
        print("\t{} PCs:  n-ieQTLs: {:,}    {:.2f}% of eQTLs has an interaction".format(pc_df.shape[1], pc_n_ieqtls, (100 / pc_df.shape[0]) * pc_eqtl_interaction_df.shape[0]))

        n_ieqtls_pics_to_pc = len([x for x in pic_eqtl_interaction_df.index if x in pc_eqtl_interaction_df.index])
        n_ieqtls_pcs_to_pics = len([x for x in pc_eqtl_interaction_df.index if x in pic_eqtl_interaction_df.index])
        overlap_eqtls_with_interaction = set(pic_eqtl_interaction_df.index).intersection(set(pc_eqtl_interaction_df.index))
        print("\t{:,}/{:,} of the PIC ieQTLs also interacts with one or more PCs".format(n_ieqtls_pics_to_pc, pic_eqtl_interaction_df.shape[0]))
        print("\t{:,}/{:,} of the PC ieQTLs also interacts with one or more PICs".format(n_ieqtls_pcs_to_pics, pc_eqtl_interaction_df.shape[0]))
        print("\tOverlap in eQTLs with an interaction: {:,}".format(len(overlap_eqtls_with_interaction)))

        print("Plot")
        df1_df2_replication_df = self.create_replication_df(df1=pic_df, df2=pc_df)
        self.plot_heatmap(df=df1_df2_replication_df,
                          xlabel="PC",
                          ylabel="PIC",
                          filename="PICs_replicating_in_PCs")

        df2_df1_replication_df = self.create_replication_df(df1=pc_df, df2=pic_df)
        self.plot_heatmap(df=df2_df1_replication_df,
                          xlabel="PIC",
                          ylabel="PC",
                          filename="PIC_replicating_in_PICs")

    @staticmethod
    def load_interaction_results(indir, cov_name):
        fdr_data = []
        for i in range(101):
            covariate = "{}{}".format(cov_name, i)

            fpath = os.path.join(indir, "{}.txt.gz".format(covariate))
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)
                fdr_df = df[["ieQTL FDR"]]
                fdr_df.columns = [covariate]
                fdr_data.append(fdr_df)
        fdr_df = pd.concat(fdr_data, axis=1)
        return fdr_df

    @staticmethod
    def create_replication_df(df1, df2):
        row_overlap = set(df1.index).intersection(set(df2.index))
        df1_subset = df1.loc[row_overlap, :]
        df2_subset = df2.loc[row_overlap, :]

        replication_df = pd.DataFrame(np.nan,
                                      index=df1_subset.columns,
                                      columns=df2_subset.columns)
        for cov1 in df1_subset.columns:
            cov1_signif_df = df2_subset.loc[df1_subset.loc[:, cov1] < 0.05, :]
            for cov2 in df2_subset.columns:
                replication_df.loc[cov1, cov2] = (cov1_signif_df.loc[:, cov2] < 0.05).sum() / cov1_signif_df.shape[0]

        df1_n_signif = (df1_subset < 0.05).sum(axis=0)
        df2_n_signif = (df2_subset < 0.05).sum(axis=0)

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
        print("  > PIC interaction input directory: {}".format(self.pic_indir))
        print("  > PC interaction input directory: {}".format(self.pc_indir))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
