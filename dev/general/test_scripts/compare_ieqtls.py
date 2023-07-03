#!/usr/bin/env python3

"""
File:         compare_ieqtls.py
Created:      2022/03/23
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
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
__program__ = "Compare ieQTLs"
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
./compare_ieqtls.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir1 = getattr(arguments, 'indir1')
        self.indir2 = getattr(arguments, 'indir2')
        self.conditional1 = getattr(arguments, 'conditional1')
        self.conditional2 = getattr(arguments, 'conditional2')
        self.n_files1 = getattr(arguments, 'n_files1')
        self.n_files2 = getattr(arguments, 'n_files2')
        self.skip1 = getattr(arguments, 'skip1')
        self.skip2 = getattr(arguments, 'skip2')
        self.name1 = getattr(arguments, 'name1')
        self.name2 = getattr(arguments, 'name2')

        if self.skip1 is None:
            self.skip1 = []
        if self.skip2 is None:
            self.skip2 = []

        current_dir = str(os.path.dirname(os.path.abspath(__file__)))

        # Prepare an output directory.
        self.outdir = os.path.join(current_dir, "compare_ieqtls")
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
        parser.add_argument("-i1",
                            "--indir1",
                            type=str,
                            required=True,
                            help="The path to the first input directory.")
        parser.add_argument("-i2",
                            "--indir2",
                            type=str,
                            required=True,
                            help="The path to the second input directory.")
        parser.add_argument("-conditional1",
                            action='store_true',
                            help="Load conditional results of analysis 1. "
                                 "Default: False.")
        parser.add_argument("-conditional2",
                            action='store_true',
                            help="Load conditional results of analysis 2. "
                                 "Default: False.")
        parser.add_argument("-nf1",
                            "--n_files1",
                            type=int,
                            default=None,
                            help="The number of files to load of analysis 1. "
                                 "Default: all.")
        parser.add_argument("-nf2",
                            "--n_files2",
                            type=int,
                            default=None,
                            help="The number of files to load of analysis 2. "
                                 "Default: all.")
        parser.add_argument("-s1",
                            "--skip1",
                            nargs="+",
                            type=str,
                            default=None,
                            help="Files to skip for analysis 1. "
                                 "Default: None.")
        parser.add_argument("-s2",
                            "--skip2",
                            nargs="+",
                            type=str,
                            default=None,
                            help="Files to skip for analysis 2. "
                                 "Default: None.")
        parser.add_argument("-n1",
                            "--name1",
                            type=str,
                            default="name1",
                            help="The name of analysis 1. "
                                 "Default: name1.")
        parser.add_argument("-n2",
                            "--name2",
                            type=str,
                            default="name2",
                            help="The name of analysis 2. "
                                 "Default: name2.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading results of {}".format(self.name1))
        fdr_df1, tvalue_df1 = self.load_results(indir=self.indir1,
                                                conditional=self.conditional1,
                                                n_files=self.n_files1,
                                                skip=self.skip1)
        fdr_count_df1 = (fdr_df1 < 0.05).astype(int)
        print("")

        print("Printing results")
        self.print_results(fdr_count_df=fdr_count_df1,
                           tvalue_df=tvalue_df1)
        print("")

        print("Plotting results")
        plot_df = tvalue_df1.copy()
        plot_df = plot_df.abs()
        plot_df[fdr_count_df1 == 0] = np.nan
        plot_dfm = plot_df.melt()
        plot_dfm.dropna(inplace=True)
        self.boxplot(df=plot_dfm,
                     xlabel="component",
                     ylabel="abs t-value",
                     title=self.name1,
                     name="{}_tvalues_per_covariate".format(self.name1))
        del plot_dfm
        print("")

        print("### Step2 ###")
        print("Loading results of {}".format(self.name2))
        fdr_df2, tvalue_df2 = self.load_results(indir=self.indir2,
                                                conditional=self.conditional2,
                                                n_files=self.n_files2,
                                                skip=self.skip2)
        fdr_count_df2 = (fdr_df2 < 0.05).astype(int)
        print(", ".join([ieqtl.split("_")[1] for ieqtl in fdr_count_df2.index]))
        print("")

        print("Printing results")
        self.print_results(fdr_count_df=fdr_count_df2,
                           tvalue_df=tvalue_df2)
        print("")

        print("Plotting results")
        plot_df = tvalue_df2.copy()
        plot_df = plot_df.abs()
        plot_df[fdr_count_df1 == 0] = np.nan
        plot_dfm = plot_df.melt()
        plot_dfm.dropna(inplace=True)
        self.boxplot(df=plot_dfm,
                     xlabel="component",
                     ylabel="abs t-value",
                     title=self.name2,
                     name="{}_tvalues_per_covariate".format(self.name2))
        del plot_dfm
        print("")

        print("### Step3 ###")
        print("Comparing results")
        ieqtls1 = set(fdr_count_df1.loc[fdr_count_df1.sum(axis=1) > 0, :].index)
        ieqtls2 = set(fdr_count_df2.loc[fdr_count_df2.sum(axis=1) > 0, :].index)
        overlap = ieqtls1.intersection(ieqtls2)
        print("  Overlap in ieQTLs: {:,}".format(len(overlap)))
        print("")
        #
        # corr_df = pd.read_csv("/groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-CenterScaledPCA/2022-03-24-BIOS-1MscBloodNL-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-UT-CD4T-Unique-BulkTechPICsCorrected-PIC1AndCD3G_gene_correlations-avgExpressionAdded.txt.gz", sep="\t", header=0, index_col=None)
        # corr_df.index = corr_df["ProbeName"]
        # print(corr_df)

        print("Unique eQTLs with interaction in {}".format(self.name1))
        unique1 = [ieqtl for ieqtl in ieqtls1 if ieqtl not in ieqtls2]
        print("  N = {:,}".format(len(unique1)))
        unique_interactions1 = fdr_count_df1.loc[unique1, :].copy()
        unique_interaction_tvalues1 = tvalue_df1.loc[unique1, :].copy()
        print("  Interactions:")
        for covariate in unique_interactions1.columns:
            ieqtls = list(unique_interactions1.loc[unique_interactions1[covariate] == 1, :].index)
            tvalues = unique_interaction_tvalues1.loc[ieqtls, covariate].copy()
            if len(ieqtls) > 0:
                print("    {}:\t{:,} ieQTLs\tt-value {:.2f} (±{:.2f})\t"
                  "N-positive: {:,}\tN-negative: {:,}".format(covariate,
                                                              len(ieqtls),
                                                              tvalues.abs().mean(),
                                                              tvalues.abs().std(),
                                                              (tvalues > 0).sum(),
                                                              (tvalues < 0).sum()
                                                              ))
            #
            # corr_subset_df = corr_df.loc[[ieqtl.split("_")[1] for ieqtl in ieqtls], [covariate]].copy()
            # print("    Positive: {}".format(", ".join(list(corr_subset_df.loc[corr_subset_df[covariate] > 0, :].index))))
            # print("    Negative: {}".format(", ".join(list(corr_subset_df.loc[corr_subset_df[covariate] < 0, :].index))))
            del ieqtls, tvalues
        print("")

        print("Unique eQTLs with interaction in {}".format(self.name2))
        unique2 = [ieqtl for ieqtl in ieqtls2 if ieqtl not in ieqtls1]
        print("  N = {:,}".format(len(unique2)))
        unique_interactions2 = fdr_count_df2.loc[unique2, :].copy()
        unique_interaction_tvalues2 = tvalue_df2.loc[unique2, :].copy()
        print("  Interactions:")
        for covariate in unique_interactions2.columns:
            ieqtls = list(unique_interactions2.loc[unique_interactions2[covariate] == 1, :].index)
            tvalues = unique_interaction_tvalues2.loc[ieqtls, covariate]
            if len(ieqtls) > 0:
                print("    {}:\t{:,} ieQTLs\tt-value {:.2f} (±{:.2f})\t"
                      "N-positive: {:,}\tN-negative: {:,}".format(covariate,
                                                                  len(ieqtls),
                                                                  tvalues.abs().mean(),
                                                                  tvalues.abs().std(),
                                                                  (tvalues > 0).sum(),
                                                                  (tvalues < 0).sum()
                                                                  ))
            #
            # corr_subset_df = corr_df.loc[[ieqtl.split("_")[1] for ieqtl in ieqtls], [covariate]].copy()
            # print("    Positive: {}".format(", ".join(list(corr_subset_df.loc[corr_subset_df[covariate] > 0, :].index))))
            # print("    Negative: {}".format(", ".join(list(corr_subset_df.loc[corr_subset_df[covariate] < 0, :].index))))
            del ieqtls, tvalues
        print("")

    def load_results(self, indir, conditional=False, n_files=None, skip=None):
        # Load the input paths.
        inpaths = glob.glob(os.path.join(indir, "*.txt.gz"))
        if conditional:
            inpaths = [inpath for inpath in inpaths if "conditional" in inpath]
        else:
            inpaths = [inpath for inpath in inpaths if "conditional" not in inpath]
        inpaths.sort(key=self.natural_keys)

        # Load the results.
        count = 0
        fdr_df_list = []
        tvalue_df_list = []
        for inpath in inpaths:
            if n_files is not None and count == n_files:
                continue

            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue
            if skip is not None and filename in skip:
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            df.index = df["SNP"] + "_" + df["gene"]

            fdr_df = df.loc[:, ["FDR"]].copy()
            fdr_df.columns = [filename]
            fdr_df_list.append(fdr_df)

            df["tvalue-interaction"] = df["beta-interaction"] / df["std-interaction"]
            tvalue_df = df.loc[:, ["tvalue-interaction"]].copy()
            tvalue_df.columns = [filename]
            tvalue_df_list.append(tvalue_df)

            del fdr_df, tvalue_df
            count += 1

        fdr_df = pd.concat(fdr_df_list, axis=1)
        tvalue_df = pd.concat(tvalue_df_list, axis=1)

        return fdr_df, tvalue_df

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

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
    def print_results(fdr_count_df, tvalue_df):
        ieqtl_tvalue_df = tvalue_df.copy()
        ieqtl_tvalue_df[fdr_count_df == 0] = np.nan

        print("  Results per covariate:")
        for covariate in fdr_count_df.columns:
            print("    {}:\t{:,} ieQTLs\tt-value {:.2f} (±{:.2f})\t"
                  "N-positive: {:,}\tN-negative: {:,}".format(covariate,
                                                              fdr_count_df[covariate].sum(),
                                                              ieqtl_tvalue_df[covariate].abs().mean(),
                                                              ieqtl_tvalue_df[covariate].abs().std(),
                                                              (ieqtl_tvalue_df[covariate] > 0).sum(),
                                                              (ieqtl_tvalue_df[covariate] < 0).sum()
                                                              ))
        print("")

        fdr_count_colsums_df = fdr_count_df.sum(axis=0)
        print("  #ieQTL stats:")
        print("    Sum: \t{:,.0f}".format(fdr_count_colsums_df.sum()))
        print("    Mean:\t{:,.1f}".format(fdr_count_colsums_df.mean()))
        print("    SD:  \t{:,.2f}".format(fdr_count_colsums_df.std()))
        print("    Max: \t{:,.0f}".format(fdr_count_colsums_df.max()))
        print("")

        print("  t-value stats:")
        print("    Sum: \t{:,.2f}".format(ieqtl_tvalue_df.abs().sum(axis=0).sum()))
        print("    Mean:\t{:,.1f}".format(ieqtl_tvalue_df.abs().mean(axis=0).mean()))
        print("    SD:  \t{:,.2f}".format(ieqtl_tvalue_df.abs().std(axis=0).std()))
        print("    Min: \t{:,.2f}".format(ieqtl_tvalue_df.abs().min(axis=0).min()))
        print("    Max: \t{:,.2f}".format(ieqtl_tvalue_df.abs().max(axis=0).max()))
        print("")

        fdr_count_rowsums_df = fdr_count_df.sum(axis=1)
        print("  Interaction stats per eQTL")
        counts = dict(zip(*np.unique(fdr_count_rowsums_df, return_counts=True)))
        eqtls_w_inter = fdr_count_df.loc[fdr_count_rowsums_df > 0, :].shape[0]
        total_eqtls = fdr_count_df.shape[0]
        for value, n in counts.items():
            if value != 0:
                print("    N-eQTLs with {} interaction:\t{:,} [{:.2f}%]".format(value, n, (100 / eqtls_w_inter) * n))
        print("    Unique: {:,} / {:,} [{:.2f}%]".format(eqtls_w_inter, total_eqtls, (100 / total_eqtls) * eqtls_w_inter))

    def boxplot(self, df, x="variable", y="value", title="", xlabel="",
                ylabel="", name="boxplot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(x=x,
                       y=y,
                       data=df,
                       cut=0,
                       color="#000000",
                       zorder=-1,
                       dodge=False,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    data=df,
                    color="white",
                    zorder=-1,
                    dodge=False,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(title,
                     fontsize=20,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.outdir, "{}.png".format(name)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  {}:".format(self.name1))
        print("    > Input directory: {}".format(self.indir1))
        print("    > Conditional: {}".format(self.conditional1))
        print("    > N-files: {}".format(self.n_files1))
        print("    > Skip: {}".format(", ".join(self.skip1)))
        print("  {}:".format(self.name2))
        print("    > Input directory: {}".format(self.indir2))
        print("    > Conditional: {}".format(self.conditional2))
        print("    > N-files: {}".format(self.n_files2))
        print("    > Skip: {}".format(", ".join(self.skip2)))
        print("  Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
