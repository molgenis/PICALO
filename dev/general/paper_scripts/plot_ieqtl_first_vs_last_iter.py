#!/usr/bin/env python3

"""
File:         plot_ieqtl_first_vs_last_iter.py
Created:      2021/05/24
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
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS

# Local application imports.

# Metadata
__program__ = "Visualisae PICALO optimization"
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
./plot_ieqtl_first_vs_last_iter.py -h

### BIOS ###

./plot_ieqtl_first_vs_last_iter.py \
    -pp /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first40ExpressionPCs.txt.gz \
    -tci /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_df.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -i rs2155218+ENSG00000236304+PIC2 \
    -n 250 \
    -e png pdf \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

### MetaBrain ###

./plot_ieqtl_first_vs_last_iter.py \
    -pp /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/ \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first80ExpressionPCs.txt.gz \
    -tci /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_df.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -i 12:31073901:rs7953706:T_A+ENSG00000013573.17+PIC1 \
    -n 10 \
    -e png pdf \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_path = getattr(arguments, 'picalo_path')
        self.geno_path = getattr(arguments, 'genotype')
        self.alleles_path = getattr(arguments, 'alleles')
        self.genotype_na = getattr(arguments, 'genotype_na')
        self.expr_path = getattr(arguments, 'expression')
        self.tcov_path = getattr(arguments, 'tech_covariate')
        self.tcov_inter_path = getattr(arguments, 'tech_covariate_with_inter')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.min_dataset_size = getattr(arguments, 'min_dataset_size')
        self.call_rate = getattr(arguments, 'call_rate')
        self.interest = getattr(arguments, 'interest')
        self.nrows = getattr(arguments, 'nrows')
        self.extensions = getattr(arguments, 'extensions')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            2.: "#E69F00",
            1.: "#0072B2",
            0.: "#D55E00"
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
        parser.add_argument("-pp",
                            "--picalo_path",
                            type=str,
                            required=True,
                            help="The path to the PICALO output directory.")
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix.")
        parser.add_argument("-al",
                            "--alleles",
                            type=str,
                            required=True,
                            help="The path to the alleles matrix")
        parser.add_argument("-na",
                            "--genotype_na",
                            type=int,
                            required=False,
                            default=-1,
                            help="The genotype value that equals a missing "
                                 "value. Default: -1.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix.")
        parser.add_argument("-tc",
                            "--tech_covariate",
                            type=str,
                            default=None,
                            help="The path to the technical covariate matrix "
                                 "(excluding an interaction with genotype). "
                                 "Default: None.")
        parser.add_argument("-tci",
                            "--tech_covariate_with_inter",
                            type=str,
                            default=None,
                            help="The path to the technical covariate matrix"
                                 "(including an interaction with genotype). "
                                 "Default: None.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix."
                                 "Default: None.")
        parser.add_argument("-mds",
                            "--min_dataset_size",
                            type=int,
                            required=False,
                            default=30,
                            help="The minimal number of samples per dataset. "
                                 "Default: >=30.")
        parser.add_argument("-cr",
                            "--call_rate",
                            type=float,
                            required=False,
                            default=0.95,
                            help="The minimal call rate of a SNP (per dataset)."
                                 "Equals to (1 - missingness). "
                                 "Default: >= 0.95.")
        parser.add_argument("-i",
                            "--interest",
                            nargs="+",
                            type=str,
                            required=True,
                            help="The IDs to plot.")
        parser.add_argument("-n",
                            "--nrows",
                            type=int,
                            required=False,
                            default=None,
                            help="Cap the number of runs to load. "
                                 "Default: None.")
        parser.add_argument("-e",
                            "--extensions",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        geno_df = self.load_file(self.geno_path, header=0, index_col=0, nrows=self.nrows)
        alleles_df = self.load_file(self.alleles_path, header=0, index_col=0, nrows=self.nrows)

        if geno_df.index.tolist() != alleles_df.index.tolist():
            print("error in genotype allele files")
            exit()

        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=self.nrows)
        tcov_df = self.load_file(self.tcov_path, header=0, index_col=0)
        tcov_inter_df = self.load_file(self.tcov_inter_path, header=0, index_col=0)
        std_df = self.load_file(self.std_path, header=0, index_col=None)
        std_df.index = std_df.iloc[:, 0]

        dataset_df = self.construct_dataset_df(std_df=std_df)
        datasets = dataset_df.columns.tolist()

        ########################################################################

        samples = std_df.iloc[:, 0].values.tolist()
        snps = geno_df.index.tolist()
        genes = expr_df.index.tolist()

        geno_m = geno_df.to_numpy(np.float64)
        expr_m = expr_df.to_numpy(np.float64)
        dataset_m = dataset_df.to_numpy(np.uint8)
        del geno_df, expr_df, dataset_df

        # Fill the missing values with NaN.
        expr_m[geno_m == self.genotype_na] = np.nan
        geno_m[geno_m == self.genotype_na] = np.nan

        ########################################################################

        tcov_m, tcov_labels = self.load_tech_cov(df=tcov_df, name="tech. cov. without interaction", std_df=std_df)
        tcov_inter_m, tcov_inter_labels = self.load_tech_cov(df=tcov_inter_df, name="tech. cov. with interaction", std_df=std_df)

        corr_m, corr_inter_m, correction_m_labels = \
            self.construct_correct_matrices(dataset_m=dataset_m,
                                            dataset_labels=datasets,
                                            tcov_m=tcov_m,
                                            tcov_labels=tcov_labels,
                                            tcov_inter_m=tcov_inter_m,
                                            tcov_inter_labels=tcov_inter_labels)

        ########################################################################

        plot_ids = {}
        max_pic = 0
        for interest in self.interest:
            snp, gene, pic = interest.split("+")
            pic_index = int(pic.replace("PIC", ""))
            if pic_index > max_pic:
                max_pic = pic_index
            eqlt_id = "{}+{}".format(snp, gene)
            if pic in plot_ids:
                plot_ids[pic].append(eqlt_id)
            else:
                plot_ids[pic] = [eqlt_id]
        print(plot_ids)

        ########################################################################

        eqtls_loaded = ["{}+{}".format(snp, gene) for snp, gene in zip(snps, genes)]
        pic_corr_m = np.copy(corr_m)
        pic_corr_inter_m = np.copy(corr_inter_m)
        for pic_index in range(1, max_pic + 1):
            pic = "PIC{}".format(pic_index)
            if pic_index > 1:
                # Loading previous run PIC.
                with open(os.path.join(self.picalo_path, pic, "component.npy"), 'rb') as f:
                    pic_a = np.load(f)
                f.close()

                if pic_corr_m is not None:
                    pic_corr_m = np.hstack((pic_corr_m, pic_a[:, np.newaxis]))
                else:
                    pic_corr_m = pic_a[:, np.newaxis]

                if pic_corr_inter_m is not None:
                    pic_corr_inter_m = np.hstack((pic_corr_inter_m, pic_a[:, np.newaxis]))
                else:
                    pic_corr_inter_m = pic_a[:, np.newaxis]

            if pic not in plot_ids:
                print("Skipping PIC{}".format(pic_index))
                continue
            pic_plot_ids = plot_ids[pic]

            for row_index, eqtl_id in enumerate(eqtls_loaded):
                if eqtl_id not in pic_plot_ids:
                    continue
                snp, gene = eqtl_id.split("+")
                print("Plotting {} - {} - {}".format(snp, gene, pic))

                # Combine data.
                df = pd.DataFrame({"genotype": geno_m[row_index, :],
                                  "expression": expr_m[row_index, :]},
                                  index=samples).merge(std_df, left_index=True, right_index=True)
                df["group"] = df["genotype"].round(0)

                # Check the call rate.
                for dataset in df["dataset"].unique():
                    sample_mask = (df["dataset"] == dataset).to_numpy(dtype=bool)
                    n_not_na = (df.loc[sample_mask, "genotype"] != self.genotype_na).astype(int).sum()
                    call_rate = n_not_na / np.sum(sample_mask)
                    if (call_rate < self.call_rate) or (n_not_na < self.min_dataset_size):
                        df.loc[sample_mask, "genotype"] = np.nan

                # Add iterations.
                iter_df = self.load_file(os.path.join(self.picalo_path, pic, "iteration.txt.gz"), header=0, index_col=0)
                iter_df = iter_df.iloc[[0, -1], :].T
                iter_df.columns = ["before", "after"]
                df = df.merge(iter_df, left_index=True, right_index=True)

                # Remove NaN.
                mask = (~df["genotype"].isna()).to_numpy(bool)
                df = df.loc[mask, :]
                print(df)

                # Correct the expression.
                X = np.ones((df.shape[0], 1))
                if pic_corr_m is not None:
                    X_m = np.copy(pic_corr_m[mask, :])
                    X = np.hstack((X, X_m))
                # if pic_corr_inter_m is not None:
                #     X_inter_m = np.copy(pic_corr_inter_m[mask, :])
                #     X = np.hstack((X, X_inter_m * geno_m[row_index, :][mask, np.newaxis]))
                df["expression"] = OLS(df["expression"], X).fit().resid

                # Check the call rate.
                for dataset in df["dataset"].unique():
                    sample_mask = (df["dataset"] == dataset).to_numpy(dtype=bool)
                    df.loc[sample_mask, "expression"] = ndtri((df.loc[sample_mask, "expression"].rank(ascending=True) - 0.5) / np.sum(sample_mask))
                    df.loc[sample_mask, "before"] = ndtri((df.loc[sample_mask, "before"].rank(ascending=True) - 0.5) / np.sum(sample_mask))
                    df.loc[sample_mask, "after"] = ndtri((df.loc[sample_mask, "after"].rank(ascending=True) - 0.5) / np.sum(sample_mask))

                # Get the allele data.
                alleles = alleles_df.iloc[row_index, :]
                zero_allele = alleles["Alleles"].split("/")[0]
                two_allele = alleles["Alleles"].split("/")[1]
                allele_map = {0.0: "{}/{}".format(zero_allele, zero_allele),
                              1.0: "{}/{}".format(zero_allele, two_allele),
                              2.0: "{}/{}".format(two_allele, two_allele)}

                # Fill the interaction plot annotation.
                snp_name = snp
                if not snp_name.startswith("rs"):
                    snp_name = snp.split(":")[2]
                annot = [
                    "{} - {}".format(snp_name, two_allele),
                    "N: {}".format(df.shape[0])]

                self.inter_plot(
                    df=df,
                    x1="before",
                    x2="after",
                    y="expression",
                    group_real="genotype",
                    group_round="group",
                    palette=self.palette,
                    allele_map=allele_map,
                    annot=annot,
                    ylabel="{} expression".format(gene),
                    filename="{}_{}_{}_{}_ieqtl_before_and_after".format(self.outfile, gene, snp, pic)
                )

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
    def construct_dataset_df(std_df):
        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]

        dataset_df = pd.DataFrame(0, index=std_df.iloc[:, 0], columns=datasets)
        for dataset in datasets:
            dataset_df.loc[(std_df.iloc[:, 1] == dataset).values, dataset] = 1
        dataset_df.index.name = "-"

        return dataset_df

    @staticmethod
    def load_tech_cov(df, name, std_df):
        if df is None:
            return None, []

        n_samples = std_df.shape[0]

        print("\tWorking on technical covariates matrix matrix '{}'".format(name))

        # Check for nan values.
        if df.isna().values.sum() > 0:
            print("\t  Matrix contains nan values")
            exit()

        # Put the samples on the rows.
        if df.shape[1] == n_samples:
            print("\t  Transposing matrix")
            df = df.T

        # Check for variables with zero std.
        variance_mask = df.std(axis=0) != 0
        n_zero_variance = variance_mask.shape[0] - variance_mask.sum()
        if n_zero_variance > 0:
            print("\t  Dropping {} rows with 0 variance".format(n_zero_variance))
            df = df.loc[:, variance_mask]

        # Convert to numpy.
        m = df.to_numpy(np.float64)
        columns = df.columns.tolist()
        del df

        covariates = columns
        print("\t  Technical covariates [{}]: {}".format(len(covariates), ", ".join(covariates)))

        return m, covariates

    @staticmethod
    def construct_correct_matrices(dataset_m, dataset_labels, tcov_m, tcov_labels,
                                   tcov_inter_m, tcov_inter_labels):
        # Create the correction matrices.
        corr_m = None
        corr_m_columns = ["Intercept"]
        corr_inter_m = None
        corr_inter_m_columns = []
        if dataset_m.shape[1] > 1:
            # Note that for the interaction term we need to include all
            # datasets.
            corr_m = np.copy(dataset_m[:, 1:])
            corr_m_columns.extend(dataset_labels[1:])

            corr_inter_m = np.copy(dataset_m)
            corr_inter_m_columns.extend(["{} x Genotype".format(label) for label in dataset_labels])

        if tcov_m is not None:
            corr_m_columns.extend(tcov_labels)
            if corr_m is not None:
                corr_m = np.hstack((corr_m, tcov_m))
            else:
                corr_m = tcov_m

        if tcov_inter_m is not None:
            corr_m_columns.extend(tcov_inter_labels)
            if corr_m is not None:
                corr_m = np.hstack((corr_m, tcov_inter_m))
            else:
                corr_m = tcov_inter_m

            corr_inter_m_columns.extend(["{} x Genotype".format(label) for label in tcov_inter_labels])
            if corr_inter_m is not None:
                corr_inter_m = np.hstack((corr_inter_m, tcov_inter_m))
            else:
                corr_inter_m = tcov_inter_m

        return corr_m, corr_inter_m, corr_m_columns + corr_inter_m_columns

    def inter_plot(self, df, x1="x1", x2="x2", y="y", group_real="group_real",
                   group_round="group_round", palette=None, allele_map=None,
                   annot=None, ylabel="", title="", filename="ieqtl_plot"):
        if len(set(df[group_round].unique()).symmetric_difference({0, 1, 2})) > 0:
            return

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(18, 10),
                                 sharex="none",
                                 sharey="none",
                                 gridspec_kw={"height_ratios": [0.1, 0.9]})

        self.single_inter_plot(fig=fig,
                               plot_ax=axes[1, 0],
                               annot_ax=axes[0, 0],
                               df=df,
                               x=x1,
                               y=y,
                               group_real=group_real,
                               group_round=group_round,
                               palette=palette,
                               allele_map=allele_map,
                               annot=annot,
                               ylabel=ylabel,
                               )
        self.single_inter_plot(fig=fig,
                               plot_ax=axes[1, 1],
                               annot_ax=axes[0, 1],
                               df=df,
                               x=x2,
                               y=y,
                               group_real=group_real,
                               group_round=group_round,
                               palette=palette,
                               allele_map=allele_map,
                               annot=annot,
                               ylabel=ylabel,
                               )

        plt.suptitle(title,
                     fontsize=30,
                     fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            print("\t\tSaving plot: {}".format(os.path.basename(outpath)))
            fig.savefig(outpath)
        plt.close()

    @staticmethod
    def single_inter_plot(fig, plot_ax, annot_ax, df, x="x", y="y",
                          group_real="group_real", group_round="group_round",
                          palette=None, allele_map=None, annot=None, xlabel="",
                          ylabel=""):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.despine(fig=fig, ax=plot_ax)
        annot_ax.set_axis_off()

        plot_annot = annot.copy()
        tmp = df[[x, y, group_real]].copy()
        tmp["intercept"] = 1
        tmp["inter"] = tmp[x] * tmp[group_real]
        ols_fit = OLS(tmp[[y]], tmp[["intercept", group_real, x, "inter"]]).fit()
        rsquared = ols_fit.rsquared
        inter_pvalue = ols_fit.pvalues[3]
        new_annot = ["R\u00b2: {:.2f}".format(rsquared),
                     "interaction p-value: {:.2e}".format(inter_pvalue)]
        plot_annot.extend(new_annot)
        del tmp, ols_fit, new_annot

        for i, group_id in enumerate([0, 1, 2]):
            subset = df.loc[df[group_round] == group_id, :].copy()
            allele = group_id
            if allele_map is not None:
                allele = allele_map[group_id]

            coef_str = "NA"
            r_annot_pos = (-1, -1)
            if len(subset.index) > 1:
                coef, p = stats.pearsonr(subset[y], subset[x])
                coef_str = "{:.2f}".format(coef)

                subset["intercept"] = 1
                betas = np.linalg.inv(subset[["intercept", x]].T.dot(subset[["intercept", x]])).dot(subset[["intercept", x]].T).dot(subset[y])
                subset["y_hat"] = np.dot(subset[["intercept", x]], betas)
                subset.sort_values(x, inplace=True)

                r_annot_pos = (subset.iloc[-1, :][x] + (subset[x].max() * 0.05),
                               subset.iloc[-1, :]["y_hat"])

                sns.regplot(x=x, y=y, data=subset, ci=None,
                            scatter_kws={'facecolors': palette[group_id],
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": palette[group_id],
                                      "alpha": 0.75},
                            ax=plot_ax
                            )

            plot_ax.annotate(
                '{}\n{}'.format(allele, coef_str),
                xy=r_annot_pos,
                color=palette[group_id],
                alpha=0.75,
                fontsize=16,
                fontweight='bold')

        if plot_annot is not None:
            for i, annot_label in enumerate(plot_annot):
                annot_ax.annotate(annot_label,
                                  xy=(0.03, 0.9 - (i * 0.3)),
                                  xycoords=annot_ax.transAxes,
                                  color="#000000",
                                  alpha=0.75,
                                  fontsize=12,
                                  fontweight='bold')

        (xmin, xmax) = (df[x].min(), df[x].max())
        (ymin, ymax) = (df[y].min(), df[y].max())
        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05

        plot_ax.set_xlim(xmin - xmargin, xmax + xmargin)
        plot_ax.set_ylim(ymin - ymargin, ymax + ymargin)

        plot_ax.set_title("",
                          fontsize=16,
                          fontweight='bold')
        plot_ax.set_ylabel(ylabel,
                           fontsize=14,
                           fontweight='bold')
        plot_ax.set_xlabel(xlabel,
                           fontsize=14,
                           fontweight='bold')

    def print_arguments(self):
        print("Arguments:")
        print("  > PICALO base path: {}".format(self.picalo_path))
        print("  > Genotype input path: {}".format(self.geno_path))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Expression input path: {}".format(self.expr_path))
        print("  > Technical covariates input path: {}".format(self.tcov_path))
        print("  > Technical covariates with interaction input path: {}".format(self.tcov_inter_path))
        print("  > Sample-dataset path: {}".format(self.std_path))
        print("  > Min. dataset size: {}".format(self.min_dataset_size))
        print("  > Genotype NA value: {}".format(self.genotype_na))
        print("  > SNP call rate: >{}".format(self.call_rate))
        print("  > Interest: {}".format(self.interest))
        print("  > Nrows: {}".format(self.nrows))
        print("  > Extension: {}".format(self.extensions))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
