#!/usr/bin/env python3

"""
File:         visualise_interaction_eqtl.py
Created:      2022/04/07
Last Changed: 2022/07/20
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
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS

# Local application imports.

# Metadata
__program__ = "Visualise Interaction eQTL"
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
./visualise_interaction_eqtl.py -h

### BIOS ###

./visualise_interaction_eqtl.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -co /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000182534_rs62085145_PIC3 ENSG00000160305_rs9306156_PIC3 ENSG00000161055_rs2453176_PIC3 ENSG00000154027_rs4949655_PIC3 ENSG00000130590_rs816933_PIC3 \
    -n 1000 \
    -e png pdf
    
./visualise_interaction_eqtl.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -co /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -i ENSG00000182534_rs62085145_Comp5 ENSG00000160305_rs9306156_Comp5 ENSG00000161055_rs2453176_Comp5 ENSG00000154027_rs4949655_Comp5 ENSG00000130590_rs816933_Comp5 \
    -n 1000 \
    -e png pdf
    
### MetaBrain ###

./visualise_interaction_eqtl.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000140873.16_16:77359175:rs891134:C_G_PIC2 ENSG00000153714.6_9:12760648:rs7025842:G_A_PIC2 ENSG00000112599.9_6:42198814:rs6907587:C_T_PIC2 \
    -n 400 \
    -e png pdf
    
./visualise_interaction_eqtl.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/calc_replication_score/2022-04-14-MetaBrain-CortexAFR-ReplicationScore-CortexEUR-cis-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/2022-04-14-MetaBrain-CortexAFR-ReplicationScore-CortexEUR-cis-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs.txt.gz \
    -i ENSG00000140873.16_16:77359175:rs891134:C_G_PIC2 ENSG00000153714.6_9:12760648:rs7025842:G_A_PIC2 ENSG00000112599.9_6:42198814:rs6907587:C_T_PIC2 \
    -n 400 \
    -e png pdf
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.geno_path = getattr(arguments, 'genotype')
        self.alleles_path = getattr(arguments, 'alleles')
        self.expr_path = getattr(arguments, 'expression')
        self.cova_path = getattr(arguments, 'covariate')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.interest = getattr(arguments, 'interest')
        self.nrows = getattr(arguments, 'nrows')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'visualise_interaction_eqtl')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            2.: "#E69F00",
            1.: "#0072B2",
            0.: "#D55E00"
        }

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
                            help="show program's version number and exit")
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the eqtl matrix")
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")
        parser.add_argument("-al",
                            "--alleles",
                            type=str,
                            required=True,
                            help="The path to the alleles matrix")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-co",
                            "--covariate",
                            type=str,
                            required=True,
                            help="The path to the covariate matrix.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            help="The path to the sample-to-dataset matrix.")
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

        print("Loading data")
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None, nrows=self.nrows)
        geno_df = self.load_file(self.geno_path, header=0, index_col=0, nrows=self.nrows)
        alleles_df = self.load_file(self.alleles_path, header=0, index_col=0, nrows=self.nrows)
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=self.nrows)
        cova_df = self.load_file(self.cova_path, header=0, index_col=0)

        if self.std_path:
            std_df = self.load_file(self.std_path, header=0, index_col=None)

            print("Filter data")
            samples = list(std_df.iloc[:, 0])
            geno_df = geno_df.loc[:, samples]
            expr_df = expr_df.loc[:, samples]
            cova_df = cova_df.loc[samples, :]

        print("Validate data")
        probes = list(eqtl_df["ProbeName"])
        snps = list(eqtl_df["SNPName"])
        samples = list(expr_df.columns)
        if list(geno_df.index) != snps:
            print("Error, genotype does not match eQTL file.")
            exit()
        if list(geno_df.columns) != samples:
            print("Error, genotype does not match expression file.")
            exit()
        if list(alleles_df.index) != snps:
            print("Error, allele does not match eQTL file.")
            exit()
        if list(expr_df.index) != probes:
            print("Error, expression does not match eQTL file.")
            exit()
        if list(cova_df.columns) != samples:
            print("Error, covariates header does not match expression file.")
            exit()

        print("Iterating over eQTLs.")
        for row_index, (_, row) in enumerate(eqtl_df.iterrows()):
            # Extract the usefull information from the row.
            snp_name = row["SNPName"]
            probe_name = row["ProbeName"]
            hgnc_name = row["HGNCName"]
            eqtl_id = probe_name + "_" + snp_name

            eqtl_covs = []
            for interest in self.interest:
                if interest.startswith(eqtl_id):
                    eqtl_covs.append(interest.replace(eqtl_id + "_", ""))

            for cov in eqtl_covs:
                print("\tWorking on: {}\t{} [{}]\t{} [{}/{} "
                      "{:.2f}%]".format(snp_name, probe_name, hgnc_name,
                                        cov,
                                        row_index + 1,
                                        eqtl_df.shape[0],
                                        (100 / eqtl_df.shape[0]) * (row_index + 1)))

                # Get the genotype / expression data.
                genotype = geno_df.iloc[[row_index], :].copy().T
                if genotype.columns != [snp_name]:
                    print("\t\tGenotype file not in identical order as eQTL file.")
                    exit()
                expression = expr_df.iloc[[row_index], :].copy().T
                if expression.columns != [probe_name]:
                    print("\t\tExpression file not in identical order as eQTL file.")
                    exit()
                data = genotype.merge(expression, left_index=True, right_index=True)
                data.columns = ["genotype", "expression"]
                data.insert(0, "intercept", 1)
                data["group"] = data["genotype"].round(0)

                # Remove missing values.
                data = data.loc[data['genotype'] != -1, :]

                # Get the allele data.
                alleles = alleles_df.iloc[row_index, :]
                if alleles.name != snp_name:
                    print("\t\tAlleles file not in identical order as eQTL file.")
                    exit()
                # A/T = 0.0/2.0
                # by default we assume T = 2.0 to be minor
                major_allele = alleles["Alleles"].split("/")[0]
                minor_allele = alleles["Alleles"].split("/")[1]

                # Add the covariate of interest.
                covariate_df = cova_df.loc[[cov], :].T
                covariate_df.columns = ["covariate"]
                data = data.merge(covariate_df, left_index=True, right_index=True)
                data["interaction"] = data["genotype"] * data["covariate"]

                # Calculate the annotations.
                counts = data["group"].value_counts()
                for x in [0.0, 1.0, 2.0]:
                    if x not in counts:
                        counts.loc[x] = 0
                zero_geno_count = (counts[0.0] * 2) + counts[1.0]
                two_geno_count = (counts[2.0] * 2) + counts[1.0]
                minor_allele_frequency = min(zero_geno_count, two_geno_count) / (zero_geno_count + two_geno_count)

                eqtl_pvalue = OLS(data["expression"], data[["intercept", "genotype"]]).fit().pvalues[1]
                eqtl_pvalue_str = "{:.2e}".format(eqtl_pvalue)
                if eqtl_pvalue == 0:
                    eqtl_pvalue_str = "<{:.1e}".format(1e-308)
                eqtl_pearsonr, _ = stats.pearsonr(data["expression"], data["genotype"])

                interaction_pvalue = OLS(data["expression"], data[["intercept", "genotype", "covariate", "interaction"]]).fit().pvalues[3]
                interaction_pvalue_str = "{:.2e}".format(interaction_pvalue)
                if interaction_pvalue == 0:
                    interaction_pvalue_str = "<{:.1e}".format(1e-308)

                print_probe_name = probe_name
                if "." in print_probe_name:
                    print_probe_name = print_probe_name.split(".")[0]

                print_snp_name = snp_name
                if ":" in print_snp_name:
                    print_snp_name = print_snp_name.split(":")[2]

                # Fill the interaction plot annotation.
                annot1 = ["N: {:,}".format(data.shape[0]),
                          "r: {:.2f}".format(eqtl_pearsonr),
                          "p-value: {}".format(eqtl_pvalue_str),
                          "MAF: {:.2f}".format(minor_allele_frequency)]
                annot2 = ["{} - {}".format(print_snp_name, minor_allele),
                          "N: {:,}".format(data.shape[0]),
                          "interaction p-value: {}".format(
                              interaction_pvalue_str),
                          "eQTL p-value: {:.2e}".format(eqtl_pvalue),
                          "MAF: {:.2f}".format(minor_allele_frequency)]

                allele_map = {0.0: "{}/{}".format(major_allele, major_allele),
                              1.0: "{}/{}".format(major_allele, minor_allele),
                              2.0: "{}/{}".format(minor_allele, minor_allele)}

                # Plot the main eQTL effect.
                self.eqtl_plot(df=data,
                               x="group",
                               y="expression",
                               palette=self.palette,
                               allele_map=allele_map,
                               xlabel=print_snp_name,
                               ylabel="{} expression".format(hgnc_name),
                               annot=annot1,
                               title="eQTL",
                               filename="{}_{}_{}_{}".format(row_index, print_probe_name, hgnc_name, print_snp_name)
                               )

                # Plot the interaction eQTL.
                self.inter_plot(df=data,
                                x="covariate",
                                y="expression",
                                group="group",
                                palette=self.palette,
                                allele_map=allele_map,
                                xlabel="{}".format(cov),
                                ylabel="{} expression".format(hgnc_name),
                                annot=annot2,
                                title="ieQTL",
                                filename="{}_{}_{}_{}_{}".format(row_index, print_probe_name, hgnc_name, print_snp_name, cov)
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

    def eqtl_plot(self, df, x="x", y="y", palette=None, allele_map=None,
                  annot=None, xlabel="", ylabel="", title="",
                  filename="eqtl_plot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.regplot(x=x,
                    y=y,
                    data=df,
                    scatter=False,
                    ci=None,
                    line_kws={"color": "#000000",
                              "linewidth": 5,
                              "alpha": 1},
                    ax=ax)
        sns.violinplot(x=x,
                       y=y,
                       data=df,
                       palette=palette,
                       cut=0,
                       zorder=-1,
                       ax=ax)
        plt.setp(ax.collections, alpha=.75)
        sns.boxplot(x=x,
                    y=y,
                    data=df,
                    whis=np.inf,
                    color="white",
                    zorder=-1,
                    ax=ax)

        if annot is not None:
            for i, annot_label in enumerate(annot):
                ax.annotate(annot_label,
                            xy=(0.03, 0.94 - (i * 0.04)),
                            xycoords=ax.transAxes,
                            color="#000000",
                            alpha=0.75,
                            fontsize=20,
                            fontweight='bold')

        if allele_map is not None:
            ax.set_xticks(range(3))
            ax.set_xticklabels([allele_map[0.0], allele_map[1.0], allele_map[2.0]])

        ax.set_title(title,
                     fontsize=22,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=20,
                      fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            print("\t\tSaving plot: {}".format(os.path.basename(outpath)))
            fig.savefig(outpath)
        plt.close()

    def inter_plot(self, df, x="x", y="y", group="group", palette=None,
                   allele_map=None, annot=None, xlabel="", ylabel="",
                   title="", filename="ieqtl_plot"):
        if len(set(df[group].unique()).symmetric_difference({0, 1, 2})) > 0:
            return

        sns.set(rc={'figure.figsize': (12, 12)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        for i, group_id in enumerate([0, 1, 2]):
            subset = df.loc[df[group] == group_id, :].copy()
            allele = group_id
            if allele_map is not None:
                allele = allele_map[group_id]

            coef_str = "NA"
            r_annot_pos = (-1, -1)
            if len(subset.index) > 1:
                coef, p = stats.pearsonr(subset[y], subset[x])
                coef_str = "{:.2f}".format(coef)

                subset["intercept"] = 1
                betas = np.linalg.inv(subset[["intercept", x]].T.dot(
                    subset[["intercept", x]])).dot(
                    subset[["intercept", x]].T).dot(subset[y])
                subset["y_hat"] = np.dot(subset[["intercept", x]], betas)
                subset.sort_values(x, inplace=True)

                r_annot_pos = (subset.iloc[-1, :][x] + (subset[x].max() * 0.05),
                               subset.iloc[-1, :]["y_hat"])

                sns.regplot(x=x, y=y, data=subset, ci=None,
                            scatter_kws={'facecolors': palette[group_id],
                                         'linewidth': 0,
                                         'alpha': 0.60},
                            line_kws={"color": palette[group_id],
                                      "linewidth": 5,
                                      "alpha": 1},
                            ax=ax
                            )

            ax.annotate(
                '{}\n{}'.format(allele, coef_str),
                xy=r_annot_pos,
                color=palette[group_id],
                alpha=0.75,
                fontsize=22,
                fontweight='bold')

        if annot is not None:
            for i, annot_label in enumerate(annot):
                ax.annotate(annot_label,
                            xy=(0.03, 0.94 - (i * 0.04)),
                            xycoords=ax.transAxes,
                            color="#000000",
                            alpha=0.75,
                            fontsize=20,
                            fontweight='bold')

        (xmin, xmax) = (df[x].min(), df[x].max())
        (ymin, ymax) = (df[y].min(), df[y].max())
        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05

        ax.set_xlim(xmin - xmargin, xmax + xmargin)
        ax.set_ylim(ymin - ymargin, ymax + ymargin)

        ax.set_title(title,
                     fontsize=22,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=20,
                      fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            print("\t\tSaving plot: {}".format(os.path.basename(outpath)))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL path: {}".format(self.eqtl_path))
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Covariate path: {}".format(self.cova_path))
        print("  > Sample-to-dataset file: {}".format(self.std_path))
        print("  > Interest: {}".format(self.interest))
        print("  > Nrows: {}".format(self.nrows))
        print("  > Extension: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
