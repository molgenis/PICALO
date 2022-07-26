#!/usr/bin/env python3

"""
File:         visualise_double_interaction_eqtl.py
Created:      2022/07/22
Last Changed: 2022/07/25
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
__program__ = "Visualise Double Interaction eQTL"
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
./visualise_double_interaction_eqtl.py -h

### BIOS ###

./visualise_double_interaction_eqtl.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -c1 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -c2 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000226278_rs12386672_Comp4+PIC8 ENSG00000188056_rs10214649_Comp2+PIC4 ENSG00000003137_rs62147573_Comp1+PIC1 ENSG00000167207_rs1981760_Comp12+PIC6 ENSG00000135845_rs2285176_Comp6+PIC5 ENSG00000226278_rs12386672_Comp3+PIC2 ENSG00000137267_rs9392465_Comp6+PIC5 ENSG00000251381_rs11022589_Comp1+PIC1 ENSG00000259235_rs964611_Comp1+PIC1 ENSG00000135953_rs2732824_Comp3+PIC2 ENSG00000198952_rs12128955_Comp3+PIC2 ENSG00000145730_rs2432162_Comp3+PIC2 ENSG00000136819_rs11788537_Comp6+PIC5 ENSG00000074803_rs964611_Comp1+PIC1 ENSG00000116688_rs28718032_Comp3+PIC2 ENSG00000129925_rs12921174_Comp3+PIC2 ENSG00000167207_rs1981760_Comp3+PIC2 ENSG00000059122_rs9928222_Comp1+PIC1 ENSG00000128973_rs1554223_Comp1+PIC1 ENSG00000197728_rs1131017_Comp1+PIC1 ENSG00000166435_rs10899051_Comp1+PIC1 ENSG00000164308_rs2910686_Comp1+PIC1 ENSG00000127946_rs1186222_Comp1+PIC1 ENSG00000100376_rs2350629_Comp1+PIC1 ENSG00000006282_rs9890200_Comp1+PIC1 \
    -n 1200 \
    -e png
    
./visualise_double_interaction_eqtl.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -c1 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -c2 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000137267_rs9392465_Comp6+PIC5 ENSG00000136819_rs11788537_Comp6+PIC5 \
    -n 300 \
    -e png pdf
    
### MetaBrain ###

./visualise_double_interaction_eqtl.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -c1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -c2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000015592.16_8:27245507:rs17366947:A_G_Comp6+PIC2 ENSG00000137944.18_1:88964737:rs1536259:A_G_Comp3+PIC1 ENSG00000140873.16_16:77359175:rs891134:C_G_Comp6+PIC2 ENSG00000188732.11_7:23681366:rs4722244:C_T_Comp6+PIC2 ENSG00000214941.8_17:15973341:rs2779212:T_C_Comp3+PIC1 ENSG00000258289.8_14:64911980:rs7160518:G_T_Comp3+PIC1 ENSG00000117983.17_11:1232808:rs11026355:G_A_Comp9+PIC3 ENSG00000165271.17_9:33467579:rs2275228:T_C_Comp9+PIC3 ENSG00000213462.5_7:64958180:rs12666841:C_T_Comp3+PIC1 ENSG00000112599.9_6:42198814:rs6907587:C_T_Comp9+PIC3 ENSG00000084072.16_1:39753393:rs1046988:C_T_Comp6+PIC2 ENSG00000112599.9_6:42198814:rs6907587:C_T_Comp6+PIC2 ENSG00000134160.13_15:31102119:rs3809579:C_T_Comp6+PIC2 ENSG00000185149.6_4:155161607:rs13132330:C_A_Comp3+PIC1 ENSG00000095261.14_9:120832525:rs13299616:T_C_Comp9+PIC3 ENSG00000107165.13_9:12761820:rs7030485:G_A_Comp3+PIC1 ENSG00000134765.9_18:31156906:rs2118343:C_G_Comp9+PIC3 ENSG00000100225.18_22:32487295:rs3827335:A_G_Comp6+PIC2 ENSG00000185163.10_12:132141880:rs60927391:G_A_Comp3+PIC1 ENSG00000243335.9_7:66625676:rs1544549:T_A_Comp6+PIC2 ENSG00000197702.14_11:12362919:rs6485683:G_A_Comp3+PIC1 ENSG00000241935.9_10:97584164:rs12416267:G_A_Comp6+PIC2 ENSG00000174652.19_19:9415237:rs7258023:G_A_Comp9+PIC3 ENSG00000105976.15_7:116805037:rs187437:G_A_Comp3+PIC1 ENSG00000100726.15_16:1488362:rs1040499:C_A_Comp3+PIC1 \
    -n 3750 \
    -e png
    
./visualise_double_interaction_eqtl.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -c1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -c2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -i ENSG00000140873.16_16:77359175:rs891134:C_G_Comp6+PIC2 ENSG00000188732.11_7:23681366:rs4722244:C_T_Comp6+PIC2 \
    -n 350 \
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
        self.cov1_path = getattr(arguments, 'covariate1')
        self.cov2_path = getattr(arguments, 'covariate2')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.interest = getattr(arguments, 'interest')
        self.nrows = getattr(arguments, 'nrows')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'visualise_double_interaction_eqtl')
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
        parser.add_argument("-c1",
                            "--covariate1",
                            type=str,
                            required=True,
                            help="The path to the covariate matrix 1.")
        parser.add_argument("-c2",
                            "--covariate2",
                            type=str,
                            required=True,
                            help="The path to the covariate matrix 2.")
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
        cov1_df = self.load_file(self.cov1_path, header=0, index_col=0)
        cov2_df = self.load_file(self.cov2_path, header=0, index_col=0)

        if self.std_path:
            std_df = self.load_file(self.std_path, header=0, index_col=None)

            print("Filter data")
            samples = list(std_df.iloc[:, 0])
            geno_df = geno_df.loc[:, samples]
            expr_df = expr_df.loc[:, samples]
            cov1_df = cov1_df.loc[samples, :]
            cov2_df = cov2_df.loc[samples, :]

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
        if list(cov1_df.columns) != samples:
            print("Error, covariates 1 header does not match expression file.")
            exit()
        if list(cov2_df.columns) != samples:
            print("Error, covariates 2 header does not match expression file.")
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
                    cov = interest.replace(eqtl_id + "_", "")
                    cov1, cov2 = cov.split("+")
                    eqtl_covs.append((cov1, cov2))

            for cov1, cov2 in eqtl_covs:
                print("\tWorking on: {}\t{} [{}]\t{}-{} [{}/{} "
                      "{:.2f}%]".format(snp_name, probe_name, hgnc_name,
                                        cov1, cov2,
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

                # Add the covariates of interest.
                for i, cov_df, cov in ((1, cov1_df, cov1), (2, cov2_df, cov2)):
                    covariate_df = cov_df.loc[[cov], :].T
                    covariate_df.columns = ["covariate{}".format(i)]
                    data = data.merge(covariate_df, left_index=True, right_index=True)
                    data["interaction{}".format(i)] = data["genotype"] * data["covariate{}".format(i)]

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

                interaction_pvalue1 = OLS(data["expression"], data[["intercept", "genotype", "covariate1", "interaction1"]]).fit().pvalues[3]
                interaction_pvalue_str1 = "{:.2e}".format(interaction_pvalue1)
                if interaction_pvalue1 == 0:
                    interaction_pvalue_str1 = "<{:.1e}".format(1e-308)

                context_genotype_r1, _ = stats.pearsonr(data["genotype"], data["covariate1"])

                interaction_pvalue2 = OLS(data["expression"], data[["intercept", "genotype", "covariate2", "interaction2"]]).fit().pvalues[3]
                interaction_pvalue_str2 = "{:.2e}".format(interaction_pvalue2)
                if interaction_pvalue2 == 0:
                    interaction_pvalue_str2 = "<{:.1e}".format(1e-308)

                context_genotype_r2, _ = stats.pearsonr(data["genotype"], data["covariate2"])

                print_probe_name = probe_name
                if "." in print_probe_name:
                    print_probe_name = print_probe_name.split(".")[0]

                print_snp_name = snp_name
                if ":" in print_snp_name:
                    print_snp_name = print_snp_name.split(":")[2]

                # Fill the interaction plot annotation.
                annot1 = ["eQTL p-value: {}".format(eqtl_pvalue_str),
                          "eQTL r: {:.2f}".format(eqtl_pearsonr),
                          "interaction p-value: {}".format(interaction_pvalue_str1),
                          "context - genotype r: {:.2f}".format(context_genotype_r1)
                          ]
                annot2 = ["eQTL p-value: {}".format(eqtl_pvalue_str),
                          "eQTL r: {:.2f}".format(eqtl_pearsonr),
                          "interaction p-value: {}".format(interaction_pvalue_str2),
                          "context - genotype r: {:.2f}".format(context_genotype_r2)
                          ]

                allele_map = {0.0: "{}/{}".format(major_allele, major_allele),
                              1.0: "{}/{}".format(major_allele, minor_allele),
                              2.0: "{}/{}".format(minor_allele, minor_allele)}

                # Plot the double interaction eQTL.
                self.double_inter_plot(df=data,
                                       x1="covariate1",
                                       x2="covariate2",
                                       y="expression",
                                       group="group",
                                       palette=self.palette,
                                       allele_map=allele_map,
                                       xlabel1=cov1,
                                       xlabel2=cov2,
                                       title="{} [{}] - {}\n MAF={:.2f}  n={:,}".format(print_snp_name, minor_allele, print_probe_name, minor_allele_frequency, data.shape[0]),
                                       annot1=annot1,
                                       annot2=annot2,
                                       ylabel="{} expression".format(hgnc_name),
                                       filename="{}_{}_{}_{}_{}_{}".format(row_index,
                                                                           print_probe_name,
                                                                           hgnc_name,
                                                                           print_snp_name,
                                                                           cov1,
                                                                           cov2)
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

    def double_inter_plot(self, df, x1="x1", x2="x2", y="y", group="group",
                          palette=None, allele_map=None, xlabel1="",
                          xlabel2="", annot1=None, annot2=None, ylabel="",
                          title="", filename="ieqtl_plot"):
        if len(set(df[group].unique()).symmetric_difference({0, 1, 2})) > 0:
            return

        fig, axes = plt.subplots(nrows=1,
                                 ncols=2,
                                 sharex='none',
                                 sharey='all',
                                 figsize=(24, 12))
        sns.set(color_codes=True)
        sns.set_style("ticks")
        sns.despine(fig=fig, ax=axes[0])
        sns.despine(fig=fig, ax=axes[1])

        self.inter_plot(ax=axes[0],
                        df=df,
                        x=x1,
                        y=y,
                        group=group,
                        palette=palette,
                        allele_map=allele_map,
                        annot=annot1,
                        xlabel=xlabel1,
                        ylabel=ylabel
                        )

        self.inter_plot(ax=axes[1],
                        df=df,
                        x=x2,
                        y=y,
                        group=group,
                        palette=palette,
                        allele_map=allele_map,
                        annot=annot2,
                        xlabel=xlabel2,
                        ylabel=ylabel
                        )

        fig.suptitle(title,
                     fontsize=25,
                     fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            print("\t\tSaving plot: {}".format(os.path.basename(outpath)))
            fig.savefig(outpath)
        plt.close()

    @staticmethod
    def inter_plot(ax, df, x="x", y="y", group="group", palette=None,
                   allele_map=None, annot=None, xlabel="", ylabel="",
                   title=""):
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

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL path: {}".format(self.eqtl_path))
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Covariate path 1: {}".format(self.cov1_path))
        print("  > Covariate path 2: {}".format(self.cov2_path))
        print("  > Sample-to-dataset file: {}".format(self.std_path))
        print("  > Interest: {}".format(self.interest))
        print("  > Nrows: {}".format(self.nrows))
        print("  > Extension: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
