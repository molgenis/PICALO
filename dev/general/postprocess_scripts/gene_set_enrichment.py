#!/usr/bin/env python3

"""
File:         gene_set_enrichment.py
Created:      2022/02/24
Last Changed: 2022/03/04
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
import glob
import json
import os
import re

# Third party imports.
import pandas as pd
import numpy as np
import requests
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Gene Set Enrichment"
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
./gene_set_enrichment.py -h

### BIOS ###

./gene_set_enrichment.py \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD
    
    
./gene_set_enrichment.py \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-04-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-AllPICsCorrected-SP140AsCov/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-04-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-AllPICsCorrected-SP140AsCov-GeneExpressionFNPD_gene_correlations-avgExpressionAdded.txt.gz \
    -o 2022-03-04-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-AllPICsCorrected-SP140AsCov-SP140AsCov-eQTLGeneCorrelationsFNPD
    
./gene_set_enrichment.py \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-04-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-AllPICsCorrected-SP140AsCov/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD
    
./gene_set_enrichment.py \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-SP140AsCov/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-SP140AsCov-GeneExpressionFNPD_gene_correlations-avgExpressionAdded.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-SP140AsCov
    
### MetaBrain ###

./gene_set_enrichment.py \
    -avge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/calc_avg_gene_expression/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -pi /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations.txt.gz \
    -o 2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.avg_ge_path = getattr(arguments, 'average_gene_expression')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')
        self.picalo_indir = getattr(arguments, 'picalo')
        self.covariates = getattr(arguments, 'covariates')
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.out_filename = getattr(arguments, 'outfile')

        self.top_n = 200
        self.min_corr = 0.1

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'gene_set_enrichment')
        self.file_outdir = os.path.join(self.outdir, self.out_filename)
        for outdir in [self.outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # MetaBrain gene network.
        self.gn_matrix_basedir = '/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-02-28-MetaBrainNetwork/output/11_ImportToWebsite'
        self.gn_matrix_subdirs = {
            'goP': '150/2020-03-23-goa_human_P.150_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt.gz',
            'goC': '575/2020-03-23-goa_human_C.575_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt.gz',
            'goF': '225/2020-03-23-goa_human_F.225_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt.gz',
            'kegg': '500/2020-03-28-c2.cp.kegg.v7.0.500_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt.gz',
            'reactome': '700/2020-03-28-Ensembl2Reactome_All_Levels.700_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt',
            'HPO': '1000/2020-03-28-HPO-phenotype-to-genes.1000_eigenvectors.predictions.bonSigOnly_termNames_tranposed.txt.gz'
        }
        self.gn_annotation_basedir = '/groups/umcg-biogen/tmp01/annotation/genenetworkAnnotations/'
        self.gn_annotation_files = {
            'goP': 'go_P_prepared_predictions.colAnnotations.txt',
            'goC': 'go_C_prepared_predictions.colAnnotations.txt',
            'goF': 'go_F_prepared_predictions.colAnnotations.txt',
            'kegg': '',
            'reactome': 'reactome_prepared_predictions.colAnnotations_filter.txt',
            'HPO': 'hpo_prepared_predictions.colAnnotations.txt'
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
        parser.add_argument("-avge",
                            "--average_gene_expression",
                            type=str,
                            required=True,
                            help="The path to the average gene expression "
                                 "matrix.")
        parser.add_argument("-mae",
                            "--min_avg_expression",
                            type=float,
                            default=None,
                            help="The minimal average expression of a gene."
                                 "Default: None.")
        parser.add_argument("-pi",
                            "--picalo",
                            type=str,
                            required=True,
                            help="The path to the PICALO output directory.")
        parser.add_argument("-c",
                            "--covariates",
                            nargs="*",
                            type=str,
                            default=None,
                            help="The covariates to analyse."
                                 "Default: all.")
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            required=True,
                            help="The path to the covariate-gene correlations "
                                 "matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()
        #
        # print("Loading gene network files.")
        # gn_matrix_df_list = []
        # for name, matrix_subdir in self.gn_matrix_subdirs.items():
        #     gn_matrix_path = os.path.join(self.gn_matrix_basedir, matrix_subdir)
        #     if not os.path.exists(gn_matrix_path):
        #         continue
        #     gn_matrix_df = self.load_file(gn_matrix_path)
        #     gn_matrix_df.index = name + ":" + gn_matrix_df.index
        #     gn_matrix_df.index.name = None
        #     gn_matrix_df_list.append(gn_matrix_df)
        # gn_matrix_df = pd.concat(gn_matrix_df_list, axis=0)
        # print(gn_matrix_df)
        #
        # gn_annot_df_list = []
        # for name, annotation_file in self.gn_annotation_files.items():
        #     gn_annotation_path = os.path.join(self.gn_annotation_basedir, annotation_file)
        #     if annotation_file == "" or not os.path.exists(gn_annotation_path):
        #         continue
        #     gn_annot_df = self.load_file(gn_annotation_path, header=None)
        #     gn_annot_df.index = name + ":" + gn_annot_df.index
        #     gn_annot_df.index.name = None
        #     gn_annot_df.columns = ["Annotation"]
        #     gn_annot_df_list.append(gn_annot_df)
        # gn_annot_df = pd.concat(gn_annot_df_list, axis=0)
        #
        # # Subset the usefull info.
        # gn_annot_df = gn_matrix_df.loc[:, []].merge(gn_annot_df, left_index=True, right_index=True, how="left")
        # gn_annot_df.fillna("NA", inplace=True)
        # gn_annot_df["category"] = [index.split(":")[0] for index in gn_annot_df.index]
        # print(gn_annot_df)
        #
        # # Prep for the willcoxon tests.
        # gn_ranked_df = gn_matrix_df.rank(axis=1)
        # tie_term_df = pd.DataFrame(np.apply_along_axis(self.calc_tie_term, 1, gn_ranked_df),
        #                            index=gn_ranked_df.index,
        #                            columns=["tie_term"])
        #
        # # Save
        # self.save_file(df=gn_annot_df, outpath=os.path.join(self.outdir, "gene_network_annotation.pkl"))
        # self.save_file(df=gn_ranked_df, outpath=os.path.join(self.outdir, "gene_network_matrix_ranked.pkl"))
        # self.save_file(df=tie_term_df, outpath=os.path.join(self.outdir, "gene_network_tie_term.pkl"))
        # del gn_matrix_df
        # # exit()
        #
        # print("Loading average gene expression.")
        # df = self.load_file(self.avg_ge_path, header=0, index_col=0)
        # df.columns = ["avg expression"]
        # df.index = [x.split(".")[0] for x in df.index]
        # print(df)
        #
        # if self.min_avg_expression is not None:
        #     print("\tFiltering on eQTLs with >{} average gene expression".format(self.min_avg_expression))
        #     pre_shape = df.shape[0]
        #     df = df.loc[df["avg expression"] > self.min_avg_expression, :]
        #     print("\t  Removed {:,} genes".format(pre_shape - df.shape[0]))
        #
        # print("Lookup gene symbols.")
        # gi_df = self.toppgene_gene_lookup(df.index)
        # gi_df.set_index("Submitted", inplace=True)
        # df = df.merge(gi_df, left_index=True, right_index=True, how="left")
        # print(df)
        #
        # print("Loading gene correlations.")
        # gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=0)
        # if "ProbeName" in gene_corr_df.columns:
        #     gene_corr_df.index = gene_corr_df["ProbeName"].str.split(".", n=1, expand=True)[0]
        #     gene_corr_df = gene_corr_df.loc[:, [col for col in gene_corr_df if col.startswith("PIC")]]
        # gene_corr_df.columns = ["{} r".format(col) for col in gene_corr_df.columns]
        # df = df.merge(gene_corr_df, left_index=True, right_index=True, how="left")
        # print(df)
        #
        # print("Loading ieQTL results data.")
        # ieqtl_results = []
        # ieqtl_result_paths = glob.glob(os.path.join(self.picalo_indir, "*.txt.gz"))
        # ieqtl_result_paths.sort(key=self.natural_keys)
        # for ieqtl_result_path in ieqtl_result_paths:
        #     covariate = os.path.basename(ieqtl_result_path).split(".")[0]
        #     if covariate in ["call_rate", "genotype_stats"]:
        #         continue
        #
        #     ieqtl_result_df = self.load_file(ieqtl_result_path, header=0, index_col=None)
        #     if ieqtl_result_df.shape[0] != len(ieqtl_result_df["gene"].unique()):
        #         print("Non unique genes in PIC interaction output.")
        #
        #     ieqtl_result_df.index = [index.split(".")[0] for index in ieqtl_result_df["gene"]]
        #
        #     snp_col = "SNP"
        #     beta_geno_col = "beta-genotype"
        #     beta_inter_col = "beta-interaction"
        #     fdr_col = "FDR"
        #     if len({snp_col, beta_geno_col, beta_inter_col, fdr_col}.intersection(set(ieqtl_result_df.columns))) != 4:
        #         snp_col = "snp"
        #         beta_geno_col = "ieQTL beta-genotype"
        #         beta_inter_col = "ieQTL beta-interaction"
        #         fdr_col = "ieQTL FDR"
        #
        #     ieqtl_result_df["direction"] = ((ieqtl_result_df[beta_geno_col] * ieqtl_result_df[beta_inter_col]) > 0).map({True: "induces", False: "inhibits"})
        #     if len(ieqtl_results) == 0:
        #         ieqtl_result_df = ieqtl_result_df[[snp_col, fdr_col, "direction"]]
        #         ieqtl_result_df.columns = ["SNP", "{} FDR".format(covariate), "{} direction".format(covariate)]
        #     else:
        #         ieqtl_result_df = ieqtl_result_df[[fdr_col, "direction"]]
        #         ieqtl_result_df.columns = ["{} FDR".format(covariate), "{} direction".format(covariate)]
        #     ieqtl_results.append(ieqtl_result_df)
        # ieqtl_result_df = pd.concat(ieqtl_results, axis=1)
        # df = df.merge(ieqtl_result_df, left_index=True, right_index=True, how="left")
        # print(df)
        #
        # print("Saving file.")
        # self.save_file(df=df,
        #                outpath=os.path.join(self.file_outdir, "info.txt.gz"))
        # exit()

        print("Loading files.")
        # gn_annot_df = self.load_file(os.path.join(self.outdir, "gene_network_annotation.pkl"))
        # gn_ranked_df = self.load_file(os.path.join(self.outdir, "gene_network_matrix_ranked.pkl"))
        # tie_term_df = self.load_file(os.path.join(self.outdir, "gene_network_tie_term.pkl"))
        # print(gn_annot_df)
        # print(gn_ranked_df)
        # print(tie_term_df)

        df = self.load_file(os.path.join(self.file_outdir, "info.txt.gz"))
        print(df)

        interest = ['ENSG00000149218', 'ENSG00000182957', 'ENSG00000166750',
                    'ENSG00000163644', 'ENSG00000120539', 'ENSG00000223960',
                    'ENSG00000138119', 'ENSG00000160216', 'ENSG00000091490',
                    'ENSG00000101224', 'ENSG00000145016', 'ENSG00000115415',
                    'ENSG00000115419', 'ENSG00000188290', 'ENSG00000146285',
                    'ENSG00000154451', 'ENSG00000237568', 'ENSG00000129347',
                    'ENSG00000181381', 'ENSG00000187231', 'ENSG00000089127',
                    'ENSG00000125148', 'ENSG00000182179', 'ENSG00000136816',
                    'ENSG00000181381', 'ENSG00000110057', 'ENSG00000245556',
                    'ENSG00000176476', 'ENSG00000185624', 'ENSG00000002549',
                    'ENSG00000067066', 'ENSG00000125952', 'ENSG00000259118',
                    'ENSG00000100596', 'ENSG00000177989', 'ENSG00000167207',
                    'ENSG00000168899', 'ENSG00000123737', 'ENSG00000145386',
                    'ENSG00000103018', 'ENSG00000260108', 'ENSG00000185298',
                    'ENSG00000120899', 'ENSG00000167208', 'ENSG00000260249',
                    'ENSG00000161929', 'ENSG00000261879', 'ENSG00000161692',
                    'ENSG00000163382', 'ENSG00000113368', 'ENSG00000136827',
                    'ENSG00000100461', 'ENSG00000188404', 'ENSG00000088298',
                    'ENSG00000186407', 'ENSG00000168961', 'ENSG00000161010',
                    'ENSG00000197226', 'ENSG00000136104', 'ENSG00000184752',
                    'ENSG00000141837', 'ENSG00000205560', 'ENSG00000254413',
                    'ENSG00000139574', 'ENSG00000170653', 'ENSG00000267281',
                    'ENSG00000105321', 'ENSG00000072694', 'ENSG00000234211',
                    'ENSG00000107960']

        covariates = [col.replace(" FDR", "") for col in df.columns if col.endswith(" FDR")]
        genes = {}
        toppgene_enrichment_list = []
        overlap_data = []
        for covariate in covariates:
            if self.covariates is not None and covariate not in self.covariates:
                continue
            print("### Analyzing {} ###".format(covariate))

            # Create output directory.
            covariate_outdir = os.path.join(self.file_outdir, covariate)
            if not os.path.exists(covariate_outdir):
                os.makedirs(covariate_outdir)

            for correlation_direction in ["positive", "negative"]:
                correlation_mask = None
                if correlation_direction == "positive":
                    correlation_mask = df["{} r".format(covariate)] > 0
                elif correlation_direction == "negative":
                    correlation_mask = df["{} r".format(covariate)] < 0
                else:
                    print("huh")
                    exit()

                # Select, filter, and sort.
                subset_df = df.loc[correlation_mask, ["avg expression",
                                                      "OfficialSymbol",
                                                      "Entrez",
                                                      "{} r".format(covariate),
                                                      "{} FDR".format(covariate),
                                                      "{} direction".format(covariate),
                                                      ]].copy()
                subset_df.columns = ["avg expression", "symbol", "entrez", "correlation", "FDR", "direction"]

                # Prep correlation matrix.
                corr_subset_df = subset_df.copy()
                corr_subset_df["abs correlation"] = corr_subset_df["correlation"].abs()
                corr_subset_df = corr_subset_df.loc[corr_subset_df["abs correlation"] >= self.min_corr, :]
                corr_subset_df.sort_values(by="abs correlation", ascending=False, inplace=True)
                print(corr_subset_df)

                if corr_subset_df.shape[0] > 0:
                    print("\t{} correlation (abs r > {:.2f})".format(correlation_direction, self.min_corr))
                    if corr_subset_df.shape[0] >= 50:
                        print("\tAnalyzing {:,} genes".format(corr_subset_df.iloc[:self.top_n, :].shape[0]))
                    else:
                        print("\tAnalyzing {:,} genes: {}".format(corr_subset_df.iloc[:self.top_n, :].shape[0], ", ".join(subset_df.index[:self.top_n])))

                    # Saving data.
                    genes["{}_{}".format(covariate, correlation_direction)] = set(corr_subset_df.index[:self.top_n])
                    print(", ".join(corr_subset_df.index))
                    self.save_file(df=corr_subset_df.iloc[:self.top_n, :],
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_data.txt.gz".format(covariate, correlation_direction)))

                    # ToppGene enrichment.
                    enrich_df = self.toppgene_func_enrichment(
                        entrez_ids=corr_subset_df.loc[~corr_subset_df["entrez"].isna(), "entrez"][:self.top_n]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}.txt.gz".format(covariate, correlation_direction)))
                    print("")
                    enrich_df["covariate"] = covariate
                    enrich_df["correlation_direction"] = correlation_direction
                    enrich_df["subset"] = "genes"
                    enrich_df["N"] = len(corr_subset_df.loc[~corr_subset_df["entrez"].isna(), "entrez"][:self.top_n])
                    toppgene_enrichment_list.append(enrich_df)
                    del enrich_df

                    # # Genenetwork enrichment.
                    # enrich_df = self.genenetwork_enrichment(
                    #     ranked_df=gn_ranked_df,
                    #     annot_df=gn_annot_df,
                    #     tie_term_df=tie_term_df,
                    #     genes=subset_df.index[:self.top_n]
                    # )
                    # self.save_file(df=enrich_df,
                    #                outpath=os.path.join(covariate_outdir,
                    #                                     "genenetwork_enrichment_{}_{}.txt.gz".format(covariate, correlation_direction)))
                    # print("")
                    # del enrich_df

                ################################################################

                # Filter on eQTL genes.
                signif_subset_df = subset_df.loc[subset_df["FDR"] < 0.05, :].copy()

                if signif_subset_df.shape[0] > 0:
                    print("\t{} correlation (ieQTL FDR < 0.05)".format(correlation_direction))
                    if signif_subset_df.shape[0] >= 50:
                        print("\tAnalyzing {:,} genes".format(signif_subset_df.iloc[:self.top_n, :].shape[0]))
                    else:
                        print("\tAnalyzing {:,} genes: {}".format(signif_subset_df.iloc[:self.top_n, :].shape[0], ", ".join(signif_subset_df.index[:self.top_n])))

                    # Saving data.
                    print(", ".join([str(symbol) for symbol in signif_subset_df["symbol"]]))
                    interest_overlap = set(interest).intersection(set(signif_subset_df.index))
                    print("overlapping with interest [{:,}/{:,}] = {}".format(len(interest_overlap), len(interest), ", ".join(list(interest_overlap))))
                    overlap_data.append([covariate, correlation_direction, signif_subset_df.shape[0], len(interest_overlap)])
                    if len(interest_overlap) > 0:
                        interest_df = self.toppgene_gene_lookup(list(interest_overlap))
                        print(interest_df)
                        print(", ".join(interest_df["OfficialSymbol"].values.tolist()))
                    self.save_file(df=signif_subset_df.iloc[:self.top_n, :],
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_ieQTL_genes_data.txt.gz".format(
                                                            covariate,
                                                            correlation_direction)))

                    # ToppGene enrichment.
                    enrich_df = self.toppgene_func_enrichment(
                        entrez_ids=signif_subset_df.loc[~signif_subset_df["entrez"].isna(), "entrez"][:self.top_n]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_ieQTL_genes_data.txt.gz".format(
                                                            covariate,
                                                            correlation_direction)))
                    print("")
                    enrich_df["covariate"] = covariate
                    enrich_df["correlation_direction"] = correlation_direction
                    enrich_df["subset"] = "ieQTL genes"
                    enrich_df["N"] = len(signif_subset_df.loc[~signif_subset_df["entrez"].isna(), "entrez"][:self.top_n])
                    toppgene_enrichment_list.append(enrich_df)
                    del enrich_df

                ################################################################

            # print("Checking overlap")
            # indices = list(genes.keys())
            # indices.sort(key=self.natural_keys)
            # overlap_df = pd.DataFrame(np.nan, index=indices, columns=indices)
            # for index1 in indices:
            #     for index2 in indices:
            #         overlap_df.loc[index1, index2] = len(genes[index1].intersection(genes[index2])) / min(len(genes[index1]), len(genes[index2]))
            # print(overlap_df)
            # self.save_file(df=overlap_df, outpath=os.path.join(covariate_outdir, "overlap.txt.gz"))

        print("")
        toppgene_enrichment_df = pd.concat(toppgene_enrichment_list, axis=0)
        toppgene_enrichment_df.sort_values(by="PValue", inplace=True)
        print(toppgene_enrichment_df)
        self.save_file(df=toppgene_enrichment_df, outpath=os.path.join(self.file_outdir,
                                                                       "toppgene_enrichment_df.txt.gz"))

        overlap_df = pd.DataFrame(overlap_data, columns=["covariate", "interaction direction", "N", "N overlap"])
        overlap_df.sort_values(by="N overlap", inplace=True, ascending=False)
        print(overlap_df)
        self.save_file(df=overlap_df, outpath=os.path.join(self.file_outdir,
                                                            "overlap_df.txt.gz"))

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith("pkl"):
            df = pd.read_pickle(inpath)
        else:
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
    def toppgene_gene_lookup(genes, chunk_size=30000):
        chuncks = [genes[i * chunk_size:(i + 1) * chunk_size] for i in range((len(genes) + chunk_size - 1) // chunk_size)]

        result_list = []
        for chunk in chuncks:
            data = json.dumps({"Symbols": list(chunk)})
            response = requests.post("https://toppgene.cchmc.org/API/lookup",
                                     headers={'Content-type': 'application/json',},
                                     data=data)
            result = json.loads(response.text)
            result_list.append(pd.DataFrame(result["Genes"]))
        return pd.concat(result_list, axis=0)

    @staticmethod
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        if outpath.endswith("pkl"):
            df.to_pickle(outpath)
        else:
            df.to_csv(outpath, sep=sep, index=index, header=header,
                      compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    @staticmethod
    def calc_tie_term(ranks):
        _, t = np.unique(ranks, return_counts=True, axis=-1)
        return (t**3 - t).sum(axis=-1)

    @staticmethod
    def toppgene_func_enrichment(entrez_ids, n_print=4):
        print("\t### ToppGene ###")

        data = json.dumps({"Genes": [int(entrez_id) for entrez_id in entrez_ids],
                           "Categories": [{"Type": "Pathway",
                                           "PValue": 0.05,
                                           "Correction": "FDR"},
                                          {"Type": "ToppCell",
                                           "PValue": 0.05,
                                           "Correction": "FDR"},
                                          ]})
        response = requests.post("https://toppgene.cchmc.org/API/enrich",
                                 headers={'Content-type': 'application/json',},
                                 data=data)
        result = json.loads(response.text)
        df = pd.DataFrame(result["Annotations"])
        if df.shape[0] > 0:
            df["Genes"] = [", ".join([gene["Symbol"] for gene in value]) for value in df["Genes"]]
            df.drop(["ID"], axis=1, inplace=True)

            print("\tResults:")
            for category in df["Category"].unique():
                print("\t  {}:".format(category))
                subset_df = df.loc[df["Category"] == category, :]
                for i, (index, row) in enumerate(subset_df.iterrows()):
                    if i > n_print:
                        break
                    print("\t  {}: [p-value: {:.2e}]  {}".format(i, row["PValue"], row["Name"]))
                print("")

        return df

    def genenetwork_enrichment(self, ranked_df, annot_df, tie_term_df, genes, n_print=4):
        print("\t### GeneNetwork ###")

        mask = ranked_df.columns.isin(genes)

        # Mann-Whitney-U test based on scipy code.
        mwu_df = ranked_df.loc[:, mask].sum(axis=1).to_frame()
        mwu_df.columns = ["R1"]
        mwu_df = mwu_df.merge(tie_term_df, left_index=True, right_index=True)
        mwu_df["n"] = np.size(mask)
        mwu_df["n1"] = np.sum(mask)
        mwu_df["n2"] = mwu_df["n"] - mwu_df["n1"]
        mwu_df["U1"] = mwu_df["R1"] - mwu_df["n1"] * (mwu_df["n1"] + 1) / 2
        mwu_df["U2"] = mwu_df["n1"] * mwu_df["n2"] - mwu_df["U1"]
        mwu_df["U"] = mwu_df[["U1", "U2"]].max(axis=1)
        mwu_df["mu"] = mwu_df["n1"] * mwu_df["n2"] / 2
        mwu_df["s"] = np.sqrt(mwu_df["n1"] * mwu_df["n2"] / 12 * ((mwu_df["n"] + 1) - mwu_df["tie_term"] / (mwu_df["n"] * (mwu_df["n"] - 1))))
        mwu_df["numerator"] = mwu_df["U"] - mwu_df["mu"] - 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            mwu_df["z"] = mwu_df["numerator"] / mwu_df["s"]
        mwu_df["z"].clip(0, 37.677120720495)
        mwu_df["p"] = stats.norm.sf(mwu_df["z"]) * 2
        mwu_df["p"].clip(0, 1)

        mwu_df = annot_df.merge(mwu_df, left_index=True, right_index=True, how="right")
        mwu_df.sort_values(by="p", inplace=True)
        mwu_df = mwu_df.loc[mwu_df["p"] < 0.05, :]

        print("\tResults:")
        for category in self.gn_matrix_subdirs.keys():
            print("\t  {}:".format(category))
            subset_df = mwu_df.loc[mwu_df["category"] == category, :]
            for i, (index, row) in enumerate(subset_df.iterrows()):
                if i > n_print:
                    break
                print("\t\t{}: [p-value: {:.2e}]  {} = {}".format(i, row["p"], ":".join(index.split(":")[1:]), row["Annotation"]))
            print("")

        return mwu_df[["p", "Annotation"]]

    def print_arguments(self):
        print("Arguments:")
        print("  > Average gene expression path: {}".format(self.avg_ge_path))
        print("  > Minimal gene expression: {}".format(self.min_avg_expression))
        print("  > PICALO directory: {}".format(self.picalo_indir))
        print("  > Covariates: {}".format(self.covariates))
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Output directory {}".format(self.outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
