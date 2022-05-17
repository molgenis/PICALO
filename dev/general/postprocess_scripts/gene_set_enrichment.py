#!/usr/bin/env python3

"""
File:         gene_set_enrichment.py
Created:      2022/02/24
Last Changed: 2022/05/17
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
import requests
from statsmodels.stats import multitest

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
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_results-avgExpressionAdded.txt.gz \
    -mae 1 \
    -mc 0.1 \
    -o 2022-04-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations

./gene_set_enrichment.py \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PIC_interactions \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_results-avgExpressionAdded.txt.gz \
    -pc /groups/umcg-biogen/tmp01/annotation/2020-02-16-gencode.v32.primary_assembly.annotation.collapsedGenes.proteincoding.txt.gz \
    -mae 1 \
    -o 2022-04-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_ZscoreFiltering

### MetaBrain ###

./gene_set_enrichment.py \
    -pi /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PIC_interactions \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_results-avgExpressionAdded.txt.gz \
    -mae 1 \
    -mc 0.1 \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations

./gene_set_enrichment.py \
    -pi /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PIC_interactions \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_results-avgExpressionAdded.txt.gz \
    -pc /groups/umcg-biogen/tmp01/annotation/2020-02-16-gencode.v32.primary_assembly.annotation.collapsedGenes.proteincoding.txt.gz \
    -mae 1 \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_ZscoreFiltering

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_indir = getattr(arguments, 'picalo')
        self.covariates = getattr(arguments, 'covariates')
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.protein_coding_path = getattr(arguments, 'protein_coding')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')
        self.min_corr = getattr(arguments, 'min_corr')
        self.alpha = getattr(arguments, 'alpha')
        self.top_n = getattr(arguments, 'top_n')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'gene_set_enrichment')
        self.file_outdir = os.path.join(self.outdir, self.out_filename)
        for outdir in [self.outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

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
        parser.add_argument("-pc",
                            "--protein_coding",
                            type=str,
                            default=None,
                            help="The path to the protein coding genes list.")
        parser.add_argument("-mae",
                            "--min_avg_expression",
                            type=int,
                            default=0,
                            help="The minimal average expression of a gene."
                                 "Default: 0.")
        parser.add_argument("-mc",
                            "--min_corr",
                            type=float,
                            default=0,
                            help="The minimal correlation of a gene "
                                 "for inclusion. Default 0.")
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
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading gene correlations.")
        df = self.load_file(self.gene_correlations_path, header=0, index_col=None)
        if "ProbeName" in df.columns:
            df.index = df["ProbeName"].str.split(".", n=1, expand=True)[0]
        print(df)

        if self.min_avg_expression > 0:
            print("\tFiltering genes with >{} average gene expression".format(self.min_avg_expression))
            pre_shape = df.shape[0]
            df = df.loc[df["avgExpression"] > self.min_avg_expression, :]
            print("\t  Removed {:,} genes".format(pre_shape - df.shape[0]))

        if self.protein_coding_path is not None:
            print("\tFiltering protein coding genes")
            protein_coding_df = self.load_file(self.protein_coding_path, header=None, index_col=None)
            pre_shape = df.shape[0]
            protein_coding_df.index = protein_coding_df.iloc[:, 0].str.split(".", n=1, expand=True)[0]
            df = df.loc[df.index.isin(protein_coding_df.index), :]
            print("\t  Removed {:,} genes".format(pre_shape - df.shape[0]))

        print("Lookup gene symbols.")
        gi_df = self.toppgene_gene_lookup(df.index)
        gi_df.set_index("Submitted", inplace=True)
        df = df.merge(gi_df, left_index=True, right_index=True, how="left")
        print(df)

        print("Loading ieQTL results data.")
        ieqtl_results = []
        ieqtl_result_paths = glob.glob(os.path.join(self.picalo_indir, "*.txt.gz"))
        ieqtl_result_paths.sort(key=self.natural_keys)
        for ieqtl_result_path in ieqtl_result_paths:
            covariate = os.path.basename(ieqtl_result_path).split(".")[0]
            if covariate in ["call_rate", "genotype_stats"]:
                continue

            ieqtl_result_df = self.load_file(ieqtl_result_path, header=0, index_col=None)
            if ieqtl_result_df.shape[0] != len(ieqtl_result_df["gene"].unique()):
                print("Non unique genes in PIC interaction output.")

            ieqtl_result_df.index = [index.split(".")[0] for index in ieqtl_result_df["gene"]]

            snp_col = "SNP"
            beta_geno_col = "beta-genotype"
            beta_inter_col = "beta-interaction"
            fdr_col = "FDR"
            if len({snp_col, beta_geno_col, beta_inter_col, fdr_col}.intersection(set(ieqtl_result_df.columns))) != 4:
                snp_col = "snp"
                beta_geno_col = "ieQTL beta-genotype"
                beta_inter_col = "ieQTL beta-interaction"
                fdr_col = "ieQTL FDR"

            ieqtl_result_df["direction"] = ((ieqtl_result_df[beta_geno_col] * ieqtl_result_df[beta_inter_col]) > 0).map({True: "induces", False: "inhibits"})
            if len(ieqtl_results) == 0:
                ieqtl_result_df = ieqtl_result_df[[snp_col, fdr_col, "direction"]]
                ieqtl_result_df.columns = ["SNP", "{} ieQTL FDR".format(covariate), "{} ieQTL direction".format(covariate)]
            else:
                ieqtl_result_df = ieqtl_result_df[[fdr_col, "direction"]]
                ieqtl_result_df.columns = ["{} ieQTL FDR".format(covariate), "{} ieQTL direction".format(covariate)]
            ieqtl_results.append(ieqtl_result_df)
        ieqtl_result_df = pd.concat(ieqtl_results, axis=1)
        df = df.merge(ieqtl_result_df, left_index=True, right_index=True, how="left")
        print(df)

        print("Recalculate correlation FDR")
        for pvalue_col in [col for col in df.columns if "pvalue" in col]:
            df[pvalue_col.replace("pvalue", "FDR")] = multitest.multipletests(df[pvalue_col], method='fdr_bh')[1]

        print("Saving file.")
        self.save_file(df=df,
                       outpath=os.path.join(self.file_outdir, "info.txt.gz"))
        # exit()
        #
        # print("Loading files.")
        # df = self.load_file(os.path.join(self.file_outdir, "info.txt.gz"))
        # print(df)

        covariates = [col.replace(" FDR", "") for col in df.columns if col.endswith(" FDR")]
        toppgene_enrichment_list = []
        for covariate in covariates:
            if self.covariates is not None and covariate not in self.covariates:
                continue
            print("### Analyzing {} ###".format(covariate))

            # Create output directory.
            covariate_outdir = os.path.join(self.file_outdir, covariate)
            if not os.path.exists(covariate_outdir):
                os.makedirs(covariate_outdir)

            for correlation_direction in ["positive", "negative"]:
                # Select, filter, and sort.
                subset_df = df.loc[:, ["avgExpression",
                                       "OfficialSymbol",
                                       "Entrez",
                                       "{} r".format(covariate),
                                       "{} pvalue".format(covariate),
                                       "{} FDR".format(covariate),
                                       "{} zscore".format(covariate),
                                       "{} ieQTL FDR".format(covariate),
                                       "{} ieQTL direction".format(covariate),
                                       ]].copy()
                subset_df.columns = ["avg expression", "symbol", "entrez", "correlation coefficient", "correlation p-value", "correlation FDR", "correlation zscore", "ieQTL FDR", "ieQTL direction"]
                subset_df = subset_df.loc[~subset_df["entrez"].isna(), :]
                subset_df["abs correlation zscore"] = subset_df["correlation zscore"].abs()
                subset_df.sort_values(by="abs correlation zscore", ascending=False, inplace=True)

                if correlation_direction == "positive":
                    subset_df = subset_df.loc[(subset_df["correlation zscore"] > 0) & (subset_df["abs correlation zscore"] > self.min_corr), :]
                elif correlation_direction == "negative":
                    subset_df = subset_df.loc[(subset_df["correlation zscore"] < 0) & (subset_df["abs correlation zscore"] > self.min_corr), :]
                else:
                    print("huh")
                    exit()

                print(subset_df)

                corr_subset_df = subset_df.iloc[:self.top_n, :].copy()
                corr_subset_df = corr_subset_df.loc[corr_subset_df["correlation FDR"] < self.alpha, :]
                if corr_subset_df.shape[0] > 0:
                    print("\t{} correlation (abs r > {:.2f})".format(correlation_direction, self.min_corr))
                    print("\tAnalyzing {:,} genes".format(corr_subset_df.shape[0]))

                    # Saving data.
                    self.save_file(df=corr_subset_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_data.txt.gz".format(covariate, correlation_direction)))

                    # ToppGene enrichment.
                    enrich_df = self.toppgene_func_enrichment(
                        entrez_ids=corr_subset_df["entrez"]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}.txt.gz".format(covariate, correlation_direction)))
                    print("")

                    if enrich_df.shape[0] > 0:
                        enrich_df["covariate"] = covariate
                        enrich_df["correlation_direction"] = correlation_direction
                        enrich_df["avg_abs_correlation_zscore_inTerm"] = [corr_subset_df.loc[corr_subset_df["symbol"].isin(genes.split(", ")), "correlation zscore"].abs().mean() for genes in enrich_df["Genes"]]
                        enrich_df["avg_abs_correlation_zscore_Overall"] = corr_subset_df["correlation zscore"].abs().mean()
                        enrich_df["subset"] = "genes"
                        enrich_df["N"] = corr_subset_df.shape[0]
                        toppgene_enrichment_list.append(enrich_df)
                    del enrich_df
                del corr_subset_df

                ################################################################

                # Filter on eQTL genes.
                signif_subset_df = subset_df.loc[subset_df["ieQTL FDR"] < self.alpha, :].iloc[:self.top_n, :].copy()
                if signif_subset_df.shape[0] > 0:
                    print("\t{} correlation (ieQTL FDR < {})".format(correlation_direction, self.alpha))
                    print("\tAnalyzing {:,} genes".format(signif_subset_df.shape[0]))

                    # Saving data.
                    self.save_file(df=signif_subset_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_ieQTL_genes_data.txt.gz".format(
                                                            covariate,
                                                            correlation_direction)))

                    # ToppGene enrichment.
                    enrich_df = self.toppgene_func_enrichment(
                        entrez_ids=signif_subset_df["entrez"]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(covariate_outdir,
                                                        "toppgene_enrichment_{}_{}_ieQTL_genes.txt.gz".format(
                                                            covariate,
                                                            correlation_direction)))
                    print("")
                    if enrich_df.shape[0] > 0:
                        enrich_df["covariate"] = covariate
                        enrich_df["correlation_direction"] = correlation_direction
                        enrich_df["avg_abs_correlation_zscore_inTerm"] = [signif_subset_df.loc[subset_df["symbol"].isin(genes.split(", ")), "correlation zscore"].abs().mean() for genes in enrich_df["Genes"]]
                        enrich_df["avg_abs_correlation_zscore_Overall"] = signif_subset_df["correlation zscore"].abs().mean()
                        enrich_df["subset"] = "ieQTL genes"
                        enrich_df["N"] = signif_subset_df.shape[0]
                        toppgene_enrichment_list.append(enrich_df)
                    del enrich_df
                del signif_subset_df

                ################################################################

        print("")
        toppgene_enrichment_df = pd.concat(toppgene_enrichment_list, axis=0)
        toppgene_enrichment_df.sort_values(by="PValue", inplace=True)
        toppgene_enrichment_df.reset_index(drop=True, inplace=True)
        print(toppgene_enrichment_df)
        self.save_file(df=toppgene_enrichment_df,
                       index=False,
                       outpath=os.path.join(self.file_outdir, "toppgene_enrichment_df.txt.gz"))
        self.save_file(df=toppgene_enrichment_df,
                       index=False,
                       outpath=os.path.join(self.file_outdir, "toppgene_enrichment_df.xlsx"))

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
    def toppgene_gene_lookup(genes, chunk_size=25000):
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
        if outpath.endswith("xlsx"):
            df.to_excel(outpath, header=header, index=index)
        else:
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

    def toppgene_func_enrichment(self, entrez_ids, n_print=4):
        print("\t### ToppGene ###")

        data = json.dumps({"Genes": [int(entrez_id) for entrez_id in entrez_ids],
                           "Categories": [{"Type": "Pathway",
                                           "PValue": self.alpha,
                                           "Correction": "FDR"},
                                          {"Type": "ToppCell",
                                           "PValue": self.alpha,
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

    def print_arguments(self):
        print("Arguments:")
        print("  > PICALO directory: {}".format(self.picalo_indir))
        print("  > Covariates: {}".format(self.covariates))
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Protein coding genes path: {}".format(self.protein_coding_path))
        print("  > Minimal gene expression: {}".format(self.min_avg_expression))
        print("  > Minimal correlation: {}".format(self.min_corr))
        print("  > Alpha: {}".format(self.alpha))
        print("  > Top-N: {}".format(self.top_n))
        print("  > Output directory {}".format(self.outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
