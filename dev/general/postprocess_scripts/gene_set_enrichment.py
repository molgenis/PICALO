#!/usr/bin/env python3

"""
File:         gene_set_enrichment.py
Created:      2022/02/24
Last Changed: 2022/02/25
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
import json
import os

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
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations.txt.gz \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD
    
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
        self.pic_start = getattr(arguments, 'pic_start')
        self.pic_end = getattr(arguments, 'pic_end')
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.enrichments = getattr(arguments, 'enrichments')
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
        parser.add_argument("-ps",
                            "--pic_start",
                            type=int,
                            default=1,
                            help="The PIC start index to analyse."
                                 "Default: 1.")
        parser.add_argument("-pe",
                            "--pic_end",
                            type=int,
                            default=5,
                            help="The PIC end index to analyse."
                                 "Default: 5.")
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            required=True,
                            help="The path to the gene correlations matrix.")
        parser.add_argument("-e",
                            "--enrichments",
                            nargs="*",
                            type=str,
                            choices=["toppgene", "genenetwork", "all"],
                            default=["all"],
                            help="Which gene enrichment analysis to perform.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading gene network files.")
        gn_matrix_df_list = []
        for name, matrix_subdir in self.gn_matrix_subdirs.items():
            gn_matrix_path = os.path.join(self.gn_matrix_basedir, matrix_subdir)
            if not os.path.exists(gn_matrix_path):
                continue
            gn_matrix_df = self.load_file(gn_matrix_path)
            gn_matrix_df.index = name + ":" + gn_matrix_df.index
            gn_matrix_df.index.name = None
            gn_matrix_df_list.append(gn_matrix_df)
        gn_matrix_df = pd.concat(gn_matrix_df_list, axis=0)
        print(gn_matrix_df)

        gn_annot_df_list = []
        for name, annotation_file in self.gn_annotation_files.items():
            gn_annotation_path = os.path.join(self.gn_annotation_basedir, annotation_file)
            if annotation_file == "" or not os.path.exists(gn_annotation_path):
                continue
            gn_annot_df = self.load_file(gn_annotation_path, header=None)
            gn_annot_df.index = name + ":" + gn_annot_df.index
            gn_annot_df.index.name = None
            gn_annot_df.columns = ["Annotation"]
            gn_annot_df_list.append(gn_annot_df)
        gn_annot_df = pd.concat(gn_annot_df_list, axis=0)

        # Subset the usefull info.
        gn_annot_df = gn_matrix_df.loc[:, []].merge(gn_annot_df, left_index=True, right_index=True, how="left")
        gn_annot_df.fillna("NA", inplace=True)
        gn_annot_df["category"] = [index.split(":")[0] for index in gn_annot_df.index]
        print(gn_annot_df)

        # Prep for the willcoxon tests.
        gn_ranked_df = gn_matrix_df.rank(axis=1)
        tie_term_df = pd.DataFrame(np.apply_along_axis(self.calc_tie_term, 1, gn_ranked_df),
                                   index=gn_ranked_df.index,
                                   columns=["tie_term"])

        # Save
        self.save_file(df=gn_annot_df, outpath=os.path.join(self.outdir, "gene_network_annotation.pkl"))
        self.save_file(df=gn_ranked_df, outpath=os.path.join(self.outdir, "gene_network_matrix_ranked.pkl"))
        self.save_file(df=tie_term_df, outpath=os.path.join(self.outdir, "gene_network_tie_term.pkl"))
        del gn_matrix_df
        # exit()

        print("Loading average gene expression.")
        df = self.load_file(self.avg_ge_path, header=0, index_col=0)
        df.columns = ["avg expression"]
        df.index = [x.split(".")[0] for x in df.index]
        print(df)

        if self.min_avg_expression is not None:
            print("\tFiltering on eQTLs with >{} average gene expression".format(self.min_avg_expression))
            pre_shape = df.shape[0]
            df = df.loc[df["avg expression"] > self.min_avg_expression, :]
            print("\t  Removed {:,} genes".format(pre_shape - df.shape[0]))

        print("Lookup gene symbols.")
        gi_df = self.toppgene_gene_lookup(df.index)
        gi_df.set_index("Submitted", inplace=True)
        df = df.merge(gi_df, left_index=True, right_index=True)
        print(df)

        print("Loading gene correlations.")
        gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=0)
        gene_corr_df.index = gene_corr_df["ProbeName"].str.split(".", n=1, expand=True)[0]
        gene_corr_df = gene_corr_df.loc[:, [col for col in gene_corr_df if col.startswith("PIC")]]
        gene_corr_df.columns = ["{} r".format(col) for col in gene_corr_df.columns]
        df = df.merge(gene_corr_df, left_index=True, right_index=True, how="left")
        print(df)

        print("Loading PICALO data.")
        pic_eqtl_list = []
        for i in range(1, 50):
            pic_eqtl_path = os.path.join(self.picalo_indir, "PIC_interactions", "PIC{}.txt.gz".format(i))
            if not os.path.exists(pic_eqtl_path):
                break

            pic_eqtl_df = self.load_file(pic_eqtl_path, header=0, index_col=None)
            if pic_eqtl_df.shape[0] != len(pic_eqtl_df["gene"].unique()):
                print("Non unique genes in PIC interaction output.")
            pic_eqtl_df.index = pic_eqtl_df["gene"].str.split(".", n=1, expand=True)[0]
            pic_eqtl_df["direction"] = ((pic_eqtl_df["beta-genotype"] * pic_eqtl_df["beta-interaction"]) > 0).map({True: "induces", False: "inhibits"})
            if len(pic_eqtl_list) == 0:
                pic_eqtl_df = pic_eqtl_df[["SNP", "FDR", "direction"]]
                pic_eqtl_df.columns = ["SNP", "PIC{} FDR".format(i), "PIC{} direction".format(i)]
            else:
                pic_eqtl_df = pic_eqtl_df[["FDR", "direction"]]
                pic_eqtl_df.columns = ["PIC{} FDR".format(i), "PIC{} direction".format(i)]
            pic_eqtl_list.append(pic_eqtl_df)
        pic_eqtl_df = pd.concat(pic_eqtl_list, axis=1)
        df = df.merge(pic_eqtl_df, left_index=True, right_index=True, how="left")
        print(df)

        print("Saving file.")
        self.save_file(df=df,
                       outpath=os.path.join(self.file_outdir, "info.txt.gz"))
        # exit()

        # print("Loading files.")
        # gn_annot_df = self.load_file(os.path.join(self.outdir, "gene_network_annotation.pkl"))
        # gn_ranked_df = self.load_file(os.path.join(self.outdir, "gene_network_matrix_ranked.pkl"))
        # tie_term_df = self.load_file(os.path.join(self.outdir, "gene_network_tie_term.pkl"))
        # print(gn_annot_df)
        # print(gn_rank_df)
        # print(tie_term_df)

        df = self.load_file(os.path.join(self.file_outdir, "info.txt.gz"))
        print(df)

        for pic_id in range(self.pic_start, self.pic_end + 1):
            print("### Analyzing PIC{} ###".format(pic_id))

            # Create output directory.
            pic_file_outdir = os.path.join(self.file_outdir, 'PIC{}'.format(pic_id))
            if not os.path.exists(pic_file_outdir):
                os.makedirs(pic_file_outdir)

            for correlation_direction in ["positive", "negative"]:
                correlation_mask = None
                if correlation_direction == "positive":
                    correlation_mask = df["PIC{} r".format(pic_id)] > 0
                elif correlation_direction == "negative":
                    correlation_mask = df["PIC{} r".format(pic_id)] < 0
                else:
                    print("huh")
                    exit()

                # Select, filter, and sort.
                subset_df = df.loc[correlation_mask, ["avg expression",
                                                      "OfficialSymbol",
                                                      "Entrez",
                                                      "PIC{} r".format(pic_id),
                                                      "PIC{} FDR".format(pic_id),
                                                      "PIC{} direction".format(pic_id),
                                                      ]].copy()
                subset_df.columns = ["avg expression", "symbol", "entrez", "correlation", "FDR", "direction"]
                subset_df["abs correlation"] = subset_df["correlation"].abs()
                subset_df = subset_df.loc[subset_df["abs correlation"] >= self.min_corr, :]
                subset_df.sort_values(by="abs correlation", ascending=False, inplace=True)

                if subset_df.shape[0] == 0:
                    continue

                print("\t{} correlation".format(correlation_direction))
                if subset_df.shape[0] >= 50:
                    print("\tAnalyzing {:,} genes".format(subset_df.iloc[:self.top_n, :].shape[0]))
                else:
                    print("\tAnalyzing {:,} genes: {}".format(subset_df.iloc[:self.top_n, :].shape[0], ", ".join(subset_df.index[:self.top_n])))

                # Saving files.
                self.save_file(df=subset_df.iloc[:self.top_n, :],
                               outpath=os.path.join(pic_file_outdir, "toppgene_enrichment_PIC{}_{}_data.txt.gz".format(pic_id, correlation_direction)))

                # ToppGene enrichment.
                if "toppgene" in self.enrichments or "all" in self.enrichments:
                    enrich_df = self.toppgene_func_enrichment(
                        entrez_ids=subset_df["entrez"][:self.top_n]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(pic_file_outdir, "toppgene_enrichment_PIC{}_{}.txt.gz".format(pic_id, correlation_direction)))
                    print("")
                    del enrich_df

                # Genenetwork enrichment.
                if "genenetwork" in self.enrichments or "all" in self.enrichments:
                    enrich_df = self.genenetwork_enrichment(
                        ranked_df=gn_ranked_df,
                        annot_df=gn_annot_df,
                        tie_term_df=tie_term_df,
                        genes=subset_df.index[:self.top_n]
                    )
                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(pic_file_outdir, "genenetwork_enrichment_PIC{}_{}.txt.gz".format(pic_id, correlation_direction)))
                    print("")
                    del enrich_df
                #
                # # Filter on interaction direction.
                # for interaction_direction in ["inhibits", "induces"]:
                #     subset_direction_df = subset_df.loc[(subset_df["FDR"] < 0.05) & (subset_df["direction"] == interaction_direction), :].copy()
                #     if subset_direction_df.shape[0] == 0:
                #         continue
                #
                #     print("\t{} correlation, context {}".format(correlation_direction, interaction_direction))
                #     print("\tAnalyzing {:,} genes: {}".format(subset_direction_df.shape[0], ", ".join(subset_direction_df.index)))
                #
                #     # Saving files.
                #     self.save_file(df=subset_direction_df,
                #                    outpath=os.path.join(pic_file_outdir, "toppgene_enrichment_PIC{}_{}_{}_data.txt.gz".format(pic_id, correlation_direction, interaction_direction)))
                #     print("")
                #
                #     # ToppGene enrichment.
                #     if "toppgene" in self.enrichments or "all" in self.enrichments:
                #         enrich_df = self.toppgene_func_enrichment(
                #             entrez_ids=subset_direction_df["entrez"]
                #         )
                #         self.save_file(df=enrich_df,
                #                        outpath=os.path.join(pic_file_outdir, "toppgene_enrichment_PIC{}_{}_{}.txt.gz".format(pic_id, correlation_direction, interaction_direction)))
                #         print("")
                #         del enrich_df
                #
                #     # Genenetwork enrichment.
                #     if "genenetwork" in self.enrichments or "all" in self.enrichments:
                #         enrich_df = self.genenetwork_enrichment(
                #             ranked_df=gn_ranked_df,
                #             annot_df=gn_annot_df,
                #             tie_term_df=tie_term_df,
                #             genes=subset_direction_df.index
                #         )
                #         self.save_file(df=enrich_df,
                #                        outpath=os.path.join(pic_file_outdir, "genenetwork_enrichment_PIC{}_{}_{}.txt.gz".format(pic_id, correlation_direction, interaction_direction)))
                #         print("")
                #         del enrich_df

            print("")

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

        data = json.dumps({"Genes": list(entrez_ids), "Categories": [{"Type": "ToppCell", "PValue": 0.05, "Correction": "FDR"}]})
        response = requests.post("https://toppgene.cchmc.org/API/enrich",
                                 headers={'Content-type': 'application/json',},
                                 data=data)
        result = json.loads(response.text)
        df = pd.DataFrame(result["Annotations"])
        if df.shape[0] > 0:
            df["Genes"] = [", ".join([gene["Symbol"] for gene in value]) for value in df["Genes"]]
            df.drop(["Category", "ID"], axis=1, inplace=True)

        print("\tToppCell:")
        for i, (_, row) in enumerate(df.iterrows()):
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

        print("\tGeneNetwork:")
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
        print("  > PICs: {}-{}".format(self.pic_start, self.pic_end))
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Enrichments: {}".format(", ".join(self.enrichments)))
        print("  > Output directory {}".format(self.outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
