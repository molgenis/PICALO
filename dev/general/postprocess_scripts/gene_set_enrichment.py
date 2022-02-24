#!/usr/bin/env python3

"""
File:         gene_set_enrichment.py
Created:      2022/02/24
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
import json
import os

# Third party imports.
import pandas as pd
import requests

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
        self.out_filename = getattr(arguments, 'outfile')

        self.top_n = 200
        self.min_corr = 0.1

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.file_outdir = os.path.join(base_dir, 'gene_set_enrichment')
        self.plot_outdir = os.path.join(self.file_outdir, 'plot')
        for outdir in [self.plot_outdir, self.file_outdir]:
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
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # print("Loading average gene expression.")
        # df = self.load_file(self.avg_ge_path, header=0, index_col=0)
        # df.columns = ["avg expression"]
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
        # df = df.merge(gi_df, left_index=True, right_index=True)
        # print(df)
        #
        # print("Loading gene correlations.")
        # gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=0)
        # gene_corr_df.index = gene_corr_df["ProbeName"].str.split(".", n=1, expand=True)[0]
        # gene_corr_df = gene_corr_df.loc[:, [col for col in gene_corr_df if col.startswith("PIC")]]
        # gene_corr_df.columns = ["{} r".format(col) for col in gene_corr_df.columns]
        # df = df.merge(gene_corr_df, left_index=True, right_index=True, how="left")
        # print(df)
        #
        # print("Loading PICALO data.")
        # pic_eqtl_list = []
        # for i in range(1, 50):
        #     pic_eqtl_path = os.path.join(self.picalo_indir, "PIC_interactions", "PIC{}.txt.gz".format(i))
        #     if not os.path.exists(pic_eqtl_path):
        #         break
        #
        #     pic_eqtl_df = self.load_file(pic_eqtl_path, header=0, index_col=None)
        #     if pic_eqtl_df.shape[0] != len(pic_eqtl_df["gene"].unique()):
        #         print("Non unique genes in PIC interaction output.")
        #     pic_eqtl_df.index = pic_eqtl_df["gene"].str.split(".", n=1, expand=True)[0]
        #     pic_eqtl_df["direction"] = ((pic_eqtl_df["beta-genotype"] * pic_eqtl_df["beta-interaction"]) > 0).map({True: "induces", False: "inhibits"})
        #     if len(pic_eqtl_list) == 0:
        #         pic_eqtl_df = pic_eqtl_df[["SNP", "FDR", "direction"]]
        #         pic_eqtl_df.columns = ["SNP", "PIC{} FDR".format(i), "PIC{} direction".format(i)]
        #     else:
        #         pic_eqtl_df = pic_eqtl_df[["FDR", "direction"]]
        #         pic_eqtl_df.columns = ["PIC{} FDR".format(i), "PIC{} direction".format(i)]
        #     pic_eqtl_list.append(pic_eqtl_df)
        # pic_eqtl_df = pd.concat(pic_eqtl_list, axis=1)
        # df = df.merge(pic_eqtl_df, left_index=True, right_index=True, how="left")
        # print(df)
        #
        # print("Saving file.")
        # self.save_file(df=df,
        #                outpath=os.path.join(self.file_outdir, "{}_info.txt.gz".format(self.out_filename)),
        #                index=False)
        # exit()

        df = self.load_file(os.path.join(self.file_outdir, "{}_info.txt.gz".format(self.out_filename)), index_col=None)
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

                enrich_df = self.toppgene_func_enrichment(subset_df["entrez"][:self.top_n])

                print("\t{} correlation".format(correlation_direction))
                print("\t  Analyzing {:,} genes".format(len(subset_df["entrez"][:self.top_n])))
                for i, (index, row) in enumerate(enrich_df.iterrows()):
                    if i > 4:
                        break
                    print("\t  {}: [p-value: {:.2e}]  {}".format(index, row["PValue"], row["Name"]))

                self.save_file(df=enrich_df,
                               outpath=os.path.join(self.file_outdir, "{}_toppgene_enrichment_PIC{}_{}.txt.gz".format(self.out_filename, pic_id, correlation_direction)),
                               index=False)
                print("")

                # Filter on interaction direction.
                for interaction_direction in ["inhibits", "induces"]:
                    subset_direction_df = subset_df.loc[(subset_df["FDR"] < 0.05) & (subset_df["direction"] == interaction_direction), :].copy()
                    if subset_direction_df.shape[0] == 0:
                        continue

                    enrich_df = self.toppgene_func_enrichment(subset_direction_df["entrez"])

                    print("\t{} correlation, context {}".format(correlation_direction, interaction_direction))
                    print("\t  Analyzing {:,} genes".format(len(subset_direction_df["entrez"])))
                    for i, (index, row) in enumerate(enrich_df.iterrows()):
                        if i > 4:
                            break
                        print("\t  {}: [p-value: {:.2e}]  {}".format(index, row["PValue"], row["Name"]))

                    self.save_file(df=enrich_df,
                                   outpath=os.path.join(self.file_outdir,
                                                        "{}_toppgene_enrichment_PIC{}_{}_{}.txt.gz".format(self.out_filename, pic_id, correlation_direction, interaction_direction)),
                                   index=False)
                    print("")
            print("")

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def toppgene_gene_lookup(genes):
        data = json.dumps({"Symbols": list(genes)})
        response = requests.post("https://toppgene.cchmc.org/API/lookup",
                                 headers={'Content-type': 'application/json',},
                                 data=data)
        result = json.loads(response.text)
        return pd.DataFrame(result["Genes"])


    @staticmethod
    def save_file(df, outpath, header=True, index=False, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    @staticmethod
    def toppgene_func_enrichment(entrez_ids):
        data = json.dumps({"Genes": list(entrez_ids), "Categories": [{"Type": "ToppCell", "PValue": 0.05, "Correction": "FDR"}]})
        response = requests.post("https://toppgene.cchmc.org/API/enrich",
                                 headers={'Content-type': 'application/json',},
                                 data=data)
        result = json.loads(response.text)
        df = pd.DataFrame(result["Annotations"])
        df["Genes"] = [", ".join([gene["Symbol"] for gene in value]) for value in df["Genes"]]
        df.drop(["Category", "ID"], axis=1, inplace=True)
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > Average gene expression path: {}".format(self.avg_ge_path))
        print("  > Minimal gene expression: {}".format(self.min_avg_expression))
        print("  > PICALO directory: {}".format(self.picalo_indir))
        print("  > PICs: {}-{}".format(self.pic_start, self.pic_end))
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
