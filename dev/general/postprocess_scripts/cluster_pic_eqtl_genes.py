#!/usr/bin/env python3

"""
File:         cluster_pic_eqtl_genes.py
Created:      2022/02/23
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
from colour import Color
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import fastcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering

# Local application imports.

# Metadata
__program__ = "Cluster PIC eQTL Genes"
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
./cluster_pic_eqtl_genes.py -h

### BIOS ###

./cluster_pic_eqtl_genes.py \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs \
    -gn /groups/umcg-wijmenga/tmp01/projects/depict2/depict2_bundle/reference_datasets/human_b37/pathway_databases/gene_network_v2_1/gene_coregulation_165_eigenvectors_protein_coding.txt.gz \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-04-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-AllPICsCorrected-SP140AsCov-GeneExpressionFNPD_gene_correlations.txt.gz \
    -gi /groups/umcg-bios/tmp01/projects/PICALO/data/ArrayAddressToSymbol.txt.gz \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_indir = getattr(arguments, 'picalo')
        self.pic_start = getattr(arguments, 'pic_start')
        self.pic_end = getattr(arguments, 'pic_end')
        self.gene_network_path = getattr(arguments, 'gene_network')
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.avg_ge_path = getattr(arguments, 'average_gene_expression')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.file_outdir = os.path.join(base_dir, 'cluster_pic_eqtl_genes')
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
        parser.add_argument("-gn",
                            "--gene_network",
                            type=str,
                            required=True,
                            help="The path to the gene network matrix.")
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            default=None,
                            help="The path to the gene correlations matrix.")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene info matrix.")
        parser.add_argument("-avge",
                            "--average_gene_expression",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the average gene expression "
                                 "matrix.")
        parser.add_argument("-mae",
                            "--min_avg_expression",
                            type=float,
                            default=None,
                            help="The minimal average expression of a gene."
                                 "Default: None.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading PICALO data.")
        snp_stats_df_list = []
        for i in range(1, 50):
            pic = "PIC{}".format(i)
            pic_eqtl_path = os.path.join(self.picalo_indir, "PIC_interactions", "PIC{}.txt.gz".format(i))
            if not os.path.exists(pic_eqtl_path):
                break

            pic_eqtl_df = self.load_file(pic_eqtl_path, header=0, index_col=None)
            pic_eqtl_df.index = pic_eqtl_df.index.astype(str) + "_" + pic_eqtl_df["gene"].str.split(".", n=1, expand=True)[0]
            pic_eqtl_df["direction"] = ((pic_eqtl_df["beta-genotype"] * pic_eqtl_df["beta-interaction"]) > 0).map({True: "induces", False: "inhibits"})

            pic_eqtl_df["N"] = 1

            if i == 1:
                unique_snp_counts_df = pic_eqtl_df[["gene", "N"]].groupby("gene").sum()
                snp_stats_df_list.append(unique_snp_counts_df)

            signif_unique_snp_counts_df = pic_eqtl_df.loc[pic_eqtl_df["FDR"] < 0.05, ["gene", "N"]].groupby("gene").sum()
            signif_unique_snp_counts_df.columns = [pic]
            snp_stats_df_list.append(signif_unique_snp_counts_df)
        snp_stats_df = pd.concat(snp_stats_df_list, axis=1)
        snp_stats_df.fillna(0, inplace=True)
        print(snp_stats_df)
        self.save_file(df=snp_stats_df,
                       outpath=os.path.join(self.file_outdir, "gene_stats_df.txt.gz".format(self.out_filename)),
                       index=True)
        exit()

        # print("Loading gene network data.")
        # gn_df = self.load_file(self.gene_network_path, header=0, index_col=0, nrows=None)
        # gn_annot_df = pd.DataFrame(np.nan, index=gn_df.columns, columns=[])
        #
        # # Load data.
        # print("Loading PICALO data.")
        # pic_eqtl_list = []
        # for i in range(1, 50):
        #     pic_eqtl_path = os.path.join(self.picalo_indir, "PIC_interactions", "PIC{}.txt.gz".format(i))
        #     if not os.path.exists(pic_eqtl_path):
        #         break
        #
        #     pic_eqtl_df = self.load_file(pic_eqtl_path, header=0, index_col=None)
        #     pic_eqtl_df.index = pic_eqtl_df.index.astype(str) + "_" + pic_eqtl_df["gene"].str.split(".", n=1, expand=True)[0]
        #     pic_eqtl_df["direction"] = ((pic_eqtl_df["beta-genotype"] * pic_eqtl_df["beta-interaction"]) > 0).map({True: "induces", False: "inhibits"})
        #     if len(pic_eqtl_list) == 0:
        #         pic_eqtl_df = pic_eqtl_df[["SNP", "gene", "FDR", "direction"]]
        #         pic_eqtl_df.columns = ["SNP", "gene", "PIC{} FDR".format(i), "PIC{} direction".format(i)]
        #     else:
        #         pic_eqtl_df = pic_eqtl_df[["FDR", "direction"]]
        #         pic_eqtl_df.columns = ["PIC{} FDR".format(i), "PIC{} direction".format(i)]
        #     pic_eqtl_list.append(pic_eqtl_df)
        # pic_eqtl_df = pd.concat(pic_eqtl_list, axis=1)
        #
        # print("Adding gene symbols.")
        # gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        # gene_dict = dict(zip(gene_info_df["ArrayAddress"], gene_info_df["Symbol"]))
        # pic_eqtl_df.insert(2, "gene symbol", pic_eqtl_df["gene"].map(gene_dict))
        # gn_annot_df["gene symbol"] = gn_annot_df.index.map(gene_dict)
        # del gene_info_df, gene_dict
        #
        # if self.avg_ge_path is not None:
        #     print("Adding average expression.")
        #     avg_ge_df = self.load_file(self.avg_ge_path, header=0, index_col=0)
        #     avg_ge_dict = dict(zip(avg_ge_df.index, avg_ge_df["average"]))
        #     pic_eqtl_df.insert(3, "avg expression", pic_eqtl_df["gene"].map(avg_ge_dict))
        #     gn_annot_df["avg expression"] = gn_annot_df.index.map(avg_ge_dict)
        #
        #     if self.min_avg_expression is not None:
        #         print("\tFiltering on eQTLs with >{} average gene expression".format(self.min_avg_expression))
        #         pre_shape = pic_eqtl_df.shape[0]
        #         pic_eqtl_df = pic_eqtl_df.loc[pic_eqtl_df["avg expression"] > self.min_avg_expression, :]
        #         print("\t  Removed {:,} eQTLs".format(pic_eqtl_df.shape[0] - pre_shape))
        #
        #         pre_shape = pic_eqtl_df.shape[0]
        #         gn_annot_df = gn_annot_df.loc[gn_annot_df["avg expression"] > self.min_avg_expression, :]
        #         gn_df = gn_df.loc[gn_annot_df.index, gn_annot_df.index]
        #         print("\t  Removed {:,} gene network genes".format(gn_annot_df.shape[0] - pre_shape))
        #     del avg_ge_df, avg_ge_dict
        #
        # if self.gene_correlations_path is not None:
        #     print("Adding gene correlations.")
        #     gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=0)
        #     for i in range(1, 50):
        #         if "PIC{}".format(i) in gene_corr_df.columns:
        #             gene_corr_dict = dict(zip(gene_corr_df["ProbeName"], gene_corr_df["PIC{}".format(i)]))
        #             pic_eqtl_df["PIC{} r".format(i)] = pic_eqtl_df["gene"].map(gene_corr_dict)
        #             gn_annot_df["PIC{} r".format(i)] = gn_annot_df.index.map(gene_corr_dict)
        #     del gene_corr_df
        # print(pic_eqtl_df)
        # print(gn_annot_df)
        #
        # print("Selecting eQTL genes")
        # overlapping_genes = [gene for gene in pic_eqtl_df["gene"] if gene in gn_df.index]
        # print("\tSelecting {:,} genes".format(len(overlapping_genes)))
        # pic_eqtl_df = pic_eqtl_df.loc[pic_eqtl_df["gene"].isin(overlapping_genes), :]
        # gn_df = gn_df.loc[overlapping_genes, overlapping_genes]
        # gn_df.index = pic_eqtl_df.index
        # print(pic_eqtl_df)
        # print(gn_df)
        #
        # print("Saving file.")
        # self.save_file(df=pic_eqtl_df,
        #                outpath=os.path.join(self.file_outdir, "pic_eqtl_annotation_df.txt.gz".format(self.out_filename)),
        #                index=True)
        # self.save_file(df=gn_annot_df,
        #                outpath=os.path.join(self.file_outdir, "gene_network_annotation_df.txt.gz".format(self.out_filename)),
        #                index=True)
        # self.save_file(df=gn_df,
        #                outpath=os.path.join(self.file_outdir, "gene_network_coregulations.pkl".format(self.out_filename)))
        # exit()

        pic_eqtl_df = self.load_file(os.path.join(self.file_outdir, "pic_eqtl_annotation_df.txt.gz".format(self.out_filename)))
        gn_annot_df = self.load_file(os.path.join(self.file_outdir, "gene_network_annotation_df.txt.gz".format(self.out_filename)))
        gn_df = self.load_file(os.path.join(self.file_outdir, "gene_network_coregulations.pkl".format(self.out_filename)))

        pic_eqtl_df = pic_eqtl_df.iloc[:500, :]
        gn_annot_df = gn_annot_df.iloc[:500, :]
        gn_df = gn_df.iloc[:500, :500]
        print(pic_eqtl_df)
        print(gn_annot_df)
        print(gn_df)

        # print("Printing groups.")
        # print("\nBackground: {}\n".format(", ".join(pic_eqtl_df["gene"])))
        # for i in range(self.pic_start, self.pic_end + 1):
        #     for correlation_direction in ["positive", "negative"]:
        #         for interaction_direction in ["inhibits", "induces"]:
        #             correlation_mask = None
        #             if correlation_direction == "positive":
        #                 correlation_mask = pic_eqtl_df["PIC{} r".format(i)] > 0
        #             elif correlation_direction == "negative":
        #                 correlation_mask = pic_eqtl_df["PIC{} r".format(i)] < 0
        #             else:
        #                 print("huh")
        #                 exit()
        #
        #             signif_pic_eqlt_genes = pic_eqtl_df.loc[
        #                     (pic_eqtl_df["PIC{} FDR".format(i)] < 0.05) & (correlation_mask) & (pic_eqtl_df["PIC{} direction".format(i)] == interaction_direction), ["SNP", "gene", "gene symbol", "PIC{} FDR".format(i), "PIC{} r".format(i), "PIC{} direction".format(i)]].copy()
        #             print("\nPIC{} - {} - {} [N={:,}]: {}\n".format(i, correlation_direction, interaction_direction, signif_pic_eqlt_genes.shape[0], ", ".join(signif_pic_eqlt_genes["gene"])))
        # exit()

        print("Create annotations")
        row_colors_df = self.prep_row_colors(annot_df=pic_eqtl_df)
        col_colors_df = self.prep_col_colors(annot_df=gn_annot_df)

        print("Visualising")
        for i in range(self.pic_start, self.pic_end + 1):
            # Create output directory.
            pic_plot_outdir = os.path.join(self.plot_outdir, 'PIC{}'.format(i))
            if not os.path.exists(pic_plot_outdir):
                os.makedirs(pic_plot_outdir)

            # Select the eQTLs that have a sginificant interaction with the PIC.
            signif_pic_eqlt_genes = pic_eqtl_df.loc[pic_eqtl_df["PIC{} FDR".format(i)] < 0.05, ["gene"]].copy()

            # Subset those genes.
            pic_gn_df = gn_df.loc[signif_pic_eqlt_genes.index, :].copy()
            signif_row_colors_df = row_colors_df.loc[signif_pic_eqlt_genes.index, ["avg expr", "PIC{} direction".format(i), "PIC{} r".format(i)]].copy()
            signif_row_colors_df.columns = ["avg expr", "direction", "correlation"]
            pic_col_colors_df = col_colors_df.loc[:, ["avg expr", "PIC{} r".format(i)]].copy()
            pic_col_colors_df.columns = ["avg expr", "correlation"]

            # Cluster.
            pic_dissimilarity_gn_df = self.zscore_to_dissimilarity(df=pic_gn_df)
            row_linkage = self.cluster(pic_dissimilarity_gn_df)

            self.plot(df=pic_dissimilarity_gn_df,
                      row_colors=signif_row_colors_df,
                      row_linkage=row_linkage,
                      col_colors=signif_row_colors_df,
                      col_linkage=row_linkage,
                      name="PIC{}_gene_network_clustermap".format(i),
                      outdir=pic_plot_outdir)
            exit()

            for direction in ["inhibits", "induces", "positive", "negative"]:
                signif_pic_eqlt_genes = None
                if direction in ["inhibits", "induces"]:
                    signif_pic_eqlt_genes = pic_eqtl_df.loc[(pic_eqtl_df["PIC{} FDR".format(i)] < 0.05) & (pic_eqtl_df["PIC{} direction".format(i)] == direction), ["PIC{} FDR".format(i), "PIC{} direction".format(i), "gene"]].copy()
                elif direction == "positive":
                    signif_pic_eqlt_genes = pic_eqtl_df.loc[(pic_eqtl_df["PIC{} FDR".format(i)] < 0.05) & (pic_eqtl_df["PIC{} r".format(i)] > 0), ["PIC{} FDR".format(i), "PIC{} r".format(i), "gene"]].copy()
                elif direction == "negative":
                    signif_pic_eqlt_genes = pic_eqtl_df.loc[(pic_eqtl_df["PIC{} FDR".format(i)] < 0.05) & (pic_eqtl_df["PIC{} r".format(i)] < 0), ["PIC{} FDR".format(i), "PIC{} r".format(i), "gene"]].copy()
                else:
                    print("huh")
                    exit()
                print(signif_pic_eqlt_genes)

                # Subset those genes.
                pic_gn_df = gn_df.loc[signif_pic_eqlt_genes.index, :].copy()
                signif_row_colors_df = row_colors_df.loc[signif_pic_eqlt_genes.index, ["avg expr", "PIC{} direction".format(i), "PIC{} r".format(i)]].copy()
                signif_row_colors_df.columns = ["avg expr", "direction", "correlation"]
                pic_col_colors_df = col_colors_df.loc[:, ["avg expr", "PIC{} r".format(i)]].copy()
                pic_col_colors_df.columns = ["avg expr", "correlation"]

                self.plot(df=pic_gn_df,
                          row_colors=signif_row_colors_df,
                          col_colors=pic_col_colors_df,
                          row_cluster=True,
                          col_cluster=True,
                          name="PIC{}_{}_gene_network_clustermap".format(i, direction),
                          outdir=pic_plot_outdir)

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
    def save_file(df, outpath, header=True, index=False, sep="\t"):
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
    def zscore_to_dissimilarity(df):
        m = df.to_numpy()

        # normalize between 0 and 1.
        m = (m - np.min(m)) / np.ptp(m)

        # inverse.
        m = 1 - m

        return pd.DataFrame(m, index=df.index, columns=df.columns)

    @staticmethod
    def cluster(df, metric="euclidean", method="centroid"):
        euclidean = metric == 'euclidean' and method in ('centroid', 'median', 'ward')
        if euclidean or method == 'single':
            linkage = fastcluster.linkage_vector(df,
                                                 method=method,
                                                 metric=metric)
        else:
            linkage = fastcluster.linkage(df,
                                          method=method,
                                          metric=metric)

        return linkage

    # @staticmethod
    # def cluster(df, affinity='precomputed', n_clusters=8):
    #     model = SpectralClustering(n_clusters=n_clusters,
    #                                affinity=affinity).fit(df.abs())
    #
    #     return model

    def prep_row_colors(self, annot_df, a=0.05):
        row_color_data = {}

        avg_expr_palette = self.create_palette_gradient(annot_df["avg expression"])
        row_color_data["avg expr"] = [avg_expr_palette[round(avg_expr, 2)] for avg_expr in annot_df["avg expression"]]

        for i in range(self.pic_start, self.pic_end + 1):
            # FDR column.
            pic_column = "PIC{} FDR".format(i)
            if pic_column in annot_df.columns:
                row_color_data[pic_column] = [Color("#000000").rgb if fdr_value <= a else Color("#FFFFFF").rgb for fdr_value in annot_df[pic_column]]

            # direction column.
            pic_column = "PIC{} direction".format(i)
            palette = {"induces": "#ac3e40", "inhibits": "#2f6ebc"}
            if pic_column in annot_df.columns:
                row_color_data[pic_column] = [Color(palette[direction]).rgb if direction in palette.keys() else Color("#000000").rgb for direction in annot_df[pic_column]]

            # r column.
            pic_column = "PIC{} r".format(i)
            if pic_column in annot_df.columns:
                pic_r_palette = self.create_palette_diverging(annot_df[pic_column])
                row_color_data[pic_column] = [pic_r_palette[round(pic_r, 2)] for pic_r in annot_df[pic_column]]

        return pd.DataFrame(row_color_data, index=annot_df.index)

    def prep_col_colors(self, annot_df):
        col_color_data = {}

        avg_expr_palette = self.create_palette_gradient(annot_df["avg expression"])
        col_color_data["avg expr"] = [avg_expr_palette[round(avg_expr, 2)] for avg_expr in annot_df["avg expression"]]

        for i in range(self.pic_start, self.pic_end + 1):
            # r column.
            pic_column = "PIC{} r".format(i)
            if pic_column in annot_df.columns:
                pic_r_palette = self.create_palette_diverging(annot_df[pic_column])
                col_color_data[pic_column] = [pic_r_palette[round(pic_r, 2)] for pic_r in annot_df[pic_column]]

        return pd.DataFrame(col_color_data, index=annot_df.index)

    @staticmethod
    def create_palette_gradient(x):
        min_value = x.min().round(2)
        max_value = x.max().round(2)
        value_range = math.ceil((max_value - min_value) * 100) + 2
        colors = sns.color_palette("Blues", value_range)
        values = [x for x in np.arange(min_value, max_value + 0.01, 0.01)]
        palette = {}
        for val, col in zip(values, colors):
            palette[round(val, 2)] = col

        return palette

    @staticmethod
    def create_palette_diverging(x):
        min_value = x.min().round(2)
        max_value = x.max().round(2)
        negative_range = math.ceil(min_value * -100)
        positive_range = math.ceil(max_value * 100)
        colors = sns.color_palette("vlag", (negative_range * 2) + 1)[:negative_range] + sns.color_palette("vlag", (positive_range * 2) + 1)[positive_range:]
        values = [x for x in np.arange(min_value, max_value + 0.01, 0.01)]
        palette = {}
        for val, col in zip(values, colors):
            palette[round(val, 2)] = col

        return palette

    def plot(self, df, vmin=None, vmax=None, center=0, row_colors=None,
             row_cluster=False, row_linkage=None, col_colors=None,
             col_cluster=False, col_linkage=None, yticklabels=False,
             xticklabels=False, name="plot", outdir=None):
        if outdir is None:
            outdir = self.plot_outdir
        sns.set(color_codes=True)

        if row_linkage is not None and not row_cluster:
            row_cluster = True
        if col_linkage is not None and not col_cluster:
            col_cluster = True

        g = sns.clustermap(df,
                           cmap=sns.diverging_palette(246, 24, as_cmap=True),
                           vmin=vmin,
                           vmax=vmax,
                           center=center,
                           row_colors=row_colors,
                           row_linkage=row_linkage,
                           row_cluster=row_cluster,
                           col_colors=col_colors,
                           col_linkage=col_linkage,
                           col_cluster=col_cluster,
                           yticklabels=yticklabels,
                           xticklabels=xticklabels,
                           figsize=(24, 12)
                           )
        plt.setp(g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(),fontsize=10))
        g.fig.subplots_adjust(bottom=0.05, top=0.7)
        plt.tight_layout()

        outpath = os.path.join(outdir, "{}.png".format(name))
        g.savefig(outpath)
        plt.close()
        print("\tSaved file '{}'.".format(os.path.basename(outpath)))

    def print_arguments(self):
        print("Arguments:")
        print("  > PICALO directory: {}".format(self.picalo_indir))
        print("  > PICs: {}-{}".format(self.pic_start, self.pic_end))
        print("  > Gene network path: {}".format(self.gene_network_path))
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Gene info: {}".format(self.gene_info_path))
        print("  > Average gene expression path: {}".format(self.avg_ge_path))
        print("  > Minimal gene expression: {}".format(self.min_avg_expression))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
