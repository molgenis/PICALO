#!/usr/bin/env python3

"""
File:         correlate_pics_with_eqtl_genes.py
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
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
import fastcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Local application imports.

# Metadata
__program__ = "Correlate PICs with eQTL Genes"
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
./correlate_pics_with_eqtl_genes.py -h

### BIOS ###

./correlate_pics_with_eqtl_genes.py \
    -pi /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs \
    -e /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/force_normalise_matrix/2021-12-10-gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS_ForceNormalised.txt.gz \
    -gi /groups/umcg-bios/tmp01/projects/PICALO/data/ArrayAddressToSymbol.txt.gz \
    -avge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/calc_avg_gene_expression/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -pa /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD

### MetaBrain ###

./correlate_components_with_genes.py \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs \
    -e /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/force_normalise_matrix/2021-12-10-MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.txt.gz \
    -gi /groups/umcg-biogen/tmp01/annotation/gencode.v32.primary_assembly.annotation.collapsedGenes.ProbeAnnotation.TSS.txt.gz \
    -avge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/calc_avg_gene_expression/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.AverageExpression.txt.gz \
    -mae 1 \
    -o 2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-PICs-eQTLGeneCorrelationsFNPD

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_indir = getattr(arguments, 'picalo')
        self.expr_path = getattr(arguments, 'expression')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.avg_ge_path = getattr(arguments, 'average_gene_expression')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.file_outdir = os.path.join(base_dir, 'correlate_pics_with_genes')
        self.plot_outdir = os.path.join(self.file_outdir, 'plot')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-e",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the gene expression matrix.")
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
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-pa",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
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
        pics_df = self.load_file(os.path.join(self.picalo_indir, "PICs.txt.gz"), header=0, index_col=0)
        # pic_eqtl_list = []
        # for i in range(1, 50):
        #     pic_eqtl_path = os.path.join(self.picalo_indir, "PIC_interactions", "PIC{}.txt.gz".format(i))
        #     if not os.path.exists(pic_eqtl_path):
        #         break
        #
        #     pic_eqtl_df = self.load_file(pic_eqtl_path, header=0, index_col=None)
        #     pic_eqtl_df.index = pic_eqtl_df["SNP"] + "_" + pic_eqtl_df["gene"]
        #     if len(pic_eqtl_list) == 0:
        #         pic_eqtl_df = pic_eqtl_df[["SNP", "gene", "FDR"]]
        #         pic_eqtl_df.columns = ["SNP", "gene", "PIC{} FDR".format(i)]
        #     else:
        #         pic_eqtl_df = pic_eqtl_df[["FDR"]]
        #         pic_eqtl_df.columns = ["PIC{} FDR".format(i)]
        #     pic_eqtl_list.append(pic_eqtl_df)
        # pic_eqtl_df = pd.concat(pic_eqtl_list, axis=1)
        #
        # print("Adding gene symbols.")
        # gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        # gene_dict = dict(zip(gene_info_df["ArrayAddress"], gene_info_df["Symbol"]))
        # pic_eqtl_df.insert(2, "gene symbol", pic_eqtl_df["gene"].map(gene_dict))
        # del gene_info_df, gene_dict
        #
        # if self.avg_ge_path is not None:
        #     print("Adding average expression.")
        #     avg_ge_df = self.load_file(self.avg_ge_path, header=0, index_col=0)
        #     avg_ge_dict = dict(zip(avg_ge_df.index, avg_ge_df["average"]))
        #     pic_eqtl_df.insert(3, "avg expression", pic_eqtl_df["gene"].map(avg_ge_dict))
        #
        #     if self.min_avg_expression is not None:
        #         print("\tFiltering on eQTLs with >{} average gene expression".format(self.min_avg_expression))
        #         pre_shape = pic_eqtl_df.shape[0]
        #         pic_eqtl_df = pic_eqtl_df.loc[pic_eqtl_df["avg expression"] > self.min_avg_expression, :]
        #         print("\t  Removed {:,} eQTLs".format(pic_eqtl_df.shape[0] - pre_shape))
        #     del avg_ge_df, avg_ge_dict
        # print(pics_df)
        # print(pic_eqtl_df)
        #
        # print("Loading expression data.")
        # expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=None)
        #
        # print("Selecting eQTL genes")
        # eqtl_genes = list(pic_eqtl_df["gene"])
        # print("\tSelecting {} genes".format(len(eqtl_genes)))
        # expr_df = expr_df.loc[eqtl_genes, :]
        # print(expr_df)
        #
        # print("Pre-processing data.")
        # # Make sure order is the same.
        # samples = set(pics_df.columns.tolist()).intersection(set(expr_df.columns.tolist()))
        # pics_df = pics_df.loc[:, samples]
        # expr_df = expr_df.loc[:, samples]
        #
        # # Safe the indices.
        # pic_labels = pics_df.index.tolist()
        #
        # # Convert to numpy.
        # pics_m = pics_df.to_numpy()
        # expr_m = expr_df.to_numpy()
        # del pics_df
        #
        # # Calculate correlating.
        # print("Correlating.")
        # corr_m = np.corrcoef(pics_m, expr_m)[:pics_m.shape[0], pics_m.shape[0]:]
        # corr_df = pd.DataFrame(corr_m,
        #                        index=["{} r".format(pic) for pic in pic_labels],
        #                        columns=pic_eqtl_df.index.tolist()).T
        #
        # print("Post-processing data.")
        # df = pic_eqtl_df.merge(corr_df, left_index=True, right_index=True)
        # print(df)
        #
        # print("Saving file.")
        # self.save_file(df=df,
        #                outpath=os.path.join(self.file_outdir, "pic_eqtl_fdr_with_gene_correlations.txt.gz".format(self.out_filename)),
        #                index=False)
        # self.save_file(df=expr_df,
        #                outpath=os.path.join(self.file_outdir, "pic_eqtl_expression.txt.gz".format(self.out_filename)),
        #                index=True)

        df = self.load_file(os.path.join(self.file_outdir, "pic_eqtl_fdr_with_gene_correlations.txt.gz".format(self.out_filename)),
                            header=0,
                            index_col=None,
                            nrows=None)
        expr_df = self.load_file(os.path.join(self.file_outdir, "pic_eqtl_expression.txt.gz".format(self.out_filename)),
                                 header=0,
                                 index_col=0,
                                 nrows=None)
        print(df)
        print(expr_df)

        print("Create annotations")
        row_colors_df = self.prep_row_colors(annot_df=df)

        std_df = self.load_file(self.std_path, header=0, index_col=None)
        col_colors_df = self.prep_col_colors(samples=expr_df.columns.tolist(),
                                             std_df=std_df,
                                             pics_df=pics_df)

        correlation_df = df.loc[:, [col for col in df.columns if col.endswith(" r")]]
        correlation_df.columns = [col.replace(" r", "") for col in correlation_df.columns]

        row_colors_df1 = row_colors_df.loc[:, ["avg expr"] + [col for col in row_colors_df.columns if col.endswith(" FDR")]].copy()
        row_colors_df1.columns = [col.replace(" FDR", "") for col in row_colors_df1]
        row_colors_df1.index = correlation_df.index

        print("Visualising")
        self.plot(df=correlation_df,
                  vmin=-1,
                  vmax=1,
                  row_colors=row_colors_df1,
                  row_cluster=True,
                  col_cluster=True,
                  xticklabels=True,
                  name="PIC_gene_correlation_clustermap")

        for i in range(1, 6):
            pic_df = df.loc[df["PIC{} FDR".format(i)] < 0.05, ["SNP", "gene", "PIC{} r".format(i)]].copy()
            pic_df.index = pic_df["SNP"] + "_" + pic_df["gene"]
            pic_expr_df = expr_df.loc[pic_df["gene"], :].copy()
            pic_row_colors_df = row_colors_df.loc[pic_df.index, ["avg expr", "PIC{} r".format(i)]].copy()
            pic_row_colors_df.columns = ["avg expr", "r"]
            pic_row_colors_df.index = pic_expr_df.index
            pic_col_colors_df = col_colors_df.loc[pic_expr_df.columns, ["dataset", "PIC{}".format(i)]].copy()

            self.plot(df=pic_expr_df,
                      row_colors=pic_row_colors_df,
                      col_colors=pic_col_colors_df,
                      row_cluster=True,
                      col_cluster=True,
                      name="PIC{}_gene_expression_clustermap".format(i))

            for direction in ["positive", "negative"]:
                pic_direction_df = None
                if direction == "positive":
                    pic_direction_df = pic_df.loc[pic_df["PIC{} r".format(i)] > 0, :]
                elif direction == "negative":
                    pic_direction_df = pic_df.loc[pic_df["PIC{} r".format(i)] < 0, :]
                else:
                    print("huh")
                    exit()
                pic_direction_expr_df = pic_expr_df.loc[pic_direction_df["gene"], :]
                pic_direction_row_colors_df = pic_row_colors_df.loc[pic_direction_expr_df.index, :]

                self.plot(df=pic_direction_expr_df,
                          row_colors=pic_direction_row_colors_df,
                          col_colors=pic_col_colors_df,
                          row_cluster=True,
                          col_cluster=True,
                          name="PIC{}_{}_gene_expression_clustermap".format(i, direction))


        # print("Clustering")
        # gene_expr_linkage = self.cluster(df=expr_df)
        #
        # print("Visualising")
        # for i in range(1, 6):
        #     row_colors_df2 = row_colors_df.loc[:, ["avg expr", "PIC{} FDR".format(i), "PIC{} r".format(i)]].copy()
        #     row_colors_df2.columns = ["avg expr", "FDR", "r"]
        #     row_colors_df2.index = expr_df.index
        #
        #     self.plot(df=expr_df,
        #               vmin=None,
        #               vmax=None,
        #               row_colors=row_colors_df2,
        #               row_linkage=gene_expr_linkage,
        #               col_cluster=True,
        #               name="PIC{}_gene_expression_clustermap".format(i))

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
    def cluster(df, metric="euclidean", method="average"):
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

    def prep_row_colors(self, annot_df, a=0.05, n_pics=5):
        row_color_data = {}

        avg_expr_palette = self.create_palette_gradient(annot_df["avg expression"])
        row_color_data["avg expr"] = [avg_expr_palette[round(avg_expr, 2)] for avg_expr in annot_df["avg expression"]]

        for i in range(1, n_pics + 1):
            pic_column = "PIC{} FDR".format(i)
            if pic_column in annot_df.columns:
                row_color_data[pic_column] = [Color("#000000").rgb if fdr_value <= a else Color("#FFFFFF").rgb for fdr_value in annot_df[pic_column]]

        for i in range(1, 50):
            pic_column = "PIC{} r".format(i)
            if pic_column in annot_df.columns:
                pic_r_palette = self.create_palette_diverging(annot_df[pic_column])
                row_color_data[pic_column] = [pic_r_palette[round(pic_r, 2)] for pic_r in annot_df[pic_column]]

        return pd.DataFrame(row_color_data, index=annot_df["SNP"] + "_" + annot_df["gene"])

    def prep_col_colors(self, samples, std_df, pics_df, n_pics=5):
        pics_df = pics_df.loc[:, samples]
        std_dict = dict(zip(std_df.iloc[:, 0], std_df.iloc[:, 1]))

        col_color_data = {"dataset": [Color(self.palette[std_dict[sample]]).rgb for sample in samples]}
        for i in range(1, n_pics + 1):
            pic_index = "PIC{}".format(i)
            if pic_index in pics_df.index:
                pic_loading_palette = self.create_palette_diverging(pics_df.loc[pic_index, :])
                col_color_data[pic_index] = [pic_loading_palette[round(pic_loading, 2)] for pic_loading in pics_df.loc[pic_index, :]]

        return pd.DataFrame(col_color_data, index=samples)

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
             col_colors=None, row_linkage=None, row_cluster=False,
             col_cluster=False, yticklabels=False, xticklabels=False,
             name="plot"):
        sns.set(color_codes=True)

        if row_linkage is not None and not row_cluster:
            row_cluster = True

        g = sns.clustermap(df,
                           cmap=sns.diverging_palette(246, 24, as_cmap=True),
                           vmin=vmin,
                           vmax=vmax,
                           center=center,
                           row_colors=row_colors,
                           col_colors=col_colors,
                           row_linkage=row_linkage,
                           row_cluster=row_cluster,
                           col_cluster=col_cluster,
                           yticklabels=yticklabels,
                           xticklabels=xticklabels,
                           figsize=(24, 12)
                           )
        plt.setp(g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(),fontsize=10))
        g.fig.subplots_adjust(bottom=0.05, top=0.7)
        plt.tight_layout()

        outpath = os.path.join(self.plot_outdir, "{}.png".format(name))
        g.savefig(outpath)
        plt.close()
        print("\tSaved file '{}'.".format(os.path.basename(outpath)))

    def print_arguments(self):
        print("Arguments:")
        print("  > PICALO directory: {}".format(self.picalo_indir))
        print("  > Gene expression path: {}".format(self.expr_path))
        print("  > Gene info: {}".format(self.gene_info_path))
        print("  > Average gene expression path: {}".format(self.avg_ge_path))
        print("  > Minimal gene expression: {}".format(self.min_avg_expression))
        print("  > STD path: {}".format(self.std_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
