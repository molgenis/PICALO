#!/usr/bin/env python3

"""
File:         pic_replication_plot.py
Created:      2021/12/13
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import os

# Third party imports.
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "PIC Replication Plot"
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
./replication_plot.py -h
"""


class main():
    def __init__(self):
        self.bios_pic_expr_corr_path = "/groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations_cleaned.txt.gz"
        self.bios_expr_path = "/groups/umcg-bios/tmp01/projects/PICALO/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.txt.gz"
        self.bios_std_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz"
        self.bios_eqtl_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"

        self.metabrain_pic_expr_corr_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-GeneExpressionFNPD_gene_correlations_cleaned.txt.gz"
        self.metabrain_expr_path = "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-01-31-expression-tables/2020-02-05-step6-covariate-removal/2021-08-27-step5-remove-covariates-per-dataset/output-PCATitration-MDSCorrectedPerDsCovarOverall-cortex-EUR/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.ProbesWithZeroVarianceRemoved.txt.gz"
        self.metabrain_std_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz"
        self.metabrain_eqtl_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"

        self.min_avg_expr = 1

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        print("Loading PIC-expression correlation data.")
        bios_pic_expr_corr_df = self.load_file(self.bios_pic_expr_corr_path, header=0, index_col=0)
        metabrain_pic_expr_corr_df = self.load_file(self.metabrain_pic_expr_corr_path, header=0, index_col=0)
        print(bios_pic_expr_corr_df)
        print(metabrain_pic_expr_corr_df)

        # print("Loading gene expression data.")
        # bios_expr_df = self.load_file(self.bios_expr_path, header=0, index_col=0)
        # bios_expr_df.index = [ensembl_id.split(".")[0] for ensembl_id in bios_expr_df.index]
        # bios_expr_df = bios_expr_df.groupby(bios_expr_df.index).first()
        # metabrain_expr_df = self.load_file(self.metabrain_expr_path, header=0, index_col=0)
        # metabrain_expr_df.index = [ensembl_id.split(".")[0] for ensembl_id in metabrain_expr_df.index]
        # metabrain_expr_df = metabrain_expr_df.groupby(metabrain_expr_df.index).first()
        # print(bios_expr_df)
        # print(metabrain_expr_df)
        #
        # print("Loading sample data.")
        # bios_std_df = self.load_file(self.bios_std_path, header=0, index_col=None)
        # bios_samples = list(bios_std_df.iloc[:, 0].values)
        # metabrain_std_df = self.load_file(self.metabrain_std_path, header=0, index_col=None)
        # metabrain_samples = list(metabrain_std_df.iloc[:, 0].values)
        #
        # print("Sample / gene selection.")
        # overlap_genes = list(set(bios_expr_df.index).intersection(set(metabrain_expr_df.index)))
        #
        # print("Calculate average expression")
        # bios_avg_expr = self.calc_avg_expr(df=bios_expr_df,
        #                                    samples=bios_samples,
        #                                    genes=overlap_genes)
        # metabrain_avg_expr = self.calc_avg_expr(df=metabrain_expr_df,
        #                                         samples=metabrain_samples,
        #                                         genes=overlap_genes)
        #
        # print("Getting genes of interest")
        # bios_expressed_genes = set(bios_avg_expr[bios_avg_expr > self.min_avg_expr].index)
        # metabrain_expressed_genes = set(metabrain_avg_expr[metabrain_avg_expr > self.min_avg_expr].index)
        # expressed_genes = bios_expressed_genes.intersection(metabrain_expressed_genes)

        print("Loading eQTL data.")
        bios_eqtl_df = self.load_file(self.bios_eqtl_path, header=0, index_col=None)
        bios_genes = set([ensembl_id.split(".")[0] for ensembl_id in bios_eqtl_df["ProbeName"]])
        metabrain_eqtl_df = self.load_file(self.metabrain_eqtl_path, header=0, index_col=None)
        metabrain_genes = set([ensembl_id.split(".")[0] for ensembl_id in metabrain_eqtl_df["ProbeName"]])

        print("Getting genes of interest")
        expressed_genes = bios_genes.intersection(metabrain_genes)

        print("Subset PIC-gene correlations.")
        plot_df = bios_pic_expr_corr_df[["PIC1"]].merge(metabrain_pic_expr_corr_df[["PIC1"]], left_index=True, right_index=True)
        plot_df.columns = ["x", "y"]
        plot_df = plot_df.loc[expressed_genes, :]

        print("Plotting")
        self.replication_plot(df=plot_df,
                              xlabel="BIOS PIC1-expression Pearson r",
                              ylabel="MetaBrain PIC1-expression Pearson r",
                              filename="BIOS_vs_MetaBrain_PrimaryeQTLs_PIC1_geneCorrelations_replication_eQTLGenes")

    @staticmethod
    def calc_avg_expr(df, samples, genes):
        df = df.loc[genes, samples]

        min_value = df.min(axis=1).min()
        if min_value <= 0:
            expr_df = np.log2(df - min_value + 1)
        else:
            expr_df = np.log2(df + 1)

        return expr_df.mean(axis=1)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def replication_plot(self, df, x="x", y="y", xlabel=None, ylabel=None,
                         title="", filename="plot"):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        # plot.
        sns.regplot(x=x, y=y, data=df, ci=None,
                    scatter_kws={'facecolors': "#808080",
                                 'linewidth': 0},
                    line_kws={"color": "#b22222"},
                    ax=ax)

        ax.axhline(0, ls='--', color="#000000", zorder=-1)
        ax.axvline(0, ls='--', color="#000000", zorder=-1)

        # calculate concordance.
        lower_quadrant = df.loc[(df[x] < 0) & (df[y] < 0), :]
        upper_quadrant = df.loc[(df[x] > 0) & (df[y] > 0), :]
        concordance = (100 / df.shape[0]) * (
                    lower_quadrant.shape[0] + upper_quadrant.shape[0])

        # Set annotation.
        pearson_coef, _ = stats.pearsonr(df[y], df[x])
        ax.annotate(
            'total N = {:,}'.format(df.shape[0]),
            xy=(0.03, 0.94),
            xycoords=ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')
        ax.annotate(
            'total r = {:.2f}'.format(pearson_coef),
            xy=(0.03, 0.90),
            xycoords=ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')
        ax.annotate(
            'concordance = {:.0f}%'.format(concordance),
            xy=(0.03, 0.86),
            xycoords=ax.transAxes,
            color="#000000",
            fontsize=14,
            fontweight='bold')

        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_title(title,
                     fontsize=18,
                     fontweight='bold')

        # Change margins.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xmargin = (xlim[1] - xlim[0]) * 0.05
        ymargin = (ylim[1] - ylim[0]) * 0.05

        new_xlim = (xlim[0] - xmargin, xlim[1] + xmargin)
        new_ylim = (ylim[0] - ymargin, ylim[1] + ymargin)

        ax.set_xlim(new_xlim[0], new_xlim[1])
        ax.set_ylim(new_ylim[0], new_ylim[1])

        outpath = os.path.join(self.outdir, "{}.png".format(filename))
        fig.savefig(outpath)
        plt.close()
        print("\tSaved figure: {} ".format(os.path.basename(outpath)))


if __name__ == '__main__':
    m = main()
    m.start()
