#!/usr/bin/env python3

"""
File:         plot_double_correlation_heatmap.py
Created:      2021/05/18
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
from scipy.special import betainc
from statsmodels.stats import multitest
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot Double Correlation Heatmap"
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
./plot_double_correlation_heatmap.py -h

./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_file1_transpose \
    -m1n PICs \
    -m2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz \
    -m2n RNA-seq_alignment_metrics \
    -b1 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -bios_file1_transpose \
    -b1n PICs \
    -b2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -b2n RNA-seq_alignment_metrics \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_vs_RNASeqAlignmentMetrics \
    -e png pdf
    
./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_file1_transpose \
    -m1n PICs \
    -m2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrain_CellFractionPercentages_forPlotting.txt.gz \
    -m2n CellFraction% \
    -b1 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -bios_file1_transpose \
    -b1n PICs \
    -b2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz \
    -b2n CellFraction% \
    -r2 2 \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_vs_CellFractionPercentages \
    -e png pdf       

./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -metabrain_file1_transpose \
    -m1n Expression_PCs \
    -m2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz \
    -m2n RNA-seq_alignment_metrics \
    -b1 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -bios_file1_transpose \
    -b1n Expression_PCs \
    -b2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -b2n RNA-seq_alignment_metrics \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_ExprPCs_vs_RNASeqAlignmentMetrics \
    -e png pdf
    
./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_file1_transpose \
    -m1n PICs \
    -m2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz \
    -m2n RNA-seq_alignment_metrics \
    -b1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -bios_file1_transpose \
    -b1n Expression_PCs \
    -b2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz \
    -b2n RNA-seq_alignment_metrics \
    -o 2022-03-24-MetaBrain_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_and_ExprPCs_vs_RNASeqAlignmentMetrics \
    -e png pdf
    
./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -metabrain_file1_transpose \
    -m1n PICs \
    -m2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -m2n RNA-seq_alignment_metrics \
    -b1 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first35ExpressionPCs.txt.gz \
    -bios_file1_transpose \
    -b1n Expression_PCs \
    -b2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz \
    -b2n RNA-seq_alignment_metrics \
    -o 2022-03-24-BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICs_and_ExprPCs_vs_RNASeqAlignmentMetrics \
    -e png pdf
    
./plot_double_correlation_heatmap.py \
    -m1 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -metabrain_file1_transpose \
    -m1n Expression_PCs \
    -m2 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrain_CellFractionPercentages_forPlotting.txt.gz \
    -m2n CellFraction% \
    -b1 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first25ExpressionPCs.txt.gz \
    -bios_file1_transpose \
    -b1n Expression_PCs \
    -b2 /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz \
    -b2n CellFraction% \
    -r2 2 \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_ExprPCs_vs_CellFractionPercentages \
    -e png pdf

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta1_path = getattr(arguments, 'metabrain_file1')
        self.meta1_transpose = getattr(arguments, 'metabrain_file1_transpose')
        self.meta1_name = getattr(arguments, 'metabrain_file1_name').replace("_", " ")

        self.meta2_path = getattr(arguments, 'metabrain_file2')
        self.meta2_transpose = getattr(arguments, 'metabrain_file2_transpose')
        self.meta2_name = getattr(arguments, 'metabrain_file2_name').replace("_", " ")

        self.bios1_path = getattr(arguments, 'bios_file1')
        self.bios1_transpose = getattr(arguments, 'bios_file1_transpose')
        self.bios1_name = getattr(arguments, 'bios_file1_name').replace("_", " ")

        self.bios2_path = getattr(arguments, 'bios_file2')
        self.bios2_transpose = getattr(arguments, 'bios_file2_transpose')
        self.bios2_name = getattr(arguments, 'bios_file2_name').replace("_", " ")

        self.rsquared_threshold = getattr(arguments, 'rsquared_threshold')
        self.extensions = getattr(arguments, 'extensions')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
                            help="show program's version number and exit.")
        parser.add_argument("-m1",
                            "--metabrain_file1",
                            type=str,
                            required=True,
                            help="The path to MetaBrain first file.")
        parser.add_argument("-metabrain_file1_transpose",
                            action='store_true',
                            help="Transpose the first MetaBrain file.")
        parser.add_argument("-m1n",
                            "--metabrain_file1_name",
                            type=str,
                            default="metabrain1",
                            help="The name of the MetaBrain first file.")

        parser.add_argument("-m2",
                            "--metabrain_file2",
                            type=str,
                            required=True,
                            help="The path to MetaBrain second file.")
        parser.add_argument("-metabrain_file2_transpose",
                            action='store_true',
                            help="Transpose the second MetaBrain file.")
        parser.add_argument("-m2n",
                            "--metabrain_file2_name",
                            type=str,
                            default="metabrain2",
                            help="The name of the MetaBrain second file.")

        parser.add_argument("-b1",
                            "--bios_file1",
                            type=str,
                            required=True,
                            help="The path to BIOS first file.")
        parser.add_argument("-bios_file1_transpose",
                            action='store_true',
                            help="Transpose the first BIOS file.")
        parser.add_argument("-b1n",
                            "--bios_file1_name",
                            type=str,
                            default="bios1",
                            help="The name of the BIOS first file.")

        parser.add_argument("-b2",
                            "--bios_file2",
                            type=str,
                            required=True,
                            help="The path to BIOS second file.")
        parser.add_argument("-bios_file2_transpose",
                            action='store_true',
                            help="Transpose the second BIOS file.")
        parser.add_argument("-b2n",
                            "--bios_file2_name",
                            type=str,
                            default="bios2",
                            help="The name of the BIOS second file.")
        parser.add_argument("-r2",
                            "--rsquared_threshold",
                            type=float,
                            default=0.99,
                            help="The rsquared threshold to remove multicolinearity."
                                 "Default: 0.99")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        meta_df = self.load_data(path1=self.meta1_path,
                                 path2=self.meta2_path,
                                 transpose1=self.meta1_transpose,
                                 transpose2=self.meta2_transpose,
                                 )
        bios_df = self.load_data(path1=self.bios1_path,
                                 path2=self.bios2_path,
                                 transpose1=self.bios1_transpose,
                                 transpose2=self.bios2_transpose,
                                 )

        # print(meta_df)
        # print(bios_df)
        #
        # meta_df.to_csv("meta_df.txt.gz", sep="\t", header=True, index=True, compression="gzip")
        # bios_df.to_csv("bios_df.txt.gz", sep="\t", header=True, index=True, compression="gzip")

        # meta_df = pd.read_csv("meta_df.txt.gz", sep="\t", header=0, index_col=0)
        # bios_df = pd.read_csv("bios_df.txt.gz", sep="\t", header=0, index_col=0)

        print("Plotting heatmap")
        self.plot_heatmap(df1=bios_df,
                          df2=meta_df,
                          xlabel1=self.bios2_name,
                          ylabel1=self.bios1_name,
                          xlabel2=self.meta2_name,
                          ylabel2=self.meta1_name)

    def load_data(self, path1, path2, transpose1, transpose2):
        df1 = self.load_file(path1, header=0, index_col=0)
        df2 = self.load_file(path2, header=0, index_col=0)

        if transpose1:
            df1 = df1.T

        if transpose2:
            df2 = df2.T

        print(df1)
        print(df2)

        overlap = list(set(df1.index).intersection(set(df2.index)))
        df1 = df1.loc[overlap, :]
        df2 = df2.loc[overlap, :]

        print("Remove non variable column")
        df1 = df1.loc[:, df1.std(axis=0) > 0]
        df2 = df2.loc[:, df2.std(axis=0) > 0]

        print("Remove multicollinearity")
        df1 = self.remove_multicollinearity(df=df1)
        df2 = self.remove_multicollinearity(df=df2)

        print("Correlating")
        corr_m, pvalue_m = self.corrcoef(m1=df1.to_numpy(),
                                         m2=df2.to_numpy())

        print("Perform BH correction")
        fdr_df = pd.DataFrame({"pvalue": pvalue_m.flatten()})
        mask = ~fdr_df["pvalue"].isnull()
        fdr_df["FDR"] = np.nan
        fdr_df.loc[mask, "FDR"] = multitest.multipletests(fdr_df.loc[mask, "pvalue"], method='fdr_bh')[1]
        fdr_m = fdr_df["FDR"].to_numpy().reshape(pvalue_m.shape)
        del fdr_df, mask

        corr_m[fdr_m > 0.05] = np.nan

        return pd.DataFrame(corr_m, index=df1.columns, columns=df2.columns)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def remove_multicollinearity(self, df):
        indices = np.arange(df.shape[1])
        max_r2 = np.inf
        while len(indices) > 1 and max_r2 > self.rsquared_threshold:
            r2 = np.array([self.calc_ols_rsquared(df=df.iloc[:, indices], idx=ix) for ix in range(len(indices))])
            max_r2 = max(r2)

            if max_r2 > self.rsquared_threshold:
                max_index = np.where(r2 == max_r2)[0][0]
                indices = np.delete(indices, max_index)

        return df.iloc[:, indices]

    @staticmethod
    def calc_ols_rsquared(df, idx):
        tmp_df = df.copy()
        tmp_df.dropna(inplace=True)
        rsquared = OLS(tmp_df.iloc[:, idx], tmp_df.loc[:, np.arange(tmp_df.shape[1]) != idx]).fit().rsquared
        del tmp_df
        return rsquared

    @staticmethod
    def corrcoef(m1, m2):
        """
        Pearson correlation over the columns.

        https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
        """
        m1_dev = m1 - np.nanmean(m1, axis=0)
        m2_dev = m2 - np.nanmean(m2, axis=0)

        m1_rss = np.nansum(m1_dev * m1_dev, axis=0)
        m2_rss = np.nansum(m2_dev * m2_dev, axis=0)

        r = np.empty((m1_dev.shape[1], m2_dev.shape[1]), dtype=np.float64)
        for i in range(m1_dev.shape[1]):
            for j in range(m2_dev.shape[1]):
                mask = np.logical_or(np.isnan(m1_dev[:, i]), np.isnan(m2_dev[:, j]))
                r[i, j] = np.sum(m1_dev[~mask, i] * m2_dev[~mask, j]) / np.sqrt(m1_rss[i] * m2_rss[j])

        rf = r.flatten()
        df = m1.shape[0] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = betainc(0.5 * df, 0.5, df / (df + ts))
        p = pf.reshape(m1.shape[1], m2.shape[1])
        return r, p

    def plot_heatmap(self, df1, df2, xlabel1="", ylabel1="", xlabel2="",
                     ylabel2=""):
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(1 * (df1.shape[1] + df2.shape[1]) + 10, 1 * max(df1.shape[0], df2.shape[0]) + 10),
                                 gridspec_kw={"width_ratios": [(1 / (df1.shape[1] + df2.shape[1])) * df1.shape[1],
                                                               (1 / (df1.shape[1] + df2.shape[1])) * df2.shape[1]],
                                              "height_ratios": [0.8, 0.2]})
        sns.set(color_codes=True)

        self.single_heatmap(ax=axes[0, 0],
                            df=df1,
                            xlabel=xlabel1,
                            ylabel=ylabel1,
                            title="blood")
        self.single_heatmap(ax=axes[0, 1],
                            df=df2,
                            xlabel=xlabel2,
                            ylabel=ylabel2,
                            title="brain")
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_{}RSquaredFiltering.{}".format(self.outfile, str(self.rsquared_threshold).replace(".", ""), extension)))
        plt.close()

    @staticmethod
    def single_heatmap(ax, df, xlabel="", ylabel="", title=""):
        annot_df = df.copy()
        annot_df = annot_df.round(2)
        annot_df.fillna("", inplace=True)

        hm = sns.heatmap(df,
                         cmap=sns.diverging_palette(246, 24,
                                                    as_cmap=True),
                         vmin=-1,
                         vmax=1,
                         center=0,
                         square=True,
                         annot=annot_df,
                         fmt='',
                         cbar=False,
                         annot_kws={"size": 14, "color": "#000000"},
                         ax=ax)

        ax.set_xlabel(xlabel, fontsize=75)
        ax.set_ylabel(ylabel, fontsize=75)
        ax.set_title(title, fontsize=100)

        hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=30, rotation=90)
        hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=30, rotation=0)

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain:")
        print("  >     (1) {}: {} {}".format(self.meta1_name, self.meta1_path, "[T]" if self.meta1_transpose else ""))
        print("  >     (2) {}: {} {}".format(self.meta2_name, self.meta2_path, "[T]" if self.meta2_transpose else ""))
        print("  > BIOS:")
        print("  >     (1) {}: {} {}".format(self.bios1_name, self.bios1_path, "[T]" if self.bios1_transpose else ""))
        print("  >     (2) {}: {} {}".format(self.bios2_name, self.bios2_path, "[T]" if self.bios2_transpose else ""))
        print("  > R2 threshold: {}".format(self.rsquared_threshold))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
