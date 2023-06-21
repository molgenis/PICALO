#!/usr/bin/env python3

"""
File:         compare_betas.py
Created:      2023/05/19
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
from colour import Color
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Compare Beta's"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@umcg.nl"
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
./compare_betas.py -h

./compare_betas.py \
    -r /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLNoCovariates/simulation1/model_betas.txt.gz \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLNoCovariates/eQTLSummaryStats.txt.gz \
    -o 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLNoCovariates

./compare_betas.py \
    -r /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction/simulation1/model_betas.txt.gz \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction/eQTLSummaryStats.txt.gz \
    -o 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction

./compare_betas.py \
    -r /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates/simulation1/model_betas.txt.gz \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates/eQTLSummaryStats.txt.gz \
    -o 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates
  
./compare_betas.py \
    -r /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates/simulation1/model_betas.txt.gz \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates/eQTLSummaryStats.txt.gz \
    -o 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates  
    
./compare_betas.py \
    -r /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate/simulation1/model_betas.txt.gz \
    -c /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate/eQTLSummaryStats.txt.gz \
    -o 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate  
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.real_path = getattr(arguments, 'real')
        self.calculated_path = getattr(arguments, 'calculated')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'compare_betas')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit")
        parser.add_argument("-r",
                            "--real",
                            type=str,
                            required=True,
                            help="The path to the real beta matrix")
        parser.add_argument("-c",
                            "--calculated",
                            type=str,
                            required=True,
                            help="The name for the calculated matrix")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")
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
        print("Loading deconvolution matrix 1.")
        real_df = self.load_file(inpath=self.real_path, header=0, index_col=0)
        calculated_df = self.load_file(inpath=self.calculated_path, header=0, index_col=None)
        real_df.columns = [x.replace("_interaction", "") + "Xgenotype" if "_interaction" in x else x for x in real_df.columns]
        print(real_df)
        print(calculated_df)

        # real_df.index = real_df["gene"] + "_" + real_df["SNP"]
        calculated_df.index = calculated_df["gene"] + "_" + calculated_df["SNP"]

        overlapping_terms = []
        for column in real_df.columns:
            beta_column = "beta-{}".format(column)
            if beta_column in calculated_df.columns:
                overlapping_terms.append((column, beta_column))
        print(overlapping_terms)
        # overlapping_terms = [("N", "N"),
        #                      ("beta-intercept", "beta-intercept"),
        #                      ("beta-genotype", "beta-genotype"),
        #                      ("beta-known_covariate0", "beta-covariate"),
        #                      ("beta-known_covariate0Xgenotype", "beta-interaction")]

        n_plots = len(overlapping_terms)
        if n_plots == 0:
            return
        ncols = math.ceil(np.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)

        print("Plotting")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            print(i)
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(overlapping_terms):
                real_term, calculated_term = overlapping_terms[i]
                compare_df = real_df[[real_term]].merge(calculated_df[[calculated_term]], left_index=True, right_index=True)
                compare_df.columns = ["real", "calculated"]

                self.plot(fig=fig,
                          ax=ax,
                          df=compare_df,
                          x="real",
                          xlabel="Real",
                          y="calculated",
                          ylabel="Calculated",
                          title=calculated_term)
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}.{}".format(self.out_filename, extension)))
        plt.close()

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, fig, ax, df, x="x", y="y", xlabel="", ylabel="", title="",
             color="#000000", ci=95, include_ylabel=True,
             lines=True):
        sns.despine(fig=fig, ax=ax)

        if not include_ylabel:
            ylabel = ""

        n = df.shape[0]
        coef = np.nan
        concordance = np.nan
        rss = np.nan

        if n > 0:
            lower_quadrant = df.loc[(df[x] < 0) & (df[y] < 0), :]
            upper_quadrant = df.loc[(df[x] > 0) & (df[y] > 0), :]
            n_concordant = lower_quadrant.shape[0] + upper_quadrant.shape[0]
            concordance = (100 / n) * n_concordant

            res = (df[x] - df[y]).to_numpy()
            rss = np.sum(res * res)

            if n > 1:
                coef, p = stats.pearsonr(df[x], df[y])

            sns.regplot(x=x, y=y, data=df, ci=ci,
                        scatter_kws={'facecolors': "#808080",
                                     'edgecolors': "#808080",
                                     'alpha': 0.60},
                        line_kws={"color": color},
                        ax=ax
                        )

        if lines:
            ax.axhline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
            ax.axvline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
            ax.axline((0, 0), slope=1, ls='--', color="#D7191C", alpha=0.3,
                      zorder=1)

        y_pos = 0.9
        if n > 0:
            ax.annotate(
                'N = {:,}'.format(n),
                xy=(0.03, 0.9),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(coef):
            ax.annotate(
                'r = {:.2f}'.format(coef),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(concordance):
            ax.annotate(
                'concordance = {:.0f}%'.format(concordance),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(rss):
            ax.annotate(
                'rss = {:.2f}'.format(rss),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        ax.set_title(title,
                     fontsize=22,
                     color=color,
                     weight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')


if __name__ == '__main__':
    m = main()
    m.start()
