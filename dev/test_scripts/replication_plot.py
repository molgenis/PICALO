#!/usr/bin/env python3

"""
File:         replication_plot.py
Created:      2021/06/04
Last Changed: 2021/07/07
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
import math
import os

# Third party imports.
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import multitest
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Replication Plot"
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
./replication_plot.py -h
"""


class main():
    def __init__(self):

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        print("Loading eQTL file.")
        eqtl_df = self.load_file("/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/cortex_eur_cis_NoENA_NoGVEX_NoCorrection/combine_eqtlprobes/eQTLprobes_combined.txt.gz", header=0, index_col=None)
        eqtl_df.index = eqtl_df["SNPName"] + eqtl_df["ProbeName"]
        print(eqtl_df)

        print("Loading alleles")
        geno_eur_df = self.load_file("/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/cortex_eur_cis_NoENA_NoGVEX_NoCorrection/create_matrices/genotype_alleles.txt.gz",
            header=0, index_col=None)
        geno_afr_df = self.load_file(
            "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/cortex_afr_cis_NoENA_NoCorrection_EURReplication/create_matrices/genotype_alleles.txt.gz"
            , header=0, index_col=None)

        flip = np.empty(geno_eur_df.shape[0])
        flip[:] = -1
        mask = geno_eur_df["MinorAllele"] == geno_afr_df["MinorAllele"]
        flip[mask] = 1

        eur_iter_df = {"component0": "iteration27",
                       "component1": "iteration23",
                       "component2": "iteration22",
                       "component3": "iteration28",
                       "component4": "iteration99",
                       }
        afr_iter_fdr = {"component0": "iteration0",
                        "component1": "iteration0",
                        "component2": "iteration0",
                        "component3": "iteration0",
                        "component4": "iteration0",
                       }

        data = {}
        signif_cutoffs = {}
        print("Loading components")
        for idx in range(5):
            component = "component{}".format(idx)

            comp_eur_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICA/output_CortexEUR_woENA_woGVEX_PCs/{}/results_{}.txt.gz".format(component, eur_iter_df[component])
            print("\t{}".format(comp_eur_path))
            if not os.path.exists(comp_eur_path):
                continue
            comp_eur_df = self.load_file(comp_eur_path, header=0, index_col=None)

            comp_afr_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICA/MetaBrain_CortexAFR_woENA_PICReplication/{}/results_{}.txt.gz".format(component, afr_iter_fdr[component])
            print("\t{}".format(comp_afr_path))
            if not os.path.exists(comp_afr_path):
                continue
            comp_afr_df = self.load_file(comp_afr_path, header=0, index_col=None)

            merged_df = eqtl_df[["SNPName", "ProbeName"]].copy()
            comp_signif_cutoff = {}
            for df, etnicity in zip([comp_eur_df, comp_afr_df], ["EUR", "AFR"]):
                df["t-value"] = df["beta-interaction"] / df["std-interaction"]
                # df["z-score"] = stats.norm.isf(df["p-value"])
                # df.loc[df["p-value"] > (1.0 - 1e-16), "z-score"] = -8.209536151601387
                # df.loc[df["p-value"] < 1e-323, "z-score"] = 38.44939448087599

                comp_signif_cutoff[etnicity] = df.loc[df["FDR"] < 0.05, "t-value"].min()
                #comp_signif_cutoff[etnicity] = df.loc[df["FDR"] < 0.05, "z-score"].min()

                df.index = df["SNP"] + df["gene"]

                merged_df = merged_df.merge(df[["t-value", "p-value", "FDR"]], how="left", left_index=True, right_index=True)
                #merged_df = merged_df.merge(df[["z-score"]], how="left", left_index=True, right_index=True)
            merged_df.columns = ["SNPName", "ProbeName", "EUR t-value", "EUR p-value", "EUR FDR", "AFR t-value", "AFR p-value", "AFR FDR"]

            # Flip.
            merged_df["AFR t-value"] = merged_df["AFR t-value"] * flip

            # Drop NA.
            merged_df.dropna(inplace=True)

            # Save.
            data[component] = merged_df
            signif_cutoffs[component] = comp_signif_cutoff

        x = "EUR t-value"
        y = "AFR t-value"

        print("Plotting interaction comparison")
        components = list(data.keys())
        components.sort()
        nplots = len(components)
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1 and ncols > 1:
                ax = axes[col_index]
            elif nrows > 1 and ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < nplots:
                plot_df = data[components[i]]
                eur_signif_df = plot_df.loc[plot_df["EUR FDR"] < 0.05, :].copy()
                eur_signif_df["AFR adj FDR"] = multitest.multipletests(eur_signif_df["AFR p-value"], method='fdr_bh')[1]
                both_signif_df = eur_signif_df.loc[eur_signif_df["AFR adj FDR"] < 0.05, :].copy()
                plot_df["hue"] = "#000000"
                plot_df.loc[both_signif_df.index.tolist(), "hue"] = "#0072B2"

                # Filter significant.
                #plot_df = plot_df.loc[plot_df["EUR FDR"] < 0.05, :]
                #plot_df = plot_df.loc[(plot_df["EUR FDR"] < 0.05) & (plot_df["AFR FDR"] < 0.05), :]
                #plot_df = plot_df.loc[plot_df[x].abs() > 2, :]
                plot_df = plot_df.loc[(plot_df[x].abs() > 2) & (plot_df[y].abs() > 2), :]
                #plot_df = subset_plot_df.loc[(subset_plot_df["EUR FDR"] < 0.05) & (subset_plot_df["AFR FDR2"] < 0.05), :]
                print(plot_df)

                # calculate concordance.
                lower_quadrant = plot_df.loc[(plot_df[x] < 0) & (plot_df[y] < 0), :]
                upper_quadrant = plot_df.loc[(plot_df[x] > 0) & (plot_df[y] > 0), :]
                concordance = (100 / plot_df.shape[0]) * (lower_quadrant.shape[0] + upper_quadrant.shape[0])

                signif_lower_quadrant = both_signif_df.loc[(both_signif_df[x] < 0) & (both_signif_df[y] < 0), :]
                signif_upper_quadrant = both_signif_df.loc[(both_signif_df[x] > 0) & (both_signif_df[y] > 0), :]
                signif_concordance = (100 / both_signif_df.shape[0]) * (signif_lower_quadrant.shape[0] + signif_upper_quadrant.shape[0])

                sns.despine(fig=fig, ax=ax)

                coef, _ = stats.spearmanr(plot_df[y], plot_df[x])

                sns.regplot(x=x, y=y, data=plot_df, ci=95,
                            scatter_kws={'facecolors': plot_df["hue"],
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": "#b22222",
                                      'linewidth': 5},
                            ax=ax)

                ax.annotate(
                    'N = {}'.format(plot_df.shape[0]),
                    xy=(0.03, 0.22),
                    xycoords=ax.transAxes,
                    color="#b22222",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'N signif. in both = {}'.format(both_signif_df.shape[0]),
                    xy=(0.03, 0.18),
                    xycoords=ax.transAxes,
                    color="#b22222",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'r = {:.2f}'.format(coef),
                    xy=(0.03, 0.14),
                    xycoords=ax.transAxes,
                    color="#b22222",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'concordance = {:.0f}%'.format(concordance),
                    xy=(0.03, 0.1),
                    xycoords=ax.transAxes,
                    color="#b22222",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'signif. concordance = {:.0f}%'.format(signif_concordance),
                    xy=(0.03, 0.06),
                    xycoords=ax.transAxes,
                    color="#b22222",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')

                ax.axhline(0, ls='--', color="#b22222", zorder=-1)
                ax.axvline(0, ls='--', color="#b22222", zorder=-1)

                xlabel = ""
                if row_index == (nrows - 1):
                    xlabel = x
                ax.set_xlabel(xlabel,
                              fontsize=20,
                              fontweight='bold')
                ylabel = ""
                if col_index == 0:
                    ylabel = y
                ax.set_ylabel(ylabel,
                              fontsize=20,
                              fontweight='bold')

                ax.set_title(components[i],
                             fontsize=25,
                             fontweight='bold')
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "MetaBrain_PICA_replication_EURtoAFR.png"))
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


if __name__ == '__main__':
    m = main()
    m.start()
