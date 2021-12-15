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
        self.d_fim_indir = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-InteractionsWithPICs/"
        self.r_fim_indir = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2021-12-14-MetaBrain-CortexAFR-CortexEURReplication-cis-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/"
        self.out_filename = "2021-12-14-MetaBrain-CortexAFR-CortexEURReplication-cis-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs"

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        print("Loading data")
        d_tvalue_df, d_pvalue_df = self.load_pic_interaction_data(indir=self.d_fim_indir)
        r_tvalue_df, r_pvalue_df = self.load_pic_interaction_data(indir=self.r_fim_indir)
        print(d_tvalue_df)
        print(r_tvalue_df)

        if list(d_tvalue_df.columns) != list(d_pvalue_df.columns):
            print("Columns do not match.")
            exit()
        if list(r_tvalue_df.columns) != list(r_pvalue_df.columns):
            print("Columns do not match.")
            exit()
        if list(d_pvalue_df.columns) != list(r_pvalue_df.columns):
            print("Columns do not match.")
            exit()

        print("Overlapping data")
        eqtls = list(set(d_pvalue_df.index).intersection(set(r_pvalue_df.index)))
        d_tvalue_df = d_tvalue_df.loc[eqtls, :]
        d_pvalue_df = d_pvalue_df.loc[eqtls, :]
        r_tvalue_df = r_tvalue_df.loc[eqtls, :]
        r_pvalue_df = r_pvalue_df.loc[eqtls, :]
        print("\tN-overlap: {}".format(len(eqtls)))

        print("Calculating FDR")
        d_fdr_df = pd.DataFrame(np.nan, index=d_pvalue_df.index, columns=d_pvalue_df.columns)
        for colname in d_pvalue_df.columns:
            d_fdr_df.loc[:, colname] = multitest.multipletests(d_pvalue_df.loc[:, colname], method='fdr_bh')[1]

        r_fdr_df = pd.DataFrame(np.nan, index=r_pvalue_df.index, columns=r_pvalue_df.columns)
        for colname in r_pvalue_df.columns:
            mask = d_fdr_df.loc[:, colname] < 0.05
            r_fdr_df.loc[mask, colname] = multitest.multipletests(r_pvalue_df.loc[mask, colname], method='fdr_bh')[1]
        del d_pvalue_df, r_pvalue_df

        print("Merging data.")
        d_df_m = self.melt_and_merge(tvalue_df=d_tvalue_df,
                                     fdr_df=d_fdr_df,
                                     name="discovery")
        r_df_m = self.melt_and_merge(tvalue_df=r_tvalue_df,
                                     fdr_df=r_fdr_df,
                                     name="replication")
        df_m = d_df_m.merge(r_df_m, on=["index", "variable"])
        print(df_m)

        print("Selecting significant results")
        df_m = df_m.loc[(df_m["discovery FDR"] < 0.05) & (df_m["replication FDR"] < 0.05), :]
        print(df_m)
        pic_replicating_interaction_counts = list(zip(*np.unique(df_m["variable"], return_counts=True)))
        pic_replicating_interaction_counts.sort(key=lambda x: -x[1])
        pics_with_replication_interactions = [x[0] for x in pic_replicating_interaction_counts if x[1] > 2]
        columns = ["PIC{}".format(i) for i in range(1, 50) if "PIC{}".format(i) in pics_with_replication_interactions]

        print("Plotting data")
        self.replication_regplot(df=df_m,
                                 col=columns,
                                 x="discovery t-value",
                                 y="replication t-value",
                                 xlabel="MetaBrain CortexEUR",
                                 ylabel="MetaBrain CortexAFR",
                                 filename=self.out_filename)

    def load_pic_interaction_data(self, indir):
        tvalue_list = []
        pvalue_list = []
        for i in range(1, 50):
            fpath = os.path.join(indir, "PIC{}.txt.gz".format(i))
            if os.path.exists(fpath):
                df = self.load_file(fpath, header=0, index_col=None)
                df.index = df["gene"] + "_" + df["snp"]

                df["ieQTL tvalue-interaction"] = df["ieQTL beta-interaction"] / df["ieQTL std-interaction"]
                tvalue_df = df[["ieQTL tvalue-interaction"]].copy()
                tvalue_df.columns = ["PIC{}".format(i)]
                tvalue_list.append(tvalue_df)

                df["ieQTL tvalue-interaction"] = df["ieQTL beta-interaction"] / df["ieQTL std-interaction"]
                pvalue_df = df[["ieQTL p-value"]].copy()
                pvalue_df.columns = ["PIC{}".format(i)]
                pvalue_list.append(pvalue_df)
        tvalue_df = pd.concat(tvalue_list, axis=1)
        pvalue_df = pd.concat(pvalue_list, axis=1)
        return tvalue_df, pvalue_df

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
    def melt_and_merge(tvalue_df, fdr_df, name):
        tvalue_df.reset_index(drop=False, inplace=True)
        tvalue_df_m = tvalue_df.melt(id_vars=["index"], value_name="{} t-value".format(name))
        fdr_df.reset_index(drop=False, inplace=True)
        fdr_df_m = fdr_df.melt(id_vars=["index"], value_name="{} FDR".format(name))
        df_m = tvalue_df_m.merge(fdr_df_m, on=["index", "variable"])
        return df_m

    def replication_regplot(self, df, col, col_colname="variable", x="x", y="y",
                            xlabel="", ylabel="", filename=""):

        if df.shape[0] <= 2:
            return

        nplots = len(col)
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
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
                plot_df = df.loc[df[col_colname] == col[i], [x, y]]
                if plot_df.shape[0] <= 2:
                    continue

                sns.despine(fig=fig, ax=ax)

                lower_quadrant = plot_df.loc[(plot_df[x] < 0) & (plot_df[y] < 0), :]
                upper_quadrant = plot_df.loc[(plot_df[x] > 0) & (plot_df[y] > 0), :]
                concordance = (100 / plot_df.shape[0]) * (lower_quadrant.shape[0] + upper_quadrant.shape[0])

                coef, _ = stats.spearmanr(plot_df[y], plot_df[x])

                sns.regplot(x=x, y=y, data=plot_df, ci=None,
                            scatter_kws={'facecolors': "#808080",
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": "#b22222",
                                      'linewidth': 5},
                            ax=ax)

                ax.axhline(0, ls='--', color="#000000", zorder=-1)
                ax.axvline(0, ls='--', color="#000000", zorder=-1)

                ax.annotate(
                    'N = {}'.format(plot_df.shape[0]),
                    xy=(0.03, 0.94),
                    xycoords=ax.transAxes,
                    color="#000000",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'r = {:.2f}'.format(coef),
                    xy=(0.03, 0.90),
                    xycoords=ax.transAxes,
                    color="#000000",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'concordance = {:.0f}%'.format(concordance),
                    xy=(0.03, 0.86),
                    xycoords=ax.transAxes,
                    color="#000000",
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')

                tmp_xlabel = ""
                if row_index == (nrows - 1):
                    tmp_xlabel = xlabel
                ax.set_xlabel(tmp_xlabel,
                              fontsize=20,
                              fontweight='bold')
                tmp_ylabel = ""
                if col_index == 0:
                    tmp_ylabel = ylabel
                ax.set_ylabel(tmp_ylabel,
                              fontsize=20,
                              fontweight='bold')

                ax.set_title(col[i],
                             fontsize=25,
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
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        outpath = os.path.join(self.outdir, "replication_plot{}.png".format(filename))
        fig.savefig(outpath)
        plt.close()
        print("\tSaved: {}".format(outpath))


if __name__ == '__main__':
    m = main()
    m.start()
