#!/usr/bin/env python3

"""
File:         check_expression_distribution.py
Created:      2021/11/30
Last Changed: 2022/02/10
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
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Check expression distribution"
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
./check_expression_distribution.py -h
"""


class main():
    def __init__(self):
        self.expression_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/expression_table_TMM_Log2Transformed.txt.gz"
        self.filter_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/genotype_stats.txt.gz"
        self.interaction_paths = {"PIC1": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC1/results_iteration049.txt.gz",
                                  "PIC2": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC2/results_iteration049.txt.gz",
                                  "PIC3": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC3/results_iteration049.txt.gz",
                                  "PIC4": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC4/results_iteration099.txt.gz",
                                  "PIC5": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC5/results_iteration099.txt.gz",
                                  "PIC6": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC6/results_iteration049.txt.gz",
                                  "PIC7": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC7/results_iteration049.txt.gz",
                                  "PIC8": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC8/results_iteration049.txt.gz",
                                  "PIC9": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC9/results_iteration049.txt.gz",
                                  "PIC10": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC10/results_iteration049.txt.gz",
                                  "PIC11": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC11/results_iteration049.txt.gz",
                                  "PIC12": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC12/results_iteration049.txt.gz",
                                  "PIC13": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC13/results_iteration049.txt.gz",
                                  "PIC14": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC14/results_iteration049.txt.gz",
                                  "PIC15": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC15/results_iteration049.txt.gz",
                                  }

        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        print("Loading data")
        expr_df = self.load_file(self.expression_path, header=0, index_col=0)
        filter_df = self.load_file(self.filter_path, header=0, index_col=None)

        print(expr_df)
        print(filter_df)

        print("Subset expression data")
        expr_df = expr_df.loc[(filter_df["mask"] == 1).to_numpy(bool), :]

        print("Calculate stats")
        n, mm, m, v, sk, kurt = stats.describe(expr_df, axis=1)
        describe_df = pd.DataFrame({"min": mm[0],
                                    "max": mm[1],
                                    "mean": m,
                                    "variance": v,
                                    "skewness": sk,
                                    "kurtosis": kurt})
        describe_df.insert(3, "median", expr_df.median(axis=1).to_numpy())
        describe_df.insert(4, "sum", expr_df.sum(axis=1).to_numpy())
        print(describe_df)

        for column in describe_df:
            self.histplot(df=describe_df, x=column, title=column, filename="_{}".format(column))

        describe_df_m_list = []
        x_order = []
        for i in range(1, 50):
            pic = "PIC{}".format(i)
            if pic not in self.interaction_paths.keys():
                continue
            interaction_path = self.interaction_paths[pic]

            print("Plotting {}".format(pic))

            interaction_df = self.load_file(interaction_path, header=0, index_col=None)
            print(interaction_df)

            if list(interaction_df["gene"].values) != list(expr_df.index):
                print("Matrices do not match.")
                exit()

            describe_df[pic] = "not signif"
            describe_df.loc[(interaction_df["FDR"] <= 0.05).to_numpy(bool), pic] = "signif"

            if pic in ["PIC1", "PIC2", "PIC15"]:
                self.plot_overview(df=describe_df,
                                   columns=["min", "max", "mean", "variance", "skewness", "kurtosis"],
                                   hue=pic,
                                   palette={"not signif": "#808080", "signif": "#009E73"},
                                   name="_{}".format(pic))

            signif_df = describe_df.loc[describe_df[pic] == "signif", :].copy()
            signif_df_m = signif_df.melt(value_vars=["min", "max", "mean", "median", "sum", "variance", "skewness", "kurtosis"])
            label = "{} [n={}]".format(pic, signif_df.shape[0])
            signif_df_m["pic"] = label
            signif_df_m["hue"] = "unique"
            if pic in ["PIC9", "PIC10", "PIC11", "PIC12", "PIC14", "PIC15"]:
                signif_df_m["hue"] = "recurring"
            x_order.append(label)
            describe_df_m_list.append(signif_df_m)

        describe_df_m = pd.concat(describe_df_m_list, axis=0)
        print(describe_df_m)

        self.plot_boxplot(df_m=describe_df_m,
                          x="pic",
                          hue="hue",
                          palette={"unique": "#009E73", "recurring": "#0072B2"},
                          col="variable",
                          col_order=["min", "max", "mean", "median", "sum", "variance", "skewness", "kurtosis"],
                          x_order=x_order)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_overview(self, df, columns, hue, palette, name=""):
        ncols = len(columns)
        nrows = len(columns)

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(10 * ncols, 10 * nrows))
        sns.set(color_codes=True)

        for i, y_col in enumerate(columns):
            for j, x_col in enumerate(columns):
                print(i, j)

                ax = axes[i, j]
                if i == 0 and j == (ncols - 1):
                    ax.set_axis_off()
                    if hue is not None and palette is not None:
                        groups_present = df[hue].unique()
                        handles = []
                        for key, value in palette.items():
                            if key in groups_present:
                                handles.append(mpatches.Patch(color=value, label=key))
                        ax.legend(handles=handles, loc=4, fontsize=25)

                elif i < j:
                    ax.set_axis_off()
                    continue
                elif i == j:
                    ax.set_axis_off()

                    ax.annotate(y_col,
                                xy=(0.5, 0.5),
                                ha='center',
                                xycoords=ax.transAxes,
                                color="#000000",
                                fontsize=40,
                                fontweight='bold')
                else:
                    sns.despine(fig=fig, ax=ax)

                    sns.scatterplot(x=x_col,
                                    y=y_col,
                                    hue=hue,
                                    data=df.loc[df[hue] == "not signif", :],
                                    s=100,
                                    palette=palette,
                                    linewidth=0,
                                    legend=False,
                                    ax=ax)
                    sns.scatterplot(x=x_col,
                                    y=y_col,
                                    hue=hue,
                                    data=df.loc[df[hue] == "signif", :],
                                    s=100,
                                    palette=palette,
                                    linewidth=0,
                                    legend=False,
                                    ax=ax)

                    ax.set_ylabel("",
                                  fontsize=20,
                                  fontweight='bold')
                    ax.set_xlabel("",
                                  fontsize=20,
                                  fontweight='bold')

        fig.savefig(os.path.join(self.outdir, "expression_describe_overview{}.png".format(name)))
        plt.close()

    def histplot(self, df, x="x", xlabel="", ylabel="", title="", filename="plot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        range = abs(df[x].max() - df[x].min())

        g = sns.histplot(data=df,
                         x=x,
                         kde=True,
                         binwidth=range / 100,
                         color="#000000",
                         ax=ax)

        ax.set_title(title,
                     fontsize=14,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.outdir, "expression_describe_histplot{}.png".format(filename)))
        plt.close()

    def plot_boxplot(self, df_m, x="x", y="value", col=None, hue=None,
                     palette=None, x_order=None, col_order=None, xlabel="",
                     ylabel=""):
        cols = [None]
        if col is not None:
            cols = df_m[col].unique().tolist()
            cols.sort()

            if col_order is None:
                col_order = cols

        if x_order is None:
            x_order = df_m[x].unique().tolist()
            x_order.sort()

        ngroups = len(cols)
        if palette is not None:
            ngroups += 1
        ncols = int(np.ceil(np.sqrt(ngroups)))
        nrows = int(np.ceil(ngroups / ncols))

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex="all",
                                 figsize=(12 * ncols, 12 * nrows)
                                 )
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

            if i < len(col_order):
                title = ""
                if col is not None:
                    subset = df_m.loc[df_m[col] == col_order[i], :]
                    title = col_order[i]
                else:
                    subset = df_m

                sns.despine(fig=fig, ax=ax)

                sns.violinplot(x=x,
                               y=y,
                               hue=hue,
                               data=subset,
                               order=x_order,
                               palette=palette,
                               cut=0,
                               dodge=False,
                               ax=ax)

                plt.setp(ax.collections, alpha=.75)

                sns.boxplot(x=x,
                            y=y,
                            hue=hue,
                            data=subset,
                            order=x_order,
                            whis=np.inf,
                            color="white",
                            dodge=False,
                            ax=ax)

                ax.set_title(title,
                             fontsize=20,
                             fontweight='bold')
                ax.set_ylabel(ylabel,
                              fontsize=20,
                              fontweight='bold')
                ax.set_xlabel(xlabel,
                              fontsize=20,
                              fontweight='bold')

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_axis_off()

                if row_index == (nrows - 1) and col_index == (ncols - 1):
                    if palette is not None:
                        handles = []
                        for label, color in palette.items():
                            handles.append(mpatches.Patch(color=color, label=label))
                        ax.legend(handles=handles, loc=4, fontsize=20)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "expression_describe_overview_boxplot.png"))
        plt.close()


if __name__ == '__main__':
    m = main()
    m.start()
