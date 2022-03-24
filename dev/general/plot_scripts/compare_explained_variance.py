#!/usr/bin/env python3

"""
File:         compare_explained_variance.py
Created:      2022/01/18
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
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Compare Explained Variance"
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
./compare_explained_variance.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_paths = getattr(arguments, 'data')
        self.interaction_paths = getattr(arguments, 'interaction')
        self.names = getattr(arguments, 'names')
        self.output_filename = getattr(arguments, 'output')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "PICs": "#0072B2",
            "PCs": "#808080",
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
        parser.add_argument("-d",
                            "--data",
                            nargs="*",
                            type=str,
                            required=False,
                            help="The paths to the input data.")
        parser.add_argument("-i",
                            "--interaction",
                            nargs="*",
                            type=str,
                            required=False,
                            help="The paths to the input data.")
        parser.add_argument("-n",
                            "--names",
                            nargs="*",
                            type=str,
                            required=False,
                            help="The names of the data files.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            default="PlotPerColumn_ColorByCohort",
                            help="The name of the output file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading explained variance data.")
        rsquared_df_list = []
        tvalue_dfm_list = []
        for name, fpath in zip(self.names, self.data_paths):
            df = self.load_file(fpath, header=0, index_col=None)
            df.index = df["snp"] + "_" + df["gene"]

            rsquared_df = df.loc[:, ["r-squared"]]
            rsquared_df.columns = [name]
            rsquared_df_list.append(rsquared_df)

            inter_coef_df = df.loc[:, [x for x in df.columns if x.endswith("xSNP coef")]]
            inter_coef_df.columns = [i + 1 for i in range(inter_coef_df.shape[1])]
            inter_stderr_df = df.loc[:, [x for x in df.columns if x.endswith("xSNP std err")]]
            inter_stderr_df.columns = [i + 1 for i in range(inter_coef_df.shape[1])]

            inter_tvalue_df = inter_coef_df / inter_stderr_df
            inter_tvalue_df["index"] = inter_tvalue_df.index
            inter_tvalue_dfm = inter_tvalue_df.melt(id_vars=["index"])
            inter_tvalue_dfm["name"] = name
            tvalue_dfm_list.append(inter_tvalue_dfm)

        rsquared_df = pd.concat(rsquared_df_list, axis=1)
        tvalue_dfm = pd.concat(tvalue_dfm_list, axis=0)

        print("Filtering on significant ieQTLs.")
        for name, fpath in zip(self.names, self.interaction_paths):
            fdr_df = self.load_file(fpath, sep=",", header=0, index_col=0)
            ieqtls = list(fdr_df.loc[fdr_df.sum(axis=1) == 0, :].index)
            rsquared_df.loc[ieqtls, name] = np.nan
            tvalue_dfm.loc[(tvalue_dfm["index"].isin(rsquared_df)) & (tvalue_dfm["name"] == name), "value"] = np.nan
        print(rsquared_df)
        print(tvalue_dfm)

        rsquared_dfm = rsquared_df.melt()
        rsquared_dfm.dropna(inplace=True)
        tvalue_dfm.dropna(inplace=True)

        print("Plot")
        self.plot_kdeplot(df=rsquared_dfm,
                          hue="variable",
                          palette=self.palette,
                          xlabel="R\u00b2",
                          ylabel="density",
                          title="Explained variance\nby context components",
                          filename="{}_kdeplot".format(self.output_filename)
                          )
        self.plot_boxplot(df=tvalue_dfm,
                          hue="name",
                          palette=self.palette,
                          xlabel="component",
                          ylabel="t-value",
                          name="{}_boxplot".format(self.output_filename)
                          )

        tvalue_dfm["value"] = tvalue_dfm["value"].abs()
        self.plot_barplot(
            df=tvalue_dfm,
            group_column="name",
            groups=self.names,
            x="variable",
            y="value",
            xlabel="",
            ylabel="",
            palette=self.palette,
            filename="{}_barplot".format(self.output_filename)
        )

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_kdeplot(self, df, x="value", hue=None, palette=None, xlabel="",
                     ylabel="", title="", filename="plot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.kdeplot(data=df,
                    x=x,
                    clip=(0, 1),
                    fill=True,
                    hue=hue,
                    palette=palette,
                    ax=ax)

        # Add the text.
        if hue is None:
            ax.annotate(
                'avg. R\u00b2 = {:.2f}'.format(df[x].mean()),
                xy=(0.03, 0.94),
                xycoords=ax.transAxes,
                color="#404040",
                fontsize=14,
                fontweight='bold')
            ax.annotate(
                'N = {:,}'.format(df.shape[0]),
                xy=(0.03, 0.94),
                xycoords=ax.transAxes,
                color="#404040",
                fontsize=14,
                fontweight='bold')
        else:
            groups = df[hue].unique()

            i = 0
            for group in groups:
                color = "#404040"
                if group in palette:
                    color = palette[group]

                ax.annotate(
                    'avg. R\u00b2 = {:.2f}'.format(df.loc[df[hue] == group, "value"].mean()),
                    xy=(0.03, 0.94 - (i * 0.04)),
                    xycoords=ax.transAxes,
                    color=color,
                    fontsize=14,
                    fontweight='bold')
                i += 1
                ax.annotate(
                    'N = {:,}'.format(df.loc[df[hue] == group, "value"].shape[0]),
                    xy=(0.03, 0.94 - (i * 0.04)),
                    xycoords=ax.transAxes,
                    color=color,
                    fontsize=14,
                    fontweight='bold')
                i += 1

        ax.set_title(title,
                     fontsize=20,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        outpath = os.path.join(self.outdir, "{}.png".format(filename))
        fig.savefig(outpath)
        plt.close()

    def plot_boxplot(self, df, x="variable", y="value", hue=None, palette=None,
                     xlabel="", ylabel="", name=""):
        width = 9
        if hue is not None:
            width = len(df[x].unique()) * 0.5

        n_rows = 1
        groups = [""]
        if hue is not None:
            groups = df[hue].unique()
            n_rows = len(groups)
        else:
            df["hue"] = ""
            hue = "hue"

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=n_rows,
                                 ncols=1,
                                 sharex="all",
                                 figsize=(width, 5 * n_rows))

        for ax, group in zip(axes, groups):
            color = "#404040"
            if group in palette:
                color = palette[group]

            if group != groups[-1]:
                xlabel = ""

            self.boxplot(fig=fig,
                         ax=ax,
                         df=df.loc[df[hue] == group, :],
                         x=x,
                         y=y,
                         color=color,
                         title=group,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         )

        plt.tight_layout()
        fig.savefig(os.path.join(self.outdir, "{}.png".format(name)))
        plt.close()

    @staticmethod
    def boxplot(fig, ax, df, x="variable", y="value", color="#404040",
                title="", xlabel="", ylabel=""):
        sns.despine(fig=fig, ax=ax)
        sns.violinplot(x=x,
                       y=y,
                       data=df,
                       color=color,
                       cut=0,
                       dodge=False,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    data=df,
                    color="white",
                    dodge=False,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_title(title,
                     fontsize=20,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

    def plot_barplot(self, df, group_column, groups, x="x", y="y", xlabel="",
                         ylabel="", palette=None, filename=""):
        if df.shape[0] <= 2:
            return

        nplots = len(groups)
        ncols = 1
        nrows = nplots

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharey="all",
                                 figsize=(12 * ncols, 12))
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
                plot_df = df.loc[df[group_column] == groups[i], :].copy()
                plot_df.dropna(inplace=True)

                color = "#404040"
                if palette is not None:
                    color = palette[groups[i]]

                sns.despine(fig=fig, ax=ax)

                g = sns.barplot(x=x,
                                y=y,
                                color=color,
                                dodge=False,
                                data=plot_df,
                                ax=ax)

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

                ax.set_title(groups[i],
                             fontsize=25,
                             fontweight='bold')
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        plt.tight_layout()
        outpath = os.path.join(self.outdir, "{}.png".format(filename))
        fig.savefig(outpath)
        plt.close()
        print("\tSaved: {}".format(outpath))

    def print_arguments(self):
        print("Arguments:")
        print("  > Data:")
        for i, (name, data_fpath, interaction_fpath) in enumerate(zip(self.names, self.data_paths, self.interaction_paths)):
            print("  > {} = {} / {}".format(name, data_fpath, interaction_fpath))
        print("  > Output filename: {}".format(self.output_filename))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
