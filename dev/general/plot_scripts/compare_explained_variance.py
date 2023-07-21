#!/usr/bin/env python3

"""
File:         compare_explained_variance.py
Created:      2022/01/18
Last Changed: 2022/07/25
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Compare Explained Variance"
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
./compare_explained_variance.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_paths = getattr(arguments, 'data')
        self.names = getattr(arguments, 'names')
        self.output_filename = getattr(arguments, 'output')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "PICs": "#0072B2",
            "PCs": "#808080",
        }

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
        parser.add_argument("-d",
                            "--data",
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
        self.print_arguments()

        print("Loading explained variance data.")
        rsquared_dfm_list = []
        coef_dfm_list = []
        std_err_dfm_list = []
        tvalue_dfm_list = []
        for name, fpath in zip(self.names, self.data_paths):
            df = self.load_file(fpath, header=0, index_col=None)
            df["index"] = df["SNPName"] + "_" + df["ProbeName"]

            for col in df.columns:
                if "coef" in col:
                    variable = col.replace(" coef", "")
                    df["{} t-value".format(variable)] = df["{} coef".format(variable)] / df["{} std error".format(variable)]

            for suffix, df_list in (("r-squared", rsquared_dfm_list),
                                    ("coef", coef_dfm_list),
                                    ("std error", std_err_dfm_list),
                                    ("t-value", tvalue_dfm_list)):
                value_vars = [col for col in df.columns if suffix in col]
                dfm = df.melt(id_vars=["index"], value_vars=value_vars).copy()
                dfm["component"] = name
                df_list.append(dfm)
            break

        print("Plot")
        for suffix, df_list in (("r-squared", rsquared_dfm_list),
                                ("coef", coef_dfm_list),
                                ("std error", std_err_dfm_list),
                                ("t-value", tvalue_dfm_list)):
            print("\t{}".format(suffix))
            df = pd.concat(df_list, axis=0)
            df.dropna(inplace=True)

            if suffix == "r-squared":
                self.plot_kdeplot(df=df,
                                  group="variable",
                                  hue="component",
                                  palette=self.palette,
                                  xlabel="R\u00b2",
                                  ylabel="density",
                                  title="Explained variance\nby context components",
                                  filename="{}_rsquared_kdeplot".format(self.output_filename)
                                  )
                self.plot_rsquared_boxplot(
                    df=df,
                    hue="component",
                    palette=self.palette,
                    xlabel="",
                    ylabel="R\u00b2",
                    filename="{}_rsquared_boxplot".format(self.output_filename)
                )
            else:
                self.plot_boxplot(
                    df=df,
                    hue="component",
                    palette=self.palette,
                    xlabel="component",
                    ylabel=suffix,
                    filename="{}_{}_boxplot".format(self.output_filename, suffix)
                )

                df["value"] = df["value"].abs()
                self.plot_barplot(
                    df=df,
                    group_column="component",
                    groups=self.names,
                    x="variable",
                    y="value",
                    xlabel="",
                    ylabel="",
                    palette=self.palette,
                    filename="{}_{}_barplot".format(self.output_filename, suffix)
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

    def plot_kdeplot(self, df, x="value", group=None, hue=None, palette=None, xlabel="",
                     ylabel="", title="", filename="plot"):
        if group is None:
            group = "group"
            df[group] = "1"

        nrows = len(df[group].unique())

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=1,
                                 sharex='none',
                                 sharey='none',
                                 figsize=(12, 9 * nrows))
        sns.set(color_codes=True)

        if nrows == 1:
            axes = [axes]

        for group_value, ax in zip(df[group].unique(), axes):
            subset = df.loc[df[group] == group_value, :].copy()

            sns.despine(fig=fig, ax=ax)

            sns.kdeplot(data=subset,
                        x=x,
                        clip=(0, 1),
                        fill=True,
                        hue=hue,
                        palette=palette,
                        ax=ax)

            # Add the text.
            if hue is None:
                ax.annotate(
                    'avg. R\u00b2 = {:.4f}'.format(subset[x].mean()),
                    xy=(0.03, 0.94),
                    xycoords=ax.transAxes,
                    color="#404040",
                    fontsize=14,
                    fontweight='bold')
                ax.annotate(
                    'N = {:,}'.format(subset.shape[0]),
                    xy=(0.03, 0.94),
                    xycoords=ax.transAxes,
                    color="#404040",
                    fontsize=14,
                    fontweight='bold')
            else:
                hue_groups = subset[hue].unique()

                i = 0
                for hue_group in hue_groups:
                    color = "#404040"
                    if hue_group in palette:
                        color = palette[hue_group]

                    ax.annotate(
                        'avg. R\u00b2 = {:.4f}'.format(subset.loc[subset[hue] == hue_group, "value"].mean()),
                        xy=(0.03, 0.94 - (i * 0.04)),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold')
                    i += 1
                    ax.annotate(
                        'N = {:,}'.format(subset.loc[subset[hue] == hue_group, "value"].shape[0]),
                        xy=(0.03, 0.94 - (i * 0.04)),
                        xycoords=ax.transAxes,
                        color=color,
                        fontsize=14,
                        fontweight='bold')
                    i += 1

            ax.set_title(group_value,
                         fontsize=20,
                         fontweight='bold')
            ax.set_ylabel(ylabel,
                          fontsize=14,
                          fontweight='bold')
            ax.set_xlabel(xlabel,
                          fontsize=14,
                          fontweight='bold')

        fig.suptitle(title,
                     fontsize=40,
                     fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def plot_rsquared_boxplot(self, df, x="variable", y="value", hue=None,
                              palette=None, xlabel="", ylabel="", title="",
                              filename=""):

        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.set(color_codes=True)

        self.boxplot(fig=fig,
                     ax=ax,
                     df=df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette=palette,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     title=title
                     )

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def plot_boxplot(self, df, x="variable", y="value", hue=None,
                     palette=None, xlabel="", ylabel="", filename=""):
        print(df)
        width = 9
        if hue is not None:
            width = len(df[x].unique()) * 0.5

        nrows = 1
        groups = [""]
        if hue is not None:
            groups = df[hue].unique()
            n_rows = len(groups)
        else:
            df["hue"] = ""
            hue = "hue"

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=1,
                                 sharex="all",
                                 figsize=(width, 5 * nrows))
        if nrows == 1:
            axes = [axes]

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
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    @staticmethod
    def boxplot(fig, ax, df, x="variable", y="value", hue=None, palette=None,
                color="#404040", title="", xlabel="", ylabel=""):
        sns.despine(fig=fig, ax=ax)
        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       data=df,
                       color=color,
                       palette=palette,
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

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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

    def print_arguments(self):
        print("Arguments:")
        print("  > Data:")
        for i, (name, data_fpath) in enumerate(zip(self.names, self.data_paths)):
            print("  > {} = {}".format(name, data_fpath))
        print("  > Output filename: {}".format(self.output_filename))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
