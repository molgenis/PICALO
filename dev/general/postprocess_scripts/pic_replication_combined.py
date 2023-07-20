#!/usr/bin/env python3

"""
File:         pic_replication_combined.py
Created:      2023/07/18
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import json
import os
import re

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.stats import multitest
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Local application imports.

"""
Syntax:


### BIOS ### 
./pic_replication_combined.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/pic_replication/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_EvenPICsInOddeQTLs_pic_replication.txt.gz /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/pic_replication/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_OddPICsInEveneQTLs_pic_replication.txt.gz \
    -dl odd_PICs even_PICs \
    -rl even_PICs odd_PICs \
    -t odd_PIC_ieQTLs even_PIC_ieQTLs \
    -p /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -e png

### MetaBrain ###


"""

# Metadata
__program__ = "PIC Replication Combined"
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


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        data_paths = getattr(arguments, 'data_paths')
        self.discovery_labels = getattr(arguments, 'discovery_labels')
        self.replication_labels = getattr(arguments, 'replication_labels')
        self.titles = getattr(arguments, 'titles')
        self.input = list(zip(data_paths, self.discovery_labels, self.replication_labels, self.titles))
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'pic_replication_combined')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
                            help="show program's version number and exit")
        parser.add_argument("-d",
                            "--data_paths",
                            nargs="+",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-dl",
                            "--discovery_labels",
                            nargs="+",
                            type=str,
                            required=True,
                            help="The name of the discovery datasets.")
        parser.add_argument("-rl",
                            "--replication_labels",
                            nargs="+",
                            type=str,
                            required=True,
                            help="The name of the replication datasets.")
        parser.add_argument("-t",
                            "--titles",
                            nargs="+",
                            type=str,
                            required=True,
                            help="The titels.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
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
        self.print_arguments()

        print("Loading interaction results.")
        ieqtl_dfs = {}
        pics = None
        for data_path, discovery_label, replication_label, title in self.input:
            ieqtl_df = self.load_file(data_path,
                                      header=0,
                                      index_col=None)
            print(ieqtl_df)

            pics = set([col.replace("DISC ", "").replace(" beta", "") for col in ieqtl_df.columns if col.startswith("DISC ") and col.endswith(" beta")])
            if pics is None:
                pics = pics
            else:
                pics = pics.intersection(pics)
            print(pics)

            ieqtl_dfs[title] = ieqtl_df

        pics = list(pics)
        pics.sort()

        print("Visualizing.")
        self.combined_plot(dfs=ieqtl_dfs,
                           pics=pics)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def combined_plot(self, dfs, pics):
        nrows = len(pics)
        ncols = len(self.titles)

        self.shared_ylim = {i: (0, 1) for i in range(nrows)}
        self.shared_xlim = {i: (0, 1) for i in range(ncols)}

        sns.set(rc={'figure.figsize': (ncols * 8, nrows * 6)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row')

        for row_index, pic in enumerate(pics):
            for col_index, title in enumerate(self.titles):
                print("\tWorking on '{}-{}'".format(pic, title))

                # Select the required columns.
                plot_df = dfs[title].loc[:, ["DISC {} pvalue".format(pic),
                                             "DISC {} FDR".format(pic),
                                             "DISC {} beta".format(pic),
                                             "DISC {} std".format(pic),
                                             "DISC {} tvalue".format(pic),
                                             "REPL {} pvalue".format(pic),
                                             "REPL {} FDR".format(pic),
                                             "REPL {} beta".format(pic),
                                             "REPL {} std".format(pic),
                                             "REPL {} tvalue".format(pic),
                                             ]].copy()
                plot_df.columns = ["DISC pvalue",
                                   "DISC FDR",
                                   "DISC beta",
                                   "DISC std",
                                   "DISC tvalue",
                                   "REPL pvalue",
                                   "REPL FDR",
                                   "REPL beta",
                                   "REPL std",
                                   "REPL tvalue"]
                plot_df = plot_df.loc[~plot_df["REPL tvalue"].isna(), :]
                plot_df.sort_values(by="DISC pvalue", inplace=True)

                plot_df["facecolors"] = [self.palette[pic] if fdr_value <= 0.05 else "#808080" for fdr_value in plot_df["REPL FDR"]]

                include_ylabel = False
                if col_index == 0:
                    include_ylabel = True

                tmp_title = ""
                if row_index == 0:
                    tmp_title = title.replace("_", " ")

                xlabel = ""
                if row_index == (nrows - 1):
                    xlabel = "{} t-value".format(self.discovery_labels[col_index].replace("_", " "))

                if col_index == 0:
                    axes[row_index, col_index].annotate(
                        pic,
                        xy=(-0.3, 0.9),
                        xycoords=axes[row_index, col_index].transAxes,
                        color=self.palette[pic],
                        fontsize=30
                    )
                xlim, ylim, _ = self.scatterplot(
                    df=plot_df.loc[plot_df["DISC FDR"] <= 0.05, :],
                    fig=fig,
                    ax=axes[row_index, col_index],
                    x="DISC tvalue",
                    y="REPL tvalue",
                    facecolors="facecolors",
                    xlabel=xlabel,
                    ylabel="{} t-value".format(self.replication_labels[col_index].replace("_", " ")),
                    title=tmp_title,
                    color=self.palette[pic],
                    include_ylabel=include_ylabel,
                    pi1_column="REPL pvalue",
                    rb_columns=[("DISC beta", "DISC std"), ("REPL beta", "REPL std")]
                )
                self.update_limits(xlim, ylim, 1, col_index)
                self.update_limits(xlim, ylim, row_index, col_index)

            for (m, n), ax in np.ndenumerate(axes):
                (xmin, xmax) = self.shared_xlim[n]
                (ymin, ymax) = self.shared_ylim[m]

                xmargin = (xmax - xmin) * 0.05
                ymargin = (ymax - ymin) * 0.05

                ax.set_xlim(xmin - xmargin - 1, xmax + xmargin)
                ax.set_ylim(ymin - ymargin, ymax + ymargin)

        # Add the main title.
        fig.suptitle("ieQTL replication within dataset\n(even / odd eQTLs split)",
                     fontsize=30,
                     color="#000000",
                     weight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_PIC_replication.{}".format(self.out_filename, extension)))
        plt.close()

    def scatterplot(self, df, fig, ax, x="x", y="y", facecolors=None,
                    label=None, max_labels=15, xlabel="", ylabel="", title="",
                    color="#000000", ci=95, include_ylabel=True,
                    pi1_column=None, rb_columns=None):
        sns.despine(fig=fig, ax=ax)

        if not include_ylabel:
            ylabel = ""

        if facecolors is None:
            facecolors = "#808080"
        else:
            facecolors = df[facecolors]

        n = df.shape[0]
        concordance = np.nan
        n_concordant = np.nan
        coef = np.nan
        pi1 = np.nan
        rb = np.nan

        if n > 0:
            lower_quadrant = df.loc[(df[x] < 0) & (df[y] < 0), :]
            upper_quadrant = df.loc[(df[x] > 0) & (df[y] > 0), :]
            n_concordant = lower_quadrant.shape[0] + upper_quadrant.shape[0]
            concordance = (100 / n) * n_concordant

            if n > 1:
                coef, p = stats.pearsonr(df[x], df[y])

                if pi1_column is not None:
                    pi1 = self.calculate_p1(p=df[pi1_column])

                if rb_columns is not None:
                    rb_est = self.calculate_rb(
                        b1=df[rb_columns[0][0]],
                        se1=df[rb_columns[0][1]],
                        b2=df[rb_columns[1][0]],
                        se2=df[rb_columns[1][1]],
                        )
                    rb = rb_est[0]

            sns.regplot(x=x, y=y, data=df, ci=ci,
                        scatter_kws={'facecolors': facecolors,
                                     'edgecolors': "#808080"},
                        line_kws={"color": color},
                        ax=ax
                        )

            if label is not None:
                texts = []
                for i, (_, point) in enumerate(df.iterrows()):
                    if i > max_labels:
                        continue
                    texts.append(ax.text(point[x],
                                         point[y],
                                         str(point[label]),
                                         color=color))
                adjust_text(texts,
                            ax=ax,
                            only_move={'points': 'x',
                                       'text': 'xy',
                                       'objects': 'x'},
                            autoalign='x',
                            expand_text=(1., 1.),
                            expand_points=(1., 1.),
                            lim=1000,
                            arrowprops=dict(arrowstyle='-', color='#808080'))

        ax.axhline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
        ax.axvline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)

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

        if not np.isnan(pi1):
            ax.annotate(
                '\u03C01 = {:.2f}'.format(pi1),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(rb):
            ax.annotate(
                'Rb = {:.2f}'.format(rb),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )

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

        stats_df = pd.DataFrame([[n, n_concordant, concordance, coef, pi1, rb]],
                                columns=["N", "N concordant", "concordance", "pearsonr", "pi1", "Rb"],
                                index=[0])

        return (df[x].min(), df[x].max()), (df[y].min(), df[y].max()), stats_df

    def update_limits(self, xlim, ylim, row, col):
        row_ylim = self.shared_ylim[row]
        if ylim[0] < row_ylim[0]:
            row_ylim = (ylim[0], row_ylim[1])
        if ylim[1] > row_ylim[1]:
            row_ylim = (row_ylim[0], ylim[1])
        self.shared_ylim[row] = row_ylim

        col_xlim = self.shared_xlim[col]
        if xlim[0] < col_xlim[0]:
            col_xlim = (xlim[0], col_xlim[1])
        if xlim[1] > col_xlim[1]:
            col_xlim = (col_xlim[0], xlim[1])
        self.shared_xlim[col] = col_xlim

    @staticmethod
    def calculate_p1(p):
        return np.nan
        importr("qvalue")
        pvals = robjects.FloatVector(p)
        lambda_seq = robjects.FloatVector([x for x in np.arange(0.05, 1, 0.05) if p.max() > x])
        pi0est = robjects.r['pi0est'](pvals, lambda_seq)
        return 1 - np.array(pi0est.rx2('pi0'))[0]

    @staticmethod
    def calculate_rb(b1, se1, b2, se2, theta=0):
        robjects.r("source('Rb.R')")
        b1 = robjects.FloatVector(b1)
        se1 = robjects.FloatVector(se1)
        b2 = robjects.FloatVector(b2)
        se2 = robjects.FloatVector(se2)
        calcu_cor_true = robjects.globalenv['calcu_cor_true']
        rb = calcu_cor_true(b1, se1, b2, se2, theta)
        return np.array(rb)[0]

    def print_arguments(self):
        print("Arguments:")
        for data_path, discovery_label, replication_label, title in self.input:
            print("  > {}:".format(title))
            print("    > Input: {}".format(data_path))
            print("    > Discovery Label: {}".format(discovery_label))
            print("    > Replication Label: {}".format(replication_label))
            print("    > Ttitle: {}".format(title))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
