#!/usr/bin/env python3

"""
File:         bryois_pic_zscore_enrichment.py
Created:      2022/04/22
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
import glob
import os
import re

# Third party imports.
import numpy as np
import pandas as pd
from functools import reduce
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

"""
Syntax:
./bryois_pic_zscore_enrichment.py -h
"""

# Metadata
__program__ = "Bryois PIC Z-score Enrichment"
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
        self.discovery_indir = getattr(arguments, 'discovery_indir')
        self.discovery_alleles = getattr(arguments, 'discovery_alleles')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        self.bryois_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/julienbryois2021/JulienBryois2021SummaryStats.txt.gz"
        self.bryois_n = 196

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'bryois_pic_replication')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

        self.bryois_ct_trans = {
            "Astrocytes": "Astrocyte",
            "EndothelialCells": "EndothelialCell",
            "ExcitatoryNeurons": "Excitatory",
            "InhibitoryNeurons": "Inhibitory",
            "Microglia": "Microglia",
            "Oligodendrocytes": "Oligodendrocyte",
            "OPCsCOPs": "OPCsCOPs",
            "Pericytes": "Pericyte"
        }

        self.palette[False] = "#808080"
        self.palette[True] = "#009E73"

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
        parser.add_argument("-di",
                            "--discovery_indir",
                            type=str,
                            required=True,
                            help="The path to the discovery deconvolution "
                                 "results input directory")
        parser.add_argument("-da",
                            "--discovery_alleles",
                            type=str,
                            required=True,
                            help="The path to the discovery genotype"
                                 " alleles matrix.")
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

        print("Loading replication data")
        replication_df = self.load_file(self.bryois_path, header=0, index_col=0)
        replication_df.drop(["SNP"], axis=1, inplace=True)
        replication_df.reset_index(drop=False, inplace=True)
        replication_df_list = []
        for suffix in ["p-value", "beta"]:
            repl_subset_df = replication_df.loc[:, ["index", "effect_allele"] + [col for col in replication_df if col.endswith(suffix)]].copy()
            repl_subset_df.columns = [col.replace(" {}".format(suffix), "") for col in repl_subset_df.columns]
            repl_subset_df.columns = [self.bryois_ct_trans[col] if col in self.bryois_ct_trans else col for col in repl_subset_df.columns]
            if suffix == "beta":
                cell_types = repl_subset_df.columns[2:]
                repl_subset_df.loc[:, cell_types] = repl_subset_df.loc[:, cell_types].abs()
                # repl_subset_df.loc[:, cell_types] = repl_subset_df.loc[:, cell_types].subtract(repl_subset_df.loc[:, cell_types].mean(axis=1), axis=0)
                repl_subset_df.loc[:, cell_types] = (repl_subset_df.loc[:, cell_types] - repl_subset_df.loc[:, cell_types].mean(axis=0)) / repl_subset_df.loc[:, cell_types].std(axis=0)

            repl_subset_df = repl_subset_df.melt(id_vars=["index", "effect_allele"])
            repl_subset_df.columns = ["index", "effect_allele", "cell type", suffix]
            replication_df_list.append(repl_subset_df)
        replication_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['index', 'effect_allele', 'cell type']), replication_df_list)
        print(replication_df)

        print("Loading interaction results.")
        inpaths = glob.glob(os.path.join(self.discovery_indir, "*.txt.gz"))
        inpaths.sort(key=self.natural_keys)
        groups = []
        data = {}
        for inpath in inpaths:
            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue

            discovery_ieqtl_df = self.load_file(inpath, header=0, index_col=None)
            discovery_ieqtl_df.index = discovery_ieqtl_df["gene"].str.split(".", expand=True)[0] + "_" + discovery_ieqtl_df["SNP"].str.split(":", expand=True)[2]

            signif = list(discovery_ieqtl_df.loc[discovery_ieqtl_df["FDR"] < 0.05, :].index)
            not_signif = list(discovery_ieqtl_df.loc[discovery_ieqtl_df["FDR"] >= 0.05, :].index)

            pic_replication_df = replication_df.copy()
            pic_replication_df["signif"] = np.nan
            pic_replication_df.loc[pic_replication_df["index"].isin(signif), "signif"] = True
            pic_replication_df.loc[pic_replication_df["index"].isin(not_signif), "signif"] = False
            pic_replication_df.dropna(inplace=True)

            self.plot_single_boxplot(
                df=pic_replication_df,
                x="cell type",
                y="beta",
                hue="signif",
                palette=self.palette,
                xlabel="",
                ylabel="abs(beta) z-score",
                title=filename,
                filename="{}_{}_beta_boxplot".format(self.out_filename,
                                                     filename)
            )

            groups.append(filename)
            data[filename] = pic_replication_df

        self.plot_multiple_boxplot(
            data=data,
            groups=groups,
            x="cell type",
            y="beta",
            hue="signif",
            palette=self.palette,
            xlabel="",
            ylabel="abs(beta) z-score",
            filename="{}_beta_boxplot".format(self.out_filename)
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

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def plot_single_boxplot(self, df, x="variable", y="value", hue=None,
                     palette=None, xlabel="", ylabel="", title="",
                            filename=""):

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       gridspec_kw={"width_ratios": [0.99, 0.01]})

        self.boxplot(fig=fig,
                     ax=ax1,
                     df=df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette=palette,
                     xlabel=xlabel,
                     ylabel=ylabel,
                     title=title)

        ax2.set_axis_off()
        if palette is not None and hue is not None:
            handles = []
            for key, value in palette.items():
                if key in df[hue].unique():
                    handles.append(mpatches.Patch(color=value, label=key))
            ax2.legend(handles=handles, fontsize=10)

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def plot_multiple_boxplot(self, data, groups, x="variable", y="value", hue=None,
                     palette=None, xlabel="", ylabel="", filename=""):
        ngroups = len(groups)
        ncols = int(np.ceil(np.sqrt((ngroups + 1))))
        nrows = int(np.ceil((ngroups + 1) / ncols))

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex="none",
                                 sharey="row",
                                 figsize=(12 * ncols, 9 * nrows))
        sns.set(color_codes=True)

        unique_hues = set()
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

            if i < ngroups:
                group = groups[i]
                df = data[group]

                if palette is not None and hue is not None:
                    unique_hues.update(df[hue].unique())

                self.boxplot(fig=fig,
                             ax=ax,
                             df=df,
                             x=x,
                             y=y,
                             hue=hue,
                             palette=palette,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=group)
            elif i == (ncols * nrows):
                if palette is not None and hue is not None:
                    handles = []
                    for key, value in palette.items():
                        if key in unique_hues:
                            handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, fontsize=10)
                else:
                    ax.set_axis_off()
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir,
                                   "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    @staticmethod
    def boxplot(fig, ax, df, x="variable", y="value", hue=None,
                palette=None, xlabel="", ylabel="", title=""):
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       data=df,
                       palette=palette,
                       cut=0,
                       dodge=True,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    hue=hue,
                    data=df,
                    color="white",
                    dodge=True,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        if hue is not None:
            new_xlabels = []
            for label in ax.get_xticklabels():
                xvalue = label.get_text()
                xvalue_subset = df.loc[df[x] == xvalue, :].copy()
                xvalue_hue_counts = xvalue_subset[hue].value_counts()
                n_false = 0
                if False in xvalue_hue_counts:
                    n_false = xvalue_hue_counts[False]

                n_true = 0
                if True in xvalue_hue_counts:
                    n_true = xvalue_hue_counts[True]

                new_xlabels.append("{}\n{:,}/{:,}".format(xvalue, n_false, n_true))
            ax.set_xticklabels(new_xlabels, rotation=0)
        else:
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

    def print_arguments(self):
        print("Arguments:")
        print("  > Discovery:")
        print("    > Input directory: {}".format(self.discovery_indir))
        print("    > Alleles path: {}".format(self.discovery_alleles))
        print("  > Replication:")
        print("    > File path: {}".format(self.bryois_path))
        print("    > N: {}".format(self.bryois_n))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
