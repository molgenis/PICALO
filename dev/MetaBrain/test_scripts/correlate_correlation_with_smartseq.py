#!/usr/bin/env python3

"""
File:         Correlate_correlations_with_smartseq.py
Created:      2021/05/25
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
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Correlate Correlations with SmartSeq"
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
./correlate_correlation_with_smartseq.py -h 
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.gc_path = getattr(arguments, 'gene_correlations')
        self.ss_path = getattr(arguments, 'smart_seq')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.colormap = {
            "GABA": "#0072B2",
            "GLUT": "#56B4E9",
            "Oligodendro/OPC": "#5d9166",
            "Astrocytes": "#9b7bb8",
            "undefined": "#000000"
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
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            required=True,
                            help="The path to the gene correlations matrix.")
        parser.add_argument("-ss",
                            "--smart_seq",
                            type=str,
                            required=True,
                            help="The path to the smart seq matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading data.")
        gc_df = self.load_file(self.gc_path, header=0, index_col=None)
        print(gc_df)
        ss_df = self.load_file(self.ss_path, skiprows=[0, 1, 2, 3, 4, 5], header=0, index_col=None)
        ss_df.index = ss_df.iloc[:, 0]
        ss_df = ss_df.iloc[:, 11:]
        ss_df = ss_df.astype(float)
        ss_df.drop_duplicates(inplace=True)
        ss_df = ss_df - ss_df.mean(axis=0)
        ss_genes = set(ss_df.index.tolist())
        ss_cell_types = ss_df.columns.tolist()
        cell_types = list(set([x.split(".")[0].split("_")[0] for x in ss_cell_types]))
        cell_types.sort()
        print(ss_df)

        print("Correlate")
        forest_data = []
        correlations = []
        indices = []
        for i in range(10):
            component = "component{}".format(i)
            component_gc_df = gc_df.loc[gc_df["component"] == component, :].copy()
            component_gc_df.dropna(inplace=True)
            if component_gc_df.shape[0] == 0:
                continue

            print("\t{}".format(component))

            component_gc_df.set_index("gene (HGNC)", inplace=True)
            overlap = set(component_gc_df.index.tolist()).intersection(ss_genes)
            print(len(overlap))
            component_gc_df = component_gc_df.loc[overlap, :]
            component_ss_df = ss_df.loc[overlap, :]

            coefficients = {}
            counts = {}
            for ss_cell_type in ss_cell_types:
                cell_type = ss_cell_type.split(".")[0].split("_")[0]
                coef, p = stats.pearsonr(component_ss_df[ss_cell_type], component_gc_df["correlation"])
                if cell_type in coefficients.keys():
                    coefs = coefficients[cell_type]
                    coefs.append(coef)
                    coefficients[cell_type] = coefs
                else:
                    coefficients[cell_type] = [coef]

                if cell_type in counts.keys():
                    counts[cell_type] += 1
                else:
                    counts[cell_type] = 1

            comp_results = []
            for cell_type in cell_types:
                comp_results.append(np.mean(coefficients[cell_type]))
                forest_data.append([component, cell_type, np.mean(coefficients[cell_type]), np.std(coefficients[cell_type]), counts[cell_type]])

            results = [(key, np.mean(value), np.std(value), np.abs(np.mean(value))) for key, value in coefficients.items()]

            results.sort(key=lambda x: -x[3])
            for ss_cell_type, mean, std, _ in results:
                print("\t  {}: mean = {:.2f} std = {:.2f}".format(ss_cell_type, mean, std))
            print("")

            correlations.append(comp_results)
            indices.append(component)

        corr_df = pd.DataFrame(correlations, index=indices, columns=cell_types)
        print(corr_df)

        self.plot_heatmap(corr_df)
        #
        # forest_df = pd.DataFrame(forest_data, columns=["component", "cell type", "mean", "std", "n"])
        # forest_df["LB"] = forest_df["mean"] - 1.96 * (forest_df["mean"] / np.sqrt(forest_df["n"]))
        # forest_df["UB"] = forest_df["mean"] + 1.96 * (forest_df["mean"] / np.sqrt(forest_df["n"]))
        # print(forest_df)
        #
        # for component in forest_df["component"].unique():
        #     self.plot_stripplot(df=forest_df.loc[forest_df["component"] == component, :], name=component)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_heatmap(self, corr_df):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        col_colors = [self.colormap[ct.split("_")[0]] for ct in corr_df.columns]

        sns.set(color_codes=True)
        g = sns.clustermap(corr_df, cmap=cmap,
                           row_cluster=False, col_cluster=True,
                           yticklabels=True, xticklabels=True, square=True,
                           vmin=-1, vmax=1, annot=corr_df.round(2),
                           col_colors=col_colors, fmt='',
                           annot_kws={"size": 16, "color": "#000000"},
                           figsize=(12, 12))
        plt.setp(
            g.ax_heatmap.set_yticklabels(
                g.ax_heatmap.get_ymajorticklabels(),
                fontsize=16, rotation=0))
        plt.setp(
            g.ax_heatmap.set_xticklabels(
                g.ax_heatmap.get_xmajorticklabels(),
                fontsize=16, rotation=90))

        plt.tight_layout()
        g.savefig(os.path.join(self.outdir, "{}_SmartSeq_corr_heatmap.png".format(self.out_filename)))
        plt.close()

    def plot_stripplot(self, df, name=""):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        df_m = df.melt(id_vars=["component", "cell type"], value_vars=["LB", "UB"])
        sns.pointplot(x="value",
                      y="cell type",
                      data=df_m,
                      join=False,
                      palette=self.colormap,
                      ax=ax)

        sns.catplot(x="mean",
                    y="cell type",
                      data=df,
                      size=25,
                      dodge=False,
                      orient="h",
                      palette=self.colormap,
                      linewidth=1,
                      edgecolor="w",
                      jitter=0,
                      ax=ax)

        ax.set_ylabel('',
                      fontsize=12,
                      fontweight='bold')
        ax.set_xlabel('Spearman r',
                      fontsize=12,
                      fontweight='bold')
        ax.set_title(name,
                     fontsize=20,
                     fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=10)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

        ax.set_xlim([-1, 1])

        fig.savefig(os.path.join(self.outdir, "{}_SmartSeq_forestplot_{}.pdf".format(self.out_filename, name)), dpi=300)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Component - gene correlations data: {}".format(self.gc_path))
        print("  > SmartSeq data: {}".format(self.ss_path))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
