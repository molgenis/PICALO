#!/usr/bin/env python3

"""
File:         correlate_components_with_genes.py
Created:      2021/05/25
Last Changed: 2021/11/03
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
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Correlate Components with Genes"
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
./correlate_components_with_genes.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.components_path = getattr(arguments, 'components')
        self.genes_path = getattr(arguments, 'genes')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.plot_outdir = os.path.join(base_dir, 'plot')
        self.file_outdir = os.path.join(base_dir, 'correlate_components_with_genes')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-c",
                            "--components",
                            type=str,
                            required=True,
                            help="The path to the components matrix.")
        parser.add_argument("-g",
                            "--genes",
                            type=str,
                            required=True,
                            help="The path to the gene expression matrix.")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene info matrix.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
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
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading data.")
        comp_df = self.load_file(self.components_path, header=0, index_col=0)
        genes_df = self.load_file(self.genes_path, header=0, index_col=0, nrows=None)
        gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        gene_dict = dict(zip(gene_info_df["ArrayAddress"], gene_info_df["Symbol"]))
        del gene_info_df

        print("Pre-processing data.")
        # Make sure order is the same.
        samples = set(comp_df.columns.tolist()).intersection(set(genes_df.columns.tolist()))
        comp_df = comp_df.loc[:, samples]
        genes_df = genes_df.loc[:, samples]

        # Safe the indices.
        components = comp_df.index.tolist()
        genes = genes_df.index.tolist()

        # Convert to numpy.
        comp_m = comp_df.to_numpy()
        genes_m = genes_df.to_numpy()
        del comp_df, genes_df

        # Calculate correlating.
        print("Correlating.")
        corr_m = np.corrcoef(comp_m, genes_m)[:comp_m.shape[0], comp_m.shape[0]:]
        corr_df = pd.DataFrame(corr_m, index=components)

        print("Post-processing data.")
        corr_df = corr_df.T
        corr_df.insert(0, "index", np.arange(0, corr_df.shape[0]))
        corr_df.insert(1, "ProbeName", genes)
        corr_df.insert(2, 'HGNCName', corr_df["ProbeName"].map(gene_dict))

        corr_df_m = corr_df.melt(id_vars=["index", "ProbeName", "HGNCName"])
        corr_df_m["abs value"] = corr_df_m["value"].abs()
        corr_df_m.sort_values(by="abs value", ascending=False, inplace=True)

        print("Saving file.")
        self.save_file(df=corr_df, outpath=os.path.join(self.file_outdir, "{}_gene_correlations.txt.gz".format(self.out_filename)),
                       index=False)
        corr_df.to_excel(os.path.join(self.file_outdir, "{}_gene_correlations.xlsx".format(self.out_filename)))
        self.save_file(df=corr_df_m, outpath=os.path.join(self.file_outdir, "{}_gene_correlations_molten.txt.gz".format(self.out_filename)),
                       index=False)

        print("Loading color data.")
        facecolors = None
        palette = {}
        if self.palette is not None and self.std_path is not None:
            std_df = self.load_file(self.std_path, header=0, index_col=None)
            std_dict = dict(zip(std_df.iloc[:, 0], std_df.iloc[:, 1]))
            facecolors = []
            for sample in samples:
                facecolors.append(self.palette[std_dict[sample]])
                palette[std_dict[sample]] = self.palette[std_dict[sample]]

        print("Plotting.")
        self.plot_distribution(df_m=corr_df_m, value='abs value', outdir=self.plot_outdir)

        n_plots_per_component = 5
        for i, component in enumerate(components):
            plot_df = pd.DataFrame(comp_m[i, :], index=samples, columns=[component])

            genes = []
            correlations = {}
            for i, (_, row) in enumerate(corr_df_m.loc[corr_df_m["variable"] == component, :].iterrows()):
                if i >= n_plots_per_component:
                    break
                plot_df[row["HGNCName"]] = genes_m[row["index"], :]
                genes.append(row["HGNCName"])
                correlations[row["HGNCName"]] = row["value"]

            self.plot_regplot(df=plot_df, x=component, columns=genes,
                              facecolors=facecolors, palette=palette,
                              outdir=self.plot_outdir)

        print("Printing overview")
        for component in components:
            print("\t{}".format(component))
            for cut_off in [0.9, 0.8, 0.7, 0.6]:
                pos_genes = corr_df.loc[corr_df[component] >= cut_off, "HGNCName"].values.tolist()
                neg_genes = corr_df.loc[corr_df[component] <= -cut_off, "HGNCName"].values.tolist()
                print("\t\t abs(correlation) > {} [{}]:".format(cut_off, len(pos_genes) + len(neg_genes)))
                print("\t\t\t positive [{}]: {}".format(len(pos_genes), ", ".join(pos_genes)))
                print("\t\t\t negative [{}]: {}".format(len(neg_genes), ", ".join(neg_genes)))
            print("")

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
    def save_file(df, outpath, header=True, index=False, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    @staticmethod
    def plot_distribution(df_m, variable='variable', value='value', outdir=None):
        outpath = "abs_correlation_distributions.png"
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)

        sns.set(style="ticks", color_codes=True)
        g = sns.FacetGrid(df_m, col=variable, sharex=True, sharey=True)
        g.map(sns.distplot, value)
        g.set_titles('{col_name}')
        plt.tight_layout()
        g.savefig(outpath)
        plt.close()

    def plot_regplot(self, df, columns, x="x", facecolors=None, palette=None,
                     outdir=None):
        outpath = "{}_correlations_with_genes.png".format(x)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)

        accent_color = "#000000"
        if facecolors is None:
            facecolors = "#000000"
            accent_color = "#b22222"

        nplots = len(columns) + 1
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

            if i < len(columns):
                y = columns[i]

                sns.despine(fig=fig, ax=ax)

                coef, _ = stats.pearsonr(df[y], df[x])

                sns.regplot(x=x, y=y, data=df, ci=95,
                            scatter_kws={'facecolors': facecolors,
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": accent_color,
                                      'linewidth': 5},
                            ax=ax)

                ax.annotate(
                    'N = {}'.format(df.shape[0]),
                    xy=(0.03, 0.94),
                    xycoords=ax.transAxes,
                    color=accent_color,
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')
                ax.annotate(
                    'r = {:.2f}'.format(coef),
                    xy=(0.03, 0.90),
                    xycoords=ax.transAxes,
                    color=accent_color,
                    alpha=1,
                    fontsize=18,
                    fontweight='bold')

                ax.axhline(0, ls='--', color=accent_color, zorder=-1)
                ax.axvline(0, ls='--', color=accent_color, zorder=-1)

                ax.set_xlabel("{}".format(x),
                              fontsize=20,
                              fontweight='bold')
                ax.set_ylabel("{} expression".format(y),
                              fontsize=20,
                              fontweight='bold')
            else:
                ax.set_axis_off()

                if palette is not None and i == (nplots - 1):
                    handles = []
                    for key, value in palette.items():
                        handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=4, fontsize=25)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Components: {}".format(self.components_path))
        print("  > Gene expression: {}".format(self.genes_path))
        print("  > Gene info: {}".format(self.gene_info_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
