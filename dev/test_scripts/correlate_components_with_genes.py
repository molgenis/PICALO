#!/usr/bin/env python3

"""
File:         correlate_components_with_genes.py
Created:      2021/05/25
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
import argparse
import time
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        base_dir = str(Path(__file__).parent.parent)
        self.plot_outdir = os.path.join(base_dir, 'plot')
        self.file_outdir = os.path.join(base_dir, 'correlate_components_with_genes')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

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

        # Make sure order is the same.
        overlap = set(comp_df.columns.tolist()).intersection(set(genes_df.columns.tolist()))
        comp_df = comp_df.loc[:, overlap]
        genes_df = genes_df.loc[:, overlap]

        # Safe the indices.
        components = comp_df.index.tolist()
        genes = genes_df.index.tolist()

        # Convert to numpy.
        comp_m = comp_df.to_numpy()
        genes_m = genes_df.to_numpy()
        del comp_df, genes_df

        # Start correlating.
        print("Starting correlating")
        n_tests = comp_m.shape[0] * genes_m.shape[0]
        count = 0
        last_print_time = None
        results = np.empty((n_tests, 4), dtype=np.float64)
        for comp_idx, comp_name in enumerate(components):
            print("Working on '{}'".format(comp_name))
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 10 or (count + 1) == n_tests:
                last_print_time = now_time
                print("\t{}/{} correlations analysed [{:.2f}%]".format((count + 1), n_tests, (100 / n_tests) * (count + 1)))

            for genes_idx, gene in enumerate(genes):
                r = self.calc_pearsonr(x=comp_m[comp_idx, :],
                                       y=genes_m[genes_idx, :])
                results[count, :] = np.array([comp_idx, genes_idx, r, np.abs(r)])

                count += 1
        # Convert to pandas.
        df = pd.DataFrame(results, columns=["component", "gene", "correlation", "abs correlation"])
        df["component"] = df["component"].map({i: x for i, x in enumerate(components)})
        df["gene"] = df["gene"].map({i: x for i, x in enumerate(genes)})
        df.insert(2, 'gene (HGNC)', df["gene"].map(gene_dict))
        df.sort_values(by="abs correlation", ascending=False, inplace=True)
        print(df)

        # Plot.
        df_m = df.melt(id_vars="component", value_vars=["abs correlation"])
        print(df_m)
        self.plot_distribution(df_m=df_m, variable='component', outdir=self.plot_outdir)

        # Save.
        self.save_file(df=df, outpath=os.path.join(self.file_outdir, "{}.txt.gz".format(self.out_filename)),
                       index=False)

        # Save per component.
        print("Results overview:")
        for component in components:
            subset = df.loc[df["component"] == component, :]
            print("\t{}".format(component))
            for cut_off in [0.9, 0.8, 0.7, 0.6]:
                pos_genes = subset.loc[subset["correlation"] >= cut_off, "gene (HGNC)"].values.tolist()
                neg_genes = subset.loc[subset["correlation"] <= -cut_off, "gene (HGNC)"].values.tolist()
                print("\t\t abs(correlation) > {} [{}]:".format(cut_off, len(pos_genes) + len(neg_genes)))
                print("\t\t\t positive [{}]: {}".format(len(pos_genes), ", ".join(pos_genes)))
                print("\t\t\t negative [{}]: {}".format(len(neg_genes), ", ".join(neg_genes)))
            print("")
            del subset

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_distribution(self, df_m, variable='variable', value='value',
                          outdir=None):
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
    def calc_pearsonr(x, y):
        x_dev = x - np.mean(x)
        y_dev = y - np.mean(y)
        dev_sum = np.sum(x_dev * y_dev)
        x_rss = np.sum(x_dev * x_dev)
        y_rss = np.sum(y_dev * y_dev)
        return dev_sum / np.sqrt(x_rss * y_rss)

    def print_arguments(self):
        print("Arguments:")
        print("  > Components: {}".format(self.components_path))
        print("  > Gene expression: {}".format(self.genes_path))
        print("  > Gene info: {}".format(self.gene_info_path))
        print("  > Plot output directory {}".format(self.plot_outdir))
        print("  > File output directory {}".format(self.file_outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
