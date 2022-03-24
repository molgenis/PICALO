#!/usr/bin/env python3

"""
File:         plot_genotype.py
Created:      2021/04/08
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
import math
import sys
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Genotype"
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
./plot_genotype.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.geno_path = getattr(arguments, 'genotype')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-g",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")
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
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading genotype data.")
        geno_df = self.load_file(self.geno_path, header=0, index_col=0)
        print(geno_df)

        print("Counting bins.")
        counts_m = np.empty((geno_df.shape[1], 7), dtype=np.uint32)
        indices = np.empty((geno_df.shape[1]), dtype=object)
        columns = ["zero", "one", "two", "missing", "not imputed", "imputed", "badly imputed"]
        for i in range(geno_df.shape[1]):
            genotypes = geno_df.iloc[:, i].to_numpy()
            n_zero = np.sum(genotypes == 0)
            n_one = np.sum(genotypes == 1)
            n_two = np.sum(genotypes == 2)
            n_missing = np.sum(genotypes == -1)
            n_not_imputed = n_zero + n_one + n_two + n_missing
            n_imputed = len(genotypes) - n_not_imputed
            badly_imputed = np.sum(np.logical_or(np.logical_and(genotypes >= 0.25, genotypes <= 0.75), np.logical_and(genotypes >= 1.25, genotypes <= 1.75)))
            counts_m[i, :] = np.array([n_zero, n_one, n_two, n_missing, n_not_imputed, n_imputed, badly_imputed])
            indices[i] = geno_df.columns[i]
        counts_df = pd.DataFrame(counts_m, index=indices, columns=columns)
        print(counts_df)

        print("Loading color data.")
        hue = None
        if self.std_path is not None:
            sa_df = self.load_file(self.std_path, header=0, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["hue"]
            counts_df = counts_df.merge(sa_df, left_index=True, right_index=True, how="left")

            hue = "hue"

        print("Post-processing data.")
        counts_df.reset_index(drop=False, inplace=True)
        counts_df_m = counts_df.melt(id_vars=["index", "hue"])
        counts_df_m["x"] = 1
        print(counts_df_m)

        self.plot_boxplot(df_m=counts_df_m,
                          variables=columns,
                          y="value",
                          hue=hue,
                          palette=self.palette,
                          appendix="_perCohort")

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_boxplot(self, df_m, variables, x="x", y="y", hue=None,
                     hue_order=None, palette=None, appendix=""):
        sizes = {}
        if hue is not None:
            sizes = dict(zip(*np.unique(df_m[hue], return_counts=True)))


        nplots = len(variables) + 1
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
            print(i)
            if nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(variables):
                sns.despine(fig=fig, ax=ax)

                subset = df_m.loc[df_m["variable"] == variables[i], :]

                sns.violinplot(x=x,
                               y=y,
                               hue=hue,
                               hue_order=hue_order,
                               cut=0,
                               data=subset,
                               palette=palette,
                               ax=ax)

                plt.setp(ax.collections, alpha=.75)

                sns.boxplot(x=x,
                            y=y,
                            hue=hue,
                            hue_order=hue_order,
                            data=subset,
                            whis=np.inf,
                            color="white",
                            ax=ax)

                plt.setp(ax.artists, edgecolor='k', facecolor='w')
                plt.setp(ax.lines, color='k')

                ax.get_legend().remove()

                ax.set_title(variables[i],
                             fontsize=25,
                             fontweight='bold')
                ax.set_ylabel("",
                              fontsize=20,
                              fontweight='bold')
                ax.set_xlabel("",
                              fontsize=20,
                              fontweight='bold')

                ax.tick_params(axis='both', which='major', labelsize=14)
            else:
                ax.set_axis_off()

                if palette is not None and i == (nplots - 1):
                    handles = []
                    for label, size in sizes.items():
                        handles.append(mpatches.Patch(color=palette[label], label="{} [n={:.0f}]".format(label, sizes[label] / len(variables))))
                    ax.legend(handles=handles, loc=4, fontsize=25)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "{}_boxplot{}.png".format(self.out_filename, appendix)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
