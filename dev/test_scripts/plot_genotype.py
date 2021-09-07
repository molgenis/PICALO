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
        self.out_filename = getattr(arguments, 'outfile')
        self.sa_path = getattr(arguments, 'sample_annotation')
        self.sample_id = getattr(arguments, 'sample_id')
        self.color_id = getattr(arguments, 'color_id')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette1 = {
            "MAYO": "#9c9fa0",
            "CMC HBCC": "#0877b4",
            "GTEx": "#0fa67d",
            "ROSMAP": "#6950a1",
            "Brain GVEx": "#48b2e5",
            "Target ALS": "#d5c77a",
            "MSBB": "#5cc5bf",
            "NABEC": "#6d743a",
            "LIBD": "#e49d26",
            "ENA": "#d46727",
            "GVEX": "#000000",
            "UCLA ASD": "#f36d2a",
            "CMC": "#eae453"
            }

        self.palette2 = {
            "included": "#0fa67d",
            "3SD filter": "#6950a1",
            "ENA": "#d46727"
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
        parser.add_argument("-g",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")
        parser.add_argument("-cid",
                            "--color_id",
                            type=str,
                            required=False,
                            default=None,
                            choices=["MetaBrain_cohort"],
                            help="The color column(s) name in the -sa / "
                                 "--sample_annotation file.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        required = False
        if "-cid" in sys.argv or "--color_id" in sys.argv:
            required = True

        parser.add_argument("-sa",
                            "--sample_annotation",
                            type=str,
                            required=required,
                            default=None,
                            help="The path to the sample annotation file.")
        parser.add_argument("-sid",
                            "--sample_id",
                            type=str,
                            required=required,
                            default=None,
                            help="The sample column name in the -sa / "
                                 "--sample_annotation file.")

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

        print("Post-processing data.")
        counts_df.reset_index(drop=False, inplace=True)
        counts_df_m = counts_df.melt(id_vars="index")
        counts_df_m["x"] = 1

        hue = None
        hue_order = None
        palette = None
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)
            sample_cohort_dict = dict(zip(sa_df[self.sample_id], sa_df[self.color_id]))
            counts_df_m["hue"] = counts_df_m["index"].map(sample_cohort_dict)
            hue = "hue"
            palette = self.palette1
            hue_order = [x for x in palette.keys()]
        print(counts_df_m)

        self.plot_boxplot(df_m=counts_df_m, variables=columns, y="value", hue=hue, hue_order=hue_order, palette=palette, appendix="_perCohort")

        removed_samples = ['2014-2625', '792_130530', '778_130528', 'Br2173_R4370', '769_130523', 'BM_22_47', 'SRR627449', '2015-1478', '770_130523', '844_130830', '2015-1540', '934_131101', '2015-857', 'CMC_HBCC_RNA_PFC_2992', '2015-2888', '846_130830', '781_130528', '956_131107', '797_130701', '2015-1507', 'UMB5144_ba41_42_22', '2015-1532', '865_130911', '2015-1341', 'SRR5589159', '885_130923', 'PENN_RNA_PFC_62', '1924_TCX', '824_130725', 'PITT_RNA_PFC_1455', '763_130520', '932_131101', 'Br1826_R3715', '11329_TCX', '796_130701', 'PITT_RNA_PFC_1256', '11327_TCX', '767_130523', 'SRR5589098', 'PITT_RNA_PFC_902']
        counts_df_m["group"] = "included"
        counts_df_m.loc[counts_df_m["hue"] == "ENA", "group"] = "ENA"
        counts_df_m.loc[counts_df_m["index"].isin(removed_samples), "group"] = "3SD filter"
        print(counts_df)
        remove_df = counts_df_m.loc[counts_df_m["index"].isin(removed_samples), ["index", "hue"]]
        remove_df.drop_duplicates(inplace=True)
        print(remove_df)
        print(dict(zip(*np.unique(remove_df["hue"].values, return_counts=True))))

        self.plot_boxplot(df_m=counts_df_m, variables=columns, y="value", hue="group", hue_order=["included", "ENA", "3SD filter"], palette=self.palette2, appendix="_included")

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
                ax.set_xlabel(self.color_id,
                              fontsize=20,
                              fontweight='bold')

                ax.tick_params(axis='both', which='major', labelsize=14)
            else:
                ax.set_axis_off()

                if palette is not None and i == (nplots - 1):
                    handles = []
                    for label in hue_order:
                        if label in sizes.keys():
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
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
