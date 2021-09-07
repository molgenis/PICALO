#!/usr/bin/env python3

"""
File:         create_upsetplot.py
Created:      2021/05/10
Last Changed: 2021/07/01
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
import itertools
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import upsetplot as up
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Upsetplot"
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
./create_upsetplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')
        self.eqtl_path = getattr(arguments, 'eqtl')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.cell_type_palette = {
            "Neuron": "#0072B2",
            "Oligodendrocyte": "#009E73",
            "EndothelialCell": "#CC79A7",
            "Macrophage": "#E69F00",
            "Astrocyte": "#D55E00"
        }

        self.comp_palette = {
            1: "#0072B2",
            2: "#009E73",
            3: "#CC79A7",
            4: "#E69F00",
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the eqtl matrix.")
        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Load eQTL data.")
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None)
        eqtl_df.index = eqtl_df["SNPName"] + ":" + eqtl_df["ProbeName"]
        eqtl_df = eqtl_df[["Iteration"]]
        eqtl_df.columns = ["eQTL level"]
        eqtl_hlines = None

        final_iter_data = {}
        for i in range(1, 11):
            pic = "PIC{}".format(i)
            print(pic)

            component_data = {}
            eqtl_level_counts_data = {}
            final_iteration = None
            for j in range(100):
                iteration = "iteration{}".format(j)
                iter_abbreviation = "iter{}".format(j)
                fpath = os.path.join(self.input_directory, pic, "results_{}_df.txt.gz".format(iteration))
                if os.path.exists(fpath):
                    df = self.load_file(fpath, header=0, index_col=None)
                    df.index = df["SNP"] + ":" + df["gene"]
                    signif_ieqtl = set(df.loc[df["FDR"] < 0.05, :].index.tolist())
                    component_data[iter_abbreviation] = signif_ieqtl
                    final_iteration = iter_abbreviation

                    eqtl_df.loc[:, iter_abbreviation] = 0
                    eqtl_df.loc[eqtl_df.index.isin(signif_ieqtl), iter_abbreviation] = 1
                    eqtl_level_counts_data[iter_abbreviation] = dict(zip(*np.unique(eqtl_df.loc[eqtl_df[iter_abbreviation] == 1, "eQTL level"], return_counts=True)))

                    if eqtl_hlines is None:
                        overlap_df = eqtl_df.loc[eqtl_df.index.isin(df.index), :]
                        eqtl_hlines = zip(*np.unique(overlap_df["eQTL level"], return_counts=True))
                        eqtl_hlines = {a: (b / overlap_df.shape[0]) * 100 for a, b in eqtl_hlines}

            if final_iteration is not None:
                final_iter_data["comp{}".format(i)] = component_data[final_iteration]

            # plot lineplot.
            if len(eqtl_level_counts_data.keys()) > 0:
                level_df = pd.DataFrame(eqtl_level_counts_data)
                level_df = (level_df / level_df.sum(axis=0)) * 100
                level_df.reset_index(drop=False, inplace=True)
                level_df_m = level_df.melt(id_vars=["index"])
                self.plot(df_m=level_df_m, x="variable", y="value", hue="index",
                          ylabel="%", palette=self.comp_palette,
                          filename=pic, outdir=self.outdir,
                          hlines=eqtl_hlines)

            n_iterations = len(component_data.keys())
            modulo = 1
            if n_iterations <= 25:
                modulo = 5
            elif 25 < n_iterations <= 50:
                modulo = 8
            elif 50 < n_iterations <= 75:
                modulo = 10
            elif n_iterations > 75:
                modulo = 15

            # filter.
            filtered_component_data = {}
            for i, iteration in enumerate(component_data.keys()):
                if (i == 0) or (i % modulo == 0) or (iteration == final_iteration):
                    filtered_component_data[iteration] = component_data[iteration]
            del component_data
            print(filtered_component_data.keys())

            if len(filtered_component_data.keys()) > 1:
                counts = self.count(filtered_component_data)
                counts = counts[counts > 0]
                print(counts)

                print("Creating plot.")
                up.plot(counts, sort_by='cardinality', show_counts=True)
                plt.savefig(os.path.join(self.outdir,
                                         "included_ieQTLs_{}_upsetplot.png".format(pic)))
                plt.close()

        counts = self.count(final_iter_data)
        counts = counts[counts > 0]
        print(counts)

        print("Creating plot.")
        up.plot(counts, sort_by='cardinality', show_counts=True)
        plt.savefig(os.path.join(self.outdir, "included_ieQTLs_finalIter_upsetplot.png"))
        plt.close()

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def count(input_data):
        combinations = []
        cols = list(input_data.keys())
        for i in range(1, len(cols) + 1):
            combinations.extend(list(itertools.combinations(cols, i)))

        abbreviations = {"CellMapNNLS_Neuron": "neuro",
                         "CellMapNNLS_Oligodendrocyte": "oligo",
                         "CellMapNNLS_EndothelialCell": "endo",
                         "CellMapNNLS_Macrophage": "macro",
                         "CellMapNNLS_Astrocyte": "astro"}
        abbr_cols = []
        for col in cols:
            if col in abbreviations.keys():
                abbr_cols.append(abbreviations[col])
            else:
                abbr_cols.append(col)

        indices = []
        data = []
        for combination in combinations:
            index = []
            for col in cols:
                if col in combination:
                    index.append(True)
                else:
                    index.append(False)

            background = set()
            for key in cols:
                if key not in combination:
                    work_set = input_data[key].copy()
                    background.update(work_set)

            overlap = None
            for key in combination:
                work_set = input_data[key].copy()
                if overlap is None:
                    overlap = work_set
                else:
                    overlap = overlap.intersection(work_set)

            duplicate_set = overlap.intersection(background)
            length = len(overlap) - len(duplicate_set)

            indices.append(index)
            data.append(length)

        s = pd.Series(data,
                      index=pd.MultiIndex.from_tuples(indices, names=abbr_cols))
        s.name = "value"
        return s

    @staticmethod
    def plot(df_m, x="x", y="y", hue=None, title="", xlabel="", ylabel="",
             palette=None, filename="plot", outdir=None, hlines=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       gridspec_kw={"width_ratios": [0.99, 0.01]})
        sns.despine(fig=fig, ax=ax1)

        g = sns.lineplot(data=df_m,
                         x=x,
                         y=y,
                         units=hue,
                         hue=hue,
                         palette=palette,
                         estimator=None,
                         legend=None,
                         ax=ax1)

        for label, value in hlines.items():
            ax1.axhline(value, ls='--', color=palette[label], zorder=-1)

        ax1.set_ylim((0, 100))

        ax1.set_title(title,
                      fontsize=14,
                      fontweight='bold')
        ax1.set_xlabel(xlabel,
                       fontsize=10,
                       fontweight='bold')
        ax1.set_ylabel(ylabel,
                       fontsize=10,
                       fontweight='bold')

        plt.setp(ax1.set_xticklabels(ax1.get_xmajorticklabels(), rotation=45))

        if palette is not None:
            handles = []
            for key, color in palette.items():
                if key in df_m[hue].values.tolist():
                    handles.append(mpatches.Patch(color=color, label=key))
            ax2.legend(handles=handles, loc="center")
        ax2.set_axis_off()

        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "{}_lineplot.png".format(filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.input_directory))
        print("  > EQTL path directory: {}".format(self.input_directory))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
