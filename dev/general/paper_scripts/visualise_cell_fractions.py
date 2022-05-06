#!/usr/bin/env python3

"""
File:         visualise_cell_fractions.py
Created:      2020/11/24
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Visualise Cell Fractions"
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

./visualise_cell_fractions.py \
    -cf /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz \
    -o 2022-05-05-BIOS_CellFractionPercentages \
    -e png pdf

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.cf_path = getattr(arguments, 'cell_fractions')
        self.outfile = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            "Basophil": "#009E73",
            "Neutrophil": "#D55E00",
            "Eosinophil": "#0072B2",
            "Granulocyte": "#808080",
            "Monocyte": "#E69F00",
            "LUC": "#F0E442",
            "Lymphocyte": "#CC79A7"
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
        parser.add_argument("-cf",
                            "--cell_fractions",
                            type=str,
                            required=True,
                            help="The path to the cell fractions matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        cf_df = self.load_file(self.cf_path)

        print("Preprocessing data.")
        cc_dfm = cf_df.melt()
        cc_dfm["value"] = cc_dfm["value"] * 100
        print(cc_dfm)

        print("Plotting.")
        self.plot(df=cc_dfm,
                  hue="variable",
                  palette=self.palette,
                  ylabel="cell fraction %",
                  name=self.outfile)

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None,
                  low_memory=True):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows, low_memory=low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def plot(self, df, x="variable", y="value", hue=None, xlabel="",
             ylabel="", name="", palette=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       sharex="none",
                                       sharey="all",
                                       gridspec_kw={"width_ratios": [0.57, 0.43]})

        self.create_boxplot(fig=fig,
                            ax=ax1,
                            df=df,
                            x=x,
                            y=y,
                            hue=hue,
                            order=["LUC", "Monocyte", "Lymphocyte", "Granulocyte"],
                            xlabel=xlabel,
                            ylabel=ylabel,
                            palette=palette)

        self.create_boxplot(fig=fig,
                            ax=ax2,
                            df=df,
                            x=x,
                            y=y,
                            hue=hue,
                            order=["Basophil", "Eosinophil", "Neutrophil"],
                            xlabel=xlabel,
                            ylabel="",
                            palette=palette)

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_boxplot.{}".format(name, extension)))
        plt.close()

    @staticmethod
    def create_boxplot(fig, ax, df, x="variable", y="value", hue=None,
                       order=None, xlabel="", ylabel="", palette=None):
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       data=df,
                       order=order,
                       palette=palette,
                       cut=0,
                       dodge=False,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    hue=hue,
                    data=df,
                    order=order,
                    whis=np.inf,
                    color="white",
                    dodge=False,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

        # Annotate the mean.
        if order is None:
            order = list(df.index)
        for i, x_value in enumerate(order):
            subset = df.loc[df[x] == x_value, y].copy()
            ax.annotate(
                '{:.0f}%'.format(subset.mean()),
                xy=(i, subset.max() + 2),
                ha="center",
                va="center",
                color="#000000",
                fontsize=15,
                fontweight='bold')

        ax.set_ylim(ax.get_ylim()[0], df[y].max() + 5)

    def print_arguments(self):
        print("Arguments:")
        print("  > Cell fraction file: {}".format(self.cf_path))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
