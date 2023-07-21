#!/usr/bin/env python3

"""
File:         plot_cf_per_pic.py
Created:      2022/05/10
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
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
from scipy import stats
from colour import Color
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot Cell Fraction per PIC"
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
./plot_cf_per_pic.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.cf_path = getattr(arguments, 'cell_fractions')
        self.pics_path = getattr(arguments, 'pics')
        self.selection = getattr(arguments, 'selection')
        self.center = getattr(arguments, 'center')
        self.outfile = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')
        self.n_bins = 10
        self.hue_order = [i for i in range(self.n_bins + 1)]

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = self.create_colormap()
        print(self.palette)
        print(self.hue_order)

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
        parser.add_argument("-p",
                            "--pics",
                            type=str,
                            required=True,
                            help="The path to the PICS matrix.")
        parser.add_argument("-s",
                            "--selection",
                            nargs="*",
                            type=str,
                            default=None,
                            help="The PICs to include.")
        parser.add_argument("-center",
                            action='store_true',
                            help="Center cell fractions.")
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
        pics_df = self.load_file(self.pics_path).T
        if self.selection is None:
            self.selection = pics_df.columns.tolist()

        print("Preprocessing data.")
        cf_df = cf_df * 100
        if self.center:
            cf_df = cf_df.subtract(cf_df.mean(axis=0), axis=1)
        cf_df["index"] = cf_df.index
        cc_dfm = cf_df.melt(id_vars=["index"],
                            var_name="cell type",
                            value_name="proportion")
        cc_dfm.dropna(inplace=True)
        del cf_df

        print("Plotting.")
        for pic in self.selection:
            print("\t{}".format(pic))
            df = cc_dfm.copy()
            loading_dict = dict(zip(pics_df.index, pics_df[pic]))
            df["loading"] = df["index"].map(loading_dict)

            # Bin the PIC loadings.
            lower_threshold = -np.inf
            x = 1
            for x in range(self.n_bins):
                upper_threshold = stats.norm.ppf(x / self.n_bins)
                df.loc[(df["loading"] > lower_threshold) & (df["loading"] <= upper_threshold), "bin"] = x
                lower_threshold = upper_threshold
            df.loc[df["loading"] > lower_threshold, "bin"] = x + 1

            self.plot_boxplot(
                df=df,
                x="cell type",
                y="proportion",
                hue="bin",
                hue_order=self.hue_order,
                palette=self.palette,
                hline=self.center,
                xlabel="",
                ylabel="cell fraction % change" if self.center else "cell fraction %",
                title=pic,
                filename="{}_{}".format(self.outfile, pic)
            )

            del df

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None,
                  low_memory=True):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows, low_memory=low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def create_colormap(self):
        values = [i for i in range(self.n_bins + 1)]
        colors = ['#' + ''.join(f'{int(i * 255):02X}' for i in x) for x in sns.diverging_palette(240, 10, n=self.n_bins + 1)]

        color_map = {}
        for val, col in zip(values, colors):
            color_map[val] = col

        print(color_map)
        return color_map

    def plot_boxplot(self, df, x="variable", y="value", hue=None, hue_order=None,
                     hline=False, palette=None, xlabel="", ylabel="", title="",
                     filename=""):

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(x=x,
                       y=y,
                       hue=hue,
                       hue_order=hue_order,
                       data=df,
                       palette=palette,
                       cut=0,
                       dodge=True,
                       ax=ax)

        plt.setp(ax.collections, alpha=.75)

        sns.boxplot(x=x,
                    y=y,
                    hue=hue,
                    hue_order=hue_order,
                    data=df,
                    color="white",
                    dodge=True,
                    ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        if hline:
            ax.axhline(0, ls='--', color="#000000", alpha=0.5, zorder=-1,
                       linewidth=2)

        ax.set_title(title,
                     fontsize=20,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Cell fraction path: {}".format(self.cf_path))
        print("  > Pics path: {}".format(self.pics_path))
        print("  > Selection: {}".format(self.selection))
        print("  > Center: {}".format(self.center))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
