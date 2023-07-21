#!/usr/bin/env python3

"""
File:         create_comparison_scatterplot.py
Created:      2021/06/30
Last Changed: 2021/09/23
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

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Create Comparison Scatterplot"
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
./create_comparison_scatterplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.nrows = getattr(arguments, 'n_rows')
        self.transpose = getattr(arguments, 'transpose')
        self.annot_path = getattr(arguments, 'annot')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        outdir = getattr(arguments, 'outdir')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, 'plot')
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-n",
                            "--n_rows",
                            type=int,
                            required=False,
                            default=None,
                            help="The number of rows to plot. Default: all.")
        parser.add_argument("-transpose",
                            action='store_true',
                            help="Transpose.")
        parser.add_argument("-a",
                            "--annot",
                            type=str,
                            required=False,
                            help="The path to the annotation matrix.")
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
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfile",
                            type=str,
                            required=False,
                            default="comparison_scatterplot",
                            help="The name of the outfile.")
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
        df = self.load_file(self.data_path, header=0, index_col=0)
        if self.transpose:
            df = df.T
        if self.nrows is not None:
            df = df.iloc[:, :self.nrows]
        columns = df.columns.tolist()
        print(df)
        print(columns)

        annot = None
        if self.annot_path is not None:
            print("Loading annotation")
            annot_df = self.load_file(self.annot_path, header=0, index_col=0)
            print(annot_df)
            annot = dict(zip(annot_df.index, annot_df.iloc[:, 0]))

        print("Loading color data.")
        hue = None
        palette = None
        if self.std_path is not None:
            sa_df = self.load_file(self.std_path, header=0, index_col=None)
            sa_df.set_index(sa_df.columns[0], inplace=True)
            sa_df.columns = ["hue"]
            sa_df["hue"] = sa_df["hue"].astype(str)
            df = df.merge(sa_df, left_index=True, right_index=True)

            hue = "hue"
            palette = self.palette

        # Plotting.
        self.plot(df=df, columns=columns, annot=annot, hue=hue, palette=palette)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, df, columns, hue=None, annot=None, palette=None):
        ncols = len(columns)
        nrows = len(columns)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(10 * ncols, 10 * nrows))
        sns.set(color_codes=True)

        for i, y_col in enumerate(columns):
            for j, x_col in enumerate(columns):
                print(i, j)

                ax = axes[i, j]
                if i == 0 and j == (ncols - 1):
                    ax.set_axis_off()
                    if hue is not None and palette is not None:
                        groups_present = df[hue].unique().tolist()
                        handles = []
                        for key, value in palette.items():
                            if key in groups_present:
                                handles.append(mpatches.Patch(color=value, label=key))
                        ax.legend(handles=handles, loc=4, fontsize=25)

                elif i < j:
                    ax.set_axis_off()
                    continue
                elif i == j:
                    ax.set_axis_off()

                    annot_str = ""
                    if annot is not None and y_col in annot:
                        annot_str = "\n{:.2f}%".format(annot[y_col])

                    ax.annotate("{}{}".format(y_col, annot_str),
                                xy=(0.5, 0.5),
                                ha='center',
                                xycoords=ax.transAxes,
                                color="#000000",
                                fontsize=40,
                                fontweight='bold')
                else:
                    sns.despine(fig=fig, ax=ax)

                    sns.scatterplot(x=x_col,
                                    y=y_col,
                                    hue=hue,
                                    data=df,
                                    s=100,
                                    palette=palette,
                                    linewidth=0,
                                    legend=False,
                                    ax=ax)

                    # for hue_value in df[hue].unique():
                    #     sns.regplot(x=x_col,
                    #                 y=y_col,
                    #                 ci=None,
                    #                 data=df.loc[df[hue] == hue_value, :],
                    #                 scatter_kws={'facecolors': palette[hue_value],
                    #                              'linewidth': 0,
                    #                              's': 60,
                    #                              'alpha': 0.75},
                    #                 line_kws={"color": palette[hue_value],
                    #                           'linewidth': 5},
                    #                 ax=ax)

                    #     ax.axhline(0, ls='--', color="#000000", zorder=-1)
                    #     ax.axvline(0, ls='--', color="#000000", zorder=-1)

                    ax.set_ylabel("",
                                  fontsize=20,
                                  fontweight='bold')
                    ax.set_xlabel("",
                                  fontsize=20,
                                  fontweight='bold')

        fig.savefig(os.path.join(self.outdir, "{}_comparison_scatterplot.png".format(self.out_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > N-rows: {}".format(self.nrows))
        print("  > Transpose: {}".format(self.transpose))
        print("  > Annotation path: {}".format(self.annot_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
