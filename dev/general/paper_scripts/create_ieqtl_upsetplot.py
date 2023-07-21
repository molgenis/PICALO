#!/usr/bin/env python3

"""
File:         create_ieqtl_upsetplot.py
Created:      2022/07/05
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import itertools
import argparse
import re
import glob
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
__program__ = "Create ieQTL Upsetplot"
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
./create_ieqtl_upsetplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')
        self.conditional = getattr(arguments, 'conditional')
        self.n_files = getattr(arguments, 'n_files')
        self.cutoff = getattr(arguments, 'cutoff')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'create_ieqtl_upsetplot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
        parser.add_argument("-n",
                            "--n_files",
                            type=int,
                            default=None,
                            help="The number of files to load. "
                                 "Default: all.")
        parser.add_argument("-c",
                            "--cutoff",
                            type=int,
                            default=None,
                            help="The minimal number of ieQTLs a file must have"
                                 "to be included. Default: none.")
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

        print("Load ieQTL data.")
        ieqtl_data = self.load_data(indir=self.input_directory, conditional=self.conditional)

        print("Counting overlap")
        counts = self.count(ieqtl_data)
        counts = counts[counts > 0]
        print(counts)

        print("Creating plot.")
        up.plot(counts, sort_by='cardinality', show_counts=True)
        for extension in self.extensions:
            plt.savefig(os.path.join(self.outdir, "{}_upsetplot.{}".format(self.out_filename, extension)))
        plt.close()

    def load_data(self, indir, conditional=False, signif_col="FDR"):
        inpaths = glob.glob(os.path.join(indir, "*.txt.gz"))
        if conditional:
            inpaths = [inpath for inpath in inpaths if inpath.endswith("_conditional.txt.gz")]
        else:
            inpaths = [inpath for inpath in inpaths if not inpath.endswith("_conditional.txt.gz")]
        inpaths.sort(key=self.natural_keys)

        ieqtls = {}
        count = 0
        filenames = []
        for i, inpath in enumerate(inpaths):
            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue

            if self.n_files is not None and count >= self.n_files:
                break

            df = self.load_file(inpath, header=0, index_col=None)
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls[filename] = set(df.loc[df[signif_col] <= 0.05, :].index)
            filenames.append(filename)
            count += 1

            del df

        # filter
        if self.cutoff is not None:
            for i, filename in enumerate(filenames[::-1]):
                n_ieqtls = len(ieqtls[filename])
                if n_ieqtls < self.cutoff:
                    del ieqtls[filename]
                else:
                    break

        return ieqtls

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

    @staticmethod
    def count(input_data):
        combinations = []
        cols = list(input_data.keys())
        for i in range(1, len(cols) + 1):
            combinations.extend(list(itertools.combinations(cols, i)))

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

        s = pd.Series(data, index=pd.MultiIndex.from_tuples(indices, names=cols))
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
            color = "#000000"
            if palette is not None:
                color = palette[label]
            ax1.axhline(value, ls='--', color=color, zorder=-1)

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
        print("  > Conditional: {}".format(self.conditional))
        print("  > N-files: {}".format(self.n_files))
        print("  > Cutoff: {}".format(self.cutoff))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
