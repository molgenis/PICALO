#!/usr/bin/env python3

"""
File:         plot_sample_size_simulations.py
Created:      2023/07/15
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import glob
import json
import re
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
__program__ = "Plot Sample Size Simulations"
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
./plot_sample_size_simulations.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.folder = getattr(arguments, 'folder')
        self.palette_path = getattr(arguments, 'palette')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), "plot_sample_size_simulations")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                palette = json.load(f)
            f.close()
            self.palette = {"Comp{}".format(i): palette["PIC{}".format(i + 1)] for i in range(0, 25)}

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
        parser.add_argument("-f",
                            "--folder",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
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
        data = []
        for fpath in glob.glob(os.path.join(self.indir, "output", "{}_*Samples".format(self.folder))):
            n_samples = int(os.path.basename(fpath).replace(self.folder + "_", "").replace("Samples", ""))
            geno_stats_inpath = os.path.join(fpath, "genotype_stats.txt.gz")
            cov_selection_inpath = os.path.join(fpath, "PIC1", "covariate_selection.txt.gz")
            if not os.path.exists(geno_stats_inpath) or not os.path.exists(cov_selection_inpath):
                continue

            # Load the genotype stats.
            geno_stats_df = self.load_file(geno_stats_inpath, header=0, index_col=0)
            n_eqtls = geno_stats_df["mask"].sum()

            # Load #ieQTLs before optimization.
            df = self.load_file(cov_selection_inpath, header=0, index_col=0)
            df.reset_index(drop=False, inplace=True)
            df.columns = ["Covariate", "ieQTLs before"]
            df["ieQTLs tested"] = n_eqtls
            # df["ieQTLs frac"] = (df["ieQTLs before"] / df["ieQTLs tested"]) * 100
            df["n_samples"] = n_samples

            # # Load #ieQTLs after optimization.
            # for pic in range(1, 10):
            #     after_ieqtls_inpath = os.path.join(after_infolder, "PIC{}".format(pic_index + 1), "info.txt.gz")
            #     if not os.path.exists(after_ieqtls_inpath):
            #         continue
            #
            #     # Load #ieQTLs after optimization.
            #     after_df = self.load_file(after_ieqtls_inpath, header=0, index_col=0)

            # print("\tN samples = {}".format(n_samples))
            # print(df)
            # print("")

            # Save.
            data.append(df)

        df = pd.concat(data, axis=0)
        groups = df["Covariate"].unique().tolist()
        groups.sort(key=self.natural_keys)
        df = df.loc[df["Covariate"].isin(groups[:5]), :]

        self.plot_combined_lineplot(
            data=df,
            x="n_samples",
            y="ieQTLs before",
            hue="Covariate",
            palette=self.palette,
            xlabel="#samples",
            ylabel="#ieQTLs before optimization",
            title="Before optimization",
            filename="ieqtls_per_sample_size_before_bla"
        )


    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def plot_combined_lineplot(self, data, x="x", y="y", hue="hue", palette=None,
                               xlabel="x", ylabel="y", title="", filename="lineplot"):
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.set(color_codes=True)

        sns.despine(fig=fig, ax=ax)

        sns.lineplot(data=data,
                     x=x,
                     y=y,
                     markers=["o"] * len(data[hue].unique()),
                     hue=hue,
                     palette=palette,
                     style=hue,
                     ax=ax)

        ax.set_title(title,
                     fontsize=25,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=20,
                      fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=14)

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_{}.{}".format(self.folder, filename, extension)))
        plt.close()

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Folder: {}".format(self.folder))
        print("  > Palette: {}".format(self.palette_path))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
