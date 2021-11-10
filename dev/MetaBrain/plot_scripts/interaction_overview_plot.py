#!/usr/bin/env python3

"""
File:         interaction_overview_plot.py
Created:      2021/10/21
Last Changed: 2021/10/25
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
import glob
import json
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
__program__ = "Interaction Overview Plot"
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
./interaction_overview_plot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
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

        print("Loading data")
        pics = []
        pic_dfs = []
        for i in range(1, 11):
            pic = "PIC{}".format(i)

            fpaths = glob.glob(os.path.join(self.input_directory, pic, "results_*.txt.gz"))
            if len(fpaths) <= 0:
                continue
            fpaths.sort()
            final_iteration_path = fpaths[-1]

            # final_iteration_path = None
            # for j in range(100):
            #     iter_path = os.path.join(self.input_directory, pic, "results_iteration{}_df.txt.gz".format(j))
            #     if os.path.exists(iter_path):
            #         final_iteration_path = iter_path
            #
            # if final_iteration_path is None:
            #     continue

            df = self.load_file(final_iteration_path, header=0, index_col=None)
            df.index = df["SNP"] + ":" + df["gene"]
            signif_ieqtl = set(df.loc[df["FDR"] < 0.05, :].index.tolist())

            signif_df = pd.DataFrame(0, index=df.index, columns=[pic])
            signif_df.loc[signif_ieqtl, pic] = 1

            pic_dfs.append(signif_df)
            pics.append(pic)

        pic_df = pd.concat(pic_dfs, axis=1)

        # Split data.
        no_interaction_df = pic_df.loc[pic_df.loc[:, pics].sum(axis=1) == 0, :]
        single_interaction_df = pic_df.loc[pic_df.loc[:, pics].sum(axis=1) == 1, :]
        multiple_interactions_df = pic_df.loc[pic_df.loc[:, pics].sum(axis=1) > 1, :]

        n_ieqtls_unique_per_pic = [(name, value) for name, value in single_interaction_df.sum(axis=0).iteritems()]
        n_ieqtls_unique_per_pic.sort(key=lambda x: -x[1])
        pics = [x[0] for x in n_ieqtls_unique_per_pic]
        n_ieqtls = [x[1] for x in n_ieqtls_unique_per_pic]

        # Construct pie data.
        data = [no_interaction_df.shape[0]] + n_ieqtls + [multiple_interactions_df.shape[0]]
        labels = ["None"] + pics + ["Multiple"]
        colors = None
        if self.palette is not None:
            colors = ["#D3D3D3"] + [self.palette[pic] for pic in pics] + ["#808080"]
        explode = [0.] + [0.1 for _ in pics] + [0.1]

        data_sum = np.sum(data)
        for i in range(len(data)):
            print("{} (N = {}) = {:.2f}%".format(labels[i], data[i], (100 / data_sum) * data[i]))

        total_n_ieqtls = pic_df.shape[0] - no_interaction_df.shape[0]
        print("Total interactions (N = {}) = {:.2f}%".format(total_n_ieqtls, (100 / pic_df.shape[0]) * total_n_ieqtls))

        self.plot(data=data, labels=labels, explode=explode, colors=colors, extension="png")
        self.plot(data=data, labels=labels, explode=explode, colors=colors, extension="pdf", label_threshold=100)

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, data, labels, explode=None, colors=None, extension='png', label_threshold=0):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        plt.pie(data,
                autopct=lambda pct: self.autopct_func(pct, data, label_threshold),
                explode=explode,
                labels=labels,
                shadow=False,
                colors=colors,
                startangle=90,
                wedgeprops={'linewidth': 1})
        ax.axis('equal')

        fig.savefig(os.path.join(self.outdir, "{}_interaction_piechart.{}".format(self.out_filename, extension)))
        plt.close()

    @staticmethod
    def autopct_func(pct, allvalues, label_threshold):
        if pct >= label_threshold:
            absolute = int(pct / 100. * np.sum(allvalues))
            return "{:.1f}%\n(N = {:,.0f})".format(pct, absolute)
        else:
            return ""

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.input_directory))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
