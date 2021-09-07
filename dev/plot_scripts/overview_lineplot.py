#!/usr/bin/env python3

"""
File:         overview_lineplot.py
Created:      2021/05/20
Last Changed: 2021/07/08
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
__program__ = "Overview Lineplot"
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
./overview_lineplot.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "Neuron": "#0072B2",
            "Oligodendrocyte": "#009E73",
            "EndothelialCell": "#CC79A7",
            "Macrophage": "#E69F00",
            "Astrocyte": "#D55E00",
            "PCT_INTRONIC_BASES": "#000000",
            "comp0": "#0072B2",
            "comp1": "#009E73",
            "comp2": "#CC79A7",
            "comp3": "#E69F00",
            "comp4": "#D55E00",
            "PIC1": "#0072B2",
            "PIC2": "#009E73",
            "PIC3": "#CC79A7",
            "PIC4": "#E69F00",
            "PIC5": "#D55E00",
            "PIC6": "#56B4E9"
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

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        comp_dict = {}
        n_ieqtls_df = pd.DataFrame(np.nan, index=["iteration{}".format(x) for x in range(100)], columns=["index"])
        n_ieqtls_df["index"] = np.arange(1, (n_ieqtls_df.shape[0] + 1))
        rss_df = pd.DataFrame(np.nan, index=["iteration{}".format(x) for x in range(100)], columns=["index"])
        rss_df["index"] = np.arange(1, (rss_df.shape[0] + 1))
        eucl_dist_df = pd.DataFrame(np.nan, index=["iteration{}".format(x) for x in range(100)], columns=["index"])
        eucl_dist_df["index"] = np.arange(1, (eucl_dist_df.shape[0] + 1))
        for i in range(1, 11):
            fpath = os.path.join(self.input_directory, "PIC{}".format(i), "info_df.txt.gz")
            if os.path.exists(fpath):
                df = self.load_file(fpath, header=0, index_col=0)
                print(df)

                comp_dict["PIC{}".format(i)] = df.loc["iteration0", "covariate"]

                n_ieqtls = df[["N"]]
                n_ieqtls.columns = ["PIC{}".format(i)]
                n_ieqtls_df = n_ieqtls_df.merge(n_ieqtls, left_index=True, right_index=True, how="left")

                eucl_dist = df[["Euclidean distance"]]
                eucl_dist.columns = ["PIC{}".format(i)]
                eucl_dist_df = eucl_dist_df.merge(eucl_dist, left_index=True, right_index=True, how="left")

                rss = np.log10(df[["RSS"]])
                rss.columns = ["PIC{}".format(i)]
                rss_df = rss_df.merge(rss, left_index=True, right_index=True, how="left")

        print(n_ieqtls_df)
        print(eucl_dist_df)
        print(rss_df)
        print(comp_dict)

        n_ieqtl_df_m = n_ieqtls_df.melt(id_vars=["index"])
        self.lineplot(df_m=n_ieqtl_df_m, x="index", y="value", hue="variable",
                      style=None, palette=self.palette,
                      title="N ieQTLs per iteration", xlabel="iteration",
                      ylabel="n ieQTLs", filename="ieQTL_counts",
                      info=comp_dict,
                      outdir=self.outdir)

        eucl_dist_df_m = eucl_dist_df.melt(id_vars=["index"])
        self.lineplot(df_m=eucl_dist_df_m, x="index", y="value", hue="variable",
                      style=None, palette=self.palette,
                      title="Eucledian distance between vector\nbefore and after iteration", xlabel="iteration",
                      ylabel="Eucledian distance", filename="ieQTL_eucl_dist_comps",
                      info=comp_dict,
                      outdir=self.outdir)

        rss_df_m = rss_df.melt(id_vars=["index"])
        self.lineplot(df_m=rss_df_m, x="index", y="value", hue="variable",
                      style=None, palette=self.palette,
                      title="RSS between vector\nbefore and after iteration", xlabel="iteration",
                      ylabel="RSS", filename="ieQTL_rss_comps",
                      info=comp_dict,
                      outdir=self.outdir)

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
    def lineplot(df_m, x="x", y="y", hue=None, style=None, palette=None, title="",
                 xlabel="", ylabel="", filename="plot", info=None, outdir=None):
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
                         style=style,
                         palette=palette,
                         estimator=None,
                         legend=None,
                         ax=ax1)

        ax1.set_title(title,
                      fontsize=14,
                      fontweight='bold')
        ax1.set_xlabel(xlabel,
                       fontsize=10,
                       fontweight='bold')
        ax1.set_ylabel(ylabel,
                       fontsize=10,
                       fontweight='bold')

        if palette is not None:
            handles = []
            for key, color in palette.items():
                if key in df_m[hue].values.tolist():
                    label = key
                    if info is not None and key in info:
                        label = "{} [{}]".format(key, info[key])
                    handles.append(mpatches.Patch(color=color, label=label))
            ax2.legend(handles=handles, loc="center")
        ax2.set_axis_off()

        plt.tight_layout()
        outpath = "{}.png".format(filename)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)
        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.input_directory))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
