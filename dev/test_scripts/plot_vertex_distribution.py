#!/usr/bin/env python3

"""
File:         plot_vertex_distribution.py
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
import sys
import os

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Plot Vertex Distribution"
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
./plot_vertex_distribution.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.vertex_data_path = getattr(arguments, 'vertex_data')
        self.optimized_data_path = getattr(arguments, 'optimized_data')
        self.sa_path = getattr(arguments, 'sample_annotation')
        self.sample_id = getattr(arguments, 'sample_id')
        self.color_id = getattr(arguments, 'color_id')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "-": "000000",
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
        parser.add_argument("-vd",
                            "--vertex_data",
                            type=str,
                            required=True,
                            help="The path to the vertex x-position matrix")
        parser.add_argument("-od",
                            "--optimized_data",
                            type=str,
                            required=True,
                            help="The path to the optmized x-position matrix")
        parser.add_argument("-cid",
                            "--color_id",
                            type=str,
                            required=False,
                            default=None,
                            choices=["MetaBrain_cohort"],
                            help="The color column(s) name in the -sa / "
                                 "--sample_annotation file.")

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

        print("Loading data.")
        vertex_df = self.load_file(self.vertex_data_path, header=0, index_col=0)
        optimized_df = self.load_file(self.optimized_data_path, header=0, index_col=0)

        print(vertex_df)
        print(optimized_df)

        sample_group_dict = {}
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)
            for group in sa_df[self.color_id].unique():
                sample_group_dict[group] = sa_df.loc[sa_df[self.color_id] == group, self.sample_id].values.tolist()
        else:
            sample_group_dict = {x: "-" for x in vertex_df.columns}

        for group, samples in sample_group_dict.items():
            if group != "Target ALS":
                continue
            overlap_samples = [x for x in samples if x in vertex_df.columns and x in optimized_df.columns]
            if len(overlap_samples) <= 0:
                continue
            sample_dict = {x: i for i, x in enumerate(overlap_samples)}
            print(group, len(overlap_samples))

            # subset.
            box_subset = vertex_df.loc[:, overlap_samples].copy()
            box_subset.reset_index(drop=False, inplace=True)
            box_subset_m = box_subset.melt(id_vars=["index"])
            box_subset_m.dropna(inplace=True)
            box_subset_m["x"] = box_subset_m["variable"].map(sample_dict)

            point_subset = optimized_df.loc[:, overlap_samples].copy()
            point_subset_m = point_subset.melt()
            point_subset_m["x"] = point_subset_m["variable"].map(sample_dict)

            self.plot(box_df=box_subset_m, point_df=point_subset_m,
                      color=self.palette[group], title=group, xlabel="samples",
                      ylabel="vertex x-pos", appendix="_{}".format(group))
            self.plot(box_df=box_subset_m, point_df=point_subset_m,
                      ylim=(-10, 10), color=self.palette[group],
                      title=group, xlabel="samples",
                      ylabel="vertex x-pos", appendix="_{}_Zoomed".format(group))

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot(self, box_df, point_df, x="x", y="value", ylim=None, color=None,
             title="", xlabel="", ylabel="", appendix=""):
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.despine(fig=fig, ax=ax)

        sns.set()
        sns.set_style("ticks")

        # sns.violinplot(x=x,
        #                y=y,
        #                color=color,
        #                data=box_df,
        #                ax=ax)
        #
        # plt.setp(ax.collections, alpha=.75)

        sns.scatterplot(x=x,
                        y=y,
                        color="#000000",
                        data=box_df,
                        ax=ax)

        sns.boxplot(x=x,
                    y=y,
                    data=box_df,
                    #whis=np.inf,
                    color="white",
                    ax=ax)

        sns.scatterplot(x=x,
                        y=y,
                        color=color,
                        data=point_df,
                        s=25,
                        ax=ax)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.get_xaxis().set_ticks([])

        ax.set_title(title,
                     color=color,
                     fontsize=16,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')

        fig.savefig(os.path.join(self.outdir, "vertex_x_distribution{}.png".format(appendix)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Vertex data path: {}".format(self.vertex_data_path))
        print("  > Optimized data path: {}".format(self.optimized_data_path))
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
