#!/usr/bin/env python3

"""
File:         create_clustermap.py
Created:      2021/04/28
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
from colour import Color
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
__program__ = "Create Matrix"
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
./create_clustermap.py -h

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.header_row = getattr(arguments, 'header_row')
        self.index_col = getattr(arguments, 'index_col')
        self.axis = getattr(arguments, 'axis')
        self.log_transform = getattr(arguments, 'log_transform')
        self.sa_path = getattr(arguments, 'sample_annotation')
        self.sample_id = getattr(arguments, 'sample_id')
        self.color_id = getattr(arguments, 'color_id')
        self.row_cluster = getattr(arguments, 'row_cluster')
        self.col_cluster = getattr(arguments, 'col_cluster')
        self.out_filename = getattr(arguments, 'outfile')

        # Change color label.
        if self.color_id is not None and "PMI" in self.color_id:
            self.color_id.remove("PMI")
            self.color_id.append("PMI_(in_hours)")

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palettes = {
            "MetaBrain_cohort" : {
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
            },
            "MetaCohort": {
                "AMP-AD": "#A9AAA9",
                "Braineac": "#E7A023",
                "Brainseq": "#CC79A9",
                "CMC": "#EEE643",
                "GTEx": "#1A9F74",
                "NABEC": "#767833",
                "TargetALS": "#DDCC78",
                "ENA": "#D56228",
                "PsychEncode": "#5AB4E5"
            },
            "sex.by.expression": {
                "F": "#DC106C",
                "M": "#03165E"
            },
            "reannotated_diangosis": {
                "Alzheimer disease": "#A9AAA9",
                "Progressive Supranuclear Palsy": "#DCDCDC",
                "Non-Neurological Control": "#DCDCDC",
                "Pathologic Aging": "#DCDCDC",
                "Mild cognitive impairement": "#DCDCDC",
                "Dementia": "#E7A023",
                "Possible alzheimer disease": "#CC79A9",
                "Probable alzheimer disease": "#CC79A9",
                "Schizophrenia": "#EEE643",
                "Bipolar": "#1A9F74",
                "Affective Disorder": "#DCDCDC",
                "Pre-fALS": "#767833",
                "ALS Spectrum MND": "#767833",
                "Parkinson's Disease (PD)": "#DDCC78",
                "Spinal Bulbar Muscular Atrophy (SBMA)": "#DCDCDC",
                "Autism Spectrum Disorder": "#5AB4E5"
            }
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-hr",
                            "--header_row",
                            type=int,
                            default=None,
                            help="Position of the data header. "
                                 "Default: None.")
        parser.add_argument("-ic",
                            "--index_col",
                            type=int,
                            default=None,
                            help="Position of the index column. "
                                 "Default: None.")
        parser.add_argument("-a",
                            "--axis",
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help="The axis that denotes the samples. "
                                 "Default: 0")
        parser.add_argument("-log_transform",
                            action='store_true',
                            help="-log10 transform the values."
                                 " Default: False.")
        parser.add_argument("-cid",
                            "--color_id",
                            nargs="*",
                            type=str,
                            required=False,
                            default=None,
                            choices=["MetaCohort", "PMI", "sex.by.expression", "reannotated_diangosis", "MetaBrain_cohort"],
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

        parser.add_argument("-row_cluster",
                            action='store_true',
                            help="Cluster the rows."
                                 " Default: False.")
        parser.add_argument("-col_cluster",
                            action='store_true',
                            help="Cluster the rows."
                                 " Default: False.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        data_df = self.load_file(self.data_path, header=self.header_row, index_col=self.index_col)
        print(data_df)

        if self.axis == 1:
            data_df = data_df.T

        if self.log_transform:
            data_df = -np.log10(data_df)

        col_colors = None
        if self.sa_path is not None:
            sa_df = self.load_file(self.sa_path, header=0, index_col=0, low_memory=False)

            print("Getting overlap.")
            overlap = set(data_df.columns).intersection(set(sa_df[self.sample_id]))
            print("\tN = {}".format(len(overlap)))
            if len(overlap) == 0:
                print("No data overlapping.")
                exit()
            data_df = data_df.loc[:, overlap]
            sa_df = sa_df.loc[sa_df[self.sample_id].isin(overlap), :]
            print(sa_df)

            print("Creating column colors")
            sa_df = sa_df.loc[:, [self.sample_id, *self.color_id]]
            sa_df.set_index(self.sample_id, inplace=True)

            col_colors_data = []
            combined_palette = []
            for col in sa_df.columns:
                label_dict = dict(zip(*np.unique(sa_df[col].values, return_counts=True)))

                if col in self.palettes:
                    palette = self.palettes[col]
                    col_colors_data.append(sa_df[col].map(palette, na_action="#000000"))
                else:
                    palette = self.create_value_palette(label_dict.keys())
                    col_colors_data.append(sa_df[col].map(palette, na_action="#000000"))
                combined_palette.extend([("{} [N={}]".format(label, label_dict[label]), color) for label, color in palette.items()])
            col_colors = pd.concat(col_colors_data, axis=1)
            print(col_colors)

            print("Plotting legend.")
            self.plot_legend(palette=combined_palette)

        print("Plotting.")
        self.plot_clustermap(df=data_df, col_colors=col_colors)

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
    def create_value_palette(data, precision=10):
        highest = int(np.nanmax(data) * precision)
        colors = [str(x).upper() for x in list(Color("#FFFFFF").range_to(Color("#B22222"), highest))]
        values = [x / precision for x in list(range(highest))]

        value_color_map = {}
        for val, col in zip(values, colors):
            value_color_map[val] = col

        return value_color_map

    def plot_clustermap(self, df, col_colors):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        xticklabels = True
        if df.shape[1] > 100:
            xticklabels = False

        yticklabels = True
        if df.shape[0] > 100:
            yticklabels = False

        sns.set(color_codes=True)
        g = sns.clustermap(df, cmap=cmap, center=0,
                           yticklabels=yticklabels, xticklabels=xticklabels,
                           row_cluster=self.row_cluster, col_cluster=self.col_cluster,
                           col_colors=col_colors, figsize=(12, 9))

        plt.setp(g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=10, rotation=0))

        g.savefig(os.path.join(self.outdir, "{}_clustermap.png".format(self.out_filename)))
        plt.close()

    def plot_legend(self, palette):
        fig, ax = plt.subplots()
        ax.set_axis_off()

        handles = []
        for label, color in palette:
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles, loc=4)

        fig.savefig(os.path.join(self.outdir, "{}_clustermap_legend.png".format(self.out_filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("    > Header: {}".format(self.header_row))
        print("    > Index col: {}".format(self.index_col))
        print("    > Axis: {}".format(self.axis))
        print("  > -log10 transform: {}".format(self.log_transform))
        print("  > Sample annotation path: {}".format(self.sa_path))
        print("     > Sample ID: {}".format(self.sample_id))
        print("     > Color ID: {}".format(self.color_id))
        print("  > Row cluster: {}".format(self.row_cluster))
        print("  > Col cluster: {}".format(self.col_cluster))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
