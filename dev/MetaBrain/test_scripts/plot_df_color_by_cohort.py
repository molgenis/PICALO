#!/usr/bin/env python3

"""
File:         plot_df_color_by_cohort.py
Created:      2021/04/12
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
import glob
import math
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
__program__ = "Plot DF Color by Cohort"
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
./plot_df_color_by_cohort.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.axis = getattr(arguments, 'axis')
        self.gte_folder = getattr(arguments, 'gte_folder')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.cohort_palette = {
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

        match_dict = {
            "GTE-EUR-AMPAD-MAYO-V2": "MAYO",
            "GTE-EUR-CMC_HBCC_set2": "CMC HBCC",
            "GTE-EUR-GTEx": "GTEx",
            "GTE-EUR-AMPAD-ROSMAP-V2": "ROSMAP",
            "GTE-EUR-BrainGVEX-V2": "Brain GVEx",
            "GTE-EUR-TargetALS": "Target ALS",
            "GTE-EUR-AMPAD-MSBB-V2": "MSBB",
            "GTE-EUR-NABEC-H610": "NABEC",
            "GTE-EUR-LIBD_1M": "LIBD",
            "GTE-EUR-ENA": "ENA",
            "GTE-EUR-LIBD_h650": "LIBD",
            "GTE-EUR-GVEX": "GVEX",
            "GTE-EUR-NABEC-H550": "NABEC",
            "GTE-EUR-CMC_HBCC_set3": "CMC HBCC",
            "GTE-EUR-UCLA_ASD": "UCLA ASD",
            "GTE-EUR-CMC": "CMC",
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
        parser.add_argument("-a",
                            "--axis",
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help="The axis that denotes the samples. "
                                 "Default: 0")
        parser.add_argument("-gte",
                            "--gte_folder",
                            type=str,
                            required=True,
                            help="The path to the folder containg 'GTE-*' files.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        outfile = os.path.basename(self.data_path).split(".")[0]

        print("Loading data.")
        cohort_dict, cohort_color_dict = self.load_cohort_info()
        df = self.load_file(self.data_path, header=0, index_col=None)

        if self.axis == 0:
            df = df.T

        variables = df.columns.tolist()
        df["cohort"] = df.index.map(cohort_dict)
        df["index"] = range(0, df.shape[0])

        print("Plot.")
        self.plot_box(df=df, variables=variables, outfile=outfile)
        self.plot_scatter(df=df, variables=variables, outfile=outfile)

    def plot_box(self, df, variables, outfile):
        df_m = df.melt(value_vars=variables)

        for showfliers in [True, False]:
            sns.set(rc={'figure.figsize': (10, 7.5)})
            sns.set_style("ticks")
            fig, ax = plt.subplots()
            sns.despine(fig=fig, ax=ax)

            sns.boxplot(data=df_m,
                        x="variable",
                        y="value",
                        color="#000000",
                        showfliers=showfliers,
                        ax=ax)

            fig.savefig(os.path.join(self.outdir, "{}_boxplot_showfliers{}.png".format(outfile, showfliers)))
            plt.close()

    def plot_scatter(self, df, variables, outfile):
        nplots = len(variables) + 1
        if nplots < 4:
            ncols = 2
        elif 4 < nplots <= 9:
            ncols = 3
        elif 9 < nplots <= 16:
            ncols = 4
        else:
            ncols = 5
        nrows = math.ceil(len(variables) / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 # sharex='col',
                                 # sharey='row',
                                 figsize=(15 * ncols, 9 * nrows))

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            ax = axes[row_index, col_index]
            if i < len(variables):
                sns.scatterplot(x="index",
                                y=variables[i],
                                hue="cohort",
                                data=df,
                                palette=self.cohort_palette,
                                linewidth=0,
                                legend=None,
                                ax=ax)

                ax.set_title(variables[i],
                             fontsize=25,
                             fontweight='bold')
                ax.set_ylabel("value",
                              fontsize=20,
                              fontweight='bold')
                ax.set_xlabel("",
                              fontsize=20,
                              fontweight='bold')
            else:
                ax.set_axis_off()
                if i == (nplots - 1):
                    handles = []
                    for key, value in self.cohort_palette.items():
                        handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=1)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.savefig(os.path.join(self.outdir, "{}_coloredByCohort.png".format(outfile)))
        plt.close()

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def load_cohort_info(self):
        match_dict = {
            "GTE-EUR-AMPAD-MAYO-V2": "MAYO",
            "GTE-EUR-CMC_HBCC_set2": "CMC HBCC",
            "GTE-EUR-GTEx": "GTEx",
            "GTE-EUR-AMPAD-ROSMAP-V2": "ROSMAP",
            "GTE-EUR-BrainGVEX-V2": "Brain GVEx",
            "GTE-EUR-TargetALS": "Target ALS",
            "GTE-EUR-AMPAD-MSBB-V2": "MSBB",
            "GTE-EUR-NABEC-H610": "NABEC",
            "GTE-EUR-LIBD_1M": "LIBD",
            "GTE-EUR-ENA": "ENA",
            "GTE-EUR-LIBD_h650": "LIBD",
            "GTE-EUR-GVEX": "GVEX",
            "GTE-EUR-NABEC-H550": "NABEC",
            "GTE-EUR-CMC_HBCC_set3": "CMC HBCC",
            "GTE-EUR-UCLA_ASD": "UCLA ASD",
            "GTE-EUR-CMC": "CMC",
        }

        gte_combined = None
        for fpath in glob.glob(os.path.join(self.gte_folder, 'GTE-*')):
            gte_df = self.load_file(fpath, index_col=None, header=None)
            gte_df.columns = ["gene_id", "expr_id"]

            gte_filename = os.path.basename(fpath).split(".")[0]
            if gte_filename in match_dict:
                gte_df["cohort"] = match_dict[gte_filename]
                gte_df["color"] = self.cohort_palette[match_dict[gte_filename]]

                if gte_combined is None:
                    gte_combined = gte_df
                else:
                    gte_combined = pd.concat([gte_combined, gte_df], axis=0)

        return dict(zip(gte_combined["expr_id"], gte_combined["cohort"])), \
               dict(zip(gte_combined["expr_id"], gte_combined["color"]))

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > Axis: {}".format(self.axis))
        print("  > GTE directory: {}".format(self.gte_folder))
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
