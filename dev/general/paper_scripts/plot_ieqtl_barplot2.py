#!/usr/bin/env python3

"""
File:         plot_ieqtl_barplot2.py
Created:      2022/07/05
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
__program__ = "Plot ieQTL Barplot 2"
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
./plot_ieqtl_barplot2.py -h

### MetaBrain ###

./plot_ieqtl_barplot2.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -conditional \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -e png pdf
    
    
### BIOS ####

./plot_ieqtl_barplot2.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -conditional \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -e png pdf
    

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'indir')
        self.conditional = getattr(arguments, 'conditional')
        self.n_files = getattr(arguments, 'n_files')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot_ieqtl_barplot2')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        unique_color = "#000000"
        if "bios" in self.input_directory:
            unique_color = "#D55E00"
        elif "biogen" in self.input_directory:
            unique_color = "#0072B2"
        self.palette = {
            "N": "#000000",
            "N-unique": unique_color,
            "N-overlap": "#808080",
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
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
        parser.add_argument("-n",
                            "--n_files",
                            type=int,
                            default=None,
                            help="The number of files to load. "
                                 "Default: all.")
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
        df = self.count(ieqtl_data)
        print(df)

        print("Creating plot.")
        self.barplot(
            df=df,
            x="index",
            y1="N",
            y2="N-unique",
            palette=self.palette,
            ylabel="#ieQTLs",
            filename=self.out_filename
        )

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
    def count(data):
        out_data = []
        for key1, value1 in data.items():
            combined_data = set()
            for key2, value2 in data.items():
                if key2 != key1:
                    combined_data.update(value2)

            n = len(value1)
            n_overlap = len(value1.intersection(combined_data))
            out_data.append([key1, n, n_overlap])

            del combined_data

        out_df = pd.DataFrame(out_data, columns=["index", "N", "N-overlap"])
        out_df["N-unique"] = out_df["N"] - out_df["N-overlap"]

        return out_df

    def barplot(self, df, x="x", y1="y1", y2="y2", palette=None, xlabel="",
                 ylabel="", title="", filename="plot"):
        sns.set(rc={'figure.figsize': (24, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.barplot(x=x,
                        y=y1,
                        color="#000000" if palette is None else palette[y1],
                        data=df,
                        ax=ax)

        g = sns.barplot(x=x,
                        y=y2,
                        color="#b22222" if palette is None else palette[y2],
                        data=df,
                        ax=ax)

        y_adjust = ax.get_ylim()[1] * 0.01
        print(y_adjust)
        for i, (_, row) in enumerate(df.iterrows()):
            print(i, row[x], row[y1], row[y2])
            g.text(i, row[y1] + y_adjust,
                   round(row[y1], 0),
                   color="#000000",
                   ha="center")
            if (row[y2] - y_adjust) > 0:
                g.text(i, row[y2] + y_adjust,
                       round(row[y2], 0),
                       color='#FFFFFF',
                       ha="center")

        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')

        fig.suptitle(title,
                     fontsize=14,
                     fontweight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.input_directory))
        print("  > Conditional: {}".format(self.conditional))
        print("  > N-files: {}".format(self.n_files))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
