#!/usr/bin/env python3

"""
File:         evaluate_model_term_importance.py
Created:      2022/05/09
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
import argparse
import glob
import math
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
__program__ = "Evaluate Model Term Importance"
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
./evaluate_model_term_importance.py -h

### BIOS ###

./evaluate_model_term_importance.py \
    -pd /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

### MetaBrain ###


"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_directory = getattr(arguments, 'picalo_directory')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'evaluate_model_term_importance')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.terms = ["intercept", "genotype", "covariate", "interaction"]

        self.palette = {
            "intercept": "#404040",
            "genotype": "#0072B2",
            "covariate": "#D55E00",
            "interaction": "#009E73"
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
        parser.add_argument("-pd",
                            "--picalo_directory",
                            type=str,
                            required=True,
                            help="The path to the PICALO output directory.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")
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

        print("Analyzing EM data")
        em_df, em_panels = self.load_em_data()
        self.barplot(
            df=em_df,
            panels=em_panels,
            panel_col="variable",
            x="term",
            y="chi_sum",
            palette=self.palette,
            order=self.terms,
            ylabel="sum(t-value ^ 2)",
            appendix="_EM_data"
        )

        print("Analyzing ieQTL data")
        ieqtl_df, ieqtl_panels = self.load_ieqtl_data()
        self.barplot(
            df=ieqtl_df,
            panels=ieqtl_panels,
            panel_col="variable",
            x="term",
            y="chi_sum",
            palette=self.palette,
            order=self.terms,
            ylabel="sum(t-value ^ 2)",
            appendix="_ieqtl_data"
        )

    def load_em_data(self):
        data = []
        panels = []
        for i in range(1, 100):
            pic = "PIC{}".format(i)
            picalo_pic_dir = os.path.join(self.picalo_directory, pic)
            if not os.path.exists(picalo_pic_dir):
                break

            fpaths = glob.glob(os.path.join(picalo_pic_dir, "results_*.txt.gz"))
            fpaths.sort()
            df = self.load_file(fpaths[-1], header=0, index_col=None)
            df = df.loc[df["FDR"] < 0.05, :]
            label = "{} [n={:,}]".format(pic, df.shape[0])
            panels.append(label)

            for term in self.terms:
                df["tvalue-{}".format(term)] = df["beta-{}".format(term)] / df["std-{}".format(term)]
                chi_sum = (df["tvalue-{}".format(term)] ** 2).sum()
                data.append([pic, label, term, chi_sum])

        df = pd.DataFrame(data, columns=["pic", "variable", "term", "chi_sum"])
        return df, panels

    def load_ieqtl_data(self):
        data = []
        panels = []
        for i in range(1, 100):
            pic = "PIC{}".format(i)
            fpath = os.path.join(self.picalo_directory, "PIC_interactions", "PIC{}.txt.gz".format(i))
            if not os.path.exists(fpath):
                break

            df = self.load_file(fpath, header=0, index_col=None)
            df = df.loc[df["FDR"] < 0.05, :]
            label = "{} [n={:,}]".format(pic, df.shape[0])
            panels.append(label)

            for term in self.terms:
                df["tvalue-{}".format(term)] = df["beta-{}".format(term)] / df["std-{}".format(term)]
                chi_sum = (df["tvalue-{}".format(term)] ** 2).sum()
                data.append([pic, label, term, chi_sum])

        df = pd.DataFrame(data, columns=["pic", "variable", "term", "chi_sum"])
        return df, panels

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith("pkl"):
            df = pd.read_pickle(inpath)
        else:
            df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                             low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def barplot(self, df, panels, panel_col, x="variable", y="value",
                order=None, palette=None, xlabel="", ylabel="", appendix=""):
        nplots = len(panels) + 1
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='all',
                                 sharey='none',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            print(i)
            if nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(panels):
                sns.despine(fig=fig, ax=ax)

                g = sns.barplot(x=x,
                                y=y,
                                data=df.loc[df[panel_col] == panels[i], :],
                                order=order,
                                palette=palette,
                                ax=ax)

                tmp_xlabel = ""
                if row_index == (nrows - 1):
                    tmp_xlabel = xlabel
                ax.set_xlabel(tmp_xlabel,
                              color="#000000",
                              fontsize=20,
                              fontweight='bold')
                tmp_ylabel = ""
                if col_index == 0:
                    tmp_ylabel = ylabel
                ax.set_ylabel(tmp_ylabel,
                              color="#000000",
                              fontsize=20,
                              fontweight='bold')

                ax.set_title(panels[i],
                             color="#000000",
                             fontsize=25,
                             fontweight='bold')

            else:
                ax.set_axis_off()

                if palette is not None and i == (nplots - 1):

                    handles = []
                    for key, value in palette.items():
                        handles.append(mpatches.Patch(color=value, label=key))
                    ax.legend(handles=handles, loc=8, fontsize=25)

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}{}.{}".format(self.out_filename, appendix, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > PICALO directory: {}".format(self.picalo_directory))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
