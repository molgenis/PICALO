#!/usr/bin/env python3

"""
File:         plot_start_nieqtls_vs_end_nieqtls.py
Created:      2021/05/18
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
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Start N-ieQTLs vs End N-ieQTLs"
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
./plot_start_nieqtls_vs_end_nieqtls.py -h

./plot_start_nieqtls_vs_end_nieqtls.py \
    -mi /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/ \
    -mf 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -bi /groups/umcg-bios/tmp01/projects/PICALO/output/ \
    -bf 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta_indir = getattr(arguments, 'metabrain_indir')
        self.meta_filename = getattr(arguments, 'metabrain_filename')
        self.bios_indir = getattr(arguments, 'bios_indir')
        self.bios_filename = getattr(arguments, 'bios_filename')
        self.outfile = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Loading palette.
        self.palette = {
            "PIC1": "#0072B2",
            "PIC2": "#009E73",
            "PIC3": "#CC79A7",
            "PIC4": "#E69F00",
            "PIC5": "#D55E00"
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
        parser.add_argument("-mi",
                            "--metabrain_indir",
                            type=str,
                            required=True,
                            help="The path to the MetaBrain PICALO output "
                                 "directory.")
        parser.add_argument("-mf",
                            "--metabrain_filename",
                            type=str,
                            required=True,
                            help="The MetaBrain PICALO filename.")
        parser.add_argument("-bi",
                            "--bios_indir",
                            type=str,
                            required=True,
                            help="The path to the BIOS PICALO outpout "
                                 "directory.")
        parser.add_argument("-bf",
                            "--bios_filename",
                            type=str,
                            required=True,
                            help="The BIOS PICALO filename.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")
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

        print("Loading data")
        meta_df = self.load_data(indir=self.meta_indir, filename=self.meta_filename)
        bios_df = self.load_data(indir=self.bios_indir, filename=self.bios_filename)

        print("Plotting")
        self.plot_regplot(df1=meta_df,
                          df2=bios_df,
                          x="start",
                          y="end",
                          hue="PIC",
                          xlabel="log #ieQTLs before optimization",
                          ylabel="log #ieQTLs after optimization",
                          title="#ieQTLs before and after\n"
                                "PICALO optimization",
                          palette=self.palette)

    def load_data(self, indir, filename):
        data = []
        for pic_index in range(1, 6):
            for comp_index in range(1, 26):
                index = "PIC{}-Comp{}".format(pic_index, comp_index)
                fpath = os.path.join(indir, "{}-{}AsCov".format(filename, index), "PIC1", "info.txt.gz")
                if not os.path.exists(fpath):
                    continue

                df = self.load_file(fpath, header=0, index_col=0)
                start = df["N"][0]
                end = np.nan
                if df.shape[0] > 1:
                    end = df["N"][-1]
                data.append(["PIC{}".format(pic_index), start, end])
                del df

        return pd.DataFrame(data, columns=["PIC", "start", "end"])

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_regplot(self, df1, df2, x="x", y="y", hue=None, palette=None,
                     xlabel=None, ylabel=None, title=""):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.set(rc={'figure.figsize': (18, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        self.plot_single_regplot(fig=fig,
                                 ax=ax1,
                                 df=df1,
                                 x=x,
                                 y=y,
                                 hue=hue,
                                 palette=palette,
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 title="blood")
        self.plot_single_regplot(fig=fig,
                                 ax=ax2,
                                 df=df2,
                                 x=x,
                                 y=y,
                                 hue=hue,
                                 palette=palette,
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 title="brain")

        fig.suptitle(title,
                     fontsize=20,
                     fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}_start_nieqtls_vs_end_nieqtls.{}".format(self.outfile, extension))
            fig.savefig(outpath)
        plt.close()

    @staticmethod
    def plot_single_regplot(fig, ax, df, x="x", y="y",
                            hue=None, palette=None,xlabel=None, ylabel=None,
                            title=""):
        sns.despine(fig=fig, ax=ax)

        group_column = hue
        if hue is None:
            df["hue"] = "#000000"
            group_column = "hue"

        group_corr_coef = {}
        group_sizes = {}
        for i, hue_group in enumerate(df[group_column].unique()):
            subset = df.loc[df[group_column] == hue_group, :]
            if subset.shape[0] < 2:
                continue

            facecolors = "#000000"
            color = "#b22222"
            if palette is not None:
                facecolors = palette[hue_group]
                color = facecolors

            sns.regplot(x=x, y=y, data=subset, ci=None,
                        scatter_kws={'facecolors': facecolors,
                                     # 's': 10,
                                     # 'alpha': 0.2,
                                     'linewidth': 0},
                        line_kws={"color": color},
                        ax=ax)

            if hue is not None:
                subset_pearson_coef, _ = stats.pearsonr(subset[y], subset[x])
                group_corr_coef[hue_group] = subset_pearson_coef
                group_sizes[hue_group] = subset.shape[0]

        pearson_coef, _ = stats.pearsonr(df[y], df[x])
        handles = [mpatches.Patch(color="#000000", label="ALL [r={:.2f}]".format(pearson_coef))]
        if hue is not None:
            for hue_group in df[group_column].unique():
                if hue_group in palette:
                    r = "NA"
                    if hue_group in group_corr_coef:
                        r = "{:.2f}".format(group_corr_coef[hue_group])
                    handles.append(mpatches.Patch(color=palette[hue_group], label="{} [r={}]".format(hue_group, r)))
        ax.legend(handles=handles, loc=7, fontsize=8)

        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_title(title,
                     fontsize=18,
                     fontweight='bold')

        ax.set_xscale('log')
        ax.set_yscale('log')

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain:")
        print("      > Input directory: {}".format(self.meta_indir))
        print("      > Filename: {}".format(self.meta_filename))
        print("  > BIOS:")
        print("      > Input directory: {}".format(self.bios_indir))
        print("      > Filename: {}".format(self.bios_filename))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
