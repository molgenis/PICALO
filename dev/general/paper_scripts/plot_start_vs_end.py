#!/usr/bin/env python3

"""
File:         plot_start_vs_end.py
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
__program__ = "Plot Start vs End"
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
./plot_start_vs_end.py -h

./plot_start_vs_end.py \
    -m /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/ \
    -b /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -o 2022-03-24-MetaBrain_and_BIOS_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -e png pdf

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta_indir = getattr(arguments, 'metabrain_indir')
        self.bios_indir = getattr(arguments, 'bios_indir')
        self.outfile = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            "blood": "#D55E00",
            "brain": "#0072B2"
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
        parser.add_argument("-bi",
                            "--bios_indir",
                            type=str,
                            required=True,
                            help="The path to the BIOS PICALO outpout "
                                 "directory.")
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
        meta_df = self.load_data(indir=self.meta_indir)
        bios_df = self.load_data(indir=self.bios_indir)

        print("Merging data")
        meta_df["hue"] = "brain"
        bios_df["hue"] = "blood"
        df = pd.concat([meta_df, bios_df], axis=0)
        print(df)

        print("Plotting")
        self.plot_stripplot(df=df,
                            xlabel="PIC",
                            ylabel="pearson correlation",
                            title="correlation before and after\nPICALO optimization",
                            palette=self.palette)

    def load_data(self, indir):
        data = []
        indices = []
        for pic_index in range(1, 100):
            pic = "PIC{}".format(pic_index)
            fpath = os.path.join(indir, pic, "iteration.txt.gz")
            if not os.path.exists(fpath):
                continue

            df = self.load_file(fpath, header=0, index_col=0)

            if df.shape[0] > 1:
                lo, r, hi = self.pearsonr_ci(x=df.iloc[0, :], y=df.iloc[-1, :])
                data.append([pic_index, lo, r, hi])
                indices.append(pic)

        return pd.DataFrame(data, index=indices, columns=["x", "low", "y", "high"])

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
    def pearsonr_ci(x, y, alpha=0.05):
        '''
        https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
        '''

        r, p = stats.pearsonr(x, y)
        r_z = np.arctanh(r)
        se = 1 / np.sqrt(x.size - 3)
        z = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se
        lo, hi = np.tanh((lo_z, hi_z))
        return lo, r, hi

    def plot_stripplot(self, df, x="x", y="y", hue="hue", low="low",
                       high="high", palette=None, xlabel="", ylabel="",
                       title="", ):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        df_m = df.melt(id_vars=[x, hue], value_vars=[low, high])
        sns.pointplot(x=x,
                      y="value",
                      hue=hue,
                      data=df_m,
                      dodge=False,
                      linewidth=4,
                      join=False,
                      palette=palette,
                      ax=ax)

        sns.stripplot(x=x,
                      y=y,
                      hue=hue,
                      data=df,
                      size=10,
                      dodge=False,
                      palette=palette,
                      linewidth=0,
                      edgecolor="w",
                      jitter=0,
                      ax=ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        handles = []
        if palette is not None:
            for label, value in palette.items():
                handles.append(mpatches.Patch(color=value, label=label))
            ax.legend(handles=handles, loc=1, fontsize=14)

        ax.set_title(title,
                     fontsize=16,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_start_vs_end_correlation_stripplot.{}".format(self.outfile, extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain input directory: {}".format(self.meta_indir))
        print("  > BIOS input directory: {}".format(self.bios_indir))
        print("  > Outfile: {}".format(self.outfile))
        print("  > Extensions: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
