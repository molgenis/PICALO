#!/usr/bin/env python3

"""
File:         compare_pvalues.py
Created:      2021/11/01
Last Changed: 2022/02/10
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Compare p-values"
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
./compare_pvalues.py -h
"""


class main():
    def __init__(self):
        # # Get the command line arguments.
        # arguments = self.create_argument_parser()
        # self.input_directory = getattr(arguments, 'indir')
        # self.palette_path = getattr(arguments, 'palette')
        # self.out_filename = getattr(arguments, 'outfile')

        self.data_path1 = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/test_scripts/main_eqtl_replication/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics_results_df.txt.gz"
        self.data_path2 = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC-Combined/PIC2/results_iteration049.txt.gz"
        self.data_path3 = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/BIOS_eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "no signif": "#808080",
            "x signif": "#0072B2",
            "y signif": "#D55E00",
            "both signif": "#009E73"
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
        df1 = self.load_file(self.data_path1, header=0, index_col=0)
        df2 = self.load_file(self.data_path2, header=0, index_col=None)
        df3 = self.load_file(self.data_path3, header=0, index_col=None)

        print(df1)
        print(df2)
        print(df3)

        df3.loc[df3["PValue"] == 0, "PValue"] = 1e-307

        df3 = df3.loc[df3["Iteration"] == 1, :]

        print("Merging data")
        # df2.index = df2["gene"] + "_" + df2["snp"]
        df2.index = df2["gene"] + "_" + df2["SNP"]
        df3.index = df3["ProbeName"] + "_" + df3["SNPName"]
        # df = df1.loc[:, ["p-value"]].merge(df2.loc[:, ["ieQTL FDR"]], left_index=True, right_index=True).merge(df3.loc[:, ["PValue"]], left_index=True, right_index=True)
        df = df1.loc[:, ["p-value"]].merge(df2.loc[:, ["FDR"]], left_index=True, right_index=True).merge(df3.loc[:, ["PValue"]], left_index=True, right_index=True)
        df.columns = ["x", "z", "y"]

        # print("Merging data")
        # df = df.loc[:, ["eQTL p-value", "ieQTL p-value", "ieQTL FDR"]]
        # df.columns = ["x", "y", "z"]

        # # # Adding color.
        # df["hue"] = self.palette["no signif"]
        # df.loc[(df["x"] <= 0.05) & (df["y"] > 0.05), "hue"] = self.palette["x signif"]
        # df.loc[(df["x"] > 0.05) & (df["y"] <= 0.05), "hue"] = self.palette["y signif"]
        # df.loc[(df["x"] <= 0.05) & (df["y"] <= 0.05), "hue"] = self.palette["both signif"]
        df["hue"] = "no signif"
        df.loc[df["z"] <= 0.05, "hue"] = "both signif"

        # Log10 transform.
        df["x"] = np.log10(df["x"]) * -1
        df["y"] = np.log10(df["y"]) * -1
        print(df)

        self.plot(df=df,
                  hue="hue",
                  palette=self.palette,
                  xlabel="eQTL -log10(p-value)",
                  ylabel="BIOS eQTL -log10(p-value)",
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
    def plot(df, x="x", y="y", hue=None, palette=None, xlabel="", ylabel="",
             title="", filename="plot", outdir=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        coef, _ = stats.spearmanr(df[x], df[y])

        sns.scatterplot(x=x,
                        y=y,
                        hue=hue,
                        data=df.loc[df[hue] == "no signif", :],
                        palette=palette,
                        linewidth=0,
                        legend=None,
                        ax=ax)
        sns.scatterplot(x=x,
                        y=y,
                        hue=hue,
                        data=df.loc[df[hue] == "both signif", :],
                        palette=palette,
                        linewidth=0,
                        legend=None,
                        ax=ax)

        ax.axhline(1.3010299956639813, ls='--', color="#000000", zorder=-1)
        ax.axvline(1.3010299956639813, ls='--', color="#000000", zorder=-1)

        # Add the text.
        ax.annotate(
            'r = {:.2f}'.format(coef),
            xy=(0.75, 0.94),
            xycoords=ax.transAxes,
            color="#404040",
            fontsize=14,
            fontweight='bold')
        ax.annotate(
            'total N = {:,}'.format(df.shape[0]),
            xy=(0.75, 0.9),
            xycoords=ax.transAxes,
            color="#404040",
            fontsize=14,
            fontweight='bold')
        if hue is not None:
            for i, group in enumerate(df[hue].unique()):
                ax.annotate(
                    'N = {:,}'.format(df.loc[df[hue] == group, :].shape[0]),
                    xy=(0.75, 0.86 - (i * 0.04)),
                    xycoords=ax.transAxes,
                    color=palette[group],
                    fontsize=14,
                    fontweight='bold')

        ax.set_title(title,
                     fontsize=20,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)

        outpath = "{}.png".format(filename)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)
        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Outpath {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
