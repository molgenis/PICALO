#!/usr/bin/env python3

"""
File:         plot_double_total_ieqtl_barplot.py
Created:      2022/07/06
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
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
__program__ = "Plot Double Total ieQTL Barplot 2"
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
./plot_double_total_ieqtl_barplot.py -h

### MetaBrain ###

./plot_double_total_ieqtl_barplot.py \
    -mpic /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -mpc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PCsAsCov-Conditional \
    -bpic /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -bpc /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PCsAsCov-Conditional \
    -conditional \
    -o 20220706_MetaBrain_BIOS_percentage_ieqtl \
    -e png pdf
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta_pic_indir = getattr(arguments, 'metabrain_pic_indir')
        self.meta_pc_indir = getattr(arguments, 'metabrain_pc_indir')
        self.bios_pic_indir = getattr(arguments, 'bios_pic_indir')
        self.bios_pc_indir = getattr(arguments, 'bios_pc_indir')
        self.conditional = getattr(arguments, 'conditional')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot_double_total_ieqtl_barplot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.palette = {
            "brain - PIC": "#0072B2",
            "brain - PC": "#808080",
            "blood - PIC": "#D55E00",
            "blood - PC": "#808080"
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
        parser.add_argument("-mpic",
                            "--metabrain_pic_indir",
                            type=str,
                            required=True,
                            help="The path to MetaBrain PIC interactions.")
        parser.add_argument("-mpc",
                            "--metabrain_pc_indir",
                            type=str,
                            required=True,
                            help="The path to MetaBrain PC interactions.")
        parser.add_argument("-bpic",
                            "--bios_pic_indir",
                            type=str,
                            required=True,
                            help="The path to BIOS PIC interactions.")
        parser.add_argument("-bpc",
                            "--bios_pc_indir",
                            type=str,
                            required=True,
                            help="The path to BIOS PC interactions.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
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
        meta_pic_data = self.load_data(indir=self.meta_pic_indir, conditional=self.conditional)
        meta_pc_data = self.load_data(indir=self.meta_pc_indir, conditional=self.conditional)
        bios_pic_data = self.load_data(indir=self.bios_pic_indir, conditional=self.conditional)
        bios_pc_data = self.load_data(indir=self.bios_pc_indir, conditional=self.conditional)

        print("Counting overlap")
        df = pd.DataFrame({"index": ["blood - PIC",
                                     "blood - PC",
                                     "brain - PIC",
                                     "brain - PC"],
                           "N": [bios_pic_data.shape[0],
                                 bios_pc_data.shape[0],
                                 meta_pic_data.shape[0],
                                 meta_pc_data.shape[0]],
                           "N-ieQTL": [(bios_pic_data.sum(axis=1) > 0).sum(),
                                       (bios_pc_data.sum(axis=1) > 0).sum(),
                                       (meta_pic_data.sum(axis=1) > 0).sum(),
                                       (meta_pc_data.sum(axis=1) > 0).sum()
                                       ]})
        print(df)

        print("Creating plot.")
        self.barplot(
            df=df,
            x="index",
            y1="N",
            y2="N-ieQTL",
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

        ieqtl_df_list = []
        for i, inpath in enumerate(inpaths):
            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df[signif_col] <= 0.05, :].index
            ieqtl_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_df.loc[ieqtls, filename] = 1
            ieqtl_df_list.append(ieqtl_df)

            del ieqtl_df

        ieqtl_df = pd.concat(ieqtl_df_list, axis=1)
        return ieqtl_df

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

    def barplot(self, df, x="x", y1="y1", y2="y2", palette=None, xlabel="",
                 ylabel="", title="", filename="plot"):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.barplot(x=x,
                        y=y1,
                        color="#000000",
                        data=df,
                        ax=ax)

        g = sns.barplot(x=x,
                        y=y2,
                        palette="#b22222" if palette is None else [palette[x] for x in df[x]],
                        data=df,
                        ax=ax)

        y_adjust = ax.get_ylim()[1] * 0.01
        print(y_adjust)
        for i, (_, row) in enumerate(df.iterrows()):
            print(i, row[x], row[y1], row[y2])
            g.text(i, row[y1] + y_adjust,
                   "{:,.0f}".format(row[y1]),
                   color="#000000",
                   ha="center")
            if (row[y2] - y_adjust) > 0:
                g.text(i, row[y2] + y_adjust,
                       "{:,.0f} [{:.2f}%]".format(row[y2], (100 / row[y1]) * row[y2]),
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
        print("  > MetaBrain input directories:")
        print("    > PIC: {}".format(self.meta_pic_indir))
        print("    > PC: {}".format(self.meta_pc_indir))
        print("  > BIOS input directories:")
        print("    > PIC: {}".format(self.bios_pic_indir))
        print("    > PC: {}".format(self.bios_pc_indir))
        print("  > Conditional: {}".format(self.conditional))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
