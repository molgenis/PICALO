#!/usr/bin/env python3

"""
File:         plot_tss.py
Created:      2022/07/07
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
__program__ = "Plot TSS"
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
./plot_tss.py -h

./plot_tss.py \
    -mpic /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -mpc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PCsAsCov-Conditional \
    -ma /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -bpic /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -bpc /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PCsAsCov-Conditional \
    -ba /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -conditional \
    -o 20220706_MetaBrain_BIOS_TSS \
    -e png pdf
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.meta_pic_indir = getattr(arguments, 'metabrain_pic_indir')
        self.meta_pc_indir = getattr(arguments, 'metabrain_pc_indir')
        self.meta_annot_path = getattr(arguments, 'metabrain_annotation')
        self.bios_pic_indir = getattr(arguments, 'bios_pic_indir')
        self.bios_pc_indir = getattr(arguments, 'bios_pc_indir')
        self.bios_annot_path = getattr(arguments, 'bios_annotation')
        self.conditional = getattr(arguments, 'conditional')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot_tss')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42


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
        parser.add_argument("-ma",
                            "--metabrain_annotation",
                            type=str,
                            required=True,
                            help="The path to MetaBrain annotation matrix.")
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
        parser.add_argument("-ba",
                            "--bios_annotation",
                            type=str,
                            required=True,
                            help="The path to BIOS annotation matrix.")
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
        meta_annotation = self.load_file(self.meta_annot_path, header=0, index_col=None)

        bios_pic_data = self.load_data(indir=self.bios_pic_indir, conditional=self.conditional)
        bios_pc_data = self.load_data(indir=self.bios_pc_indir, conditional=self.conditional)
        bios_annotation1 = self.load_file(self.bios_annot_path, header=0, index_col=None)
        bios_annotation1["Gene"] = [x.split(".")[0] for x in bios_annotation1["ProbeName"]]
        bios_annotation2 = self.load_file("/groups/umcg-bios/tmp01/projects/PICALO/data/Homo_sapiens.GRCh37.75.gtf-genepos.txt.gz", header=0, index_col=None)
        bios_annotation = bios_annotation1.merge(bios_annotation2, on="Gene", how="left")

        print("Pre-processing.")
        print(meta_annotation)
        meta_annotation.index = meta_annotation["SNPName"] + "_" + meta_annotation["ProbeName"]
        # meta_annotation["TSS dist"] = meta_annotation["SNPChrPos"] - meta_annotation["ProbeCenterChrPos"]

        meta_pic_df = meta_pic_data.merge(meta_annotation, left_index=True, right_index=True, how="left")
        meta_pc_df = meta_pc_data.merge(meta_annotation, left_index=True, right_index=True, how="left")

        bios_annotation.index = bios_annotation["SNPName"] + "_" + bios_annotation["ProbeName"]
        # bios_annotation["TSS dist"] = bios_annotation["SNPChrPos"] - bios_annotation["TSS"]

        bios_pic_df = bios_pic_data.merge(bios_annotation, left_index=True, right_index=True, how="left")
        bios_pc_df = bios_pc_data.merge(bios_annotation, left_index=True, right_index=True, how="left")

        plot_data = []
        for (tissue, data, df) in (("blood", "PIC", bios_pic_df),
                                   ("blood", "PC", bios_pc_df),
                                   ("brain", "PIC", meta_pic_df),
                                   ("brain", "PC", meta_pc_df)):
            variables = [col for col in df.columns if col.startswith("PIC") or col.startswith("Comp")]
            # no_inter_tss_dist = df.loc[df.loc[:, variables].sum(axis=1) == 0, "TSS dist"].tolist()
            # for value in no_inter_tss_dist:
            #     plot_data.append([tissue, data, 0, "no interaction", value])
            #
            # yes_inter_tss_dist = df.loc[df.loc[:, variables].sum(axis=1) > 0, "TSS dist"].tolist()
            # for value in yes_inter_tss_dist:
            #     plot_data.append([tissue, data, 1, "yes interaction", value])

            for i, variable in enumerate(variables):
                tss_dist = df.loc[df.loc[:, variable] == 1, "OverallZScore"].tolist()
                for value in tss_dist:
                    plot_data.append([tissue, data, 1 + i, variable, value * value])

        df = pd.DataFrame(plot_data, columns=["tissue", "data", "index", "variable", "value"])
        print(df)

        print("Creating plot.")
        self.create_boxplot(
            df=df,
            rows=["blood", "brain"],
            x="index",
            y="value",
            hue="data",
            ylabel="overall z-score ^ 2",
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

    def create_boxplot(self, df, rows, x="variable", y="value", hue=None,
                       xlabel="", ylabel="", filename="plot"):
        nrows = len(rows)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=1,
                                 figsize=(24, 9 * nrows))
        sns.set(color_codes=True)
        sns.set_style("ticks")

        for row, ax in zip(rows, axes):
            palette = None
            if row == "brain":
                palette = {"PIC": "#0072B2", "PC": "#808080"}
            elif row == "blood":
                palette = {"PIC": "#D55E00", "PC": "#808080"}
            else:
                pass

            sns.despine(fig=fig, ax=ax)

            sns.violinplot(x=x,
                           y=y,
                           hue=hue,
                           data=df.loc[df["tissue"] == row, :],
                           palette=palette,
                           cut=0,
                           dodge=True,
                           ax=ax)

            plt.setp(ax.collections, alpha=.75)

            sns.boxplot(x=x,
                        y=y,
                        hue=hue,
                        data=df.loc[df["tissue"] == row, :],
                        whis=np.inf,
                        color="white",
                        dodge=True,
                        ax=ax)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel(xlabel,
                          fontsize=14,
                          fontweight='bold')
            ax.set_ylabel(ylabel,
                          fontsize=14,
                          fontweight='bold')
            ax.set_title(row,
                         fontsize=16,
                         fontweight='bold')

        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > MetaBrain input directories:")
        print("    > PIC: {}".format(self.meta_pic_indir))
        print("    > PC: {}".format(self.meta_pc_indir))
        print("    > Annotation: {}".format(self.meta_annot_path))
        print("  > BIOS input directories:")
        print("    > PIC: {}".format(self.bios_pic_indir))
        print("    > PC: {}".format(self.bios_pc_indir))
        print("    > Annotation: {}".format(self.bios_annot_path))
        print("  > Conditional: {}".format(self.conditional))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
