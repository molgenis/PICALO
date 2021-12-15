#!/usr/bin/env python3

"""
File:         overlap_pic_interactions_with_ct_interactions.py
Created:      2021/12/13
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
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.stats import multitest
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Overlap PIC Interactions with CellType Interactions"
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
./overlap_pic_interactions_with_ct_interactions.py -h

./overlap_pic_interactions_with_ct_interactions.py -pi /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/ -di /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/decon-eqtl_scripts/decon_eqtl/2021-12-07-CortexEUR-cis-NormalisedMAF5-LimitedConfigs-PsychENCODEProfile-NoDev-InhibitorySummedWithOtherNeuron/deconvolutionResults.txt.gz -o 2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-ComparedTo-2021-12-07-CortexEUR-cis-NormalisedMAF5-LimitedConfigs-PsychENCODEProfile-NoDev-InhibitorySummedWithOtherNeuron
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.pic_interactions_path = getattr(arguments, 'pic_interactions')
        self.decon_eqtl_interactions_path = getattr(arguments, 'decon_eqtl_interactions')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-pi",
                            "--pic_interactions",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-di",
                            "--decon_eqtl_interactions",
                            type=str,
                            required=True,
                            help="T")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading decon-eQTL data")
        decon_eqtl_df = self.load_file(self.decon_eqtl_interactions_path, header=0, index_col=0)
        print(decon_eqtl_df)

        print("\tSubset")
        columns = [x for x in decon_eqtl_df.columns if "pvalue" in x]
        decon_eqtl_df = decon_eqtl_df[columns]

        print("\tBH FDR")
        decon_eqtl_df = self.bh_correct(decon_eqtl_df)
        print(decon_eqtl_df)

        print("Loading PICALO data")
        pic_list = []
        for i in range(1, 50):
            fpath = os.path.join(self.pic_interactions_path, "PIC_interactions", "PIC{}.txt.gz".format(i))
            if os.path.exists(fpath):
                df = self.load_file(fpath, header=0, index_col=None)

                df.index = df["gene"] + "_" + df["SNP"]
                df = df[["FDR"]]
                df.columns = ["PIC{}".format(i)]
                pic_list.append(df)
        pic_df = pd.concat(pic_list, axis=1)
        print(pic_df)

        print("Comparing")
        df1_df2_replication_df = self.create_replication_df(df1=pic_df, df2=decon_eqtl_df)
        self.plot_heatmap(df=df1_df2_replication_df,
                          xlabel="PICALO",
                          ylabel="Decon-eQTL",
                          filename="Decon_eQTL_replicating_in_PICALO")

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
    def bh_correct(pvalue_df):
        df = pvalue_df.copy()
        fdr_data = []
        indices = []
        for col in df.columns:
            if col.endswith("_pvalue"):
                fdr_data.append(multitest.multipletests(df.loc[:, col], method='fdr_bh')[1])
                indices.append(col.replace("_pvalue", ""))
        fdr_df = pd.DataFrame(fdr_data, index=indices, columns=df.index)

        return fdr_df.T

    @staticmethod
    def create_replication_df(df1, df2):
        row_overlap = set(df1.index).intersection(set(df2.index))
        df1_subset = df1.loc[row_overlap, :]
        df2_subset = df2.loc[row_overlap, :]

        replication_df = pd.DataFrame(np.nan,
                                      index=df1_subset.columns,
                                      columns=df2_subset.columns)
        for ct1 in df1_subset.columns:
            ct1_df = df2_subset.loc[df1_subset.loc[:, ct1] < 0.05, :]
            for ct2 in df2_subset.columns:
                replication_df.loc[ct1, ct2] = (ct1_df.loc[:, ct2] < 0.05).sum() / ct1_df.shape[0]

        df1_n_signif = (df1_subset < 0.05).sum(axis=0)
        df2_n_signif = (df2_subset < 0.05).sum(axis=0)

        replication_df.index = ["{} [n={}]".format(ct, df1_n_signif.loc[ct]) for ct in df1_subset.columns]
        replication_df.columns = ["{} [n={}]".format(ct, df2_n_signif.loc[ct]) for ct in df2_subset.columns]

        return replication_df

    def plot_heatmap(self, df, xlabel="", ylabel="", filename=""):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        fig, axes = plt.subplots(nrows=2,
                                 ncols=2,
                                 figsize=(1 * df.shape[1] + 5, 1 * df.shape[0] + 5),
                                 gridspec_kw={"width_ratios": [0.2, 0.8],
                                              "height_ratios": [0.8, 0.2]})
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for _ in range(4):
            ax = axes[row_index, col_index]
            if row_index == 0 and col_index == 1:

                sns.heatmap(df, cmap=cmap, vmin=-1, vmax=1, center=0,
                            square=True, annot=df.round(2), fmt='',
                            cbar=False, annot_kws={"size": 16, "color": "#000000"},
                            ax=ax)

                plt.setp(ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20, rotation=0))
                plt.setp(ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20, rotation=90))

                ax.set_xlabel(xlabel, fontsize=14)
                ax.xaxis.set_label_position('top')

                ax.set_ylabel(ylabel, fontsize=14)
                ax.yaxis.set_label_position('right')
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > 1:
                col_index = 0
                row_index += 1

        # plt.tight_layout()
        fig.savefig(os.path.join(self.outdir, "{}_heatmap.png".format(filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > PIC interactions path: {}".format(self.pic_interactions_path))
        print("  > Decon-eQTL interactions path: {}".format(self.decon_eqtl_interactions_path))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
