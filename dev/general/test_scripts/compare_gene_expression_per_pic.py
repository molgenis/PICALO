#!/usr/bin/env python3

"""
File:         compare_gene_expression_per_pic.py
Created:      2022/05/03
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
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Compare Gene Expression per PIC"
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
./compare_gene_expression_per_pic.py -h

### BIOS ###

./compare_gene_expression_per_pic.py \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -o 2022-05-03-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations

### MetaBrain ###


"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.min_corr = getattr(arguments, 'min_corr')
        self.top_n = getattr(arguments, 'top_n')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'compare_gene_expression_per_pic')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "all": "#404040",
            "pos": "#009E73",
            "neg": "#D55E00"
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
        parser.add_argument("-gc",
                            "--gene_correlations",
                            type=str,
                            required=True,
                            help="The path to the covariate-gene correlations "
                                 "matrix.")
        parser.add_argument("-mc",
                            "--min_corr",
                            type=float,
                            default=0.1,
                            help="The minimal correlation of a gene "
                                 "for inclusion. Default 0.1.")
        parser.add_argument("-tn",
                            "--top_n",
                            type=int,
                            default=200,
                            help="The top n genes to include in the "
                                 "enrichment analysis. Default: 200.")
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

        print("Loading data")
        pic_gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=0)
        print(pic_gene_corr_df)

        print("Pre-processing data")
        df_list = []
        panels = []
        for i in range(1, 100):
            pic = "PIC{}".format(i)
            if not pic in pic_gene_corr_df.columns:
                break

            subset_df = pic_gene_corr_df[["ProbeName", "avgExpression", "HGNCName", pic]].copy()
            subset_df.columns = ["ProbeName", "avgExpression", "HGNCName", "correlation"]
            subset_df["direction"] = "all"
            subset_df = subset_df.loc[subset_df["correlation"].abs() > self.min_corr, :]
            subset_df.sort_values(by="correlation", inplace=True)

            pos_df = subset_df.loc[subset_df["correlation"] > 0, :].iloc[:self.top_n, :].copy()
            pos_df["direction"] = "pos"

            neg_df = subset_df.loc[subset_df["correlation"] < 0, :].iloc[-self.top_n:, :].copy()
            neg_df["direction"] = "neg"

            _, p2 = stats.ttest_ind(pos_df["avgExpression"], neg_df["avgExpression"])

            label = "{} [p: {:.2e}]\nall n={:,} / pos n={:,} / neg n={:,}".format(pic, p2, subset_df.shape[0], pos_df.shape[0], neg_df.shape[0])
            subset_df["covariate"] = label
            pos_df["covariate"] = label
            neg_df["covariate"] = label

            if subset_df.shape[0] > 0:
                df_list.append(subset_df)
                panels.append(label)
            if pos_df.shape[0] > 0:
                df_list.append(pos_df)
            if neg_df.shape[0] > 0:
                df_list.append(neg_df)
        df = pd.concat(df_list, axis=0)
        print(df)

        print("Plotting data")
        self.boxplot(df=df,
                     panels=panels,
                     x="covariate",
                     y="avgExpression",
                     hue="direction",
                     palette=self.palette,
                     xlabel="",
                     ylabel="average expression",
                     )

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

    def boxplot(self, df, panels, x="variable", y="value", hue=None,
                palette=None, xlabel="", ylabel=""):
        nplots = len(panels)
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='all',
                                 sharey='all',
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

                subset = df.loc[df[x] == panels[i], :]

                sns.violinplot(x=x,
                               y=y,
                               hue=hue,
                               data=subset,
                               palette=palette,
                               dodge=True,
                               ax=ax)

                plt.setp(ax.collections, alpha=.75)

                sns.boxplot(x=x,
                            y=y,
                            hue=hue,
                            data=subset,
                            color="white",
                            dodge=True,
                            ax=ax)

                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                plt.setp(ax.artists, edgecolor='k', facecolor='w')
                plt.setp(ax.lines, color='k')

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

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        plt.tight_layout()
        for extension in self.extensions:
            outpath = os.path.join(self.outdir, "{}.{}".format(self.out_filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > Minimal correlation: {}".format(self.min_corr))
        print("  > Top-N: {}".format(self.top_n))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
