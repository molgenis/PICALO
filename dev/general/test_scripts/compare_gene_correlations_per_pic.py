#!/usr/bin/env python3

"""
File:         compare_gene_correlations_per_pic.py
Created:      2022/05/06
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
__program__ = "Compare Gene Correlations per PIC"
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
./compare_gene_correlations_per_pic.py -h

### BIOS ###

### MetaBrain ###

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -eq /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-03-31-CortexEUR-and-AFR-noENA-trans-0PCs-NegativeToZero-DatasetAndRAMCorrected/combine_eqtlprobes/eQTLprobes_combined.txt.gz \
    -me 0 \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_trans_eQTLs
    
./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_cis_eQTLs_TMMLog2GT1

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -ii /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_interaction_cis_eQTLs_TMMLog2GT1


"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.inter_indir = getattr(arguments, 'interaction_indir')
        self.min_expr = getattr(arguments, 'min_expr')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'compare_gene_correlations_per_pic')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.eqtl_path is None and self.inter_indir is None:
            print('Error, eqtl or inter indir should be not none.')
            exit()

        self.palette = {
            "all genes": "#404040",
            "eqtl genes": "#009E73"
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
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the eqtl matrix. Default: None.")
        parser.add_argument("-ii",
                            "--interaction_indir",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the interaction directory. "
                                 "Defaul: None.")
        parser.add_argument("-me",
                            "--min_expr",
                            type=float,
                            default=1,
                            help="The minimal expression of a gene "
                                 "for inclusion. Default 1.")
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
        pic_gene_corr_df = self.load_file(self.gene_correlations_path, header=0, index_col=None)
        print(pic_gene_corr_df)

        eqtl_df = None
        if self.eqtl_path is not None:
            eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None)
            print(eqtl_df)

        print("Pre-processing data")
        df_list = []
        panels = []
        for i in range(1, 100):
            pic_r_column = "PIC{} r".format(i)
            if not pic_r_column in pic_gene_corr_df.columns:
                break

            subset_df = pic_gene_corr_df[["ProbeName", "avgExpression", "HGNCName", pic_r_column]].copy()
            subset_df.columns = ["ProbeName", "avgExpression", "HGNCName", "correlation"]
            subset_df = subset_df.loc[subset_df["avgExpression"] > self.min_expr, :]

            subset_df["subset"] = "all genes"
            if self.inter_indir is not None:
                inter_path = os.path.join(self.inter_indir, "PIC{}_conditional.txt.gz".format(i))
                if not os.path.exists(inter_path):
                    print("Could not find interaction file '{}'".format(
                        os.path.basename(inter_path)))
                    exit()

                inter_df = self.load_file(inter_path, header=0, index_col=None)
                subset_df.loc[subset_df["ProbeName"].isin(inter_df.loc[inter_df["FDR"] < 0.05, "gene"]), "subset"] = "eqtl genes"
            else:
                subset_df.loc[subset_df["ProbeName"].isin(eqtl_df["ProbeName"]), "subset"] = "eqtl genes"
            df_list.append(subset_df)

            counts = subset_df["subset"].value_counts()
            n_eqtl_genes = 0
            if "eqtl genes" in counts:
                n_eqtl_genes = counts["eqtl genes"]
            n_other_genes = subset_df.shape[0] - n_eqtl_genes

            p2 = np.nan
            if n_eqtl_genes > 0:
                _, p2 = stats.ttest_ind(subset_df.loc[subset_df["subset"] == "all genes", "correlation"],
                                        subset_df.loc[subset_df["subset"] == "eqtl genes", "correlation"])
                if p2 == 0:
                    p2 = 2.2250738585072014e-308

            label = "PIC{} [p: {:.2e}]\n{:,} / {:,}".format(i, p2, n_other_genes, n_eqtl_genes)

            subset_df["covariate"] = label
            df_list.append(subset_df)
            panels.append(label)
        df = pd.concat(df_list, axis=0)
        print(df)

        print("Plotting data")
        self.boxplot(df=df,
                     panels=panels,
                     x="covariate",
                     y="correlation",
                     hue="subset",
                     palette=self.palette,
                     xlabel="",
                     ylabel="correlation",
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

            ax.axhline(0, ls='--', color="#000000", alpha=0.5, zorder=-1, linewidth=3)

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
        print("  > eQTL path: {}".format(self.eqtl_path))
        print("  > Interaction input directory: {}".format(self.inter_indir))
        print("  > Minimal expression: {}".format(self.min_expr))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
