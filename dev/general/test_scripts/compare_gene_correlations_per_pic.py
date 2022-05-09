#!/usr/bin/env python3

"""
File:         compare_gene_correlations_per_pic.py
Created:      2022/05/06
Last Changed: 2022/05/09
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
import matplotlib.patches as mpatches

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

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -ce /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_eqtl_file/BIOS_Primary_eQTLProbesFDR0.05-ProbeLevel_GT1.0AvgExprFilter.txt.gz \
    -te /groups/umcg-bios/tmp01/projects/PICALO/data/2018-09-04-trans-eQTLsFDR0.05-CohortInfoRemoved-BonferroniAdded-eQTLGen.txt.gz \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_cis_and_eQTLGen_trans_eQTLs_TMMLog2GT1

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/correlate_components_with_genes/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -ci /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_interaction_cis_eQTLs_TMMLog2GT1


### MetaBrain ###

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -ce /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2021-12-07-CortexEUR-cis/combine_eqtlprobes/eQTLprobes_combined.txt.gz \
    -te /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-03-31-CortexEUR-and-AFR-noENA-trans-100PCs-NegativeToZero-DatasetAndRAMCorrected/combine_eqtlprobes/eQTLprobes_combined.txt.gz \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_cis_and_100PCsRemovedtrans_eQTLs_TMMLog2GT1

./compare_gene_correlations_per_pic.py \
    -gc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/correlate_components_with_genes/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_gene_correlations-avgExpressionAdded.txt.gz \
    -ci /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -o 2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_interaction_cis_eQTLs_TMMLog2GT1


"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.gene_correlations_path = getattr(arguments, 'gene_correlations')
        self.cis_eqtl_path = getattr(arguments, 'cis_eqtl')
        self.trans_eqtl_path = getattr(arguments, 'trans_eqtl')
        self.cis_inter_indir = getattr(arguments, 'cis_inter_indir')
        self.min_expr = getattr(arguments, 'min_expr')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(base_dir, 'compare_gene_correlations_per_pic')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.cis_eqtl_path is None and self.cis_inter_indir is None:
            print('Error, eqtl or inter indir should be not none.')
            exit()
        if self.cis_eqtl_path is not None and self.cis_inter_indir is not None:
            print('Error, eqtl or inter indir should be not none.')
            exit()

        self.palette = {
            "all genes": "#404040",
            "cis genes": "#0072B2",
            "trans genes": "#D55E00",
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
        parser.add_argument("-ce",
                            "--cis_eqtl",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the cis-eqtl matrix. "
                                 "Default: None.")
        parser.add_argument("-te",
                            "--trans_eqtl",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the trans-eqtl matrix. "
                                 "Default: None.")
        parser.add_argument("-ci",
                            "--cis_inter_indir",
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

        cis_eqtl_df = None
        if self.cis_eqtl_path is not None:
            cis_eqtl_df = self.load_file(self.cis_eqtl_path, header=0, index_col=None)
            print(cis_eqtl_df)

        trans_eqtl_df = None
        if self.trans_eqtl_path is not None:
            trans_eqtl_df = self.load_file(self.trans_eqtl_path, header=0, index_col=None)
            print(trans_eqtl_df)

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
            n_genes = subset_df.shape[0]

            # Add the cis-eQTL info.
            n_cis_genes = 0
            cis_p2 = np.nan
            cis_df = None
            if self.cis_inter_indir is not None or cis_eqtl_df is not None:
                if self.cis_inter_indir is not None:
                    inter_path = os.path.join(self.cis_inter_indir, "PIC{}_conditional.txt.gz".format(i))
                    if not os.path.exists(inter_path):
                        print("Could not find interaction file '{}'".format(
                            os.path.basename(inter_path)))
                        exit()

                    inter_df = self.load_file(inter_path, header=0, index_col=None)
                    cis_df = subset_df.loc[subset_df["ProbeName"].isin(inter_df.loc[inter_df["FDR"] < 0.05, "gene"]), :].copy()
                else:
                    cis_df = subset_df.loc[subset_df["ProbeName"].isin(cis_eqtl_df["ProbeName"]), :].copy()

                cis_df["subset"] = "cis genes"
                n_cis_genes = cis_df.shape[0]
                cis_p2 = max(2.2250738585072014e-308, stats.ttest_ind(subset_df["correlation"], cis_df["correlation"])[1])

            # Add the trans-eQTL info.
            n_trans_genes = 0
            trans_p2 = np.nan
            trans_df = None
            if trans_eqtl_df is not None:
                trans_df = subset_df.loc[subset_df["ProbeName"].isin(trans_eqtl_df["Gene"]), :].copy()
                trans_df["subset"] = "trans genes"
                n_trans_genes = trans_df.shape[0]
                trans_p2 = max(2.2250738585072014e-308, stats.ttest_ind(subset_df["correlation"], trans_df["correlation"])[1])

            label = "PIC{}\nT-test: {:.2e} / {:.2e}\n{:,} / {:,} / {:,}".format(i, cis_p2, trans_p2, n_genes, n_cis_genes, n_trans_genes)

            subset_df["covariate"] = label
            df_list.append(subset_df)

            if cis_df is not None:
                cis_df["covariate"] = label
                df_list.append(cis_df)

            if trans_df is not None:
                trans_df["covariate"] = label
                df_list.append(trans_df)

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

    def boxplot(self, df, panels, x="variable", y="value", hue=None, hue_order=None,
                palette=None, xlabel="", ylabel=""):
        nplots = len(panels) + 1
        ncols = math.ceil(np.sqrt(nplots))
        nrows = math.ceil(nplots / ncols)

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='all',
                                 sharey='row',
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
                               hue_order=hue_order,
                               data=subset,
                               palette=palette,
                               dodge=True,
                               ax=ax)

                plt.setp(ax.collections, alpha=.75)

                sns.boxplot(x=x,
                            y=y,
                            hue=hue,
                            hue_order=hue_order,
                            data=subset,
                            color="white",
                            dodge=True,
                            ax=ax)

                ax.axhline(0, ls='--', color="#000000", alpha=0.5, zorder=-1,
                           linewidth=3)

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
            outpath = os.path.join(self.outdir, "{}.{}".format(self.out_filename, extension))
            fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Gene correlations path: {}".format(self.gene_correlations_path))
        print("  > cis-eQTL path: {}".format(self.cis_eqtl_path))
        print("  > trans-eQTL path: {}".format(self.trans_eqtl_path))
        print("  > cis-interaction input directory: {}".format(self.cis_inter_indir))
        print("  > Minimal expression: {}".format(self.min_expr))
        print("  > Output directory {}".format(self.outdir))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
