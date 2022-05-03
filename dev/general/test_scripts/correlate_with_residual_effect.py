#!/usr/bin/env python3

"""
File:         correlate_with_residual_effect.py
Created:      2022/04/28
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
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Correlate With Redisual Effect"
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
./correlate_with_residual_effect.py -h

### BIOS ###

./correlate_with_residual_effect.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz \
    -p /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -od 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_NormalRes \
    -on 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_NormalRes
    
### MetaBrain ###

./correlate_with_residual_effect.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/MetaBrain_CorrelationsWithAverageExpression.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -od 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-NormalRes \
    -on 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-NormalRes
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.pics_path = getattr(arguments, 'pics')
        outdir = getattr(arguments, 'outdir')
        self.outname = getattr(arguments, 'outname')

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.file_outdir = os.path.join(base_dir, 'correlate_with_residual_effect', outdir)
        self.plot_outdir = os.path.join(self.file_outdir, 'plot')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-p",
                            "--pics",
                            type=str,
                            required=True,
                            help="The path to the PICS matrix.")
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-on",
                            "--outname",
                            type=str,
                            required=True,
                            help="The name of the output files.")

        return parser.parse_args()

    def start(self):
        print("Starting program")
        self.print_arguments()

        print("Loading data")
        data_df = self.load_file(self.data_path)
        pics_df = self.load_file(self.pics_path)

        print("Preprocessing data")
        pics_df = pics_df.T
        if data_df.shape[0] < data_df.shape[1]:
            data_df = data_df.T

        samples = [sample for sample in pics_df.index if sample in data_df.index]
        print("\tUsing {} samples".format(len(samples)))
        pics_df = pics_df.loc[samples, :]
        data_df = data_df.loc[samples, :]

        print("Modelling")
        correlation_m = np.empty((data_df.shape[1], pics_df.shape[1]), dtype=np.float64)
        pvalue_m = np.empty((data_df.shape[1], pics_df.shape[1]), dtype=np.float64)
        for i, colname in enumerate(data_df.columns):
            # Creat mask.
            mask = ~data_df.loc[:, colname].isna()
            n = mask.sum()

            print("\t{} [N={:,}]".format(colname, n))

            for j, pic in enumerate(pics_df.columns):
                data_vector = data_df.loc[mask, colname].copy()

                if j > 0:
                    # Correct for previus PICs.
                    ols = OLS(data_df.loc[mask, colname], pics_df.loc[mask, :].iloc[:, :j])
                    results = ols.fit()
                    data_vector = results.predict()

                # Correlate.
                coef, pvalue = stats.pearsonr(data_vector, pics_df.loc[mask, pic])
                correlation_m[i, j] = coef
                pvalue_m[i, j] = pvalue

        correlation_df = pd.DataFrame(correlation_m,
                                      index=data_df.columns,
                                      columns=pics_df.columns
                                      ).T
        pvalue_df = pd.DataFrame(pvalue_m,
                                 index=data_df.columns,
                                 columns=pics_df.columns
                                 ).T

        print("Saving file.")
        self.save_file(df=correlation_df,
                       outpath=os.path.join(self.file_outdir, "{}_correlation_df.txt.gz".format(self.outname)))

        print("Visualising")
        correlation_annot_df = correlation_df.copy()
        correlation_annot_df = correlation_annot_df.round(2).astype(str)
        correlation_df[pvalue_df > 0.05] = 0
        self.plot_heatmap(df=correlation_df,
                          annot_df=correlation_annot_df,
                          vmin=-1,
                          vmax=1,
                          xlabel="PICs",
                          ylabel="data",
                          title="Pearson correlations",
                          filename="{}_correlation_heatmap".format(self.outname))

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def plot_barplot(self, df, x="x", y="y", xlabel="", ylabel="", title="",
                     palette=None, filename=""):
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 12))

        sns.despine(fig=fig, ax=ax)

        color = None
        if palette is None:
            color = "#808080"

        g = sns.barplot(x=x,
                        y=y,
                        data=df,
                        color=color,
                        palette=palette,
                        dodge=False,
                        ax=ax)

        ax.set_title(title,
                     fontsize=22,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_outdir, "{}.png".format(filename)))
        plt.close()

    def plot_heatmap(self, df, annot_df, vmin=None, vmax=None, xlabel="",
                     ylabel="", title="", filename=""):
        sns.set_style("ticks")
        annot_df.fillna("", inplace=True)

        fig, ax = plt.subplots(figsize=(df.shape[1], df.shape[0]))
        sns.set(color_codes=True)

        sns.heatmap(df,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=sns.diverging_palette(246, 24, as_cmap=True),
                    cbar=False,
                    center=0,
                    square=True,
                    annot=annot_df,
                    fmt='',
                    annot_kws={"size": 14, "color": "#000000"},
                    ax=ax)

        plt.setp(ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20,
                                    rotation=0))
        plt.setp(ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20,
                                    rotation=90))

        ax.set_xlabel(xlabel, fontsize=14)
        ax.xaxis.set_label_position('top')

        ax.set_ylabel(ylabel, fontsize=14)
        ax.yaxis.set_label_position('right')

        fig.suptitle(title,
                     fontsize=22,
                     fontweight='bold')

        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_outdir, "{}.png".format(filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data path: {}".format(self.data_path))
        print("  > PICs path: {}".format(self.pics_path))
        print("  > Output name: {}".format(self.outname))
        print("  > Plot output directory: {}".format(self.plot_outdir))
        print("  > File output directory: {}".format(self.file_outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()