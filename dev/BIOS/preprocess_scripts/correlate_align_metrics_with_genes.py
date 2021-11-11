#!/usr/bin/env python3

"""
File:         correlate_align_metrics_with_genes.py
Created:      2021/11/09
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
import time
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

# Local application imports.

# Metadata
__program__ = "Correlate Alignment Metrics with Genes"
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
./correlate_align_metrics_with_genes.py -h

./correlate_align_metrics_with_genes.py -am /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_alignment_metrics.txt.gz -e /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -o BIOS-cis-noRNAPhenoNA-NoMDSOutlier-CorrectForAllMetrics
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.am_path = getattr(arguments, 'alignment_metrics')
        self.expr_path = getattr(arguments, 'expression')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'correlate_align_metrics_with_genes')
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
        parser.add_argument("-am",
                            "--alignment_metrics",
                            type=str,
                            required=True,
                            help="The path to RNAseq alignment metrics "
                                 "matrix.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading data.")
        am_df = self.load_file(self.am_path, header=0, index_col=0)
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=None)

        am_df = am_df.loc[:, [x for x in am_df.columns if x != "Sex"]]

        print("Pre-processing data.")
        am_df = am_df.T

        # Make sure order is the same.
        samples = set(am_df.columns.tolist()).intersection(set(expr_df.columns.tolist()))
        am_df = am_df.loc[:, samples]
        expr_df = expr_df.loc[:, samples]

        # Filter on columns without std.
        am_df = am_df.loc[am_df.std(axis=1) > 0, :]
        expr_df = expr_df.loc[expr_df.std(axis=1) > 0, :]

        # Remove columns that are a linear combination of others.
        am_df = self.remove_multicollinearity(am_df.T).T

        print("Correcting gene expression data.")
        # correction_df = am_df.loc[["fastqc_clean.R1_clean_GC_mean", "fastqc_raw.R1_raw_GC_mean", "fastqc_clean.R2_clean_GC_mean", "fastqc_clean.R1_clean_GC_std", "fastqc_clean.R2_clean_GC_std", "fastqc_raw.R2_raw_GC_mean", "fastqc_raw.R1_raw_GC_std", "prime_bias.MEDIAN_5PRIME_BIAS", "fastqc_raw.R2_raw_GC_std", "prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS", "star.pct_mapped_many", "star.pct_mapped_multiple", "star.rate_insertion_per_base", "star.avg_insertion_length", "star.avg_deletion_length", "bam.genome_insert_std", "star.num_mapped_many", "star.rate_mismatch_per_base", "star.pct_unique_mapped", "fastqc_raw.R1_raw_GC_mean"], :].T
        # correction_df = am_df.loc[["fastqc_clean.R1_clean_GC_mean", "fastqc_raw.R1_raw_GC_mean", "fastqc_clean.R2_clean_GC_mean", "fastqc_clean.R1_clean_GC_std", "fastqc_clean.R2_clean_GC_std", "fastqc_raw.R2_raw_GC_mean", "fastqc_raw.R1_raw_GC_std", "prime_bias.MEDIAN_5PRIME_BIAS", "fastqc_raw.R2_raw_GC_std", "prime_bias.MEDIAN_5PRIME_TO_3PRIME_BIAS"], :].T
        # print(correction_df)
        # expr_df = self.calculate_residuals(df=expr_df, correction_df=correction_df)

        # Safe the indices.
        metrics = am_df.index.tolist()
        genes = expr_df.index.tolist()

        # Convert to numpy.
        am_m = am_df.to_numpy()
        expr_m = expr_df.to_numpy()
        del am_df, expr_df

        print("Correlating.")
        corr_m = np.corrcoef(am_m, expr_m)[:am_m.shape[0], am_m.shape[0]:]
        corr_df = pd.DataFrame(corr_m, index=metrics, columns=genes)
        print(corr_df)

        print("Saving file.")
        self.save_file(df=corr_df,
                       outpath=os.path.join(self.outdir, "{}_correlations.txt.gz".format(self.out_filename)))
        # corr_df = self.load_file(os.path.join(self.outdir, "{}_correlations.txt.gz".format(self.out_filename)), header=0, index_col=0)

        print("Ranking top metrics.")
        summary_df = pd.DataFrame({"max abs correlation": corr_df.abs().max(axis=1),
                                   "mean abs correlation": corr_df.abs().mean(axis=1),
                                   "std abs correlation": corr_df.abs().std(axis=1)})
        summary_df.sort_values(by="max abs correlation", inplace=True, ascending=False)
        print(summary_df)

        print("Saving file.")
        self.save_file(df=summary_df,
                       outpath=os.path.join(self.outdir, "{}_summary.txt.gz".format(self.out_filename)))

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def remove_multicollinearity(self, df, threshold=1):
        indices = np.arange(df.shape[1])
        max_r2 = np.inf
        while len(indices) > 1 and max_r2 >= threshold:
            r2 = np.array([self.calc_ols_rsquared(df.iloc[:, indices], ix) for ix in range(len(indices))])
            max_r2 = max(r2)

            if max_r2 >= threshold:
                max_index = np.where(r2 == max_r2)[0][0]
                print("Removing {}".format(df.columns[max_index]))
                indices = np.delete(indices, max_index)

        return df.iloc[:, indices]

    @staticmethod
    def calc_ols_rsquared(df, idx):
        return OLS(df.iloc[:, idx], df.loc[:, np.arange(df.shape[1]) != idx]).fit().rsquared

    @staticmethod
    def calculate_residuals(df, correction_df):
        corrected_m = np.empty_like(df, dtype=np.float64)
        last_print_time = None
        n_tests = df.shape[0]
        for i in range(n_tests):
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 10 or (i + 1) == n_tests:
                last_print_time = now_time
                print("\t{}/{} ieQTLs analysed [{:.2f}%]".format((i + 1), n_tests, (100 / n_tests) * (i + 1)))

            ols = OLS(df.iloc[i, :], correction_df)
            results = ols.fit()
            # print(results.summary())
            corrected_m[i, :] = results.resid

        return pd.DataFrame(corrected_m, index=df.index, columns=df.columns)

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

    def print_arguments(self):
        print("Arguments:")
        print("  > Expression: {}".format(self.expr_path))
        print("  > Alignment metrics: {}".format(self.am_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
