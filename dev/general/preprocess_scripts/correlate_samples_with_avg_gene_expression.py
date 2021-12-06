#!/usr/bin/env python3

"""
File:         correlate_samples_with_avg_gene_expression.py
Created:      2021/11/15
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
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Correlate Samples with Average "
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
./correlate_samples_with_avg_gene_expression.py -h

./correlate_samples_with_avg_gene_expression.py -e /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/pre_process_metabrain_expression_matrix/MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier/data/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.txt.gz -op MetaBrain

./correlate_samples_with_avg_gene_expression.py -e /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_bios_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_20RNAseqAlignemntMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.txt.gz -op BIOS
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.expr_path = getattr(arguments, 'expression')
        self.log2 = getattr(arguments, 'log2')
        self.outfile_prefix = getattr(arguments, 'outfile_prefix')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'correlate_samples_with_avg_gene_expression')
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
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-log2",
                            action='store_true',
                            help="Combine the created files with force."
                                 " Default: False.")
        parser.add_argument("-op",
                            "--outfile_prefix",
                            type=str,
                            required=False,
                            default="Data",
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        print("Loading data.")
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=None)

        print("Pre-processing data.")
        # Convert to numpy.
        samples = expr_df.columns.tolist()
        expr_m = expr_df.to_numpy()
        del expr_df

        # Remove zero variance genes.
        print("\tRemove zero variance genes.")
        mask = np.std(expr_m, axis=1) != 0
        print("\tUsing {}/{} probes.".format(np.sum(mask), np.size(mask)))
        expr_m = expr_m[mask, :]

        # Log2 transform.
        if self.log2:
            print("\tLog2 transform the data.")
            min_value = np.min(expr_m, axis=1).min()
            if min_value <= 0:
                expr_m = np.log2(expr_m - min_value + 1)
            else:
                expr_m = np.log2(expr_m + 1)

        # set samples on index
        expr_m = expr_m.T

        # calculate average matrix
        expr_avg_a = np.mean(expr_m, axis=0)

        print("Correlating.")
        corr_m = np.corrcoef(expr_m, expr_avg_a)[:expr_m.shape[0], expr_m.shape[0]:]
        corr_df = pd.DataFrame(corr_m, index=samples, columns=["AvgExprCorrelation"])
        print(corr_df)

        print("Saving file.")
        self.save_file(df=corr_df,
                       outpath=os.path.join(self.outdir, "{}_CorrelationsWithAverageExpression.txt.gz".format(self.outfile_prefix)))

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
        print("  > Log2: {}".format(self.log2))
        print("  > Outputfile prefix: {}".format(self.outfile_prefix))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
