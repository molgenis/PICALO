#!/usr/bin/env python3

"""
File:         calc_avg_gene_expression.py
Created:      2022/02/17
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

# Local application imports.

# Metadata
__program__ = "Calculate Average Gene Expression"
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
./calc_avg_gene_expression.py -h

### METABRAIN ###

./calc_avg_gene_expression.py \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-01-31-expression-tables/2020-02-05-step6-covariate-removal/2021-08-27-step5-remove-covariates-per-dataset/output-PCATitration-MDSCorrectedPerDsCovarOverall-cortex-EUR/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.ProbesWithZeroVarianceRemoved.txt.gz
    
### BIOS ###    

./calc_avg_gene_expression.py \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.txt.gz

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.expression_path = getattr(arguments, 'expression')

        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'calc_avg_gene_expression')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if not self.expression_path.endswith(".txt.gz"):
            print("Expression path should end with '.txt.gz'")
            exit()

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
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            default=None,
                            help="The path to the expression matrix in TMM"
                                 "format.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading expression data")
        expr_df = self.load_file(self.expression_path, header=0, index_col=0)
        std_df = self.load_file(self.std_path, header=0, index_col=None)
        samples = std_df.iloc[:, 0].values.tolist()

        print("Sample selection.")
        expr_df = expr_df.loc[:, samples]
        print("\tUsing {} samples".format(len(samples)))

        print("Log2 transform.")
        min_value = expr_df.min(axis=1).min()
        if min_value <= 0:
            expr_df = np.log2(expr_df - min_value + 1)
        else:
            expr_df = np.log2(expr_df + 1)

        print("Calculate average.")
        avg_df = expr_df.mean(axis=1).to_frame()
        avg_df.columns = ["average"]

        print("Saving file.")
        print(avg_df)
        filename = os.path.basename(self.expression_path).replace(".txt.gz", "")
        self.save_file(df=avg_df, outpath=os.path.join(self.outdir, "{}.Log2Transformed.AverageExpression.txt.gz".format(filename)))

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
        print("  > Expression: {}".format(self.expression_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
