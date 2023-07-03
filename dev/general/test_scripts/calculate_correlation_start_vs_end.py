#!/usr/bin/env python3

"""
File:         calculate_correlation_start_vs_end.py
Created:      2022/01/14
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import pandas as pd
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Count Correlation Start VS End"
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
./calculate_correlation_start_vs_end.py -h

### MetaBrain ###

./calculate_correlation_start_vs_end.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/

### BIOS ###

./calculate_correlation_start_vs_end.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')

        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'count_n_ieqtls')
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to input directory.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading PICALO results")
        coefficient_data = []
        for i in range(101):
            covariate = "PIC{}".format(i)
            fpath2 = os.path.join(self.indir, covariate, "iteration.txt.gz")
            print(fpath2)
            if os.path.exists(fpath2):
                df = pd.read_csv(fpath2, sep="\t", header=0, index_col=0)
                coef, _ = stats.spearmanr(df.loc[df.index[0], :], df.loc[df.index[-1], :])
                coefficient_data.append([covariate, coef])

        coef_df = pd.DataFrame(coefficient_data, columns=["covariate", "coef"])
        print(coef_df)

        n = 5
        print(coef_df.iloc[:n, :])
        print("Spearman correlations first {}:".format(n))
        print("\tMean: {:.2f}".format(coef_df.iloc[:n, :]["coef"].mean()))
        print("\tSD: {:.2f}".format(coef_df.iloc[:n, :]["coef"].std()))

        print(coef_df.iloc[n:, :])
        print("Spearman correlations rest")
        print("\tMean: {:.2f}".format(coef_df.iloc[n:, :]["coef"].mean()))
        print("\tSD: {:.2f}".format(coef_df.iloc[n:, :]["coef"].std()))

        print("Spearman correlations:")
        print("\tMean: {:.2f}".format(coef_df["coef"].mean()))
        print("\tSD: {:.2f}".format(coef_df["coef"].std()))

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
