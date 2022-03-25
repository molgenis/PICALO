#!/usr/bin/env python3

"""
File:         calculate_correlation_start_vs_end.py
Created:      2022/01/14
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
import pandas as pd
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Count Correlation Start VS End"
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
./calculate_correlation_start_vs_end.py -h

### MetaBrain ###

./calculate_correlation_start_vs_end.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/

### BIOS ###

./calculate_correlation_start_vs_end.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/
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
            fpath1 = os.path.join(self.indir, "PIC_interactions", "{}.txt.gz".format(covariate))
            fpath2 = os.path.join(self.indir, covariate, "iteration.txt.gz")
            print(fpath2)
            if os.path.exists(fpath1) and os.path.exists(fpath2):
                df = pd.read_csv(fpath2, sep="\t", header=0, index_col=0)
                coef, _ = stats.spearmanr(df.loc[df.index[0], :], df.loc[df.index[-1], :])
                coefficient_data.append([covariate, coef])

        coef_df = pd.DataFrame(coefficient_data, columns=["covariate", "coef"])
        print(coef_df)

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
