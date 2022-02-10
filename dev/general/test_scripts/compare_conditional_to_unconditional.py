#!/usr/bin/env python3

"""
File:         compare_conditional_vs_unconditional.py
Created:      2021/12/21
Last Changed: 2022/02/10
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
import glob
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.


# Metadata
__program__ = "Compare Conditional vs Unconditional"
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
./compare_conditional_to_unconditional.py -h

./compare_conditional_to_unconditional.py -i /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')

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
        print("Loading conditional results")
        conditional_fdr_data = []
        for i in range(101):
            pic = "PIC{}".format(i)

            fpaths = glob.glob(os.path.join(self.indir, pic, "results_*.txt.gz"))
            fpaths.sort()
            if len(fpaths) > 0:
                fpath = fpaths[-1]

                if os.path.exists(fpath):
                    df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)
                    fdr_df = df[["FDR"]]
                    fdr_df.columns = [pic]
                    conditional_fdr_data.append(fdr_df)
        conditional_fdr_df = pd.concat(conditional_fdr_data, axis=1)
        print(conditional_fdr_df)

        print("### Step2 ###")
        print("Loading unconditional results")
        unconditional_fdr_data = []
        for i in range(101):
            pic = "PIC{}".format(i)

            fpath = os.path.join(self.indir, "PIC_interactions", "{}.txt.gz".format(pic))
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)
                fdr_df = df[["FDR"]]
                fdr_df.columns = [pic]
                unconditional_fdr_data.append(fdr_df)
        unconditional_fdr_df = pd.concat(unconditional_fdr_data, axis=1)
        print(unconditional_fdr_df)

        print("### Step3 ###")
        print("Compare")
        conditional_total_ieqtls = 0
        unconditional_total_ieqtls = 0
        for i in range(101):
            pic = "PIC{}".format(i)
            if pic in conditional_fdr_df.columns and pic in unconditional_fdr_df.columns:
                n_conditional_ieqtls = conditional_fdr_df.loc[conditional_fdr_df[pic] <= 0.05, :].shape[0]
                n_unconditional_ieqtls = unconditional_fdr_df.loc[unconditional_fdr_df[pic] <= 0.05, :].shape[0]
                print("{}:\tconditional: {:,}\tunconditional: {:,}".format(pic, n_conditional_ieqtls, n_unconditional_ieqtls))

                conditional_total_ieqtls += n_conditional_ieqtls
                unconditional_total_ieqtls += n_unconditional_ieqtls
        print("------------------------")
        print("Total:\tconditional: {:,}\tunconditional: {:,}".format(conditional_total_ieqtls, unconditional_total_ieqtls))


    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
