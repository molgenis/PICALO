#!/usr/bin/env python3

"""
File:         count_n_ieqtls.py
Created:      2021/12/20
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
__program__ = "Count N-ieQTLs"
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
./count_n_ieqtls.py -h

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-FIrst100ExprPCsAsCov/

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICsAsCov

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-First33ExprPCsAsCov
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')

        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'count_n_ieqtls')
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
        n_signif_data = []
        indices = []
        for i in range(101):
            covariate = "PIC{}.txt.gz".format(i)
            fpath = os.path.join(self.indir, covariate)
            print(fpath)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)
                n_signif = df.loc[df["ieQTL FDR"] < 0.05, :].shape[0]
                print(covariate, n_signif)
                n_signif_data.append(n_signif)
                indices.append(covariate)
        df = pd.DataFrame(n_signif_data, indices)
        print(df)
        print(os.path.join(self.outdir, "{}.xlsx".format(os.path.basename(self.indir))))
        df.to_excel(os.path.join(self.outdir, "{}.xlsx".format(os.path.basename(self.indir))))

        n_signif_a = np.array(n_signif_data)
        print(np.sum(n_signif_a))
        print(np.mean(n_signif_a))
        print(np.std(n_signif_a))
        print(np.max(n_signif_a))

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
