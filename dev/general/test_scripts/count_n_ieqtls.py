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

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-FIrst100ExprPCsAsCov-PICsNotRemoved/

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-100TMMLog2ExprPCs-PICsNotRemoved/

./count_n_ieqtls.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-FIrst100ExprPCsAsCov/

./count_n_ieqtls.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-FIrst100ExprPCsAsCov-PICsNotRemoved/

./count_n_ieqtls.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs-100TMMLog2ExprPCs-PICsNotRemoved/

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-PICsAsCov

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs-First33ExprPCsAsCov

./count_n_ieqtls.py -i /groups/umcg-bios/tmp01/projects/PICALO/output/2021-12-09-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/PIC_interactions

./count_n_ieqtls.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2021-12-09-MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/PIC_interactions
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
        ieqtl_df_list = []
        for i in range(101):
            covariate = "PIC{}".format(i)
            filename = "{}.txt.gz".format(covariate)
            fpath = os.path.join(self.indir, filename)
            print(fpath)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t", header=0, index_col=None)

                signif_col = None
                if "ieQTL FDR" in df.columns:
                    signif_col = "ieQTL FDR"
                    df.index = df["snp"] + "_" + df["gene"]
                elif "FDR" in df.columns:
                    signif_col = "FDR"
                    df.index = df["SNP"] + "_" + df["gene"]
                else:
                    print("No signif column found")
                    exit()

                ieqtls = df.loc[df[signif_col] < 0.05, :].index
                ieqtl_df = pd.DataFrame(0, index=df.index, columns=[covariate])
                ieqtl_df.loc[ieqtls, covariate] = 1

                ieqtl_df_list.append(ieqtl_df)

                del ieqtl_df
        ieqtl_df = pd.concat(ieqtl_df_list, axis=1)

        bla = ieqtl_df.copy()
        bla["snp"] = [x.split("_")[0] for x in bla.index]
        bla["gene"] = [x.split("_")[1] for x in bla.index]
        bla.to_excel("PICs.xlsx")
        del bla
        cov_sum = ieqtl_df.sum(axis=0)
        print(cov_sum)
        exit()

        print("Stats per covariate:")
        print("\tSum: {:,}".format(cov_sum.sum()))
        print("\tMean: {:.1f}".format(cov_sum.mean()))
        print("\tSD: {:.2f}".format(cov_sum.std()))
        print("\tMax: {:.2f}".format(cov_sum.max()))

        print("Stats per eQTL")
        counts = dict(zip(*np.unique(ieqtl_df.sum(axis=1), return_counts=True)))
        eqtls_w_inter = ieqtl_df.loc[ieqtl_df.sum(axis=1) > 0, :].shape[0]
        total_eqtls = ieqtl_df.shape[0]
        for value, n in counts.items():
            if value != 0:
                print("\tN-eQTLs with {} interaction: {:,} [{:.2f}%]".format(value, n, (100 / eqtls_w_inter) * n))
        print("\tUnique: {:,} / {:,} [{:.2f}%]".format(eqtls_w_inter, total_eqtls, (100 / total_eqtls) * eqtls_w_inter))

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
