#!/usr/bin/env python3

"""
File:         save_ieqtl_fdr_matrix.py
Created:      2022/07/05
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import glob
import argparse
import re
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Save ieQTL FDR Matrix"
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
./save_ieqtl_fdr_matrix.py -h

./save_ieqtl_fdr_matrix.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov
    
./save_ieqtl_fdr_matrix.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional \
    -conditional \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov-Conditional
    
./save_ieqtl_fdr_matrix.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov
    
./save_ieqtl_fdr_matrix.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional \
    -conditional \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov-Conditional

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.conditional = getattr(arguments, 'conditional')
        self.outname = getattr(arguments, 'outname')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'save_ieqtl_fdr_matrix')
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
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Perform conditional analysis. Default: False.")
        parser.add_argument("-o",
                            "--outname",
                            type=str,
                            required=True,
                            help="The name of the output file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading results")
        ieqtl_fdr_df_list = []
        inpaths = glob.glob(os.path.join(self.indir, "*.txt.gz"))
        if self.conditional:
            inpaths = [inpath for inpath in inpaths if inpath.endswith("_conditional.txt.gz")]
        else:
            inpaths = [inpath for inpath in inpaths if not inpath.endswith("_conditional.txt.gz")]
        inpaths.sort(key=self.natural_keys)
        for i, inpath in enumerate(inpaths):
            filename = os.path.basename(inpath).split(".")[0].replace("_conditional", "")
            if filename in ["call_rate", "genotype_stats"]:
                continue

            df = self.load_file(inpath, header=0, index_col=None)
            signif_col = "FDR"
            df.index = df["SNP"] + "_" + df["gene"]

            ieqtls = df.loc[df[signif_col] <= 0.05, :].index
            ieqtl_fdr_df = pd.DataFrame(0, index=df.index, columns=[filename])
            ieqtl_fdr_df.loc[ieqtls, filename] = 1
            ieqtl_fdr_df_list.append(ieqtl_fdr_df)

            del ieqtl_fdr_df

        ieqtl_fdr_df = pd.concat(ieqtl_fdr_df_list, axis=1)
        print(ieqtl_fdr_df)

        print("Saving file")
        self.save_file(df=ieqtl_fdr_df, outpath=os.path.join(self.outdir, self.outname + ".txt.gz"))

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
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

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
        print("  > Input directory: {}".format(self.indir))
        print("  > Conditional: {:,}".format(self.conditional))
        print("  > Outname: {}".format(self.outname))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
