#!/usr/bin/env python3

"""
File:         export_n_ieqtls.py
Created:      2022/06/04
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
from pathlib import Path
import glob
import re
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Export N-ieQTLs"
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
./export_n_ieqtls.py -h
"""

class main():
    def __init__(self):
        # Define the input paths.
        self.brain_indir = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/"
        self.brain_filename = "2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA"
        self.blood_indir = "/groups/umcg-bios/tmp01/projects/PICALO/fast_interaction_mapper"
        self.blood_filename = "2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA"

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'export_n_ieqtls')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, "number_of_ieqtls.xlsx")

    def start(self):
        with pd.ExcelWriter(self.outpath) as writer:

            for indir, filename, symbol, name in [(self.blood_indir, self.blood_filename, "-", "blood"), (self.brain_indir, self.brain_filename, "_", "brain")]:
                print("Loading {} data".format(name))

                pic_df_default = self.load_data(indir=os.path.join(indir, filename + symbol + "PICsAsCov"))
                pic_df_cond = self.load_data(indir=os.path.join(indir, filename + symbol + "PICsAsCov-Conditional"), conditional=True)

                pc_df_default = self.load_data(indir=os.path.join(indir, filename + symbol + "PCsAsCov"))
                pc_df_cond = self.load_data(indir=os.path.join(indir, filename + symbol + "PCsAsCov-Conditional"), conditional=True)

                df = pd.concat([pic_df_default, pic_df_cond, pc_df_default, pc_df_cond], axis=1)
                df.columns = ["PIC - default", "PIC - conditional", "PC - default", "PC - conditional"]
                df.insert(0, "component", [i + 1 for i in range(df.shape[0])])
                df.to_excel(writer, sheet_name=name, na_rep="NA", index=False)
                print("Saving sheet '{}' with shape {}".format(name, df.shape))

    def load_data(self, indir, conditional=False):
        print(indir)
        ieqtl_fdr_df_list = []
        inpaths = glob.glob(os.path.join(indir, "*.txt.gz"))
        if conditional:
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
        ieqtls_count_df = ieqtl_fdr_df.sum(axis=0).to_frame()
        ieqtls_count_df.columns = ["#ieQTLs"]
        ieqtls_count_df.reset_index(drop=True, inplace=True)
        return ieqtls_count_df

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


if __name__ == '__main__':
    m = main()
    m.start()
