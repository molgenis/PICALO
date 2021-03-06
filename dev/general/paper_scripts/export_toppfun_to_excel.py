#!/usr/bin/env python3

"""
File:         export_toppfun_to_excel.py
Created:      2022/06/24
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
import re
import os

# Third party imports.
import pandas as pd

# Local application imports.

"""
Syntax:
./export_toppfun_to_excel.py -h

### BIOS ###

./export_toppfun_to_excel.py \
    -b /groups/umcg-bios/tmp01/projects/PICALO/postprocess_scripts/gene_set_enrichment/2022-04-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_ZscoreFiltering/toppgene_enrichment_df.txt.gz \
    -m /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/gene_set_enrichment/2022-04-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_FNPDGeneCorrelations_ZscoreFiltering/toppgene_enrichment_df.txt.gz \
    -o 2022-04-13-SupplementaryTable4_v1
"""


# Metadata
__program__ = "Export ToppFun to Excel"
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


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.bios_path = getattr(arguments, 'bios')
        self.metabrain_path = getattr(arguments, 'metabrain')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'export_toppfun_to_excel')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, self.outfile + ".xlsx")

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
        parser.add_argument("-b",
                            "--bios",
                            type=str,
                            required=True,
                            help="The path to BIOS file.")
        parser.add_argument("-m",
                            "--metabrain",
                            type=str,
                            required=True,
                            help="The path to MetaBrain file.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load data.
        bios_df = self.load_file(self.bios_path, header=0, index_col=None)
        metabrain_df = self.load_file(self.metabrain_path, header=0, index_col=None)

        print(bios_df)
        print(metabrain_df)

        col_trans_dict = {
            "PValue": "p-value",
            "QValueFDRBH": "BH-FDR",
            "TotalGenes": "number of genes in category",
            "GenesInTerm": "number of genes in annotation",
            "GenesInQuery": "number of input genes in category",
            "GenesInTermInQuery": "number of genes from input",
            "covariate": "PIC",
            "correlation_direction": "correlation direction",
            "avg_abs_correlation_zscore_inTerm": "average absolute correlation z-score of genes in term",
            "avg_abs_correlation_zscore_Overall": "average absolute correlation z-score of genes from input"
        }
        for df in [bios_df, metabrain_df]:
            df.insert(0, "PIC", df["covariate"])
            df.insert(11, "HGNC symbols of genes from input", df["Genes"])
            df.drop(["QValueFDRBY", "QValueBonferroni", "Genes", "covariate", "N"], axis=1, inplace=True)
            df.columns = [col_trans_dict[col] if col in col_trans_dict else col for col in df.columns]
            print(df)

        with pd.ExcelWriter(self.outpath) as writer:
            for df, name in [(bios_df, "blood"), (metabrain_df, "brain")]:
                for category in ["ToppCell", "Pathway"]:
                    for subset in ["genes", "ieQTL genes"]:
                        susbet_df = df.loc[(bios_df["Category"] == category) & (df["subset"] == subset), :].copy()
                        susbet_df.drop(["Category", "subset"], axis=1, inplace=True)
                        sheet_name = "{} - {} - {}".format(name, category, subset)

                        susbet_df.to_excel(writer,
                                           sheet_name=sheet_name,
                                           na_rep="NA",
                                           index=False)
                        print("Saving sheet '{}' with shape {}".format(sheet_name, susbet_df.shape))
                        print("")

                        del susbet_df, sheet_name

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("Loaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def print_arguments(self):
        print("Arguments:")
        print("  > Bios input file: {}".format(self.bios_path))
        print("  > MetaBrain input file: {}".format(self.metabrain_path))
        print("  > Output file: {}".format(self.outfile))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
