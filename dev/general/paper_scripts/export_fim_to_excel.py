#!/usr/bin/env python3

"""
File:         export_fim_to_excel.py
Created:      2022/06/23
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
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
./export_fim_to_excel.py -h
"""


# Metadata
__program__ = "Export FIM to Excel"
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


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.conditional = getattr(arguments, 'conditional')
        self.alleles_path = getattr(arguments, 'alleles')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'export_fim_to_excel')
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to input directory.")
        parser.add_argument("-conditional",
                            action='store_true',
                            help="Load conditional files. Default: False.")
        parser.add_argument("-al",
                            "--alleles",
                            type=str,
                            required=True,
                            help="The path to the alleles matrix")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene information matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the output file")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Load alleles data.
        alleles_df = self.load_file(self.alleles_path, index_col=None)
        alleles_df["Affect Allele"] = alleles_df["Alleles"].str.split("/", n=1, expand=True)[1]
        snp_to_alles_dict = dict(zip(alleles_df["SNP"], alleles_df["Alleles"]))
        snp_to_ae_dict = dict(zip(alleles_df["SNP"], alleles_df["Affect Allele"]))
        del alleles_df

        # Load HGNC translate data.
        gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        gene_info_df["gene"] = gene_info_df["ArrayAddress"].str.split(".", n=1, expand=True)[0]
        gene_dict = dict(zip(gene_info_df["gene"], gene_info_df["Symbol"]))
        del gene_info_df

        with pd.ExcelWriter(self.outpath) as writer:
            # call_rate_df = self.load_file(os.path.join(self.indir, "call_rate.txt.gz"), header=0, index_col=None)
            # call_rate_df.columns = [col.replace("CR", "call rate") for col in call_rate_df.columns]
            #
            # call_rate_df.to_excel(writer, sheet_name="Call Rate", na_rep="NA", index=False)
            # print("\tSaving sheet 'Call Rate' with shape {}".format(call_rate_df.shape))
            # print("")
            # del call_rate_df

            ####################################################################

            genotype_stats_df = self.load_file(os.path.join(self.indir, "genotype_stats.txt.gz"), header=0, index_col=None)
            # genotype_stats_df.to_excel(writer, sheet_name="Genotype Statistics", na_rep="NA", index=False)
            # print("\tSaving sheet 'Genotype Statistics' with shape {}".format(genotype_stats_df.shape))
            # print("")

            maf_dict = dict(zip(genotype_stats_df["SNP"], genotype_stats_df["MAF"]))
            hw_dict = dict(zip(genotype_stats_df["SNP"], genotype_stats_df["HW pval"]))
            del genotype_stats_df

            for i in range(1, 100):
                pic = "PIC{}".format(i)
                pic_path = os.path.join(self.indir, "{}{}.txt.gz".format(pic, "_conditional" if self.conditional else ""))
                print(pic_path)
                if not os.path.exists(pic_path):
                    continue

                pic_df = self.load_file(pic_path, header=0, index_col=None)

                pic_df.columns = [col if col != "FDR" else "BH-FDR" for col in pic_df.columns]
                pic_df.drop(["covariate"], axis=1, inplace=True)
                pic_df.insert(1, "allele", pic_df["SNP"].map(snp_to_alles_dict))
                pic_df.insert(2, "affect allele", pic_df["SNP"].map(snp_to_ae_dict))
                pic_df.insert(3, "MAF", pic_df["SNP"].map(maf_dict))
                pic_df.insert(4, "HW p-value", pic_df["SNP"].map(hw_dict))
                pic_df["gene"] = pic_df["gene"].str.split(".", n=1, expand=True)[0]
                pic_df.insert(6, "symbol", pic_df["gene"].map(gene_dict))

                pic_df.sort_values(by="BH-FDR", inplace=True)

                # Save
                pic_df.to_excel(writer, sheet_name=pic, na_rep="NA", index=False)
                print("Saving sheet '{}' with shape {}".format(pic, pic_df.shape))
                print("")

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
        print("  > Input directory: {}".format(self.indir))
        print("  > Conditional: {}".format(self.conditional))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Gene info path: {}".format(self.gene_info_path))
        print("  > Output file: {}".format(self.outfile))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
