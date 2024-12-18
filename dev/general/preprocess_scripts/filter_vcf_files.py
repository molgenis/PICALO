#!/usr/bin/env python3

"""
File:         filter_vcf_files.py
Created:      2024/01/19
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import gzip
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Filter VCF files"
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
./filter_vcf_files.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.gte_path = getattr(arguments, 'genotype_to_expression')
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'filter_vcf_files')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__,
                                         )

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
                            default=None,
                            help="The path to input directory.")
        parser.add_argument("-e",
                            "--eqtl",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to eqtl file.")
        parser.add_argument("-gte",
                            "--genotype_to_expression",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the genotype-to-expression"
                                 " link matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The output filename.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading GTE data.")
        gte_df = self.load_file(self.gte_path, header=0, index_col=None)
        print(gte_df)

        print("Loading eQTL data.")
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None)
        print(eqtl_df)

        print("Filtering VCF data.")
        for dataset in gte_df["dataset"].unique():
        #for dataset in ["PAN"]:
            in_dataset = dataset
            if dataset == "NTR_AFFY":
                in_dataset = "NTR_Affy6"
            if dataset == "NTR_GONL":
                in_dataset = "NTR_GoNL"
            print("  Processing '{}'".format(dataset))

            outpath = os.path.join(self.outdir, "{}.{}.vcf.gz".format(self.outfile, dataset))
            if os.path.exists(outpath):
                print("    Output already exists")
                continue


            query_samples = set(gte_df.loc[gte_df["dataset"] == dataset, "genotype_id"].values)
            print("    Looking for {:,} samples\n".format(len(query_samples)))
            if len(query_samples) == 0:
                continue

            dataset_df_list = []
            for chr in range(1, 23):
                print("    Processing 'chr{}'".format(chr))
                query_snps = set(eqtl_df.loc[eqtl_df["SNPChr"] == chr, "SNPChrPos"].values)
                print("      Looking for {:,} SNPs\n".format(len(query_snps)))

                columns = None
                data = []

                vcf_path = os.path.join(self.indir, in_dataset, "postimpute", "chr{}.filtered.vcf.gz".format(chr))
                if not os.path.exists(vcf_path):
                    continue

                column_mask = None
                pos_index = None
                fh = gzip.open(vcf_path, 'rt')
                for i, line in enumerate(fh):
                    if (i == 0) or (i % 10000 == 0):
                        print("      parsed {:,} lines, saved {:,} lines".format(i, len(data)))
                    if len(data) == (len(query_snps) + 1):
                        break

                    # Skip comments.
                    if line.startswith("##"):
                        continue

                    # Find query samples from header.
                    values = np.array(line.strip("\n").split("\t"), dtype=object)
                    if line.startswith("#CHROM"):
                        pos_index = np.where(values == "POS")[0][0]
                        column_mask = np.array([True if (value in query_samples or value in ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]) else False for value in values ])
                        columns = values[column_mask]
                        continue

                    # Check if SNP is in query.
                    if int(values[pos_index]) in query_snps:
                        data.append(values[column_mask])

                print("      processed {:,} lines, saved {:,} lines".format(i, len(data)))

                df = pd.DataFrame(data, columns=columns)
                print(df)
                dataset_df_list.append(df)

                print("")

            # Merge dataset df.
            dataset_df = pd.concat(dataset_df_list, axis=0)
            print(dataset_df)

            self.save_file(df=dataset_df, outpath=outpath)

            missing_samples = query_samples - set(dataset_df.columns)
            print("      Missing {:,} samples".format(len(missing_samples)))
            print(missing_samples)

            del dataset_df_list, dataset_df
            print("")

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
    def save_file(df, outpath, header=True, index=False, sep="\t"):
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
        print("  > GTE path: {}".format(self.gte_path))
        print("  > eQTL path: {}".format(self.eqtl_path))
        print("  > Output filename: {}".format(self.outfile))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
