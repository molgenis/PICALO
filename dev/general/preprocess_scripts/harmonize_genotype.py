#!/usr/bin/env python3

"""
File:         harmonzize_genotype.py
Created:      2021/11/13
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
import gzip
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Harmonize Genotype"
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
./harmonize_genotype.py -h

./harmonize_genotype.py -dg /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-05-26-eqtls-rsidfix-popfix/cis/2020-05-26-Cortex-EUR/genotypedump/GenotypeData.txt.gz -rg /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-05-26-eqtls-rsidfix-popfix/cis/2020-05-26-Cortex-AFR/genotypedump_EUR_SNPs/GenotypeData.txt.gz -o 2020-05-26-Cortex-AFR-ReplicatonOfCortexEUR-GenotypeData
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.discovery_genotype_path = getattr(arguments, 'discovery_genotype')
        self.replication_genotype_path = getattr(arguments, 'replication_genotype')
        self.output_filename = getattr(arguments, 'output_filename')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'harmonize_genotype')
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
        parser.add_argument("-dg",
                            "--discovery_genotype",
                            type=str,
                            required=True,
                            help="The path to the discovery genotype matrix")
        parser.add_argument("-rg",
                            "--replication_genotype",
                            type=str,
                            required=True,
                            help="The path to the discovery genotype matrix")
        parser.add_argument("-o",
                            "--output_filename",
                            type=str,
                            required=True,
                            help="The name of the output file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading genotype data.")
        d_data_df = self.load_file(self.discovery_genotype_path, header=0, index_col=0)
        d_data_df = d_data_df.groupby(d_data_df.index).first()
        r_data_df = self.load_file(self.replication_genotype_path, header=0, index_col=0)
        r_data_df = r_data_df.groupby(r_data_df.index).first()
        print(d_data_df)
        print(r_data_df)

        print("Split the dataframes.")
        d_alleles_df = d_data_df.iloc[:, :2]
        r_alleles_df = r_data_df.iloc[:, :2]
        r_geno_df = r_data_df.iloc[:, 2:]
        del d_data_df, r_data_df

        print("Access the minor allele")
        alleles_df = r_alleles_df[["MinorAllele"]].merge(d_alleles_df[["MinorAllele"]], left_index=True, right_index=True)
        alleles_df["flip"] = alleles_df.iloc[:, 0] != alleles_df.iloc[:, 1]
        flip_mask = alleles_df["flip"].to_numpy(dtype=bool)
        snps_of_interest = list(alleles_df.index)

        print("Processing genotype file.")
        r_geno_df = r_geno_df.loc[snps_of_interest, :]

        # Flip genotypes.
        geno_m = r_geno_df.to_numpy()
        missing_mask = geno_m == -1
        geno_m[flip_mask, :] = 2 - geno_m[flip_mask, :]
        geno_m[missing_mask] = -1
        r_geno_df = pd.DataFrame(geno_m, index=r_geno_df.index, columns=r_geno_df.columns)
        r_geno_df.index.name = None
        del geno_m

        # Delete rows with all NaN.
        mask = r_geno_df.isnull().all(1).to_numpy()
        r_geno_df = r_geno_df.loc[~mask, :]

        print("Combine alleles and genotype file.")
        out_data_df = d_alleles_df.loc[snps_of_interest, :].merge(r_geno_df, left_index=True, right_index=True)
        print(out_data_df)

        print("\tSaving output file.")
        self.save_file(df=out_data_df,
                       outpath=os.path.join(self.outdir, "{}.txt.gz".format(self.output_filename)))

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
        print("  > Discovery genotype path: {}".format(self.discovery_genotype_path))
        print("  > Replication genotype path: {}".format(self.replication_genotype_path))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Output filename: {}".format(self.output_filename))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
