#!/usr/bin/env python3

"""
File:         pre_process_bios_gtd.py
Created:      2021/10/28
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
import glob
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Pre-Process BIOS GTD"
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
./pre_process_bios_gtd.py -i /groups/umcg-bios/tmp01/resources/genotypes-hrc-imputed-trityper -o ../data

./pre_process_bios_gtd.py -i /groups/umcg-bios/prm02/projects/HRC_imputed_trityper/1000G_harmonized/ -o .
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'input')
        self.outdir = getattr(arguments, 'output')

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
                            "--input",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            required=True,
                            help="The path to the output directory.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading individual data")

        # Find the datasets folders (i.e. subdirectories).
        datasets = [name for name in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, name))]

        # Loop over datasets
        individual_dfs = []
        for dataset in datasets:
            print(dataset)
            # for chrom_id in range(1, 23, 1):
            #     print("chr{}".format(chrom_id))
            #     fpath = os.path.join(self.input_directory, dataset, "chr{}".format(chrom_id), "Individuals.txt")
            #     if not os.path.exists(fpath):
            #         print("Not found.")
            #         continue
            #     individual_df = self.load_file(fpath, index_col=None, header=None)
            #     individual_df.columns = ["sample"]
            #     individual_df["dataset"] = dataset
            #
            #     individual_dfs.append(individual_df)


            fpath = os.path.join(self.input_directory, dataset, "Individuals.txt")
            if not os.path.exists(fpath):
                print("Not found.")
                continue
            individual_df = self.load_file(fpath, index_col=None, header=None)
            individual_df.columns = ["sample"]
            individual_df["dataset"] = dataset

            individual_dfs.append(individual_df)

        # Merge the data frames.
        gtd_df = pd.concat(individual_dfs, axis=0)
        gtd_df.drop_duplicates(inplace=True)
        print(gtd_df)

        self.save_file(df=gtd_df, outpath=os.path.join(self.outdir, "BIOS_GenotypeToDataset.txt.gz"), index=False)

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith(".pkl"):
            df = pd.read_pickle(inpath)
        else:
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
        print("  > Input directory: {}".format(self.input_directory))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
