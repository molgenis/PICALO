#!/usr/bin/env python3

"""
File:         prepare_bios_eqtl_file.py
Created:      2021/11/11
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
__program__ = "Prepare BIOS eQTL file"
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
./prepare_bios_eqtl_file.py -h

./prepare_bios_eqtl_file.py -e /groups/umcg-bios/tmp01/projects/PICALO/data/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt.gz -s /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport/SNP_dataset_matrix.txt.gz -p EQTLGen_

./prepare_bios_eqtl_file.py -e /groups/umcg-bios/tmp01/projects/PICALO/data/gene_level_eQTLs_independent_effects_interactions.txt.gz -s /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata/brittexport/SNP_dataset_matrix.txt.gz -p BIOS_
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.snps_path = getattr(arguments, 'snps')
        self.n_iterations = getattr(arguments, 'n_iterations')
        self.n_datasets = getattr(arguments, 'n_datasets')
        self.prefix = getattr(arguments, 'prefix')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'prepare_bios_eqtl_file')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.dataset_order = ['RS', 'LL', 'LLS_660Q', 'NTR_AFFY', 'LLS_OmniExpr', 'CODAM', 'PAN', 'NTR_GONL', 'GONL']
        self.dataset_sizes = np.array([765, 733, 400, 341, 255, 184, 169, 138, 74])

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
        parser.add_argument("-e",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the replication eqtl matrix.")
        parser.add_argument("-s",
                            "--snps",
                            type=str,
                            required=True,
                            help="The path to the genotype directory")
        parser.add_argument("-ni",
                            "--n_iterations",
                            type=int,
                            default=4,
                            help="The number of eQTL iterations to include. "
                                 "Default: 4.")
        parser.add_argument("-nd",
                            "--n_datasets",
                            type=int,
                            default=2,
                            help="The number of required datasets per SNP. "
                                 "Default: 2.")
        parser.add_argument("-p",
                            "--prefix",
                            type=str,
                            required=True,
                            help="Prefix for the output file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None)
        snps_df = self.load_file(self.snps_path, header=0, index_col=0)
        print(eqtl_df)
        print(snps_df)

        print("Preprocessing data")
        # Translating column names.
        trans_dict = {"P-value": "PValue",
                      "SNP": "SNPName",
                      "SNP Chr": "SNPChr",
                      "SNP Chr Position": "SNPChrPos",
                      "Gene": "ProbeName",
                      "Gene Chr": "ProbeChr",
                      "Gene Chr position": "ProbeCenterChrPos",
                      "GeneId": "ProbeName",
                      "GeneChr": "ProbeChr",
                      "GeneCenterChrPos": "ProbeCenterChrPos",
                      "SNP Alleles": "SNPType",
                      "Assesed Allele": "AlleleAssessed",
                      "Z-score": "OverallZScore",
                      "Gene name": "HGNCName",
                      "eQTLLevelNumber": "Iteration"
                      }
        eqtl_df.columns = [trans_dict[x] if x in trans_dict else x for x in eqtl_df.columns]

        # Remove interaction columns.
        drop_columns = []
        for i in range(20):
            column = "M{}".format(i)
            if column in eqtl_df.columns:
                drop_columns.append(column)
        eqtl_df.drop(drop_columns, axis=1, inplace=True)
        print(eqtl_df)

        # Adding iteration column if not there.
        if "Iteration" not in eqtl_df.columns:
            eqtl_df["Iteration"] = 1

        # Removing rows with a too high iteration.
        eqtl_df = eqtl_df.loc[eqtl_df["Iteration"] <= self.n_iterations, :]

        # Calculating the number of samples and datasets per SNP.
        snps_df["N datasets"] = snps_df.sum(axis=1).to_frame()
        snps_df["N samples"] = np.dot(snps_df.loc[:, self.dataset_order].to_numpy(), self.dataset_sizes)
        snps_df.index.name = None
        print(snps_df)
        present_snps = set(snps_df.index)

        print("Parsing eQTL file.")
        mask = np.zeros(eqtl_df.shape[0], dtype=bool)
        found_genes = set()
        for i, (_, row) in enumerate(eqtl_df.iterrows()):
            if (i == 0) or (i % 1000000 == 0):
                print("\tprocessed {} lines".format(i))

            if "," not in row["ProbeName"] and row["ProbeName"] not in found_genes and row["SNPName"] in present_snps:
                mask[i] = True
                found_genes.add(row["ProbeName"])

        top_eqtl_df = eqtl_df.loc[mask, :]
        top_eqtl_df = top_eqtl_df.merge(snps_df, left_on="SNPName", right_index=True, how="left")
        top_eqtl_df["index"] = top_eqtl_df.index

        # Filter on having enough datasets.
        top_eqtl_df = top_eqtl_df.loc[top_eqtl_df["N datasets"] >= 2, :]

        print("Saving file.")
        print(top_eqtl_df)
        self.save_file(df=top_eqtl_df, outpath=os.path.join(self.outdir, "{}eQTLProbesFDR0.05-ProbeLevel.txt.gz".format(self.prefix)), index=False)

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
        print("  > eQTL: {}".format(self.eqtl_path))
        print("  > SNPs: {}".format(self.snps_path))
        print("  > Prefix: {}".format(self.prefix))
        print("  > N-iterations: {}".format(self.n_iterations))
        print("  > N-datasets: {}".format(self.n_iterations))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
