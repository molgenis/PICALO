#!/usr/bin/env python3

"""
File:         split_expression_matrix.py
Created:      2023/07/17
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Split Expression Matrix"
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
./split_expression_matrix.py -h

### BIOS ###
./split_expression_matrix.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.txt.gz \
    -gi /groups/umcg-biogen/tmp01/annotation/GeneReference/GencodeV32/gencode.v32.primary_assembly.annotation.collapsedGenes.ProbeAnnotation.TSS.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/filter_gte_file/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier/SampleToDataset.txt.gz \
    -of 2023-07-17-gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM

### MetaBrain ###
./split_expression_matrix.py \
    -d /groups/umcg-biogen/prm03/projects/2022-DeKleinEtAl/output/2020-01-31-expression-tables/2020-02-04-step5-center-scale/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.txt.gz \
    -gi /groups/umcg-biogen/tmp01/annotation/GeneReference/GencodeV32/gencode.v32.primary_assembly.annotation.collapsedGenes.ProbeAnnotation.TSS.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/filter_gte_file/MetaBrain_CortexEUR_NoENA/SampleToDataset.txt.gz \
    -of 2023-07-17-MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        outdir = getattr(arguments, 'outdir')
        self.outfile = getattr(arguments, 'outfile')

        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "split_expression_matrix")
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=False,
                            help="The path to the gene info matrix.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfile",
                            type=str,
                            required=False,
                            default="output",
                            help="The name of the output filename.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Processing data files.")
        gene_info_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        gene_chr_dict = dict(zip([x.split(".")[0] for x in gene_info_df["ArrayAddress"]], gene_info_df["Chr"]))
        std_df = self.load_file(self.std_path, header=0, index_col=None)
        expr_df = self.load_file(self.data_path, header=0, index_col=0)

        print("Splitting data files in odd and even chromosomes.")
        even_mask = np.zeros(expr_df.shape[0], dtype=bool)
        odd_mask = np.zeros(expr_df.shape[0], dtype=bool)
        for i, gene in enumerate(expr_df.index):
            if gene in gene_chr_dict:
                chr = gene_chr_dict[gene]
                try:
                    chr = int(chr)
                except ValueError:
                    continue

                if chr % 2 == 0:
                    even_mask[i] = True
                else:
                    odd_mask[i] = True
        print("\tFound {:,} genes on the even chromosomes [{:.2f}%].".format(np.sum(even_mask), (100 / expr_df.shape[0]) * np.sum(even_mask)))
        print("\tFound {:,} genes on the odd chromosomes [{:.2f}%].".format(np.sum(odd_mask), (100 / expr_df.shape[0]) * np.sum(odd_mask)))

        print("Filtering samples")
        samples = std_df.iloc[:, 0].tolist()
        sample_mask = np.zeros(expr_df.shape[1], dtype=bool)
        for i, sample in enumerate(expr_df.columns):
            if sample in samples:
                sample_mask[i] = True
        print("\tFound {:,} samples [{:.2f}%].".format(np.sum(sample_mask), (100 / expr_df.shape[1]) * np.sum(sample_mask)))

        print("")
        for mask, label in ((even_mask, "even"), (odd_mask, "odd")):
            print("Processing {}".format(label))
            self.save_file(df=expr_df.loc[mask, sample_mask], outpath=os.path.join(self.outdir, "{}.{}.txt.gz".format(self.outfile, label)))
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
        print("  > Data path: {}".format(self.data_path))
        print("  > Gene info path: {}".format(self.gene_info_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
