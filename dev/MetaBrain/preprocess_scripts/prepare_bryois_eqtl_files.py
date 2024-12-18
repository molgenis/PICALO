#!/usr/bin/env python3

"""
File:         prepare_bryois_eqtl_files.py
Created:      2022/01/26
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Prepare Bryois eQTL files"
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
./prepare_bryois_eqtl_files.py -h 
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.bryois_path = getattr(arguments, 'bryois')
        self.min_avg_expression = getattr(arguments, 'min_avg_expression')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.expression_path = getattr(arguments, 'expression')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'prepare_bryois_eqtl_files')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        if self.min_avg_expression is not None:
            if self.std_path is None:
                print("Argument -std / --sample_to_dataset is required if -mae / --min_avg_expression is not None.")
                exit()
            if self.expression_path is None:
                print("Argument -ex / --expression is required if -mae / --min_avg_expression is not None.")
                exit()

        self.bryois_ct_trans = {
            'Astrocytes': 'Astrocytes',
            'Endothelial cells': 'EndothelialCells',
            'Excitatory neurons': 'ExcitatoryNeurons',
            'Inhibitory neurons': 'InhibitoryNeurons',
            'Microglia': 'Microglia',
            'OPCs / COPs': 'Oligodendrocytes',
            'Oligodendrocytes': 'OPCsCOPs',
            'Pericytes': 'Pericytes'
        }

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
                            "--bryois",
                            type=str,
                            required=True,
                            help="The path to the bryois eqtl matrix.")
        parser.add_argument("-mae",
                            "--min_avg_expression",
                            type=float,
                            default=None,
                            help="The minimal average expression of a gene."
                                 "Default: None.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            default=None,
                            help="The path to the expression matrix in TMM"
                                 "format.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        eqtl_df = self.load_file(self.bryois_path, header=0, index_col=None)
        print(eqtl_df)
        print(eqtl_df["cell_type"].value_counts())

        print("Loading expression data")
        expr_df = self.load_file(self.expression_path, header=0, index_col=0)
        ensembl_version_trans_dict = {key.split(".")[0]: key for key in expr_df.index}

        print("Constructing eQTL matrix.")
        eqtl_df["ProbeName"] = eqtl_df["ensembl"].map(ensembl_version_trans_dict)
        eqtl_df = eqtl_df[["cell_type", "bpval", "SNPName", "ProbeName", "symbol", "slope", "adj_p", "dist_to_TSS"]]
        eqtl_df.columns = ["cell_type", "PValue", "SNPName", "ProbeName", "HGNCName", "Beta", "FDR", "TSSDistance"]
        eqtl_df["ID"] = eqtl_df["SNPName"] + "_" + eqtl_df["ProbeName"]

        print("Removing non-unique eQTLs.")
        mask = []
        for id in eqtl_df["ID"]:
            if eqtl_df.loc[eqtl_df["ID"] == id, :].shape[0] == 1:
                mask.append(True)
            else:
                mask.append(False)
        eqtl_df = eqtl_df.loc[mask, :]
        print(eqtl_df["cell_type"].value_counts())

        # Filter on having high enough expression.
        file_appendix = ""
        if self.min_avg_expression is not None:
            std_df = self.load_file(self.std_path, header=0, index_col=None)
            samples = std_df.iloc[:, 0].values.tolist()

            present_genes = []
            missing_genes = []
            for gene in eqtl_df["ProbeName"]:
                if gene in expr_df.index:
                    present_genes.append(gene)
                else:
                    missing_genes.append(gene)
            if len(missing_genes) > 0:
                print("Warning: {} genes are not in the expression matrix: {}".format(len(missing_genes), ", ".join(missing_genes)))
                eqtl_df = eqtl_df.loc[eqtl_df["ProbeName"].isin(present_genes), :]
            del missing_genes

            print("Sample / gene selection.")
            expr_df = expr_df.loc[present_genes, samples]

            print("Log2 transform.")
            min_value = expr_df.min(axis=1).min()
            if min_value <= 0:
                expr_df = np.log2(expr_df - min_value + 1)
            else:
                expr_df = np.log2(expr_df + 1)

            print("Calculate average.")
            averages = expr_df.mean(axis=1)

            print("Remove eQTLs")
            eqtl_df = eqtl_df.loc[(averages > self.min_avg_expression).to_numpy(), :]

            file_appendix = "_GT{}AvgExprFilter".format(self.min_avg_expression)

        print("Saving eQTL files.")
        for ct in eqtl_df["cell_type"].unique():
            print("\tSaving {}".format(ct))
            ct_eqtl_df = eqtl_df.loc[eqtl_df["cell_type"] == ct, :].copy()
            ct_eqtl_df = ct_eqtl_df.iloc[:, 1:]
            print("\t  Shape: {}".format(ct_eqtl_df.shape))
            if ct_eqtl_df.shape[0] > 0:
                ct_eqtl_df.to_csv(os.path.join(self.outdir, "bryois-eQTLProbesFDR0.05-ProbeLevel{}-{}.txt.gz".format(file_appendix, self.bryois_ct_trans[ct])),
                                  sep="\t",
                                  header=True,
                                  index=False,
                                  compression="gzip")

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
        print("  > Bryois eQTL: {}".format(self.bryois_path))
        print("  > Minimal average expression: >{}".format(self.min_avg_expression))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Expression: {}".format(self.expression_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
