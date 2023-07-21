#!/usr/bin/env python3

"""
File:         generate_starting_vectors.py
Created:      2021/07/12
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
import numpy as np
import pandas as pd
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Generate Starting Vectors"
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
./generate_starting_vectors.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.real_contexts_path = getattr(arguments, 'real_contexts')
        rho = getattr(arguments, 'rho')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        if rho is None:
            self.rho = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        else:
            self.rho = [rho]


        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "generate_starting_vectors", outfolder)
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
        parser.add_argument("-rc",
                            "--real_contexts",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-r",
                            "--rho",
                            type=float,
                            required=False,
                            default=None,
                            help="")
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfolder",
                            type=str,
                            required=False,
                            default="output",
                            help="The name of the output folder.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        contexts_df = self.load_file(self.real_contexts_path, header=0, index_col=0)
        n_contexts = contexts_df.shape[0]
        n_samples = contexts_df.shape[1]
        print(contexts_df)

        context_indices = contexts_df.index.tolist()
        vector_indices = ["Start{}".format(i) for i in range(n_contexts)]
        samples = contexts_df.columns.tolist()

        print("\tConvert to numpy.")
        contexts_m = contexts_df.to_numpy()

        combined_vectors_m = np.empty(shape=(n_contexts * len(self.rho), n_samples))
        combined_vector_indices = []
        for i, rho in enumerate(self.rho):
            vectors_m = np.empty(shape=(n_contexts, n_samples))
            for j in range(n_contexts):
                vector =  rho * contexts_m[j, :] + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, size=(n_samples,))
                vectors_m[j, :] = vector
                combined_vectors_m[(i * n_contexts) + j, :] = vector
                combined_vector_indices.append("{}_Rho{}".format(context_indices[j], str(rho).replace(".", "")))

            vectors_df = pd.DataFrame(vectors_m, index=vector_indices, columns=samples)
            print(vectors_df)
            self.save_file(df=vectors_df, outpath=os.path.join(self.outdir, "starting_vector_rho{}.txt.gz".format(str(rho).replace(".", ""))))

        vectors_df = pd.DataFrame(combined_vectors_m, index=combined_vector_indices, columns=samples)
        print(vectors_df)

        print("Correlating")
        corr_m = np.corrcoef(combined_vectors_m, contexts_m)[:combined_vectors_m.shape[0], combined_vectors_m.shape[0]:]
        corr_df = pd.DataFrame(corr_m, index=combined_vector_indices, columns=context_indices)
        print(corr_df)

        print("Save")
        self.save_file(df=vectors_df, outpath=os.path.join(self.outdir, "starting_vectors.txt.gz"))

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
    def correlate(m1, m2):
        return np.corrcoef(m1, m2)[:m1.shape[0], m1.shape[0]:]


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
        print("  > Real contexts: {}".format(self.real_contexts_path))
        print("  > Rho: {}".format(self.rho))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
