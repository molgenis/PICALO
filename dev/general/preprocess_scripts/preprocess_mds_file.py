#!/usr/bin/env python3

"""
File:         preprocess_mds_file.py
Created:      2020/10/26
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

# Local application imports.

# Metadata
__program__ = "Preprocess MDS file"
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
./preprocess_mds_file.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.gte_path = getattr(arguments, 'gene_to_exression')
        self.output_prefix = getattr(arguments, 'output_prefix')

        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'preprocess_mds_file')
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
        parser.add_argument("-gte",
                            "--gene_to_exression",
                            type=str,
                            required=True,
                            help="The path to the GtE file.")
        parser.add_argument("-op",
                            "--output_prefix",
                            type=str,
                            required=False,
                            default="",
                            help="The path to the GtE file.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data file")
        columns = None
        lines = []
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                data = [x for x in line.rstrip().split(" ") if x != ""]

                if i == 0:
                    columns = data
                else:
                    lines.append(data)
        f.close()
        df = pd.DataFrame(lines, columns=columns)
        print(df)

        print("Loading GtE file")
        gte_df = pd.read_csv(self.gte_path, sep="\t", header=None, index_col=None)
        gte_dict = dict(zip(gte_df.iloc[:, 0], gte_df.iloc[:, 1]))

        print("Pre-process")
        df.set_index("IID", inplace=True)
        df.index.name = "-"
        df = df.loc[:, ["C1", "C2", "C3", "C4"]]

        print("Translating")
        df.index = [gte_dict[genotype_id] for genotype_id in df.index]
        print(df)

        print("Saving file")
        outpath = os.path.join(self.outdir, self.output_prefix + os.path.basename(self.data_path).split(".")[0] + ".txt.gz")
        df.to_csv(outpath, compression="gzip", sep="\t", header=True, index=True)
        print("\tSaved dataframe: {} "
                    "with shape: {}".format(os.path.basename(outpath),
                                            df.shape))

    def print_arguments(self):
        print("Arguments:")
        print("  > Data: {}".format(self.data_path))
        print("  > GtE: {}".format(self.gte_path))
        print("  > Output prefix: {}".format(self.output_prefix))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
