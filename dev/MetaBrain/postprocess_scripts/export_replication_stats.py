#!/usr/bin/env python3

"""
File:         export_replication_stats.py
Created:      2022/06/18
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
import re
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Export Replication Stats"
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
./export_replication_stats.py -h
"""

class main():
    def __init__(self):
        # Define the input paths.
        self.replication_stats_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/afr_pic_replication/replication_stats.txt.gz"
        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'export_replication_stats')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, "replication_stats.xlsx")

    def start(self):
        with pd.ExcelWriter(self.outpath) as writer:
            df = self.load_file(self.replication_stats_path, header=0, index_col=0)
            replication_stats = pd.pivot_table(df.loc[
                                               df["label"] == "discovery significant", :],
                                               values='value',
                                               index='col',
                                               columns='variable')
            replication_stats = replication_stats[["N", "pearsonr", "concordance", "Rb", "pi1"]]

            order = []
            for i in range(1, 50):
                index = "PIC{}".format(i)
                if index in replication_stats.index:
                    order.append(index)
            replication_stats = replication_stats.loc[order, :]
            print(replication_stats)

            replication_stats.to_excel(writer, sheet_name="brain replication stats", na_rep="NA", index=False)
            print("Saving sheet 'brain replication stats' with shape {}".format(df.shape))

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    @staticmethod
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    m = main()
    m.start()
