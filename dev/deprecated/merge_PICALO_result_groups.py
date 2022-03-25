#!/usr/bin/env python3

"""
File:         merge_PICALO_result_groups.py
Created:      2021/11/24
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
import subprocess
import os

# Third party imports.
import pandas as pd

# Local application imports.

"""
Syntax:
./merge_PICALO_result_groups.py -h
"""

# Metadata
__program__ = "Merge PICALO Job Results"
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


class main():
    def __init__(self):
        self.indir = "/groups/umcg-bios/tmp01/projects/PICALO/test_scripts/merge_PICALO_job_results"
        self.name = "2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics"
        self.groups = {"1": ["PC2"],
                       "2": ["PC2"],
                       "3": ["PC1", "PC4", "PC8"],
                       "4": ["PC1"],
                       "5": ["PC1", "PC5"],
                       "6": ["PC1", "PC6", "PC11", "PC18"],
                       "7": ["PC1", "PC5"],
                       "8": ["PC1", "PC16"],
                       "9": ["PC1"],
                       "10": ["PC1"],
                       "11": ["PC1", "PC6", "PC8"],
                       "12": ["PC1"],
                       "13": ["PC1", "PC5"],
                       "14": ["PC1"],
                       "15": ["PC1", "PC8"]
                       }
        self.selected = {"1": "PC2",
                         "2": "PC2",
                         "3": "PC4",
                         "4": "PC1",
                         "5": "PC5",
                         "6": "PC1",
                         "7": "PC1",
                         "8": "PC1",
                         "9": "PC1",
                         "10": "PC1",
                         "11": "PC1",
                         "12": "PC1",
                         "13": "PC5",
                         "14": "PC1",
                         "15": "PC1"
                         }

        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'merge_PICALO_result_groups')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        df_list = []
        for pic, columns in self.groups.items():
            fpath = os.path.join(self.indir, self.name + "-PIC" + pic, "data", "PICBasedOnPCX.txt.gz")
            df = self.load_file(fpath, header=0, index_col=0)
            subset = df.loc[:, columns]

            new_colnames = []
            for i, colname in enumerate(subset.columns):
                appendix = ""
                if self.selected[pic] == colname:
                    appendix= "-X"
                new_colnames.append("PIC{}-{}-Group{}{}".format(pic, colname, i, appendix))
            subset.columns = new_colnames
            df_list.append(subset)
        df = pd.concat(df_list, axis=1)
        print(df)

        outpath = os.path.join(self.outdir, self.name + ".txt.gz")
        self.save_file(df=df.T, outpath=outpath)

        print("Plotting")
        # Plot correlation_heatmap of components.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', outpath, "-rn", "PIC_groups", "-o", "2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PICGroups"]
        self.run_command(command)

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

    @staticmethod
    def run_command(command):
        print(" ".join(command))
        subprocess.call(command)

if __name__ == '__main__':
    m = main()
    m.start()
