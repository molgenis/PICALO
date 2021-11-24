#!/usr/bin/env python3

"""
File:         merge_PICALO_job_results.py
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
from pathlib import Path
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

"""
Syntax:
./merge_PICALO_job_results.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/ -j 2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC1

./merge_PICALO_job_results.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/ -j 2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC2

./merge_PICALO_job_results.py -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/ -j 2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC3
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
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'input')
        self.job_name = getattr(arguments, 'job_name')
        output = getattr(arguments, 'output')
        if output is None:
            output = self.job_name
        self.data_outdir = os.path.join(Path().resolve(), 'merge_PICALO_job_results', output, "data")
        self.plot_outdir = os.path.join(Path().resolve(), 'merge_PICALO_job_results', output, "plot")
        for outdir in [self.data_outdir, self.plot_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

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
        parser.add_argument("-i",
                            "--input",
                            type=str,
                            required=True,
                            help="The path to the input folder.")
        parser.add_argument("-j",
                            "--job_name",
                            type=str,
                            required=True,
                            help="The name of the jobs. Replace the PC index"
                                 "with <N>")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output folder.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("loading PIC data")
        pic_df_list = []
        for pc_index in np.arange(1, 26):
            fpath = os.path.join(self.indir, self.job_name + "-PC{}AsCov".format(pc_index), "components.txt.gz")
            if os.path.exists(fpath):
                pic_df = self.load_file(inpath=fpath, header=0, index_col=0).T
                pic_df.columns = ["PC{}".format(pc_index)]
                pic_df_list.append(pic_df)
        pic_df = pd.concat(pic_df_list, axis=1)
        print(pic_df)

        print("Saving data.")
        self.save_file(df=pic_df, outpath=os.path.join(self.data_outdir, "PICBasedOnPCX.txt.gz"))
        del pic_df

        print("loading PIC info data")
        info_df_m_list = []
        for pc_index in np.arange(1, 26):
            fpath = os.path.join(self.indir, self.job_name + "-PC{}AsCov".format(pc_index), "PIC1", "info.txt.gz")
            print(fpath)
            if os.path.exists(fpath):
                info_df = self.load_file(inpath=fpath, header=0, index_col=0)
                info_df["index"] = np.arange(1, (info_df.shape[0] + 1))
                info_df["component"] = "PC{}".format(pc_index)
                info_df_m = info_df.melt(id_vars=["index", "covariate", "component"])
                info_df_m_list.append(info_df_m)

        if len(info_df_m_list) > 1:
            info_df_m = pd.concat(info_df_m_list, axis=0)
        else:
            info_df_m = info_df_m_list[0]
        info_df_m["log10 value"] = np.log10(info_df_m["value"])

        print("Plotting")
        for variable in info_df_m["variable"].unique():
            print("\t{}".format(variable))

            subset_m = info_df_m.loc[info_df_m["variable"] == variable, :]
            if variable == ["N Overlap", "Overlap %"]:
                subset_m = subset_m.loc[subset_m["index"] != 1, :]

            self.lineplot(df_m=subset_m, x="index", y="value", hue="component",
                          xlabel="iteration", ylabel=variable,
                          filename=variable.replace(" ", "_").lower() + "_lineplot",
                          outdir=self.plot_outdir)

            if "Likelihood" in variable:
                self.lineplot(df_m=subset_m, x="index", y="log10 value",
                              hue="component",
                              xlabel="iteration", ylabel="log10 " + variable,
                              filename=variable.replace(" ", "_").lower() + "_lineplot_log10",
                              outdir=self.plot_outdir)

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
    def lineplot(df_m, x="x", y="y", hue=None, title="", xlabel="", ylabel="",
                 filename="plot", outdir=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.lineplot(data=df_m,
                         x=x,
                         y=y,
                         units=hue,
                         hue=hue,
                         estimator=None,
                         legend="brief",
                         ax=ax)

        ax.set_title(title,
                     fontsize=14,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')

        plt.tight_layout()
        outpath = "{}.png".format(filename)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)
        fig.savefig(outpath)
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Input path: {}".format(self.indir))
        print("  > Job name: {}".format(self.job_name))
        print("  > Data output directory: {}".format(self.data_outdir))
        print("  > Plot output directory: {}".format(self.plot_outdir))


if __name__ == '__main__':
    m = main()
    m.start()
