#!/usr/bin/env python3

"""
File:         create_replication_heatmap.py
Created:      2022/12/20
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
import re

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Create Replication Heatmap"
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
./create_replication_heatmap.py -h

./create_replication_heatmap.py \
    -w /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/bryois_pic_replication
    
./create_replication_heatmap.py \
    -w /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/postprocess_scripts/klein_ct_eqtl_replication/PICALODiscovery
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.work_dir = getattr(arguments, 'work_dir')
        self.extensions = getattr(arguments, 'extensions')

        self.cell_type_abbreviations = {
            "Astrocytes": "AST",
            "Astrocyte": "AST",
            "EndothelialCells": "END",
            "EndothelialCell": "END",
            "ExcitatoryNeurons": "EX",
            "Excitatory": "EX",
            "InhibitoryNeurons": "IN",
            "Inhibitory": "IN",
            "Microglia": "MIC",
            "Oligodendrocytes": "OLI",
            "Oligodendrocyte": "OLI",
            "OPCsCOPs": "OPC",
            "Pericytes": "PER",
            "OtherNeuron": "NEU",
        }

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

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
        parser.add_argument("-w",
                            "--work_dir",
                            type=str,
                            required=True,
                            help="The path to the working directory.")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        df = self.load_file(os.path.join(self.work_dir, "replication_stats.txt.gz"), header=0, index_col=0)
        print(df)

        data = {}
        for variable in df["variable"].unique():
            subset = pd.pivot_table(df.loc[(df["label"] == "discovery significant") & (df["variable"] == variable), :],
                                    values='value',
                                    index='cell type',
                                    columns='PIC')

            cell_types = list(subset.index)
            cell_types.sort()
            pics = list(subset.columns)
            pics.sort(key=self.natural_keys)
            subset = subset.loc[cell_types, pics].iloc[:, :5]
            subset.index = [self.cell_type_abbreviations[ct] for ct in cell_types]

            data[variable] = subset

        self.plot(n_df=data["N"],
                  stats_df=[("N", data["N"], None, None, 0, 0, ""),
                            ("concordance", data["concordance"], 0, 100, 50, 0, "%"),
                            ("Rb", data["Rb"], -1, 1, 0, 2, ""),
                            ("π1", data["pi1"], 0, 1, 0.5, 2, "")]
                  )

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
    def natural_keys(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def plot(self, n_df, stats_df):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)

        ncols = len(stats_df)

        sns.set(rc={'figure.figsize': (1 * sum([stats[1].shape[0] for stats in stats_df]) + 10, 1 * max([stats[1].shape[1] for stats in stats_df]) + 10)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=1,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none')
        sns.set(color_codes=True)

        for i, (label, df, vmin, vmax, center, precision, value_appendix) in enumerate(stats_df):
            ax = axes[i]

            annot_df = pd.DataFrame("", index=df.index, columns=df.columns)
            for index in df.index:
                for column in df.columns:
                    annot_df.loc[index, column] = "{:.{}f}{}".format(df.loc[index, column], precision, value_appendix)
                    # annot_df.loc[index, column] = "{:.{}f}{}\nn={:,.0f}".format(df.loc[index, column], precision, value_appendix, n_df.loc[index, column])

            sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                        square=True, annot=annot_df, fmt='', yticklabels=True if i == 0 else False,
                        cbar=False, annot_kws={"size": 25, "color": "#000000"},
                        ax=ax)

            plt.setp(ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=30, rotation=0))
            plt.setp(ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=30, rotation=45))

            ax.set_xlabel("", fontsize=14)
            ax.xaxis.set_label_position('top')

            ax.set_ylabel("", fontsize=14)
            ax.yaxis.set_label_position('right')

            ax.set_title(label,
                         fontsize=30,
                         color="#000000",
                         weight='bold')

        fig.suptitle("PIC ieQTL replication in single-nucleus eQTLs\nBryois et al. 2021",
                     fontsize=40,
                     color="#000000",
                     weight='bold')

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.work_dir, "replication_stats_heatmap.{}".format(extension)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Working directory: {}".format(self.work_dir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
