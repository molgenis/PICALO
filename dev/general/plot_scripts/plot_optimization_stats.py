#!/usr/bin/env python3

"""
File:         plot_optimization_stats.py
Created:      2022/07/21
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
import glob
import argparse
import re
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Plot Optimization Stats"
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
./plot_optimization_stats.py -h

### BIOS ###

./plot_optimization_stats.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

### MetaBrain ###

./count_n_ieqtls.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        components = getattr(arguments, 'components')
        out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extensions')

        if components is None:
            components = []
            for i in range(1, 50):
                component = "PIC{}".format(i)
                if os.path.exists(os.path.join(self.indir, component)):
                    components.append(component)
        self.components = components

        # Set variables.
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.data_outdir = os.path.join(base_dir, 'plot_optimization_stats', out_filename, "data")
        self.plot_outdir = os.path.join(base_dir, 'plot_optimization_stats', out_filename, "plot")
        for outdir in [self.data_outdir, self.plot_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        self.palette = {
            "not signif": "#808080",
            "before signif": "#0072B2",
            "after signif": "#D55E00",
            "both signif": "#009E73"
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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to input directory.")
        parser.add_argument("-c",
                            "--components",
                            nargs="*",
                            type=str,
                            required=False,
                            help="The components to plot.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")
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

        print("Parsing components")
        for comp_index, component in enumerate(self.components):
            if comp_index > 9:
                break

            print("  Processing {}".format(component))
            iterations_files = glob.glob(os.path.join(self.indir, component, "results_iteration*.txt.gz"))
            iterations_files.sort(key=self.natural_keys)

            print("\tLoading data")
            first_df = self.load_file(iterations_files[0], header=0, index_col=None)
            first_df.index = first_df["SNP"] + "_" + first_df["gene"]

            last_df = self.load_file(iterations_files[-1], header=0, index_col=None)
            last_df.index = last_df["SNP"] + "_" + last_df["gene"]
            last_df.drop(["SNP", "gene", "covariate", "N"], axis=1, inplace=True)

            first_df.columns = ["{} before".format(col) if col not in ["SNP", "gene", "covariate", "N"] else col for col in first_df.columns]
            last_df.columns = ["{} after".format(col) for col in last_df.columns]

            print("\tAdding z-scores")
            for df, appendix in ((first_df, "before"), (last_df, "after")):
                p_values = np.copy(df["p-value {}".format(appendix)].to_numpy())
                p_values = p_values / 2
                p_values[p_values > (1 - 1e-16)] = (1 - 1e-16)
                p_values[p_values < 2.4703282292062328e-324] = 2.4703282292062328e-324
                mask = np.ones_like(p_values)
                mask[df["beta-interaction {}".format(appendix)] > 0] = -1
                df["zscore-interaction {}".format(appendix)] = stats.norm.ppf(p_values) * mask

            print("\tMerging file")
            df = first_df.merge(last_df, left_index=True, right_index=True)

            print("\tSaving file")
            self.save_file(df=df, outpath=os.path.join(self.data_outdir, "{}.txt.gz".format(component)))

            print("\tFilter significant")
            group_map = {}
            for index, row in df.iterrows():
                if (row["FDR before"] <= 0.05) & (row["FDR after"] > 0.05):
                    group_map[index] = "before signif"
                elif (row["FDR before"] > 0.05) & (row["FDR after"] <= 0.05):
                    group_map[index] = "after signif"
                elif (row["FDR before"] <= 0.05) & (row["FDR after"] <= 0.05):
                    group_map[index] = "both signif"
                else:
                    group_map[index] = "not signif"
            df["hue"] = df.index.map(group_map)
            df = df.loc[df["hue"] != "not signif"]

            print("\tPlotting")
            sns.set_style("ticks")
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(48, 48))
            sns.set(color_codes=True)

            for row_index, term in enumerate(["interaction", "covariate", "genotype", "intercept"]):
                self.scatterplot(fig=fig,
                                 ax=axes[row_index, 0],
                                 x="beta-{} before".format(term),
                                 y="beta-{} after".format(term),
                                 df=df,
                                 hue="hue",
                                 palette=self.palette,
                                 ylabel=term,
                                 title="Beta" if row_index == 0 else "")

                self.scatterplot(fig=fig,
                                 ax=axes[row_index, 1],
                                 x="std-{} before".format(term),
                                 y="std-{} after".format(term),
                                 df=df,
                                 hue="hue",
                                 palette=self.palette,
                                 title="Standard error" if row_index == 0 else "")

                if term == "interaction":
                    self.scatterplot(fig=fig,
                                     ax=axes[row_index, 2],
                                     x="zscore-{} before".format(term),
                                     y="zscore-{} after".format(term),
                                     df=df,
                                     hue="hue",
                                     palette=self.palette,
                                     title="Z-score")
                else:
                    axes[row_index, 2].set_axis_off()

                annotation_ax = axes[row_index, 3]
                annotation_ax.set_axis_off()

                if row_index == 3:
                    group_counts = df["hue"].value_counts()
                    for i, (index, value) in enumerate(group_counts.iteritems()):
                        annotation_ax.annotate(
                            '{} = {:,}'.format(index, value),
                            xy=(0.03, 0.94 - (i * 0.08)),
                            xycoords=annotation_ax.transAxes,
                            color=self.palette[index],
                            fontsize=40,
                            fontweight='bold')

            fig.suptitle(component,
                         fontsize=80,
                         fontweight='bold')

            for extension in self.extensions:
                fig.savefig(os.path.join(self.plot_outdir, "{}.{}".format(component, extension)))
            plt.close()

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
    def scatterplot(fig, ax, df, x="x", y="y", hue=None, palette=None,
                    xlabel="", ylabel="", title=""):
        sns.despine(fig=fig, ax=ax)

        group_column = hue
        if hue is None:
            df["hue"] = "#000000"
            group_column = "hue"

        for i, hue_group in enumerate(df[group_column].unique()):
            subset = df.loc[df[group_column] == hue_group, :].copy()
            if subset.shape[0] < 1:
                continue

            sns.scatterplot(x=x,
                            y=y,
                            data=subset,
                            color=palette[hue_group],
                            linewidth=0,
                            legend=False,
                            ax=ax)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xlim[0] < 0:
            ax.axvline(0, ls='--', color="#000000", alpha=0.15, zorder=-1)
        if ylim[0] < 0:
            ax.axhline(0, ls='--', color="#000000", alpha=0.15, zorder=-1)
        ax.axline((0, 0), slope=1, ls='--', color="#000000", alpha=0.15, zorder=-1)

        ax.set_xlabel(xlabel,
                      fontsize=30,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=30,
                      fontweight='bold')
        ax.set_title(title,
                     fontsize=40,
                     fontweight='bold')

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Components: {}".format(", ".join(self.components)))
        print("  > Data outpath {}".format(self.data_outdir))
        print("  > Plot outpath {}".format(self.plot_outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
