#!/usr/bin/env python3

"""
File:         plot_n_ieqtls_in_simulations.py
Created:      2023/07/15
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import glob
import json
import re
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy import optimize
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
from datetime import datetime

# Local application imports.

# Metadata
__program__ = "Plot Rho Simulations"
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
./plot_rho_simulations.py -h

### MetaBrain ###

### Bios ###

./plot_rho_simulations.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/ \
    -f 2023-07-15-BIOS-SimulationOf-2022-03-24 \
    -p /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.folder = getattr(arguments, 'folder')
        self.palette_path = getattr(arguments, 'palette')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), "plot_rho_simulations")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                palette = json.load(f)
            f.close()
            self.palette = {(int(key.replace("PIC", "")) - 1):value for key, value in palette.items() if key.startswith("PIC")}

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
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-f",
                            "--folder",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
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

        print("Loading data.")
        data = {}
        groups = []
        efficiency_data = []
        for fpath in glob.glob(os.path.join(self.indir, "output", "{}-*-AllRandomVectors-1PIC".format(self.folder))):
            folder = os.path.basename(fpath).replace("-AllRandomVectors-1PIC", "")
            n_covariates = int(folder.split("-")[-1].replace("first", "").replace("ExprPCForceNormalised", ""))
            if n_covariates != 3:
                continue

            before_ieqtl_inpath = os.path.join(fpath, "PIC1", "covariate_selection.txt.gz")
            context_inpath = os.path.join(self.indir, "simulate_expression2", folder, "simulated_covariates.txt.gz")
            if not os.path.exists(before_ieqtl_inpath) or not os.path.exists(context_inpath):
                continue

            # Load #ieQTLs before optimization.
            df = self.load_file(before_ieqtl_inpath, header=0, index_col=0)
            df.columns = ["ieQTLs before"]
            df["covariate"] = [int(x.split("_")[1]) for x in df.index]
            df["rho"] = [int(x.split("_")[2].replace("Rho", "")) / 10 for x in df.index]
            df.reset_index(drop=True, inplace=True)

            # Load context.
            context_df = self.load_file(context_inpath, header=0, index_col=0)
            # print(context_df)

            # Load after optimization.
            df["ieQTLs after"] = np.nan
            df["coef"] = np.nan
            df["time"] = np.nan
            for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                # if rho != 0.4:
                #     continue

                # Parse the log file.
                after_infolder = os.path.join(self.indir, "output", "{}-RandomVector-R{}".format(folder, str(rho).replace(".", "")))
                log_inpath = os.path.join(after_infolder, "log.log")
                job_logfile = os.path.join("/groups/umcg-bios/tmp01/projects/PICALO/run_PICALO_simulations/2023-07-15-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/{}Covariates/jobs/output/PICALO_SIMULATION_RUN_R{}_{}COVS.out".format(n_covariates, str(rho).replace(".", ""), n_covariates))
                if not os.path.exists(after_infolder) or not os.path.exists(log_inpath) or not os.path.exists(job_logfile):
                    continue
                time, n_iterations, n_pics = self.parse_log(log_inpath)
                mem = self.parse_joblog(job_logfile)
                efficiency_data.append([n_covariates, time, n_iterations, n_pics, mem])

                pics_inpath = os.path.join(after_infolder, "PICs.txt.gz")
                if not os.path.exists(pics_inpath):
                    continue

                # Load contexts after optimization.
                pics_df = self.load_file(pics_inpath, header=0, index_col=0)
                # print(pics_df)

                # Correlate contexts with PICs.
                corr_df = pd.concat([context_df, pics_df], axis=0).T.corr().iloc[n_covariates:, :n_covariates]
                # print(corr_df)

                # Check which context this most likely is.
                solution = optimize.linear_sum_assignment(1 - corr_df.abs())
                matches = {}
                for row_id, col_id in zip(*solution):
                    matches[row_id] = col_id
                # print(matches)

                for pic_index, context_index  in matches.items():
                    after_ieqtls_inpath = os.path.join(after_infolder, "PIC{}".format(pic_index + 1), "info.txt.gz")
                    if not os.path.exists(after_ieqtls_inpath):
                        continue

                    # Load #ieQTLs after optimization.
                    after_df = self.load_file(after_ieqtls_inpath, header=0, index_col=0)

                    # Save the #ieQTLs.
                    df.loc[(df["rho"] == rho) & (df["covariate"] == matches[pic_index]), "ieQTLs after"] = int(after_df.iloc[-1, :]["N"])
                    df.loc[(df["rho"] == rho) & (df["covariate"] == matches[pic_index]), "coef"] = corr_df.iloc[pic_index, context_index]

            if n_covariates == 1:
                print(df)

            total_ieQTLs = df.iloc[-1, ]["ieQTLs before"]
            df["%found before"] = df["ieQTLs before"] / total_ieQTLs
            df["%found after"] = df["ieQTLs after"]  / total_ieQTLs

            print("\tN covariates = {}".format(n_covariates))
            print(df)
            print("")

            # Loading simulated beta's.
            for fpath in glob.glob(os.path.join(self.indir, "fast_eqtl_mapper", "{}*".format(self.folder[:10]))):
                if fpath.endswith("first{}ExprPCForceNormalised".format(n_covariates)):
                    stats_df = self.load_file(os.path.join(fpath, "eQTLSummaryStats.txt.gz"), header=0, index_col=None)
                    stats_df = stats_df.loc[:, [col for col in stats_df.columns if col.endswith("Xgenotype")]]
                    stats_df.columns = [col.replace("Xgenotype", "") for col in stats_df.columns]
                    stats_df = stats_df.melt()
                    stats_df["covariate"] = [int(value.split("-")[1].replace("Comp", "")) - 1 for value in stats_df["variable"]]
                    stats_df["variable"] = [value.split("-")[0] for value in stats_df["variable"]]
                    print(stats_df)

                    # Plot.
                    # self.plot_covariate(df=df, stats_df=stats_df, filename="{}covariates".format(n_covariates))
                    self.plot_covariate_simple(df=df, filename="{}covariates_simple".format(n_covariates))

            # Save.
            data[n_covariates] = df
            groups.append(n_covariates)
        groups.sort()

        efficiency_df = pd.DataFrame(efficiency_data, columns=["#covariates", "time", "#iterations", "#PICs", "mem"])
        print(efficiency_df)
        efficiency_df["time per iter"] = efficiency_df["time"] / efficiency_df["#iterations"]
        efficiency_df["time per cov"] = efficiency_df["time"] / efficiency_df["#covariates"]
        efficiency_df.loc[efficiency_df["#iterations"] == 0, "time per iter"] = np.nan
        efficiency_plot_df = efficiency_df.groupby("#covariates").mean()
        efficiency_plot_df.reset_index(drop=False, inplace=True)
        efficiency_plot_df["time per iter per cov"] = efficiency_plot_df["time per iter"] / efficiency_plot_df["#covariates"]
        efficiency_plot_df["#iterations per pic"] = efficiency_plot_df["#iterations"] / efficiency_plot_df["#PICs"]
        print(efficiency_plot_df)
        print(efficiency_plot_df.mean(axis=0))

        # self.plot_combined_lineplot(
        #     data={"efficiency": efficiency_plot_df},
        #     groups=["efficiency"],
        #     x="#covariates",
        #     y="time per iter",
        #     xlabel="#covariates",
        #     ylabel="time (seconds) per EM round",
        #     title="Computational efficiency",
        #     subtitle_suffix="",
        #     filename="comp_efficiency"
        # )
        # exit()

        self.plot_combined_lineplot(
            data=data,
            groups=groups,
            x="rho",
            y="ieQTLs before",
            hue="covariate",
            palette=self.palette,
            xlabel="r",
            ylabel="# ieQTLs",
            title="Before optimization",
            subtitle_suffix=" covariates",
            filename="ieqtls_per_rho_before"
        )

        self.plot_combined_lineplot(
            data=data,
            groups=groups,
            x="rho",
            y="ieQTLs after",
            hue="covariate",
            palette=self.palette,
            xlabel="r",
            ylabel="# ieQTLs",
            title="After optimization",
            subtitle_suffix=" covariates",
            filename="ieqtls_per_rho_after"
        )

        self.plot_combined_lineplot(
            data=data,
            groups=groups,
            x="rho",
            y="coef",
            hue="covariate",
            palette=self.palette,
            xlabel="r",
            ylabel="r with context",
            title="Context reconstruction",
            subtitle_suffix=" covariates",
            filename="ieqtls_per_rho_corr"
        )

        self.plot_combined_lineplot(
            data=data,
            groups=groups,
            x="ieQTLs before",
            y="coef",
            hue="covariate",
            palette=self.palette,
            xlabel="# ieQTLs",
            ylabel="r with context",
            title="Context reconstruction",
            subtitle_suffix=" covariates",
            filename="ieqtls_per_ieqtl_before_corr"
        )

        self.plot_combined_lineplot(
            data=data,
            groups=groups,
            x="ieQTLs after",
            y="coef",
            hue="covariate",
            palette=self.palette,
            xlabel="# ieQTLs",
            ylabel="r with context",
            title="Context reconstruction",
            subtitle_suffix=" covariates",
            filename="ieqtls_per_ieqtl_after_corr"
        )

    def plot_covariate(self, df, stats_df, filename="plot"):

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=3,
                                 ncols=2,
                                 sharex="none",
                                 sharey="row",
                                 figsize=(24, 27))
        sns.set(color_codes=True)

        for row_index, col_index, x, y, xlabel, ylabel in (
                (0, 0, "rho", "ieQTLs before", "starting vector r", "# ieQTLs before optimization"),
                (0, 1, "rho", "ieQTLs after", "starting vector r", "# ieQTLs after optimization"),
                (1, 0, "ieQTLs before", "coef", "#ieQTLs before optimization", "r with context"),
                (1, 1, "ieQTLs after", "coef", "#ieQTLs after optimization", "r with context"),
                (2, 0, "rho", "coef", "starting vector r", "r with context")):
            ax = axes[row_index, col_index]
            if x is None or y is None:
                ax.set_axis_off()
                continue
            sns.despine(fig=fig, ax=ax)

            sns.lineplot(data=df,
                         x=x,
                         y=y,
                         markers=["o"] * len(df["covariate"].unique()),
                         hue="covariate",
                         palette=self.palette,
                         style="covariate",
                         ax=ax)

            ax.set_ylabel(ylabel,
                          fontsize=20,
                          fontweight='bold')
            ax.set_xlabel(xlabel,
                          fontsize=20,
                          fontweight='bold')

            # ax.set_xlim(0, 1)
            # if row_index == 0:
            #     ax.set_ylim(0, 3250)
            # elif row_index == 2 and col_index == 0:
            #     pass
            # else:
            #     ax.set_ylim(0, 1)

            ax.tick_params(axis='both', which='major', labelsize=14)

        ax = axes[2, 1]
        sns.despine(fig=fig, ax=ax)

        sns.violinplot(data=stats_df,
                       x="variable",
                       y="value",
                       hue="covariate",
                       palette=self.palette,
                       ax=ax)

        ax.set_ylabel("",
                      fontsize=20,
                      fontweight='bold')
        ax.set_xlabel("",
                      fontsize=20,
                      fontweight='bold')

        ax.tick_params(axis='both', which='major', labelsize=14)



        grid = plt.GridSpec(3, 2)
        self.create_subtitle(fig, grid[0, ::], "Effect of initial guess on #ieQTLs")
        self.create_subtitle(fig, grid[1, ::], "Effect of #ieQTLs on correlation with real context")
        self.create_subtitle(fig, grid[2, ::], "Effect of initial guess on correlation with real context")

        plt.tight_layout()
        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_{}.{}".format(self.folder, filename, extension)))
        plt.close()

    def plot_covariate_simple(self, df, filename):
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.set(color_codes=True)

        sns.despine(fig=fig, ax=ax)

        sns.lineplot(data=df,
                     x="rho",
                     y="ieQTLs after",
                     hue="covariate",
                     palette=self.palette,
                     style="covariate",
                     ax=ax)

        ax.set_ylim(0, 3250)
        ax.set_title("", fontsize=20, fontweight='bold')
        ax.set_ylabel("detected number of ieQTLs", fontsize=15, fontweight='bold')
        ax.set_xlabel("correlation of start vector with actual context", fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)

        for covariate in df["covariate"].unique():
            max_value = df.loc[(df["covariate"] == covariate) & (df["rho"] == 1.0), "ieQTLs after"].values[0]
            ax.axhline(max_value, ls='--', color=self.palette[covariate], zorder=-1)


        fig.suptitle("",
                     fontsize=40,
                     fontweight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_{}.{}".format(self.folder, filename, extension)))
        plt.close()

    @staticmethod
    def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f'{title}\n', fontsize=25, fontweight='bold')
        # hide subplot
        row.set_frame_on(False)
        row.axis('off')

    def plot_combined_lineplot(self, data, groups, x="x", y="y", hue=None, palette=None,
                               xlabel="x", ylabel="y", title="", subtitle_suffix="",
                               filename="lineplot"):
        ngroups = len(groups)
        ncols = int(np.ceil(np.sqrt((ngroups))))
        nrows = int(np.ceil((ngroups) / ncols))

        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex="none",
                                 sharey="none",
                                 figsize=(12 * ncols, 9 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1 and ncols > 1:
                ax = axes[col_index]
            elif nrows > 1 and ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(groups):
                sns.despine(fig=fig, ax=ax)

                sns.lineplot(data=data[groups[i]],
                             x=x,
                             y=y,
#                             markers=["o"] * len(data[groups[i]][hue].unique()),
                             hue=hue,
                             palette=palette,
                             style=hue,
                             ax=ax)

                ax.set_title("{}{}".format(groups[i], subtitle_suffix),
                             fontsize=25,
                             fontweight='bold')
                ax.set_ylabel(ylabel,
                              fontsize=20,
                              fontweight='bold')
                ax.set_xlabel(xlabel,
                              fontsize=20,
                              fontweight='bold')

                ax.tick_params(axis='both', which='major', labelsize=14)
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        fig.suptitle(title,
                     fontsize=40,
                     fontweight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_{}.{}".format(self.folder, filename, extension)))
        plt.close()

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
    def parse_log(inpath):
        first_line = None
        last_line = None
        n_iterations = 0
        n_pics = 0
        with open(inpath, 'r') as f:
            for line in f:
                if first_line is None:
                    first_line = line
                if "Finding ieQTLs" in line:
                    n_iterations += 1
                if "Identifying PIC" in line:
                    n_pics += 1
                last_line = line
        f.close()
        start_time =  datetime.strptime(re.search("(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})", first_line).group(0), "%Y-%m-%d %H:%M:%S")
        end_time =  datetime.strptime(re.search("(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})", last_line).group(0), "%Y-%m-%d %H:%M:%S")
        return (end_time - start_time).total_seconds(), n_iterations, n_pics

    @staticmethod
    def parse_joblog(inpath):
        second_last_line = None
        last_line = None
        with open(inpath, 'r') as f:
            for line in f:
                second_last_line = last_line
                last_line = line
        f.close()
        mem = [x for x in second_last_line.split(" ") if x != ""]
        if len(mem) != 9:
            return np.nan
        mem = mem[5]
        if "K" in mem:
            mem = int(mem.replace("K", "")) * 1000
        return mem / 1e9

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("  > Folder: {}".format(self.folder))
        print("  > Palette: {}".format(self.palette_path))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
