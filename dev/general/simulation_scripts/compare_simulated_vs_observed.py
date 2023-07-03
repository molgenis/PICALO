#!/usr/bin/env python3

"""
File:         compare_simulated_vs_observed.py
Created:      2023/06/29
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
from colour import Color
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Local application imports.

# Metadata
__program__ = "Compare Simulated vs Observed"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@umcg.nl"
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
./compare_simulated_vs_observed.py -h

./compare_simulated_vs_observed.py \
    -si /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised/simulation1/ \
    -ob /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised-Simulation1 \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-OneCovariate 
    
./compare_simulated_vs_observed.py \
    -si /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised/simulation1/ \
    -ob /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised-Simulation1-FilteredeQTLs \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-OneCovariate-FilteredeQTLs
    
./compare_simulated_vs_observed.py \
    -si /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised/simulation1/ \
    -ob /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-firstExprPCForceNormalised \
    -picalo \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-OneCovariate-PICALO
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.simulation_path = getattr(arguments, 'simulated')
        self.observation_path = getattr(arguments, 'observed')
        self.picalo = getattr(arguments, 'picalo')
        self.extensions = getattr(arguments, 'extension')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'compare_simulated_vs_observed')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit")
        parser.add_argument("-si",
                            "--simulated",
                            type=str,
                            required=True,
                            help="The path to the simulation folder.")
        parser.add_argument("-ob",
                            "--observed",
                            type=str,
                            required=True,
                            help="The path to the observation folder.")
        parser.add_argument("-picalo",
                            action='store_true',
                            help="Observed result is from PICALO. "
                                 "Default: False.")
        parser.add_argument("-ex",
                            "--extension",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")
        parser.add_argument("-of",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        nrows = None

        print("Loading data.")
        simulated_beta_df = self.load_file(inpath=os.path.join(self.simulation_path, "model_betas.txt.gz"), header=0, index_col=0, nrows=nrows)
        simulated_std_df = self.load_file(inpath=os.path.join(self.simulation_path, "model_std.txt.gz"), header=0, index_col=0, nrows=nrows)
        print(simulated_beta_df)
        print(simulated_beta_df.columns.tolist())
        print(simulated_std_df)
        print(simulated_std_df.columns.tolist())

        simulated_rsquared_df = self.load_file(inpath=os.path.join(self.simulation_path, "model_rsquared.txt.gz"), header=0, index_col=0, nrows=nrows)
        simulated_rsquared_df.columns = ["r-squared"]
        print(simulated_rsquared_df)

        simulated_beta_df = simulated_beta_df.merge(simulated_rsquared_df, left_index=True, right_index=True)
        simulated_std_df["r-squared"] = 0

        observed_df = None
        genotype_stats_df = None
        if self.picalo:
            observed_df = self.load_file(inpath=os.path.join(self.observation_path, "PIC_interactions", "PIC1.txt.gz"), header=0, index_col=None, nrows=nrows)
            for prefix in ["beta", "std"]:
                observed_df.columns = [col.replace("{}-covariate".format(prefix), "{}-hidden_covariate0".format(prefix)).replace("{}-interaction".format(prefix), "{}-hidden_covariate_interaction0".format(prefix)) for col in observed_df.columns]

            # genotype_stats_df = self.load_file(inpath=os.path.join(self.observation_path, "genotype_stats.txt.gz"), header=0, index_col=0, nrows=nrows)
            # genotype_stats_df.index = observed_df.index
            # # print(genotype_stats_df)
        else:
            observed_df = self.load_file(inpath=os.path.join(self.observation_path, "eQTLSummaryStats.txt.gz"), header=0, index_col=None, nrows=nrows)
            for prefix in ["beta", "std"]:
                observed_df.columns = [col.replace("{}-hidden_covariate".format(prefix), "{}-hidden_covariate_interaction".format(prefix)).replace("Xgenotype", "") if (col.startswith("{}-hidden_covariate".format(prefix)) and col.endswith("Xgenotype")) else col for col in observed_df.columns]

            # genotype_stats_df = self.load_file(inpath=os.path.join(self.observation_path, "GenotypeStats.txt.gz"), header=0, index_col=0, nrows=nrows)
            # genotype_stats_df.index = observed_df.index
            # # print(genotype_stats_df)
        print(observed_df)

        observed_df.index = observed_df["gene"] + "_" + observed_df["SNP"]
        observed_beta_df = observed_df.loc[:, [col for col in observed_df.columns if col.startswith("beta-") or col == "r-squared"]]
        observed_beta_df.columns = [col.replace("beta-", "") for col in observed_beta_df.columns]

        observed_std_df = observed_df.loc[:, [col for col in observed_df.columns if col.startswith("std-")]]
        if not self.picalo:
            observed_std_df["r-squared"] = 0
        observed_std_df.columns = [col.replace("std-", "") for col in observed_std_df.columns]
        del observed_df

        print(observed_beta_df)
        print(observed_beta_df.columns.tolist())
        print(observed_std_df)
        print(observed_std_df.columns.tolist())

        overlapping_terms = [col for col in simulated_beta_df.columns if col in observed_beta_df.columns]
        print("Terms:")
        print(overlapping_terms)

        print("Plotting comparison")
        self.plot_model_comparison(simulated_beta_df=simulated_beta_df,
                                   simulated_std_df=simulated_std_df,
                                   observed_beta_df=observed_beta_df,
                                   observed_std_df=observed_std_df,
                                   overlapping_terms=overlapping_terms)

        # delta_beta_df = (simulated_beta_df - observed_beta_df).abs()
        # # print(delta_beta_df)
        #
        # print("Plotting difference vs genotype stats")
        # self.plot_delta_per_stats(delta_beta_df=delta_beta_df,
        #                           genotype_stats_df=genotype_stats_df)

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def plot_model_comparison(self, simulated_beta_df, simulated_std_df,
                              observed_beta_df, observed_std_df,
                              overlapping_terms):
        n_plots = len(overlapping_terms)
        if n_plots == 0:
            return
        ncols = math.ceil(np.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)

        print("Plotting")
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        row_index = 0
        col_index = 0
        for i in range(ncols * nrows):
            print(i)
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[col_index]
            elif ncols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            if i < len(overlapping_terms):
                term = overlapping_terms[i]
                compare_df = simulated_beta_df[[term]].merge(simulated_std_df[[term]], left_index=True, right_index=True).merge(observed_beta_df[[term]], left_index=True, right_index=True).merge(observed_std_df[[term]], left_index=True, right_index=True)
                compare_df.columns = ["simulated-beta", "simulated-std", "observed-beta", "observed-std"]

                self.plot(fig=fig,
                          ax=ax,
                          df=compare_df,
                          x="simulated",
                          xlabel="Simulated",
                          y="observed",
                          ylabel="Observed",
                          scatter_appendix="beta",
                          ci_appendix="std",
                          title=term)
            else:
                ax.set_axis_off()

            col_index += 1
            if col_index > (ncols - 1):
                col_index = 0
                row_index += 1

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_ModelComparison.{}".format(self.out_filename, extension)))

    def plot_delta_per_stats(self, delta_beta_df, genotype_stats_df):
        model_terms = delta_beta_df.columns.tolist()
        stats_terms = genotype_stats_df.columns.tolist()

        nrows = len(model_terms)
        ncols = len(stats_terms)

        n_plots = ncols * nrows
        if n_plots == 0:
            return

        print("Plotting")
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='none',
                                 sharey='none',
                                 figsize=(12 * ncols, 12 * nrows))
        sns.set(color_codes=True)

        for row_index, model_term in enumerate(model_terms):
            for col_index, stats_term in enumerate(stats_terms):
                print(row_index, col_index)
                ax = axes[row_index, col_index]

                compare_df = genotype_stats_df[[stats_term]].merge(delta_beta_df[[model_term]], left_index=True, right_index=True)

                self.plot(fig=fig,
                          ax=ax,
                          df=compare_df,
                          x=stats_term,
                          xlabel="",
                          y=model_term,
                          ylabel=model_term if col_index == 0 else "",
                          title=stats_term if row_index == 0 else "",
                          include_concordance=False,
                          lines=False)

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_DeltaVsGenotypeStats.{}".format(self.out_filename, extension)))

    def plot(self, fig, ax, df, x="x", y="y", xlabel="", ylabel="",
             scatter_appendix="", ci_appendix="", title="",
             color="#000000", ci=95, include_ylabel=True,
             include_concordance=True, lines=True):
        sns.despine(fig=fig, ax=ax)

        if not include_ylabel:
            ylabel = ""

        n = df.shape[0]
        coef = np.nan
        concordance = np.nan

        scatter_x = x
        if scatter_appendix != "":
            scatter_x += "-" + scatter_appendix

        scatter_y = y
        if scatter_appendix != "":
            scatter_y += "-" + scatter_appendix

        ci_x = x
        if scatter_appendix != "":
            ci_x += "-" + ci_appendix

        ci_y = y
        if scatter_appendix != "":
            ci_y += "-" + ci_appendix

        if n > 0:
            if n > 1:
                coef, p = stats.pearsonr(df[scatter_x], df[scatter_y])

                if include_concordance:
                    upper_left_quadrant = df.loc[(df[scatter_x] < 0) & (df[scatter_y] > 0), :]
                    upper_right_quadrant = df.loc[(df[scatter_x] > 0) & (df[scatter_y] > 0), :]
                    lower_left_quadrant = df.loc[(df[scatter_x] < 0) & (df[scatter_y] < 0), :]
                    lower_right_quadrant = df.loc[(df[scatter_x] > 0) & (df[scatter_y] < 0), :]

                    if coef < 0:
                        n_concordant = upper_left_quadrant.shape[0] + lower_right_quadrant.shape[0]
                        concordance = (100 / n) * n_concordant
                    elif coef > 0:
                        n_concordant = lower_left_quadrant.shape[0] + upper_right_quadrant.shape[0]
                        concordance = (100 / n) * n_concordant
                    else:
                        pass

            if scatter_appendix != "":
                ax.errorbar(df[scatter_x], df[scatter_y], df[ci_x], df[ci_y],
                            linestyle='None', marker='^', color="#808080",
                            alpha=0.2)

            sns.regplot(x=scatter_x, y=scatter_y, data=df, ci=ci,
                        scatter_kws={'facecolors': "#808080",
                                     'edgecolors': "#808080",
                                     'alpha': 0.60},
                        line_kws={"color": color},
                        ax=ax
                        )

        if lines:
            ax.axhline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
            ax.axvline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
            ax.axline((0, 0), slope=1 if coef >0 else -1, ls='--', color="#D7191C", alpha=0.3,
                      zorder=1)

        y_pos = 0.9
        if n > 0:
            ax.annotate(
                'N = {:,}'.format(n),
                xy=(0.03, 0.9),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(coef):
            ax.annotate(
                'r = {:.2f}'.format(coef),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if include_concordance and not np.isnan(concordance):
            ax.annotate(
                'concordance = {:.0f}%'.format(concordance),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        ax.set_title(title,
                     fontsize=22,
                     color=color,
                     weight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

    def print_arguments(self):
        print("Arguments:")
        print("  > Simulation path: {}".format(self.simulation_path))
        print("  > Observed path: {}".format(self.simulation_path))
        print("  > PICALO: {}".format(self.picalo))
        print("  > Extension(s): {}".format(", ".join(self.extensions)))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
