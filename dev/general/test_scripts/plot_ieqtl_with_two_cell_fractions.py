#!/usr/bin/env python3

"""
File:         plot_ieqtl_with_two_cell_fractions.py
Created:      2020/11/24
Last Changed: 2022/02/10
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
from colour import Color
import argparse
import os

# Third party imports.
import pandas as pd
import seaborn as sns
import matplotlib
from statsmodels.stats import multitest
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Local application imports.

# Metadata
__program__ = "Plot ieQTL with 2 Cell Fractions"
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
./plot_ieqtl_with_two_cell_fractions.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.geno_path = getattr(arguments, 'genotype')
        self.alleles_path = getattr(arguments, 'alleles')
        self.expr_path = getattr(arguments, 'expression')
        self.cf_path = getattr(arguments, 'cell_fractions')
        self.ocf_path = getattr(arguments, 'optimized_cell_fractions')
        self.decon_path = getattr(arguments, 'decon')
        self.alpha = getattr(arguments, 'alpha')
        self.interest = getattr(arguments, 'interest')
        self.nrows = getattr(arguments, 'nrows')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'plot')

        self.colormap = {
            "minor": "#E69F00",
            "center": "#0072B2",
            "major": "#D55E00",
            "Neuron": "#0072B2",
            "Oligodendrocyte": "#009E73",
            "EndothelialCell": "#CC79A7",
            "Microglia": "#E69F00",
            "Macrophage": "#E69F00",
            "Astrocyte": "#D55E00"
        }

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Create color map.
        self.group_color_map, self.value_color_map = self.create_color_map(self.colormap)

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
                            help="show program's version number and exit")
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the eqtl matrix")
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")
        parser.add_argument("-al",
                            "--alleles",
                            type=str,
                            required=True,
                            help="The path to the alleles matrix")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-cf",
                            "--cell_fractions",
                            type=str,
                            required=True,
                            help="The path to the original cell_fractions "
                                 "matrix")
        parser.add_argument("-ocf",
                            "--optimized_cell_fractions",
                            type=str,
                            required=True,
                            help="The path to the optimized cell_fractions "
                                 "matrix")
        parser.add_argument("-d",
                            "--decon",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-a",
                            "--alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The significance cut-off. Default: 0.05.")
        parser.add_argument("-i",
                            "--interest",
                            nargs="+",
                            type=str,
                            required=False,
                            default=None,
                            help="The HGNCSymbols to plot. Default: none.")
        parser.add_argument("-n",
                            "--nrows",
                            type=int,
                            required=False,
                            default=None,
                            help="Cap the number of runs to load. "
                                 "Default: None.")
        parser.add_argument("-e",
                            "--extension",
                            nargs="+",
                            type=str,
                            choices=["png", "pdf", "eps"],
                            default=["png"],
                            help="The figure file extension. "
                                 "Default: 'png'.")

        return parser.parse_args()

    @staticmethod
    def create_color_map(colormap):
        major_small = list(Color(colormap["major"]).range_to(Color("#FFFFFF"), 12))[2]
        center_small = list(Color(colormap["center"]).range_to(Color("#FFFFFF"), 12))[2]

        palette = list(Color(colormap["major"]).range_to(Color(major_small), 50)) + \
                  list(Color(major_small).range_to(Color(colormap["center"]), 50)) + \
                  list(Color(colormap["center"]).range_to(Color(center_small), 50)) + \
                  list(Color(center_small).range_to(Color(colormap["minor"]), 51))
        colors = [str(x).upper() for x in palette]
        values = [x / 100 for x in list(range(201))]
        group_color_map = {0.0: colormap["major"], 1.0: colormap["center"], 2.0: colormap["minor"]}
        value_color_map = {}
        for val, col in zip(values, colors):
            value_color_map[val] = col
        return group_color_map, value_color_map

    def start(self):
        self.print_arguments()
        eqtl_df, geno_df, alleles_df, expr_df, cf_df, ocf_df, decon_df = self.load()
        _, decon_fdr_df = self.bh_correct(decon_df)
        self.plot_eqtls(eqtl_df, geno_df, alleles_df, expr_df, cf_df, ocf_df, decon_fdr_df)

    def load(self):
        print("Loading input files.")
        eqtl_df = self.load_file(self.eqtl_path, index_col=None, nrows=self.nrows)
        geno_df = self.load_file(self.geno_path, nrows=self.nrows)
        alleles_df = self.load_file(self.alleles_path, nrows=self.nrows)
        expr_df = self.load_file(self.expr_path, nrows=self.nrows)
        cf_df = self.load_file(self.cf_path)
        ocf_df = self.load_file(self.ocf_path)

        decon_df = self.load_file(self.decon_path)

        return eqtl_df, geno_df, alleles_df, expr_df, cf_df, ocf_df, decon_df

    @staticmethod
    def bh_correct(pvalue_df):
        df = pvalue_df.copy()
        pval_data = []
        fdr_data = []
        indices = []
        for col in df.columns:
            if col.endswith("_pvalue"):
                pval_data.append(df.loc[:, col])
                fdr_data.append(multitest.multipletests(df.loc[:, col], method='fdr_bh')[1])
                indices.append(col.replace("_pvalue", ""))
        pval_df = pd.DataFrame(pval_data, index=indices, columns=df.index)
        fdr_df = pd.DataFrame(fdr_data, index=indices, columns=df.index)

        return pval_df.T, fdr_df.T

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def plot_eqtls(self, eqtl_df, geno_df, alleles_df, expr_df, cf_df, ocf_df,
                   decon_df):
        print("Plotting interaction eQTL plots.")

        print("Iterating over eQTLs.")
        for i, (index, row) in enumerate(eqtl_df.iterrows()):
            # Extract the usefull information from the row.
            fdr_value = row["FDR"]
            snp_name = row["SNPName"]
            probe_name = row["ProbeName"]
            hgnc_name = row["HGNCName"]
            eqtl_type = row["CisTrans"]

            # Check if the SNP has an interaction effect.
            interaction_effect = decon_df.loc["{}_{}".format(probe_name, snp_name), :]
            if not interaction_effect.name.startswith(probe_name) or not interaction_effect.name.endswith(snp_name):
                print("\t\tCannot find probe-snp combination in decon results.")
                exit()
            interaction_effect = interaction_effect.to_frame()
            interaction_effect.columns = ["FDR"]
            interaction_effect = interaction_effect.loc[
                                 interaction_effect["FDR"] <= self.alpha, :]
            interaction_effect = interaction_effect.reindex(
                interaction_effect["FDR"].abs().sort_values(
                    ascending=True).index)

            if (self.interest is not None and hgnc_name not in self.interest) or (interaction_effect.shape[0] == 0):
                continue

            print("\tWorking on: {}\t{}\t{} [{}/{} "
                  "{:.2f}%]".format(snp_name, probe_name, hgnc_name,
                                    i + 1,
                                    eqtl_df.shape[0],
                                    (100 / eqtl_df.shape[0]) * (i + 1)))

            # Get the genotype / expression data.
            genotype = geno_df.iloc[i, :].T.to_frame()
            if genotype.columns != [snp_name]:
                print("\t\tGenotype file not in identical order as eQTL file.")
                exit()
            expression = expr_df.iloc[i, :].T.to_frame()
            if expression.columns != [probe_name]:
                print("\t\tExpression file not in identical order as eQTL file.")
                exit()
            data = genotype.merge(expression, left_index=True, right_index=True)
            data.columns = ["genotype", "expression"]
            data["group"] = data["genotype"].round(0)

            # Remove missing values.
            data = data.loc[(data['genotype'] >= 0.0) &
                            (data['genotype'] <= 2.0), :]

            # Get the allele data.
            alleles = alleles_df.iloc[i, :]
            if alleles.name != snp_name:
                print("\t\tAlleles file not in identical order as eQTL file.")
                exit()
            # A/T = 0.0/2.0
            # by default we assume T = 2.0 to be minor
            major_allele = alleles["Alleles"].split("/")[0]
            minor_allele = alleles["Alleles"].split("/")[1]

            # Check if we need to flip the genotypes.
            counts = data["group"].value_counts()
            for x in [0.0, 1.0, 2.0]:
                if x not in counts:
                    counts.loc[x] = 0
            zero_geno_count = (counts[0.0] * 2) + counts[1.0]
            two_geno_count = (counts[2.0] * 2) + counts[1.0]
            if two_geno_count > zero_geno_count:
                # Turns out that 0.0 was the minor.
                minor_allele = alleles["Alleles"].split("/")[0]
                major_allele = alleles["Alleles"].split("/")[1]
                data["genotype"] = 2.0 - data["genotype"]
                data["group"] = 2.0 - data["group"]

            allele_map = {0.0: "{}/{}".format(major_allele, major_allele),
                          1.0: "{}/{}".format(major_allele, minor_allele),
                          2.0: "{}/{}".format(minor_allele, minor_allele)}
            data["alleles"] = data["group"].map(allele_map)

            # Determine the minor allele frequency.
            minor_allele_frequency = min(zero_geno_count, two_geno_count) / (
                        zero_geno_count + two_geno_count)

            # Add the color.
            data["round_geno"] = data["genotype"].round(2)
            data["value_hue"] = data["round_geno"].map(self.value_color_map)
            data["group_hue"] = data["group"].map(self.group_color_map)

            # Prepare output directory.
            if len(interaction_effect.index) > 0:
                eqtl_interaction_outdir = os.path.join(self.outdir,
                                                       "{}_{}_{}_{}".format(
                                                           index, snp_name,
                                                           probe_name,
                                                           hgnc_name))
                if not os.path.exists(eqtl_interaction_outdir):
                    os.makedirs(eqtl_interaction_outdir)

                for cov_name, (fdr,) in interaction_effect.iterrows():
                    print("\t\t{}: FDR = {}".format(cov_name, fdr))

                    # Initialize the plot.
                    sns.set(rc={'figure.figsize': (24, 9)})
                    sns.set_style("ticks")
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
                    for ax in [ax1, ax2, ax3]:
                        sns.despine(fig=fig, ax=ax)

                    # Plot original cell fractions interaction eQTL.
                    eqtl_data = data.copy()
                    cov_data = cf_df.loc[:, cov_name].to_frame()
                    eqtl_data = eqtl_data.merge(cov_data, left_index=True,
                                                right_index=True)

                    self.plot_inter_eqtl(df=eqtl_data,
                                         ax=ax1,
                                         group_color_map=self.group_color_map,
                                         allele_map=allele_map,
                                         x=cov_name,
                                         y="expression",
                                         xlabel="original {}".format(cov_name),
                                         ylabel="{} ({}) expression".format(probe_name, hgnc_name),
                                         title="original"
                                         )

                    # # Plot normalized cell fractions interaction eQTL.
                    # eqtl_data = data.copy()
                    # cov_data = self.force_normal_series(cf_df.loc[:, cov_name], as_series=True).to_frame()
                    # cov_data.columns = [cov_name]
                    # eqtl_data = eqtl_data.merge(cov_data, left_index=True,
                    #                             right_index=True)
                    #
                    # self.plot_inter_eqtl(df=eqtl_data,
                    #                      ax=ax2,
                    #                      group_color_map=self.group_color_map,
                    #                      allele_map=allele_map,
                    #                      x=cov_name,
                    #                      y="expression",
                    #                      xlabel="normalized {}".format(cov_name),
                    #                      ylabel="{} ({}) expression".format(probe_name, hgnc_name),
                    #                      title="normalized"
                    #                      )
                    #
                    # del eqtl_data, cov_data

                    # Plot optimized cell fractions interaction eQTL.
                    eqtl_data = data.copy()
                    cov_data = ocf_df.loc[:, cov_name].to_frame()
                    eqtl_data = eqtl_data.merge(cov_data, left_index=True,
                                                right_index=True)

                    self.plot_inter_eqtl(df=eqtl_data,
                                         ax=ax3,
                                         group_color_map=self.group_color_map,
                                         allele_map=allele_map,
                                         x=cov_name,
                                         y="expression",
                                         xlabel="optimized {}".format(cov_name),
                                         ylabel="{} ({}) expression".format(probe_name, hgnc_name),
                                         title="optimized"
                                         )

                    # Safe the plot.
                    plt.tight_layout()
                    for extension in self.extensions:
                        filename = "{}_inter_eqtl_{}_{}_{}_{}.{}".format(
                            i,
                            snp_name,
                            probe_name,
                            hgnc_name,
                            cov_name,
                            extension)
                        print("\t\tSaving plot: {}".format(filename))
                        fig.savefig(os.path.join(eqtl_interaction_outdir, filename), dpi=300)
                    plt.close()
            else:
                print("\t\tNo significant interactions.")

    @staticmethod
    def force_normal_series(s, as_series=False):
        normal_s = stats.norm.ppf((s.rank(ascending=True) - 0.5) / s.size)
        if as_series:
            return pd.Series(normal_s, index=s.index)
        else:
            return normal_s

    @staticmethod
    def plot_inter_eqtl(df, ax, group_color_map, allele_map, x="x", y="y", xlabel="", ylabel="", title=""):
        # Calculate R2.
        tmp_X = df[["genotype", x]].copy()
        tmp_X["genotype_x_{}".format(x)] = tmp_X["genotype"] * tmp_X[x]
        tmp_X["intercept"] = 1
        tmp_y = df[y].copy()

        reg = LinearRegression().fit(tmp_X, tmp_y)
        r2_score = reg.score(tmp_X, tmp_y)

        # calculate axis limits.
        ymin_value = df[y].min()
        ymin = ymin_value - abs(ymin_value * 0.2)
        ymax_value = df[y].max()
        ymax = ymax_value + abs(ymax_value * 0.6)

        xmin_value = df[x].min()
        xmin = xmin_value - max(abs(xmin_value * 0.1), 0.05)
        xmax_value = df[x].max()
        xmax = xmax_value + max(abs(xmax_value * 0.1), 0.05)

        label_pos = {0.0: 0.94, 1.0: 0.90, 2.0: 0.86}
        for i, genotype in enumerate([1.0, 0.0, 2.0]):
            subset = df.loc[df["round_geno"] == genotype, :].copy()
            color = group_color_map[genotype]
            allele = allele_map[genotype]

            coef_str = "NA"
            p_str = "NA"
            if len(subset.index) > 1:
                # Regression.
                coef, p = stats.spearmanr(subset[y], subset[x])
                coef_str = "{:.2f}".format(coef)
                p_str = "p = {:.2e}".format(p)

                # Plot.
                sns.regplot(x=x, y=y, data=subset,
                            scatter_kws={'facecolors': subset['value_hue'],
#                                         'edgecolors': subset['group_hue'],
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": color, "alpha": 0.75},
                            ax=ax
                            )

            # Add the text.
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.annotate(
                '{}: r = {} [{}]'.format(allele, coef_str, p_str),
                # xy=(0.03, 0.94 - ((i / 100) * 4)),
                xy=(0.03, label_pos[genotype]),
                xycoords=ax.transAxes,
                color=color,
                alpha=0.75,
                fontsize=12,
                fontweight='bold')

        ax.text(0.5, 1.06,
                title,
                fontsize=18, weight='bold', ha='center', va='bottom',
                transform=ax.transAxes)
        ax.text(0.5, 1.02,
                "R2: {:.2f}".format(r2_score),
                fontsize=12, alpha=0.75, ha='center', va='bottom',
                transform=ax.transAxes)

        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')
        print("check")

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL path: {}".format(self.eqtl_path))
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Cell fractions path: {}".format(self.cf_path))
        print("  > Optimized cell fractions path: {}".format(self.ocf_path))
        print("  > Deconvolution path: {}".format(self.decon_path))
        print("  > Alpha: {}".format(self.alpha))
        print("  > Interest: {}".format(self.interest))
        print("  > Nrows: {}".format(self.nrows))
        print("  > Extension: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
