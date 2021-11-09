#!/usr/bin/env python3

"""
File:         visualise_interaction_eqtl.py
Created:      2020/11/09
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
import json
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import betainc

# Local application imports.

# Metadata
__program__ = "Visualise Interaction eQTL"
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
./visualise_interaction_eqtl.py -ge ../../preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/genotype_table.txt.gz -ex ../../preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/expression_table_CovCorrected.txt.gz -i interest.txt -n 6000
"""


class main():
    def __init__(self):
        arguments = self.create_argument_parser()
        self.geno_path = getattr(arguments, 'genotype')
        self.expr_path = getattr(arguments, 'expression')
        self.cov_path = getattr(arguments, 'covariates')
        self.alpha = getattr(arguments, 'alpha')
        self.interest_path = getattr(arguments, 'interest')
        self.nrows = getattr(arguments, 'nrows')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'visualise_interaction_eqtl')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Set the right pdf font for exporting.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Loading palette.
        self.palette = {0.0: "#E69F00", 1.0: "#0072B2", 2.0: "#D55E00"}

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
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the deconvolution matrix")
        parser.add_argument("-co",
                            "--covariates",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the covariates matrix. Default:"
                                 "same as -ex / --expression.")
        parser.add_argument("-a",
                            "--alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The significance cut-off. Default: 0.05.")
        parser.add_argument("-i",
                            "--interest",
                            type=str,
                            required=True,
                            help="The interaction eQTL to plot.")
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

    def start(self):
        self.print_arguments()

        print("Loading data.")
        geno_df = self.load_file(self.geno_path, header=0, index_col=0, nrows=self.nrows)
        geno_df = geno_df.groupby(geno_df.index).first()
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=self.nrows)
        expr_df = expr_df.groupby(expr_df.index).first()

        cov_df = expr_df
        if self.cov_path is not None:
            print("No -c /--covariates given. Using gene expression as "
                  "covariates.")
            cov_df = self.load_file(self.cov_path, header=0, index_col=0, nrows=self.nrows)

        interest_df = self.load_file(self.interest_path, sep=",", header=0, index_col=None)

        ########################################################################

        print("Checking loaded data.")
        if list(geno_df.columns) != list(expr_df.columns):
            print("Unequal input matrix.")
            exit()

        loaded_snps = set(geno_df.index)
        loaded_genes = set(expr_df.index)
        loaded_covs = set(cov_df.index)

        for _, (snp_name, probe_name, _, cov_name, _) in interest_df.iterrows():
            if snp_name not in loaded_snps:
                print("Cannot find SNP {}".format(snp_name))
                exit()

            if probe_name not in loaded_genes:
                print("Cannot find gene {}".format(probe_name))
                exit()

            if cov_name not in loaded_covs:
                print("Cannot find covariate {}.".format(cov_name))
                exit()

        ########################################################################

        X = np.ones((geno_df.shape[1], 4), dtype=np.float64)
        for _, (snp_name, probe_name, probe_label, cov_name, cov_label) in interest_df.iterrows():
            y = expr_df.loc[probe_name, :].to_numpy()

            X[:, 1] = geno_df.loc[snp_name, :].to_numpy()
            X[:, 2] = cov_df.loc[cov_name, :].to_numpy()
            X[:, 3] = X[:, 1] * X[:, 2]

            mask = X[:, 1] != -1

            # Calculate stats.
            eqtl_stats = self.calculate_eqtl_stats(X=X[mask, :2], y=y[mask])
            ieqtl_stats = self.calculate_interaction_eqtl_stats(X=X[mask, :], y=y[mask])

            df = pd.DataFrame(X[mask, :], columns=["intercept", "genotype", "covariate", "interaction"])
            df["expression"] = y[mask]
            df["group"] = df["genotype"].round(0)
            print(df)

            # Plot.
            self.create_overview_figure(df=df,
                                        snp=snp_name,
                                        plot1_annot=eqtl_stats,
                                        plot2_annot=ieqtl_stats,
                                        gene="{} ({})".format(probe_name, probe_label),
                                        cov="{} ({})".format(cov_name, cov_label),
                                        title="{}:{}:{}".format(probe_name, snp_name, cov_name),
                                        outdir=self.outdir)

    @staticmethod
    def load_file(path, header, index_col, sep="\t", nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    @staticmethod
    def calculate_eqtl_stats(X, y):
        n = X.shape[0]
        df = X.shape[1]

        # Calculate alternative model.
        inv_m = np.linalg.inv(X.T.dot(X))
        betas = inv_m.dot(X.T).dot(y)
        y_hat = np.dot(X, betas)

        res = y - y_hat
        rss = np.sum(res * res)

        std = np.sqrt(rss / (n - df) * np.diag(inv_m))
        t_values = betas / std

        # Calculate null model.
        null_y_hat = np.mean(y)
        null_res = y - null_y_hat
        null_rss = np.sum(null_res * null_res)

        # Calculate p-value.
        if rss >= null_rss:
            return 1
        dfn = 1
        dfd = n - df
        f_value = ((null_rss - rss) / dfn) / (rss / dfd)
        p_value = betainc(dfd / 2, dfn / 2, 1 - ((dfn * f_value) / ((dfn * f_value) + dfd)))
        if p_value == 0:
            p_value = 2.2250738585072014e-308

        return ["N = {:,}".format(n),
                "Betas = {}".format(", ".join(["{:.2f}".format(x) for x in betas])),
                "SD = {}".format(", ".join(["{:.2f}".format(x) for x in std])),
                "t-values = {}".format(", ".join(["{:.2f}".format(x) for x in t_values])),
                "p-value = {:.2e}".format(p_value)]

    @staticmethod
    def calculate_interaction_eqtl_stats(X, y):
        n = X.shape[0]
        df = X.shape[1]

        # Calculate alternative model.
        inv_m = np.linalg.inv(X.T.dot(X))
        betas = inv_m.dot(X.T).dot(y)
        y_hat = np.dot(X, betas)

        res = y - y_hat
        rss = np.sum(res * res)

        std = np.sqrt(rss / (n - df) * np.diag(inv_m))
        t_values = betas / std

        # Calculate null model.
        null_y_hat = np.dot(X[:, :3], np.linalg.inv(X[:, :3].T.dot(X[:, :3])).dot(X[:, :3].T).dot(y))
        null_res = y - null_y_hat
        null_rss = np.sum(null_res * null_res)

        # Calculate p-value.
        if rss >= null_rss:
            return 1
        dfn = 1
        dfd = n - df
        f_value = ((null_rss - rss) / dfn) / (rss / dfd)
        p_value = betainc(dfd / 2, dfn / 2, 1 - ((dfn * f_value) / ((dfn * f_value) + dfd)))
        if p_value == 0:
            p_value = 2.2250738585072014e-308

        return ["N = {:,}".format(n),
                "Betas = {}".format(", ".join(["{:.2f}".format(x) for x in betas])),
                "SD = {}".format(", ".join(["{:.2f}".format(x) for x in std])),
                "t-values = {}".format(", ".join(["{:.2f}".format(x) for x in t_values])),
                "p-value = {:.2e}".format(p_value)]

    def create_overview_figure(self, df, snp, gene, cov, title, outdir,
                               plot1_annot=None, plot2_annot=None):
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1,
                                       ncols=2,
                                       figsize=(24, 9))

        self.eqtl_plot(fig=fig,
                       ax=ax1,
                       df=df,
                       x="group",
                       y="expression",
                       palette=self.palette,
                       xlabel=snp,
                       ylabel=gene,
                       title="eQTL",
                       annot=plot1_annot,
                       )

        self.inter_plot(fig=fig,
                        ax=ax2,
                        df=df,
                        x="covariate",
                        y="expression",
                        group="group",
                        palette=self.palette,
                        xlabel=cov,
                        ylabel="",
                        ci=None,
                        title="interaction",
                        annot=plot2_annot
                        )

        plt.suptitle(title, fontsize=18)

        fig.savefig(os.path.join(outdir, "{}_overview_plot.png".format(title.replace(":", "-"))))
        plt.close()

    @staticmethod
    def eqtl_plot(fig, ax, df, x="x", y="y", palette=None, xlabel="",
                  ylabel="", title="", annot=None):
        sns.despine(fig=fig, ax=ax)

        sns.regplot(x=x,
                    y=y,
                    data=df,
                    scatter=False,
                    line_kws={"color": "#000000"},
                    ax=ax)
        sns.boxplot(x=x,
                    y=y,
                    data=df,
                    palette=palette,
                    zorder=-1,
                    ax=ax)

        ax.annotate(
            'N = {:,}'.format(df.shape[0]),
            xy=(0.03, 0.94),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=12,
            fontweight='bold')
        if annot is not None:
            for i, annot_label in enumerate(annot):
                ax.annotate(annot_label,
                            xy=(0.03, 0.94 - (i * 0.04)),
                            xycoords=ax.transAxes,
                            color="#000000",
                            alpha=0.75,
                            fontsize=12,
                            fontweight='bold')

        ax.set_title(title,
                     fontsize=16,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

    @staticmethod
    def inter_plot(fig, ax, df, x="x", y="y", group="group", palette=None,
                   ci=95, xlabel="", ylabel="", title="", annot=None):
        if len(set(df[group].unique()).symmetric_difference({0, 1, 2})) > 0:
            return

        sns.despine(fig=fig, ax=ax)

        for i, group_id in enumerate([0, 1, 2]):
            subset = df.loc[df[group] == group_id, :]
            n = subset.shape[0]

            coef_str = "NA"
            if len(subset.index) > 1:
                # Regression.
                coef, p = stats.spearmanr(subset[y], subset[x])
                coef_str = "{:.2f}".format(coef)

                # Plot.
                sns.regplot(x=x, y=y, data=subset, ci=ci,
                            scatter_kws={'facecolors': palette[group_id],
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": palette[group_id], "alpha": 0.75},
                            ax=ax
                            )

            # Add the text.
            ax.annotate(
                '{}: r = {} [n={}]'.format(group_id, coef_str, n),
                xy=(0.03, 0.94 - (i * 0.04)),
                xycoords=ax.transAxes,
                color=palette[group_id],
                alpha=0.75,
                fontsize=12,
                fontweight='bold')

        if annot is not None:
            for i, annot_label in enumerate(annot):
                ax.annotate(annot_label,
                            xy=(0.03, 0.82 - (i * 0.04)),
                            xycoords=ax.transAxes,
                            color="#000000",
                            alpha=0.75,
                            fontsize=12,
                            fontweight='bold')

        ax.set_title(title,
                     fontsize=16,
                     fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

    def print_arguments(self):
        print("Arguments:")
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Covariates path: {}".format(self.cov_path))
        print("  > Alpha: {}".format(self.alpha))
        print("  > Interest path: {}".format(self.interest_path))
        print("  > Nrows: {}".format(self.nrows))
        print("  > Extension: {}".format(self.extensions))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
