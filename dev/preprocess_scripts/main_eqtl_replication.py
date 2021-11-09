#!/usr/bin/env python3

"""
File:         main_eqtl_replication.py
Created:      2021/11/08
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
from scipy import stats
from statsmodels.regression.linear_model import OLS
from scipy.special import betainc

# Local application imports.


# Metadata
__program__ = "Main eQTL Replication"
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
./main_eqtl_replication.py -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/genotype_table.txt.gz -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/genotype_alleles_table.txt.gz -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_BIOS_PICALO_files/BIOS-cis-noRNAPhenoNA-NoMDSOutlier-20RnaAlignment/expression_table_CovCorrected.txt.gz -o eQTLReplication-BIOS-withMDSCorrection-noRNAPhenoNa-20RnaAlignment
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl_path = getattr(arguments, 'eqtl')
        self.geno_path = getattr(arguments, 'genotype')
        self.alleles_path = getattr(arguments, 'alleles')
        self.expr_path = getattr(arguments, 'expression')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'main_eqtl_replication')
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
                            help="show program's version number and exit")
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the replication eqtl matrix.")
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
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        nrows = None
        eqtl_df = self.load_file(self.eqtl_path, header=0, index_col=None, nrows=nrows)
        eqtl_df.index = eqtl_df["ProbeName"] + "_" + eqtl_df["SNPName"]
        print(eqtl_df)
        geno_df = self.load_file(self.geno_path, header=0, index_col=0, nrows=nrows)
        alleles_df = self.load_file(self.alleles_path, header=0, index_col=0, nrows=nrows)
        expr_df = self.load_file(self.expr_path, header=0, index_col=0, nrows=nrows)

        print("Checking matrices")
        if list(geno_df.index) != list(eqtl_df["SNPName"].values):
            print("Unequal input matrix.")
            exit()
        if list(expr_df.index) != list(eqtl_df["ProbeName"].values):
            print("Unequal input matrix.")
            exit()
        if list(geno_df.columns) != list(expr_df.columns):
            print("Unequal input matrix.")
            exit()

        print("Modelling discovery expression ~ genotype")
        X = np.ones((geno_df.shape[1], 2), dtype=np.float64)
        results_m = np.empty((eqtl_df.shape[0], 12), dtype=np.float64)
        for i in range(eqtl_df.shape[0]):
            genotype = geno_df.iloc[i, :].to_numpy()
            expression = expr_df.iloc[i, :].to_numpy()
            mask = genotype != -1
            if np.std(genotype[mask]) != 0 and np.std(expression[mask]) != 0:
                X[:, 1] = genotype
                results_m[i, :] = self.calculate(X=X[mask, :], y=expression[mask])

                # ols = OLS(expression, X)
                # results = ols.fit()
                # print(results.summary())
                # exit()
            else:
                results_m[i, :] = np.array([np.sum(mask), 2] + [np.nan] * 10)
        results_df = pd.DataFrame(results_m, index=eqtl_df.index, columns=["N", "df", "RSS", "beta intercept", "beta genotype", "std intercept", "std genotype", "t-value intercept", "t-value genotype", "f-value", "p-value", "z-score"])
        results_df["AA"] = alleles_df.loc[:, "AltAllele"].to_numpy()
        self.save_file(df=results_df, outpath=os.path.join(self.outdir, "{}_results_df.txt.gz".format(self.out_filename)))
        results_df = self.load_file(os.path.join(self.outdir, "{}_results_df.txt.gz".format(self.out_filename)), header=0, index_col=0)
        print(results_df)

        print("Combining data")
        eqtl_data_df = eqtl_df.loc[:, ["AssessedAllele", "Pvalue", "Zscore"]].copy()
        eqtl_data_df.columns = ["eqtl AA", "eQTL p-value", "eQTL z-score"]
        plot_df = results_df.loc[:, ["AA", "p-value", "t-value genotype"]].merge(eqtl_data_df, left_index=True, right_index=True)
        plot_df.dropna(inplace=True)

        # flip.
        plot_df["flip"] = (plot_df["eqtl AA"] == plot_df["AA"]).map({True: 1, False: -1})
        plot_df["t-value genotype flipped"] = plot_df["t-value genotype"] * plot_df["flip"]

        print("Comparing")
        self.plot_replication(df=plot_df,
                              x="eQTL z-score",
                              y="t-value genotype flipped",
                              xlabel="eQTL file z-score",
                              ylabel="t-value",
                              name="{}_zscore_vs_tvalue_replication".format(self.out_filename))

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    def calculate(self, X, y):
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

        z_score = stats.norm.ppf(p_value)

        return np.array([n, df, rss] + betas.tolist() + std.tolist() + t_values.tolist() + [f_value, p_value, z_score])

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
    def plot_eqtl(df, palette, ax, title="", xlabel="", ylabel="", annotate=None):
        # Calculate the correlation.
        coef, _ = stats.pearsonr(df["genotype"], df["expression"])

        # Plot the scatter / box plot.
        sns.regplot(x="genotype", y="expression", data=df,
                    scatter=False,
                    line_kws={"color": "#000000"},
                    ax=ax
                    )
        sns.boxplot(x="round_geno", y="expression", data=df,
                    palette=palette,
                    showfliers=False,
                    zorder=1,
                    ax=ax)

        ax.annotate(
            'N = {:,}'.format(df.shape[0]),
            xy=(0.03, 0.94),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=12,
            fontweight='bold')
        ax.annotate(
            'r = {:.2f}'.format(coef),
            xy=(0.03, 0.90),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=12,
            fontweight='bold')

        if annotate is not None:
            for i, (label, value, rounding) in enumerate(annotate):
                ax.annotate(
                    '{} = {:{}}'.format(label, value, rounding),
                    xy=(0.03, 0.86 - (0.04 * i)),
                    xycoords=ax.transAxes,
                    color="#000000",
                    alpha=0.75,
                    fontsize=12,
                    fontweight='bold')

        ax.set_title(title,
                     fontsize=22,
                     weight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

    @staticmethod
    def plot_inter_eqtl(df, palette, ax, title="", xlabel="", ylabel="",
                        annotate=None):

        for i, genotype in enumerate([0.0, 1.0, 2.0]):
            subset = df.loc[df["round_geno"] == genotype, :].copy()
            color = palette[genotype]
            coef = np.nan
            if len(subset.index) > 1:
                # Calculate the correlation.
                coef, _ = stats.pearsonr(subset["cell count"], subset["expression"])

                # Plot the scatter / box plot.
                sns.regplot(x="cell count", y="expression", data=subset,
                            scatter_kws={'facecolors': color,
                                         'linewidth': 0,
                                         'alpha': 0.75},
                            line_kws={"color": color, "alpha": 0.75},
                            ax=ax
                            )

            ax.annotate(
                '{}: r = {:.2f} [N = {:,}]'.format(genotype, coef, subset.shape[0]),
                xy=(0.03, 0.94 - (0.04 * i)),
                xycoords=ax.transAxes,
                color=color,
                alpha=0.75,
                fontsize=12,
                fontweight='bold')

        if annotate is not None:
            for i, (label, value, rounding) in enumerate(annotate):
                ax.annotate(
                    '{} = {:{}}'.format(label, value, rounding),
                    xy=(0.03, 0.82 - (0.04 * i)),
                    xycoords=ax.transAxes,
                    color="#000000",
                    alpha=0.75,
                    fontsize=12,
                    fontweight='bold')

        ax.set_title(title,
                     fontsize=22,
                     weight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

    def plot_replication(self, df, x="x", y="y", hue=None, xlabel="",
                         ylabel="", name="", title=""):
        if df.shape[0] <= 2:
            return

        facecolors = "#000000"
        if hue is not None:
            facecolors = df[hue]

        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.set(color_codes=True)

        sns.despine(fig=fig, ax=ax)

        lower_quadrant = df.loc[(df[x] < 0) & (df[y] < 0), :]
        upper_quadrant = df.loc[(df[x] > 0) & (df[y] > 0), :]
        concordance = (100 / df.shape[0]) * (lower_quadrant.shape[0] + upper_quadrant.shape[0])

        coef, _ = stats.pearsonr(df[y], df[x])

        sns.regplot(x=x, y=y, data=df, ci=None,
                    scatter_kws={'facecolors': facecolors,
                                 'linewidth': 0,
                                 'alpha': 0.75},
                    line_kws={"color": "#0072B2",
                              'linewidth': 5},
                    ax=ax)

        ax.annotate(
            'N = {}'.format(df.shape[0]),
            xy=(0.03, 0.94),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=18,
            fontweight='bold')
        ax.annotate(
            'r = {:.2f}'.format(coef),
            xy=(0.03, 0.90),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=18,
            fontweight='bold')
        ax.annotate(
            'concordance = {:.0f}%'.format(concordance),
            xy=(0.03, 0.86),
            xycoords=ax.transAxes,
            color="#000000",
            alpha=1,
            fontsize=18,
            fontweight='bold')

        ax.axhline(0, ls='--', color="#000000", zorder=-1)
        ax.axvline(0, ls='--', color="#000000", zorder=-1)

        ax.set_xlabel(xlabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=20,
                      fontweight='bold')
        ax.set_title(title,
                     fontsize=25,
                     fontweight='bold')

        outpath = os.path.join(self.outdir, "{}.png".format(name))
        fig.savefig(outpath)
        plt.close()
        print("\tSaved: {}".format(outpath))

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL file: {}".format(self.eqtl_path))
        print("  > Genotype path: {}".format(self.geno_path))
        print("  > Alleles path: {}".format(self.alleles_path))
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
