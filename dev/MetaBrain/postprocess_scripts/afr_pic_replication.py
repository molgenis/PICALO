#!/usr/bin/env python3

"""
File:         pic_replication.py
Created:      2022/04/14
Last Changed: 2022/04/19
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
import argparse
import json
import glob
import os
import re

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.stats import multitest
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Local application imports.

"""
Syntax:
./afr_pic_replication.py \
    -di /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov \
    -da /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ri /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA_CortexEURPICLoadingsAsCov \
    -ra /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_Replication \
    -e png pdf
    
./afr_pic_replication.py \
    -di /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PICsAsCov \
    -da /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -ri /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA_noFNPD_CortexEURPICLoadingsAsCov \
    -ra /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexAFR_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_CortexEURPrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_Replication_noFNPD \
    -e png pdf
"""

# Metadata
__program__ = "AFR PIC Replication"
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
        self.discovery_indir = getattr(arguments, 'discovery_indir')
        self.discovery_alleles = getattr(arguments, 'discovery_alleles')
        self.replication_indir = getattr(arguments, 'replication_indir')
        self.replication_alleles = getattr(arguments, 'replication_alleles')
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'afr_pic_replication')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-di",
                            "--discovery_indir",
                            type=str,
                            required=True,
                            help="The path to the discovery deconvolution "
                                 "results input directory")
        parser.add_argument("-da",
                            "--discovery_alleles",
                            type=str,
                            required=True,
                            help="The path to the discovery genotype"
                                 " alleles matrix.")
        parser.add_argument("-ri",
                            "--replication_indir",
                            type=str,
                            required=True,
                            help="The path to the replication deconvolution "
                                 "results input directory")
        parser.add_argument("-ra",
                            "--replication_alleles",
                            type=str,
                            required=True,
                            help="The path to the discovery genotype"
                                 " alleles matrix.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            required=True,
                            help="The name of the outfile.")
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

        print("Loading discovery data")
        discovery_geno_stats_df = self.load_file(os.path.join(self.discovery_indir, "genotype_stats.txt.gz"), header=0, index_col=0)
        discovery_alleles_df = self.load_file(self.discovery_alleles, header=0, index_col=0)

        if discovery_geno_stats_df.index.tolist() != discovery_alleles_df.index.tolist():
            print("Error, genotype stats and alleles df to not match")
            exit()

        discovery_df = pd.concat([discovery_geno_stats_df[["N", "MAF"]], discovery_alleles_df[["Alleles"]]], axis=1)
        del discovery_geno_stats_df, discovery_alleles_df

        discovery_df["AlleleAssessed"] = discovery_df["Alleles"].str.split("/", n=1, expand=True)[1]
        discovery_df.index.name = "SNP"
        discovery_df.reset_index(drop=False, inplace=True)
        discovery_df.drop_duplicates(inplace=True)
        discovery_df.columns = ["SNP", "EUR N", "EUR MAF", "Alleles", "AlleleAssessed"]
        print(discovery_df)

        print("Loading replication data")
        replication_geno_stats_df = self.load_file(os.path.join(self.replication_indir, "genotype_stats.txt.gz"), header=0, index_col=0)
        repilication_alleles_df = self.load_file(self.replication_alleles, header=0, index_col=0)

        if replication_geno_stats_df.index.tolist() != repilication_alleles_df.index.tolist():
            print("Error, genotype stats and alleles df to not match")
            exit()

        replication_df = pd.concat([replication_geno_stats_df[["N", "MAF"]], repilication_alleles_df[["Alleles"]]], axis=1)
        del replication_geno_stats_df, repilication_alleles_df

        replication_df["AlleleAssessed"] = replication_df["Alleles"].str.split("/", n=1, expand=True)[1]
        replication_df.drop(["Alleles"], axis=1, inplace=True)
        replication_df.index.name = "SNP"
        replication_df.reset_index(drop=False, inplace=True)
        replication_df.drop_duplicates(inplace=True)
        replication_df.columns = ["SNP", "AFR N", "AFR MAF", "AFR AlleleAssessed"]
        print(replication_df)

        print("Merging data")
        df = discovery_df.merge(replication_df, on="SNP", how="left")
        flip_dict = dict(zip(df["SNP"], (df["AlleleAssessed"] == df["AFR AlleleAssessed"]).map({True: 1, False: -1})))
        df = df.loc[:, ["SNP", "Alleles", "AlleleAssessed", "EUR N", "EUR MAF", "AFR N", "AFR MAF"]]
        print(df)
        del discovery_df, replication_df

        print("Loading interaction results.")
        ieqtl_data_list = []
        for i in range(1, 100):
            discovery_path = os.path.join(self.discovery_indir, "PIC{}.txt.gz".format(i))
            replication_path = os.path.join(self.replication_indir, "PIC{}.txt.gz".format(i))

            if not os.path.exists(discovery_path) or not os.path.exists(replication_path):
                continue
            print("\tPIC{}".format(i))

            discovery_ieqtl_df = self.load_file(discovery_path, header=0, index_col=None)
            discovery_ieqtl_df.index = discovery_ieqtl_df["SNP"] + "_" + discovery_ieqtl_df["gene"]
            discovery_ieqtl_df["tvalue-interaction"] = discovery_ieqtl_df["beta-interaction"] / discovery_ieqtl_df["std-interaction"]
            discovery_ieqtl_df = discovery_ieqtl_df[["beta-interaction", "std-interaction", "tvalue-interaction", "p-value", "FDR"]]
            discovery_ieqtl_df.columns = ["EUR PIC{} beta".format(i), "EUR PIC{} std".format(i), "EUR PIC{} tvalue".format(i), "EUR PIC{} pvalue".format(i), "EUR PIC{} FDR".format(i)]

            replication_ieqtl_df = self.load_file(replication_path, header=0, index_col=None)
            replication_ieqtl_df.index = replication_ieqtl_df["SNP"] + "_" + replication_ieqtl_df["gene"]
            replication_ieqtl_df["tvalue-interaction"] = replication_ieqtl_df["beta-interaction"] / replication_ieqtl_df["std-interaction"]
            replication_ieqtl_df["flip"] = replication_ieqtl_df["SNP"].map(flip_dict)
            replication_ieqtl_df["tvalue-interaction"] = replication_ieqtl_df["tvalue-interaction"] * replication_ieqtl_df["flip"]
            replication_ieqtl_df = replication_ieqtl_df[["beta-interaction", "std-interaction", "tvalue-interaction", "p-value"]]
            replication_ieqtl_df.columns = ["AFR PIC{} beta".format(i), "AFR PIC{} std".format(i), "AFR PIC{} tvalue".format(i), "AFR PIC{} pvalue".format(i)]

            ieqtl_df = discovery_ieqtl_df.merge(replication_ieqtl_df, left_index=True, right_index=True, how="left")

            ieqtl_df["AFR PIC{} FDR".format(i)] = np.nan
            discovery_mask = (ieqtl_df["EUR PIC{} FDR".format(i)] <= 0.05).to_numpy()
            print("\t  Discovery N-ieqtls: {:,}".format(np.sum(discovery_mask)))
            replication_mask = (~ieqtl_df["AFR PIC{} pvalue".format(i)].isna()).to_numpy()
            mask = np.logical_and(discovery_mask, replication_mask)
            n_overlap = np.sum(mask)
            if n_overlap > 1:
                ieqtl_df.loc[mask, "AFR PIC{} FDR".format(i)] = multitest.multipletests(ieqtl_df.loc[mask, "AFR PIC{} pvalue".format(i)], method='fdr_bh')[1]
            n_replicating = ieqtl_df.loc[ieqtl_df["AFR PIC{} FDR".format(i)] <= 0.05, :].shape[0]
            print("\t  Replication N-ieqtls: {:,} / {:,} [{:.2f}%]".format(n_replicating, n_overlap, (100 / n_overlap) * n_replicating))

            ieqtl_data_list.append(ieqtl_df)

        ieqtl_df = pd.concat(ieqtl_data_list, axis=1)
        ieqtl_df["SNP"] = ["_".join(x.split("_")[:-1]) for x in ieqtl_df.index]
        print(ieqtl_df)

        print("Merging data.")
        df = df.merge(ieqtl_df, on="SNP", how="right")
        print(df)

        print("Saving output")
        self.save_file(df=df,
                       outpath=os.path.join(self.outdir, "{}_pic_replication.txt.gz".format(self.out_filename)),
                       index=False)

        # df = self.load_file(os.path.join(self.outdir, "pic_replication.txt.gz"),
        #                     header=0,
        #                     index_col=None)

        print("Visualizing")
        pics = [col.replace("EUR ", "").replace(" FDR", "") for col in df.columns if col.startswith("EUR") and col.endswith("FDR")]
        pics.sort(key=self.natural_keys)
        for pic in pics:
            if pic not in self.palette:
                self.palette[pic] = "#000000"

        chuncks = [pics[i * 5:(i + 1) * 5] for i in range((len(pics) + 5 - 1) // 5)]
        for chunck in chuncks:
            print(chunck)
            self.plot(df=df,
                      cols=chunck,
                      plot_appendix="_{}_to_{}".format(chunck[0], chunck[-1]))

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

    @staticmethod
    def save_file(df, outpath, header=True, index=True, sep="\t", na_rep="NA",
                  sheet_name="Sheet1"):
        if outpath.endswith('xlsx'):
            df.to_excel(outpath,
                        sheet_name=sheet_name,
                        na_rep=na_rep,
                        header=header,
                        index=index)
        else:
            compression = 'infer'
            if outpath.endswith('.gz'):
                compression = 'gzip'

            df.to_csv(outpath, sep=sep, index=index, header=header,
                      compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    def plot(self, df, cols, plot_appendix=""):
        nrows = 3
        ncols = len(cols)

        self.shared_ylim = {i: (0, 1) for i in range(nrows)}
        self.shared_xlim = {i: (0, 1) for i in range(ncols)}

        sns.set(rc={'figure.figsize': (ncols * 8, nrows * 6)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row')

        for col_index, col in enumerate(cols):
            print("\tWorking on '{}'".format(col))

            # Select the required columns.
            plot_df = df.loc[:, ["EUR {} pvalue".format(col),
                                 "EUR {} FDR".format(col),
                                 "EUR {} beta".format(col),
                                 "EUR {} std".format(col),
                                 "EUR {} tvalue".format(col),
                                 "AFR {} pvalue".format(col),
                                 "AFR {} FDR".format(col),
                                 "AFR {} beta".format(col),
                                 "AFR {} std".format(col),
                                 "AFR {} tvalue".format(col),
                                 ]].copy()
            plot_df.columns = ["EUR pvalue",
                               "EUR FDR",
                               "EUR beta",
                               "EUR std",
                               "EUR tvalue",
                               "AFR pvalue",
                               "AFR FDR",
                               "AFR beta",
                               "AFR std",
                               "AFR tvalue"]
            plot_df = plot_df.loc[~plot_df["AFR tvalue"].isna(), :]
            plot_df.sort_values(by="EUR pvalue", inplace=True)

            include_ylabel = False
            if col_index == 0:
                include_ylabel = True

            print("\tPlotting row 1.")
            xlim, ylim, stats1 = self.scatterplot(
                df=plot_df,
                fig=fig,
                ax=axes[0, col_index],
                x="EUR tvalue",
                y="AFR tvalue",
                xlabel="",
                ylabel="AFR t-value",
                title=col,
                color=self.palette[col],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 0, col_index)

            print("\tPlotting row 2.")
            xlim, ylim, stats2 = self.scatterplot(
                df=plot_df.loc[plot_df["EUR FDR"] <= 0.05, :],
                fig=fig,
                ax=axes[1, col_index],
                x="EUR tvalue",
                y="AFR tvalue",
                xlabel="",
                ylabel="AFR t-value",
                title="",
                color=self.palette[col],
                include_ylabel=include_ylabel,
                pi1_column="AFR pvalue",
                rb_columns=[("EUR beta", "EUR std"), ("AFR beta", "AFR std")]
            )
            self.update_limits(xlim, ylim, 1, col_index)

            print("\tPlotting row 3.")
            xlim, ylim, stats3 = self.scatterplot(
                df=plot_df.loc[plot_df["AFR FDR"] <= 0.05, :],
                fig=fig,
                ax=axes[2, col_index],
                x="EUR tvalue",
                y="AFR tvalue",
                xlabel="EUR t-value",
                ylabel="AFR t-value",
                title="",
                color=self.palette[col],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 2, col_index)
            print("")

        for (m, n), ax in np.ndenumerate(axes):
            (xmin, xmax) = self.shared_xlim[n]
            (ymin, ymax) = self.shared_ylim[m]

            xmargin = (xmax - xmin) * 0.05
            ymargin = (ymax - ymin) * 0.05

            ax.set_xlim(xmin - xmargin - 1, xmax + xmargin)
            ax.set_ylim(ymin - ymargin, ymax + ymargin)

        # Add the main title.
        fig.suptitle("EUR PIC replication in AFR",
                     fontsize=40,
                     color="#000000",
                     weight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_PIC_replication{}.{}".format(self.out_filename, plot_appendix, extension)))
        plt.close()

    @staticmethod
    def pvalue_to_zscore(df, beta_col, p_col, prefix=""):
        p_values = df[p_col].to_numpy()
        zscores = stats.norm.ppf(p_values / 2)
        mask = np.ones_like(p_values)
        mask[df[beta_col] > 0] = -1
        df["{}z-score".format(prefix)] = zscores * mask
        df.loc[df[p_col] == 1, "{}z-score".format(prefix)] = 0
        df.loc[df[p_col] == 0, "{}z-score".format(prefix)] = -40.

    @staticmethod
    def zscore_to_beta(df, z_col, maf_col, n_col, prefix=""):
        chi = df[z_col] * df[z_col]
        a = 2 * df[maf_col] * (1 - df[maf_col]) * (df[n_col] + chi)
        df["{}beta".format(prefix)] = df[z_col] / a ** (1/2)
        df["{}se".format(prefix)] = 1 / a ** (1/2)

    @staticmethod
    def log_modulus_beta(series):
        s = series.copy()
        data = []
        for index, beta in s.T.iteritems():
            data.append(np.log(abs(beta) + 1) * np.sign(beta))
        new_df = pd.Series(data, index=s.index)

        return new_df

    def scatterplot(self, df, fig, ax, x="x", y="y", facecolors=None,
                    label=None, max_labels=15, xlabel="", ylabel="", title="",
                    color="#000000", ci=95, include_ylabel=True,
                    pi1_column=None, rb_columns=None):
        sns.despine(fig=fig, ax=ax)

        if not include_ylabel:
            ylabel = ""

        if facecolors is None:
            facecolors = "#808080"
        else:
            facecolors = df[facecolors]

        n = df.shape[0]
        concordance = np.nan
        n_concordant = np.nan
        coef = np.nan
        pi1 = np.nan
        rb = np.nan

        if n > 0:
            lower_quadrant = df.loc[(df[x] < 0) & (df[y] < 0), :]
            upper_quadrant = df.loc[(df[x] > 0) & (df[y] > 0), :]
            n_concordant = lower_quadrant.shape[0] + upper_quadrant.shape[0]
            concordance = (100 / n) * n_concordant

            if n > 1:
                coef, p = stats.pearsonr(df[x], df[y])

                if pi1_column is not None:
                    pi1 = self.calculate_p1(p=df[pi1_column])

                if rb_columns is not None:
                    rb_est = self.calculate_rb(
                        b1=df[rb_columns[0][0]],
                        se1=df[rb_columns[0][1]],
                        b2=df[rb_columns[1][0]],
                        se2=df[rb_columns[1][1]],
                        )
                    rb = rb_est[0]

            sns.regplot(x=x, y=y, data=df, ci=ci,
                        scatter_kws={'facecolors': facecolors,
                                     'edgecolors': "#808080"},
                        line_kws={"color": color},
                        ax=ax
                        )

            if label is not None:
                texts = []
                for i, (_, point) in enumerate(df.iterrows()):
                    if i > max_labels:
                        continue
                    texts.append(ax.text(point[x],
                                         point[y],
                                         str(point[label]),
                                         color=color))
                adjust_text(texts,
                            ax=ax,
                            only_move={'points': 'x',
                                       'text': 'xy',
                                       'objects': 'x'},
                            autoalign='x',
                            expand_text=(1., 1.),
                            expand_points=(1., 1.),
                            lim=1000,
                            arrowprops=dict(arrowstyle='-', color='#808080'))

        ax.axhline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)
        ax.axvline(0, ls='--', color="#D7191C", alpha=0.3, zorder=-1)

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

        if not np.isnan(concordance):
            ax.annotate(
                'concordance = {:.0f}%'.format(concordance),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(pi1):
            ax.annotate(
                '\u03C01 = {:.2f}'.format(pi1),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(rb):
            ax.annotate(
                'Rb = {:.2f}'.format(rb),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold'
            )

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

        stats_df = pd.DataFrame([[n, n_concordant, concordance, coef, pi1, rb]],
                                columns=["N", "N concordant", "concordance", "pearsonr", "pi1", "Rb"],
                                index=[0])

        return (df[x].min(), df[x].max()), (df[y].min(), df[y].max()), stats_df

    def update_limits(self, xlim, ylim, row, col):
        row_ylim = self.shared_ylim[row]
        if ylim[0] < row_ylim[0]:
            row_ylim = (ylim[0], row_ylim[1])
        if ylim[1] > row_ylim[1]:
            row_ylim = (row_ylim[0], ylim[1])
        self.shared_ylim[row] = row_ylim

        col_xlim = self.shared_xlim[col]
        if xlim[0] < col_xlim[0]:
            col_xlim = (xlim[0], col_xlim[1])
        if xlim[1] > col_xlim[1]:
            col_xlim = (col_xlim[0], xlim[1])
        self.shared_xlim[col] = col_xlim

    @staticmethod
    def calculate_p1(p):
        importr("qvalue")
        pvals = robjects.FloatVector(p)
        lambda_seq = robjects.FloatVector([x for x in np.arange(0.05, 1, 0.05) if p.max() > x])
        pi0est = robjects.r['pi0est'](pvals, lambda_seq)
        return 1 - np.array(pi0est.rx2('pi0'))[0]

    @staticmethod
    def calculate_rb(b1, se1, b2, se2, theta=0):
        robjects.r("source('Rb.R')")
        b1 = robjects.FloatVector(b1)
        se1 = robjects.FloatVector(se1)
        b2 = robjects.FloatVector(b2)
        se2 = robjects.FloatVector(se2)
        calcu_cor_true = robjects.globalenv['calcu_cor_true']
        rb = calcu_cor_true(b1, se1, b2, se2, theta)
        return np.array(rb)[0]

    def print_arguments(self):
        print("Arguments:")
        print("  > Discovery:")
        print("    > Input directory: {}".format(self.discovery_indir))
        print("    > Alleles path: {}".format(self.discovery_alleles))
        print("  > Replication:")
        print("    > Input directory: {}".format(self.replication_indir))
        print("    > Alleles path: {}".format(self.replication_alleles))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
