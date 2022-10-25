#!/usr/bin/env python3

"""
File:         bryois_ct_eqtl_replication.py
Created:      2022/10/25
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
import argparse
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.stats import multitest
import rpy2.robjects as robjects
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Local application imports.

"""
Syntax:    
./bryois_ct_eqtl_replication.py \
    -di /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-10-21-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_CFAsCov \
    -da /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -gi /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/gencode.v32.primary_assembly.annotation-genelengths.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -t No_PIC_correction \
    -o 2022-10-21-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_CFAsCov \
    -e png

./bryois_ct_eqtl_replication.py \
    -di /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-10-21-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_TechPICsCorrected_CFAsCov \
    -da /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -gi /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/gencode.v32.primary_assembly.annotation-genelengths.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -t PIC1_and_PIC4_corrected \
    -o 2022-10-21-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_TechPICsCorrected_CFAsCov \
    -e png

./bryois_ct_eqtl_replication.py \
    -di /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_interaction_mapper/2022-10-25-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PIC1Corrected_CFAsCov \
    -da /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -gi /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/gencode.v32.primary_assembly.annotation-genelengths.txt.gz \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -t PIC1_corrected \
    -o 2022-10-25-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA_PIC1Corrected_CFAsCov \
    -e png
    
"""

# Metadata
__program__ = "Decon-eQTL Bryois Replication Plot"
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
        self.gene_info_path = getattr(arguments, 'gene_info')
        self.palette_path = getattr(arguments, 'palette')
        self.title = getattr(arguments, 'title').replace("_", " ")
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        self.bryois_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/julienbryois2021/JulienBryois2021SummaryStats.txt.gz"
        self.bryois_n = 196

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'bryois_ct_eqtl_replication')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.matched_cell_types = [
            ("Astrocyte", "Astrocytes"),
            ("EndothelialCell", "EndothelialCells"),
            ("Excitatory", "ExcitatoryNeurons"),
            ("Inhibitory", "InhibitoryNeurons"),
            ("Microglia", "Microglia"),
            ("Oligodendrocyte", "Oligodendrocytes")
        ]

        self.palette = {
            "Astrocyte": "#D55E00",
            "EndothelialCell": "#CC79A7",
            "Excitatory": "#56B4E9",
            "Inhibitory": "#0072B2",
            "Microglia": "#E69F00",
            "Oligodendrocyte": "#009E73",
            "OtherNeuron": "#2690ce"
        }

        self.shared_xlim = None
        self.shared_ylim = None

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
        parser.add_argument("-gi",
                            "--gene_info",
                            type=str,
                            required=True,
                            help="The path to the gene info matrix.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-t",
                            "--title",
                            type=str,
                            required=False,
                            default="ieQTL replication in Bryois et al. 2021",
                            help="The title for the plot. Replace '_' with ' '.")
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

        print("Loading gene translate data")
        gene_trans_df = self.load_file(self.gene_info_path, header=0, index_col=None)
        gene_trans_dict = dict(zip(gene_trans_df.iloc[:, 0].str.split(".", n=1, expand=True)[0], gene_trans_df.iloc[:, 1]))
        del gene_trans_df

        print("Loading discovery data")
        discovery_geno_stats_df = self.load_file(os.path.join(self.discovery_indir, "genotype_stats.txt.gz"), header=0, index_col=0)
        discovery_alleles_df = self.load_file(self.discovery_alleles, header=0, index_col=0)

        if discovery_geno_stats_df.index.tolist() != discovery_alleles_df.index.tolist():
            print("Error, genotype stats and alleles df to not match")
            exit()

        discovery_snp_info_df = pd.concat([discovery_geno_stats_df[["N", "HW pval", "MA", "MAF"]], discovery_alleles_df[["Alleles"]]], axis=1)
        del discovery_geno_stats_df, discovery_alleles_df

        ma_list = []
        for _, row in discovery_snp_info_df.iterrows():
            if row["MA"] == 0:
                ma_list.append(row["Alleles"].split("/")[0])
            elif row["MA"] == 2:
                ma_list.append(row["Alleles"].split("/")[1])
            else:
                ma_list.append(np.nan)
        discovery_snp_info_df["MA"] = ma_list

        discovery_snp_info_df["AlleleAssessed"] = discovery_snp_info_df["Alleles"].str.split("/", n=1, expand=True)[1]
        discovery_snp_info_df.index.name = "SNP"
        discovery_snp_info_df.reset_index(drop=False, inplace=True)
        discovery_snp_info_df.drop_duplicates(inplace=True)
        discovery_snp_info_df.columns = ["SNP", "MetaBrain N", "MetaBrain HW pval", "MetaBrain Minor allele", "MetaBrain MAF", "Alleles", "AlleleAssessed"]
        discovery_snp_info_df["SNP"] = discovery_snp_info_df["SNP"].str.split(":", expand=True)[2]

        print("Loading replication data")
        replication_df = self.load_file(self.bryois_path, header=0, index_col=0)
        replication_snp_info_df = replication_df[["SNP", "effect_allele"]].copy()
        replication_snp_info_df.columns = ["SNP", "Bryois AlleleAssessed"]
        replication_snp_info_df["Bryois N"] = self.bryois_n
        replication_df.drop(["SNP", "effect_allele"], axis=1, inplace=True)
        replication_df.columns = ["Bryois {}".format(col) for col in replication_df.columns]

        print("Merging data")
        df = discovery_snp_info_df.merge(replication_snp_info_df, on="SNP")
        flip_dict = dict(zip(df["SNP"], (df["AlleleAssessed"] == df["Bryois AlleleAssessed"]).map({True: 1, False: -1})))
        del discovery_snp_info_df, replication_snp_info_df
        print(df)

        print("Loading interaction results.")
        ieqtl_df_list = []
        for (discovery_ct, replication_ct) in self.matched_cell_types:
            # Discovery data.
            ieqtl_df = self.load_file(os.path.join(self.discovery_indir, "{}.txt.gz".format(discovery_ct)), header=0, index_col=None)
            ieqtl_df["SNP"] = [x.split(":")[2] for x in ieqtl_df["SNP"]]
            ieqtl_df["gene"] = [x.split(".")[0] for x in ieqtl_df["gene"]]
            ieqtl_df.index = ieqtl_df["gene"] + "_" + ieqtl_df["SNP"]
            ieqtl_df["flip"] = ieqtl_df["SNP"].map(flip_dict)
            ieqtl_df = ieqtl_df.loc[:, ["flip", "beta-interaction", "p-value", "FDR"]]
            ieqtl_df.columns = ["flip", "MetaBrain {} interaction beta".format(discovery_ct), "MetaBrain {} p-value".format(discovery_ct), "MetaBrain {} FDR".format(discovery_ct)]

            # Replication data.
            ieqtl_df = ieqtl_df.merge(replication_df[["Bryois {} beta".format(replication_ct), "Bryois {} p-value".format(replication_ct)]].copy(), left_index=True, right_index=True, how="left")
            ieqtl_df["Bryois {} beta".format(replication_ct)] = ieqtl_df["Bryois {} beta".format(replication_ct)] * ieqtl_df["flip"]
            ieqtl_df.drop(["flip"], axis=1, inplace=True)

            ieqtl_df["Bryois {} FDR".format(replication_ct)] = np.nan
            discovery_mask = (ieqtl_df["MetaBrain {} FDR".format(discovery_ct)] <= 0.05).to_numpy()
            replication_mask = (~ieqtl_df["Bryois {} p-value".format(replication_ct)].isna()).to_numpy()
            mask = np.logical_and(discovery_mask, replication_mask)
            n_overlap = np.sum(mask)
            if n_overlap > 1:
                ieqtl_df.loc[mask, "Bryois {} FDR".format(discovery_ct)] = multitest.multipletests(ieqtl_df.loc[mask, "Bryois {} p-value".format(replication_ct)], method='fdr_bh')[1]
            n_replicating = ieqtl_df.loc[ieqtl_df["Bryois {} FDR".format(replication_ct)] <= 0.05, :].shape[0]
            print("\t  {} replication N-ieqtls: {:,} / {:,} [{:.2f}%]".format(replication_ct, n_replicating, n_overlap, (100 / n_overlap) * n_replicating))

            ieqtl_df_list.append(ieqtl_df)

        ieqtl_df = pd.concat(ieqtl_df_list, axis=1)
        ieqtl_df["SNP"] = [x.split("_")[1] for x in ieqtl_df.index]
        ieqtl_df["Gene"] = [x.split("_")[0] for x in ieqtl_df.index]
        del ieqtl_df_list
        print(ieqtl_df)

        print("Merging data.")
        df = df.merge(ieqtl_df, on="SNP", how="right")

        print("Adding gene symbols.")
        df["Gene symbol"] = df["Gene"].map(gene_trans_dict)

        print("Visualizing")
        replication_stats_df = self.plot(df=df,
                                         matched_cell_types=self.matched_cell_types)
        self.save_file(df=replication_stats_df, outpath=os.path.join(self.outdir, "{}_replication_stats.txt.gz".format(self.out_filename)))

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

    def plot(self, df, matched_cell_types):
        nrows = 3
        ncols = len(matched_cell_types)

        self.shared_ylim = {i: (0, 1) for i in range(nrows)}
        self.shared_xlim = {i: (0, 1) for i in range(ncols)}

        replication_stats = []

        sns.set(rc={'figure.figsize': (ncols * 8, nrows * 6)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row')

        for col_index, (discovery_ct, replication_ct) in enumerate(matched_cell_types):
            print("\tWorking on '{}'".format(discovery_ct))

            # Select the required columns.
            plot_df = df.loc[:, ["Gene symbol",
                                 "MetaBrain N",
                                 "MetaBrain MAF",
                                 "MetaBrain {} interaction beta".format(discovery_ct),
                                 "MetaBrain {} p-value".format(discovery_ct),
                                 "MetaBrain {} FDR".format(discovery_ct),
                                 "Bryois N",
                                 "Bryois {} beta".format(replication_ct),
                                 "Bryois {} p-value".format(replication_ct),
                                 "Bryois {} FDR".format(replication_ct),
                                 ]].copy()
            plot_df.columns = ["Gene symbol",
                               "MetaBrain N",
                               "MetaBrain MAF",
                               "MetaBrain interaction beta",
                               "MetaBrain pvalue",
                               "MetaBrain FDR",
                               "Bryois N",
                               "Bryois eQTL beta",
                               "Bryois pvalue",
                               "Bryois FDR"
                               ]
            plot_df = plot_df.loc[~plot_df["Bryois pvalue"].isna(), :]
            plot_df.sort_values(by="MetaBrain pvalue", inplace=True)

            # Calculate the discovery standard error.
            for prefix, beta_col in zip(["MetaBrain", "Bryois"], ["interaction beta", "eQTL beta"]):
                self.pvalue_to_zscore(df=plot_df,
                                      beta_col="{} {}".format(prefix, beta_col),
                                      p_col="{} pvalue".format(prefix),
                                      prefix="{} ".format(prefix))
                self.zscore_to_beta(df=plot_df,
                                    z_col="{} z-score".format(prefix),
                                    maf_col="MetaBrain MAF",
                                    n_col="{} N".format(prefix),
                                    prefix="{} zscore-to-".format(prefix))

            # Convert the interaction beta to log scale.
            plot_df["MetaBrain interaction beta"] = self.log_modulus_beta(plot_df["MetaBrain interaction beta"])
            plot_df["Bryois eQTL beta"] = self.log_modulus_beta(plot_df["Bryois eQTL beta"])
            print(plot_df)

            include_ylabel = False
            if col_index == 0:
                include_ylabel = True

            if col_index == 0:
                for row_index, panel in enumerate(["A", "B", "C"]):
                    axes[row_index, col_index].annotate(
                        panel,
                        xy=(-0.3, 0.9),
                        xycoords=axes[row_index, col_index].transAxes,
                        color="#000000",
                        fontsize=40
                    )

            print("\tPlotting row 1.")
            xlim, ylim, stats1 = self.scatterplot(
                df=plot_df,
                fig=fig,
                ax=axes[0, col_index],
                x="MetaBrain interaction beta",
                y="Bryois eQTL beta",
                xlabel="",
                ylabel="Bryois log eQTL beta",
                title=discovery_ct,
                color=self.palette[discovery_ct],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 0, col_index)

            print("\tPlotting row 2.")
            xlim, ylim, stats2 = self.scatterplot(
                df=plot_df.loc[plot_df["MetaBrain FDR"] <= 0.05, :],
                fig=fig,
                ax=axes[1, col_index],
                x="MetaBrain interaction beta",
                y="Bryois eQTL beta",
                xlabel="",
                ylabel="Bryois log eQTL beta",
                title="",
                color=self.palette[discovery_ct],
                include_ylabel=include_ylabel,
                pi1_column="Bryois pvalue",
                rb_columns=[("MetaBrain zscore-to-beta", "MetaBrain zscore-to-se"), ("Bryois zscore-to-beta", "Bryois zscore-to-se")]
            )
            self.update_limits(xlim, ylim, 1, col_index)

            print("\tPlotting row 3.")
            xlim, ylim, stats3 = self.scatterplot(
                df=plot_df.loc[plot_df["Bryois FDR"] <= 0.05, :],
                fig=fig,
                ax=axes[2, col_index],
                x="MetaBrain interaction beta",
                y="Bryois eQTL beta",
                label="Gene symbol",
                xlabel="MetaBrain log interaction beta",
                ylabel="Bryois log eQTL beta",
                title="",
                color=self.palette[discovery_ct],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 2, col_index)
            print("")

            for stats, label in zip([stats1, stats2, stats3], ["all", "discovery significant", "both significant"]):
                stats_m = stats.melt()
                stats_m["label"] = label
                stats_m["cell type"] = discovery_ct
                replication_stats.append(stats_m)

        for (m, n), ax in np.ndenumerate(axes):
            (xmin, xmax) = self.shared_xlim[n]
            (ymin, ymax) = self.shared_ylim[m]

            xmargin = (xmax - xmin) * 0.05
            ymargin = (ymax - ymin) * 0.05

            ax.set_xlim(xmin - xmargin - 1, xmax + xmargin)
            ax.set_ylim(ymin - ymargin, ymax + ymargin)

        # Add the main title.
        fig.suptitle(self.title,
                     fontsize=40,
                     color="#000000",
                     weight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "{}_bryois_ct_eqtl_replication_plot.{}".format(self.out_filename, extension)))
        plt.close()

        # Construct the replication stats data frame.
        replication_stats_df = pd.concat(replication_stats, axis=0)
        replication_stats_df.dropna(inplace=True)

        return replication_stats_df

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
        robjects.r("source('qvalue_truncp.R')")
        p = robjects.FloatVector(p)
        qvalue_truncp = robjects.globalenv['qvalue_truncp']
        pi0 = qvalue_truncp(p)[0]
        return 1 - np.array(pi0)

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
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")

if __name__ == '__main__':
    m = main()
    m.start()
