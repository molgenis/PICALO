#!/usr/bin/env python3

"""
File:         bryois_pic_replication.py
Created:      2022/10/18
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
./klein_ct_eqtl_replication.py -h
"""

# Metadata
__program__ = "De Klein Cell Type eQTL Replication"
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


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.picalo_indir = getattr(arguments, 'picalo_indir')
        self.picalo_alleles = getattr(arguments, 'picalo_alleles')
        self.discovery = getattr(arguments, 'discovery')
        self.replication = "Klein" if self.discovery == "PICALO" else "PICALO"
        self.palette_path = getattr(arguments, 'palette')
        self.out_filename = getattr(arguments, 'outfile')
        self.extensions = getattr(arguments, 'extension')

        self.klein_path = "/groups/umcg-biogen/prm03/projects/2022-DeKleinEtAl/output/2020-10-12-deconvolution/deconvolution/decon-eqtl_scripts/decon_eqtl/2022-03-03-CortexEUR-cis-ForceNormalised-MAF5-4SD-CompleteConfigs-NegativeToZero-DatasetAndRAMCorrected/merged_decon_results.txt.gz"

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'klein_ct_eqtl_replication', '{}Discovery'.format(self.discovery))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.palette = {
            "Astrocyte": "#D55E00",
            "EndothelialCell": "#CC79A7",
            "Excitatory": "#56B4E9",
            "Inhibitory": "#0072B2",
            "Microglia": "#E69F00",
            "Oligodendrocyte": "#009E73",
            "OtherNeuron" : "#2690ce",
        }

        self.cell_type_abbreviations = {
            "Astrocyte": "AST",
            "EndothelialCell": "END",
            "Excitatory": "EX",
            "Inhibitory": "IN",
            "Microglia": "MIC",
            "Oligodendrocyte": "OLI",
            "OtherNeuron": "NEU",
        }

        self.shared_xlim = None
        self.shared_ylim = None

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
                            help="show program's version number and exit")
        parser.add_argument("-pi",
                            "--picalo_indir",
                            type=str,
                            required=True,
                            help="The path to the PICALO input directory")
        parser.add_argument("-pa",
                            "--picalo_alleles",
                            type=str,
                            required=True,
                            help="The path to the PICALO genotype"
                                 " alleles matrix.")
        parser.add_argument("-d",
                            "--discovery",
                            type=str,
                            required=False,
                            choices=["PICALO", "Klein"],
                            default="PICALO",
                            help="Which dataset to consider discovery.")
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
        picalo_geno_stats_df = self.load_file(os.path.join(self.picalo_indir, "genotype_stats.txt.gz"), header=0, index_col=None)
        picalo_alleles_df = self.load_file(self.picalo_alleles, header=0, index_col=None)

        if picalo_geno_stats_df["SNP"].tolist() != picalo_alleles_df["SNP"].tolist():
            print("Error, genotype stats and alleles df to not match")
            exit()

        picalo_snp_info_df = pd.concat([picalo_geno_stats_df, picalo_alleles_df[["Alleles"]]], axis=1)
        picalo_snp_info_df = picalo_snp_info_df.loc[picalo_snp_info_df["mask"] == 1, :]
        picalo_snp_info_df.drop(["mask"], axis=1, inplace=True)
        picalo_snp_info_df.reset_index(inplace=True, drop=True)
        del picalo_geno_stats_df, picalo_alleles_df

        ma_list = []
        for _, row in picalo_snp_info_df.iterrows():
            if row["MA"] == 0:
                ma_list.append(row["Alleles"].split("/")[0])
            elif row["MA"] == 2:
                ma_list.append(row["Alleles"].split("/")[1])
            else:
                ma_list.append(np.nan)
        picalo_snp_info_df["MA"] = ma_list

        picalo_snp_info_df["AlleleAssessed"] = picalo_snp_info_df["Alleles"].str.split("/", n=1, expand=True)[1]
        picalo_snp_info_df.columns = ["PICALO {}".format(col) for col in picalo_snp_info_df.columns]
        print(picalo_snp_info_df)
        print(picalo_snp_info_df.columns.tolist())

        print("Loading de Klein et al. 2021 data")
        klein_ieqtl_df = self.load_file(self.klein_path, header=0, index_col=None)
        klein_ieqtl_df.index = klein_ieqtl_df["Gene"] + "_" + klein_ieqtl_df["SNP"]
        klein_cell_types = [col.replace(" pvalue", "") for col in klein_ieqtl_df.columns if " pvalue" in col]
        drop_columns = []
        for ct in klein_cell_types:
            drop_columns.extend(["{} Perm-FDR".format(ct), "{} beta".format(ct)])
        klein_ieqtl_df.drop(drop_columns, axis=1, inplace=True)
        klein_ieqtl_df.columns = ["Klein {}".format(col) for col in klein_ieqtl_df.columns]

        # cell types.
        klein_cell_types.sort()
        max_klein_ct_length = max([len(str(ct)) for ct in klein_cell_types])

        print("Loading interaction results.")
        replication_stats_df_list = []
        ieqtl_dfs = {}
        pics = []
        for i in [1, 2, 5]:
            pic = "PIC{}".format(i)
            picalo_pic_path = os.path.join(self.picalo_indir, "{}.txt.gz".format(pic))

            if not os.path.exists(picalo_pic_path):
                continue
            print("\t{}".format(pic))

            picalo_ieqtl_df = self.load_file(picalo_pic_path, header=0, index_col=None)
            picalo_ieqtl_df = picalo_ieqtl_df.loc[:, ["gene", "beta-interaction", "std-interaction", "p-value", "FDR"]]
            picalo_ieqtl_df.columns = ["PICALO {}".format(col) for col in picalo_ieqtl_df.columns]
            picalo_ieqtl_df = pd.concat([picalo_snp_info_df, picalo_ieqtl_df], axis=1)
            picalo_ieqtl_df.index = picalo_ieqtl_df["PICALO gene"] + "_" + picalo_ieqtl_df["PICALO SNP"]
            print(picalo_ieqtl_df)
            print(picalo_ieqtl_df.columns.tolist())

            ieqtl_df = picalo_ieqtl_df.merge(klein_ieqtl_df, left_index=True, right_index=True, how="inner")
            print(ieqtl_df)
            print(ieqtl_df.columns.tolist())
            if ieqtl_df.shape[0] == 0:
                continue
            if ieqtl_df["PICALO SNP"].tolist() != ieqtl_df["Klein SNP"].tolist():
                print("Error, SNPs don't match")
                continue
            if ieqtl_df["PICALO gene"].tolist() != ieqtl_df["Klein Gene"].tolist():
                print("Error, gene's don't match")
                continue
            print(ieqtl_df)
            print(ieqtl_df.columns.tolist())

            # Flip towards discovery.
            if self.discovery == "PICALO":
                ieqtl_df["flip"] = (ieqtl_df["PICALO AlleleAssessed"] == ieqtl_df["Klein Allele assessed"]).map({True: 1, False: -1})
                for klein_ct in klein_cell_types:
                    ieqtl_df["Klein {} interaction beta".format(klein_ct)] = ieqtl_df["Klein {} interaction beta".format(klein_ct)] * ieqtl_df["flip"]

                discovery_mask = (ieqtl_df["PICALO FDR"] <= 0.05).to_numpy()
                print("\t  Discovery N-ieqtls: {:,}".format(np.sum(discovery_mask)))
                for klein_ct in klein_cell_types:
                    ieqtl_df["Klein {} FDR".format(klein_ct)] = np.nan
                    replication_mask = (~ieqtl_df["Klein {} pvalue".format(klein_ct)].isna()).to_numpy()
                    mask = np.logical_and(discovery_mask, replication_mask)
                    n_overlap = np.sum(mask)
                    if n_overlap > 1:
                        ieqtl_df.loc[mask, "Klein {} FDR".format(klein_ct)] = multitest.multipletests(ieqtl_df.loc[mask, "Klein {} pvalue".format(klein_ct)], method='fdr_bh')[1]
                    n_replicating = ieqtl_df.loc[ieqtl_df["Klein {} FDR".format(klein_ct)] <= 0.05, :].shape[0]
                    print("\t  {}{} replication N-ieqtls: {:,} / {:,} [{:.2f}%]".format(klein_ct, (max_klein_ct_length - len(str(klein_ct))) * " ", n_replicating, n_overlap, (100 / n_overlap) * n_replicating))
            elif self.discovery == "Klein":
                ieqtl_df["flip"] = (ieqtl_df["Klein Allele assessed"] == ieqtl_df["PICALO AlleleAssessed"]).map({True: 1, False: -1})
                ieqtl_df["PICALO beta-interaction"] = ieqtl_df["PICALO beta-interaction"] * ieqtl_df["flip"]

                replication_mask = (~ieqtl_df["PICALO p-value"].isna()).to_numpy()
                for klein_ct in klein_cell_types:
                    discovery_mask = (ieqtl_df["Klein {} BH-FDR".format(klein_ct)] <= 0.05).to_numpy()
                    print("\t  Discovery N-ieqtls: {:,}".format(np.sum(discovery_mask)))
                    ieqtl_df["PICALO {} FDR".format(klein_ct)] = np.nan
                    mask = np.logical_and(discovery_mask, replication_mask)
                    n_overlap = np.sum(mask)
                    if n_overlap > 1:
                        ieqtl_df.loc[mask, "PICALO {} FDR".format(klein_ct)] = multitest.multipletests(ieqtl_df.loc[mask, "PICALO p-value"], method='fdr_bh')[1]
                    n_replicating = ieqtl_df.loc[ieqtl_df["PICALO {} FDR".format(klein_ct)] <= 0.05, :].shape[0]
                    print("\t  {}{} replication N-ieqtls: {:,} / {:,} [{:.2f}%]".format(klein_ct, (max_klein_ct_length - len(str(klein_ct))) * " ", n_replicating, n_overlap, (100 / n_overlap) * n_replicating))
            else:
                print("Error")
                exit()

            print("Visualizing.")
            replication_stats_df = self.plot(df=ieqtl_df,
                                             cell_types=klein_cell_types,
                                             pic=pic)
            print(replication_stats_df)
            replication_stats_df["PIC"] = pic
            replication_stats_df_list.append(replication_stats_df)

            # Saving for next.
            ieqtl_dfs[pic] = ieqtl_df
            pics.append(pic)

        replication_stats_df = pd.concat(replication_stats_df_list, axis=0)
        print(replication_stats_df)
        # replication_stats_df = pd.pivot_table(replication_stats_df.loc[replication_stats_df["label"] == "discovery significant", :],
        #                                       values='value',
        #                                       index='col',
        #                                       columns='variable')
        # replication_stats_df = replication_stats_df[["N", "pearsonr", "concordance", "Rb", "pi1"]]
        # replication_stats_df.columns = ["N", "Pearson r", "Concordance", "Rb", "pi1"]
        self.save_file(df=replication_stats_df, outpath=os.path.join(self.outdir, "replication_stats.txt.gz"))

        print("Visualizing.")
        self.combined_plot(dfs=ieqtl_dfs,
                           pics=pics,
                           cell_types=klein_cell_types)

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

    def plot(self, df, cell_types, pic):
        nrows = 3
        ncols = len(cell_types)

        self.shared_ylim = {i: (0, 1) for i in range(nrows)}
        self.shared_xlim = {i: (0, 1) for i in range(ncols)}

        replication_stats = []

        sns.set(rc={'figure.figsize': (ncols * 8, nrows * 6)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row')

        suptitle = "{} ieQTL replication in cell type ieQTLs\nfrom de Klein et al. 2021".format(pic) if self.discovery == "PICALO" else "cell type ieQTL replication in {} ieQTLs\nfrom PICALO".format(pic)

        for col_index, ct in enumerate(cell_types):
            print("\tWorking on '{}'".format(ct))

            picalo_fdr_col = None
            klein_fdr_col = None
            discovery_beta_col = None
            discovery_std_col = None
            replication_beta_col = None
            replication_std_col = None
            if self.discovery == "PICALO":
                picalo_fdr_col = "PICALO FDR"
                klein_fdr_col = "Klein {} FDR".format(ct)
                discovery_beta_col = "PICALO beta"
                discovery_std_col = "PICALO std"
                replication_beta_col = "Klein zscore-to-beta"
                replication_std_col = "Klein zscore-to-se"
            elif self.discovery == "Klein":
                picalo_fdr_col = "PICALO {} FDR".format(ct)
                klein_fdr_col = "Klein {} BH-FDR".format(ct)
                discovery_beta_col = "Klein zscore-to-beta"
                discovery_std_col = "Klein zscore-to-se"
                replication_beta_col = "PICALO beta"
                replication_std_col = "PICALO std"
            else:
                print("Error")
                exit()

            # Select the required columns.
            plot_df = df.loc[:, ["Klein Gene symbol",
                                 "PICALO N",
                                 "PICALO MAF",
                                 "PICALO beta-interaction",
                                 "PICALO std-interaction",
                                 "PICALO p-value",
                                 picalo_fdr_col,
                                 "Klein N",
                                 "Klein MAF",
                                 "Klein {} interaction beta".format(ct),
                                 "Klein {} pvalue".format(ct),
                                 klein_fdr_col,
                                 ]].copy()
            plot_df.columns = ["Gene symbol",
                               "PICALO N",
                               "PICALO MAF",
                               "PICALO beta",
                               "PICALO std",
                               "PICALO pvalue",
                               "PICALO FDR",
                               "Klein N",
                               "Klein MAF",
                               "Klein beta",
                               "Klein pvalue",
                               "Klein FDR"
                               ]
            plot_df = plot_df.loc[~plot_df["{} pvalue".format(self.replication)].isna(), :]
            plot_df.sort_values(by="{} pvalue".format(self.discovery), inplace=True)

            # Calculate the discovery standard error.
            self.pvalue_to_zscore(df=plot_df,
                                  beta_col="Klein beta",
                                  p_col="Klein pvalue",
                                  prefix="Klein ")
            self.zscore_to_beta(df=plot_df,
                                z_col="Klein z-score",
                                maf_col="Klein MAF",
                                n_col="Klein N",
                                prefix="Klein zscore-to-")

            # Convert the interaction beta to log scale.
            plot_df["PICALO beta"] = self.log_modulus_beta(plot_df["PICALO beta"])
            plot_df["Klein beta"] = self.log_modulus_beta(plot_df["Klein beta"])
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
                x="{} beta".format(self.discovery),
                y="{} beta".format(self.replication),
                xlabel="",
                ylabel="{} log interaction beta".format(self.replication),
                title=ct,
                title_color=self.palette[ct],
                accent_color=self.palette[ct],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 0, col_index)

            print("\tPlotting row 2.")
            xlim, ylim, stats2 = self.scatterplot(
                df=plot_df.loc[plot_df["{} FDR".format(self.discovery)] <= 0.05, :],
                fig=fig,
                ax=axes[1, col_index],
                x="{} beta".format(self.discovery),
                y="{} beta".format(self.replication),
                xlabel="",
                ylabel="{} log interaction beta".format(self.replication),
                title="",
                title_color=self.palette[ct],
                accent_color=self.palette[ct],
                include_ylabel=include_ylabel,
                pi1_column="{} pvalue".format(self.replication),
                rb_columns=[
                    (discovery_beta_col, discovery_std_col),
                    (replication_beta_col, replication_std_col)]
            )
            self.update_limits(xlim, ylim, 1, col_index)

            print("\tPlotting row 3.")
            xlim, ylim, stats3 = self.scatterplot(
                df=plot_df.loc[plot_df["{} FDR".format(self.replication)] <= 0.05, :],
                fig=fig,
                ax=axes[2, col_index],
                x="{} beta".format(self.discovery),
                y="{} beta".format(self.replication),
                # label="Gene symbol",
                xlabel="{} log interaction beta".format(self.discovery),
                ylabel="{} log interaction beta".format(self.replication),
                title="",
                title_color=self.palette[ct],
                accent_color=self.palette[ct],
                include_ylabel=include_ylabel
            )
            self.update_limits(xlim, ylim, 2, col_index)
            print("")

            for stats, label in zip([stats1, stats2, stats3],
                                    ["all", "discovery significant",
                                     "both significant"]):
                stats_m = stats.melt()
                stats_m["label"] = label
                stats_m["cell type"] = ct
                replication_stats.append(stats_m)

        for (m, n), ax in np.ndenumerate(axes):
            (xmin, xmax) = self.shared_xlim[n]
            (ymin, ymax) = self.shared_ylim[m]

            xmargin = (xmax - xmin) * 0.05
            ymargin = (ymax - ymin) * 0.05

            ax.set_xlim(xmin - xmargin - 1, xmax + xmargin)
            ax.set_ylim(ymin - ymargin, ymax + ymargin)

        # Add the main title.
        fig.suptitle(suptitle,
            fontsize=40,
            color="#000000",
            weight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "klein_ct_eqtl_replication_plot_{}_{}Discovery.{}".format(pic, self.discovery, extension)))
        plt.close()

        # Construct the replication stats data frame.
        replication_stats_df = pd.concat(replication_stats, axis=0)
        replication_stats_df.dropna(inplace=True)

        return replication_stats_df

    def combined_plot(self, dfs, pics, cell_types):
        nrows = len(cell_types)
        ncols = len(pics)

        self.shared_ylim = {i: (0, 1) for i in range(nrows)}
        self.shared_xlim = {i: (0, 1) for i in range(ncols)}

        sns.set(rc={'figure.figsize': (ncols * 8, nrows * 6)})
        sns.set_style("ticks")
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 sharex='col',
                                 sharey='row')

        suptitle = "PIC ieQTL replication in cell type ieQTLs\nfrom de Klein et al. 2021" if self.discovery == "PICALO" else "cell type ieQTL replication in PIC ieQTLs\nfrom PICALO"

        for col_index, pic in enumerate(pics):
            for row_index, ct in enumerate(cell_types):
                print("\tWorking on '{}-{}'".format(pic, ct))
                df = dfs[pic]

                picalo_fdr_col = None
                discovery_beta_col = None
                discovery_std_col = None
                replication_beta_col = None
                replication_std_col = None
                if self.discovery == "PICALO":
                    picalo_fdr_col = "PICALO FDR"
                    klein_fdr_col = "Klein {} FDR".format(ct)
                    discovery_beta_col = "PICALO beta"
                    discovery_std_col = "PICALO std"
                    replication_beta_col = "Klein zscore-to-beta"
                    replication_std_col = "Klein zscore-to-se"
                elif self.discovery == "Klein":
                    picalo_fdr_col = "PICALO {} FDR".format(ct)
                    klein_fdr_col = "Klein {} BH-FDR".format(ct)
                    discovery_beta_col = "Klein zscore-to-beta"
                    discovery_std_col = "Klein zscore-to-se"
                    replication_beta_col = "PICALO beta"
                    replication_std_col = "PICALO std"
                else:
                    print("Error")
                    exit()

                # Select the required columns.
                plot_df = df.loc[:, ["Klein Gene symbol",
                                     "PICALO N",
                                     "PICALO MAF",
                                     "PICALO beta-interaction",
                                     "PICALO std-interaction",
                                     "PICALO p-value",
                                     picalo_fdr_col,
                                     "Klein N",
                                     "Klein MAF",
                                     "Klein {} interaction beta".format(ct),
                                     "Klein {} pvalue".format(ct),
                                     klein_fdr_col,
                                     ]].copy()
                plot_df.columns = ["Gene symbol",
                                   "PICALO N",
                                   "PICALO MAF",
                                   "PICALO beta",
                                   "PICALO std",
                                   "PICALO pvalue",
                                   "PICALO FDR",
                                   "Klein N",
                                   "Klein MAF",
                                   "Klein beta",
                                   "Klein pvalue",
                                   "Klein FDR"
                                   ]
                plot_df = plot_df.loc[~plot_df["{} pvalue".format(self.replication)].isna(),:]
                plot_df.sort_values(by="{} pvalue".format(self.discovery), inplace=True)

                # Calculate the discovery standard error.
                self.pvalue_to_zscore(df=plot_df,
                                      beta_col="Klein beta",
                                      p_col="Klein pvalue",
                                      prefix="Klein ")
                self.zscore_to_beta(df=plot_df,
                                    z_col="Klein z-score",
                                    maf_col="Klein MAF",
                                    n_col="Klein N",
                                    prefix="Klein zscore-to-")

                # Convert the interaction beta to log scale.
                plot_df["PICALO beta"] = self.log_modulus_beta(plot_df["PICALO beta"])
                plot_df["Klein beta"] = self.log_modulus_beta(plot_df["Klein beta"])
                print(plot_df)

                plot_df["facecolors"] = [self.palette[ct] if fdr_value <= 0.05 else "#808080" for fdr_value in plot_df["{} FDR".format(self.replication)]]

                include_ylabel = False
                if col_index == 0:
                    include_ylabel = True

                title = ""
                if row_index == 0:
                    title = pic

                xlabel = ""
                if row_index == (len(cell_types) - 1):
                    xlabel = "{} log beta".format(self.discovery)

                if col_index == 0:
                    axes[row_index, col_index].annotate(
                        self.cell_type_abbreviations[ct],
                        xy=(-0.5, 0.9),
                        xycoords=axes[row_index, col_index].transAxes,
                        color=self.palette[ct],
                        fontsize=40
                    )
                xlim, ylim, _ = self.scatterplot(
                    df=plot_df.loc[plot_df["{} FDR".format(self.discovery)] <= 0.05, :],
                    fig=fig,
                    ax=axes[row_index, col_index],
                    x="{} beta".format(self.discovery),
                    y="{} beta".format(self.replication),
                    facecolors="facecolors",
                    # label="Gene symbol",
                    # max_labels=10,
                    xlabel=xlabel,
                    ylabel="{} log interaction beta".format(self.replication),
                    title=title,
                    title_color=self.palette[ct],
                    accent_color=self.palette[ct],
                    include_ylabel=include_ylabel,
                    pi1_column="{} pvalue".format(self.replication),
                    rb_columns=[
                        (discovery_beta_col, discovery_std_col),
                        (replication_beta_col, replication_std_col)]
                )
                self.update_limits(xlim, ylim, 1, col_index)
                self.update_limits(xlim, ylim, row_index, col_index)
                print("")

        for (m, n), ax in np.ndenumerate(axes):
            (xmin, xmax) = self.shared_xlim[n]
            (ymin, ymax) = self.shared_ylim[m]

            xmargin = (xmax - xmin) * 0.05
            ymargin = (ymax - ymin) * 0.05

            ax.set_xlim(xmin - xmargin - 1, xmax + xmargin)
            ax.set_ylim(ymin - ymargin, ymax + ymargin)

        # Add the main title.
        fig.suptitle(suptitle,
                     fontsize=40,
                     color="#000000",
                     weight='bold')

        for extension in self.extensions:
            fig.savefig(os.path.join(self.outdir, "klein_ct_eqtl_replication_plot_{}Discovery.{}".format(self.discovery, extension)))
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
        df["{}beta".format(prefix)] = df[z_col] / a ** (1 / 2)
        df["{}se".format(prefix)] = 1 / a ** (1 / 2)

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
                    title_color="#000000", accent_color="#000000", ci=95,
                    include_ylabel=True, pi1_column=None, rb_columns=None):
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

                if rb_columns is not None and n > 2:
                    rb_est = self.calculate_rb(
                        b1=df[rb_columns[0][0]],
                        se1=df[rb_columns[0][1]],
                        b2=df[rb_columns[1][0]],
                        se2=df[rb_columns[1][1]],
                    )
                    rb = min(rb_est[0], 1)


            sns.regplot(x=x, y=y, data=df, ci=ci,
                        scatter_kws={'facecolors': facecolors,
                                     'edgecolors': "#808080"},
                        line_kws={"color": accent_color},
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
                                         color=accent_color))

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
                color=accent_color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(coef):
            ax.annotate(
                'r = {:.2f}'.format(coef),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=accent_color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(concordance):
            ax.annotate(
                'concordance = {:.0f}%'.format(concordance),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=accent_color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(pi1):
            ax.annotate(
                '\u03C01 = {:.2f}'.format(pi1),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=accent_color,
                fontsize=14,
                fontweight='bold'
            )
            y_pos -= 0.05

        if not np.isnan(rb):
            ax.annotate(
                'Rb = {:.2f}'.format(rb),
                xy=(0.03, y_pos),
                xycoords=ax.transAxes,
                color=accent_color,
                fontsize=14,
                fontweight='bold'
            )

        ax.set_title(title,
                     fontsize=22,
                     color=title_color,
                     weight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=14,
                      fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=14,
                      fontweight='bold')

        stats_df = pd.DataFrame([[n, n_concordant, concordance, coef, pi1, rb]],
                                columns=["N", "N concordant", "concordance",
                                         "pearsonr", "pi1", "Rb"],
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
        print("  > PICALO input directory: {}".format(self.picalo_indir))
        print("  > PICALO alleles: {}".format(self.picalo_alleles))
        print("  > Discovery: {}".format(self.discovery))
        print("  > Replication: {}".format(self.replication))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory: {}".format(self.outdir))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
