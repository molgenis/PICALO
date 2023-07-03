#!/usr/bin/env python3

"""
File:         merge_PICALO_job_results.py
Created:      2021/11/24
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import subprocess
import os

# Third party imports.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

"""
Syntax:
./merge_PICALO_job_results.py -h 
"""

# Metadata
__program__ = "Merge PICALO Job Results"
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
        self.indir = getattr(arguments, 'input')
        self.job_name = getattr(arguments, 'job_name')
        self.pic = getattr(arguments, 'pic')
        output = getattr(arguments, 'output')
        if output is None:
            output = self.job_name
        base_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.data_outdir = os.path.join(base_dir, 'merge_PICALO_job_results', output, "data")
        self.plot_outdir = os.path.join(base_dir, 'merge_PICALO_job_results', output, "plot")
        for outdir in [self.data_outdir, self.plot_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # ("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25")
        self.pic_groups = {
            1: None,
            2: None,
            3: [("PC1", "PC2", "PC3", "PC7", "PC14", "PC17", "PC20", "PC22", "PC23", "PC24"),
                ("PC4"),
                ("PC5", "PC6", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC15", "PC16", "PC18", "PC19", "PC21", "PC25")],
            4: None,
            5: [("PC1", "PC2", "PC3", "PC4", "PC7", "PC8", "PC9", "PC10", "PC12", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                  ("PC5", "PC6", "PC11", "PC13")],
            6: [("PC1", "PC2", "PC3", "PC4", "PC5", "PC7", "PC8", "PC9", "PC10", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                  ("PC6"),
                  ("PC11"),
                  ("PC18")],
            7: [("PC1", "PC2", "PC3", "PC4", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                  ("PC5")],
            8: [("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                  ("PC16")],
            9: None,
            10: None,
            11: [("PC1", "PC2", "PC3", "PC4", "PC5", "PC7", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                 ("PC6"),
                 ("PC8")],
            12: None,
            13: [("PC1", "PC2", "PC3", "PC4", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                 ("PC5", "PC6")],
            14: None,
            15: [("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20", "PC21", "PC22", "PC23", "PC24", "PC25"),
                 ("PC8")]
        }

        self.palette = {
            -1: "#808080",
            0: "#0072B2",
            1: "#009E73",
            2: "#D55E00",
            3: "#E69F00"
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
                            "--input",
                            type=str,
                            required=True,
                            help="The path to the input folder.")
        parser.add_argument("-j",
                            "--job_name",
                            type=str,
                            required=True,
                            help="The name of the jobs. Replace the PC index"
                                 "with <N>")
        parser.add_argument("-p",
                            "--pic",
                            type=int,
                            required=False,
                            default=None,
                            help="The PIC.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output folder.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        pic_groups = None
        if self.pic is not None and self.pic in self.pic_groups.keys():
            pic_groups = self.pic_groups[self.pic]

        print("loading PIC info data")
        info_df_m_list = []
        pcs = []
        start_n_ieqtls = []
        end_n_ieqtls = []
        for pc_index in np.arange(1, 26):
            fpath = os.path.join(self.indir, self.job_name + "-PC{}AsCov".format(pc_index), "PIC1", "info.txt.gz")
            if os.path.exists(fpath):
                info_df = self.load_file(inpath=fpath, header=0, index_col=0)
                info_df["index"] = np.arange(1, (info_df.shape[0] + 1))
                info_df["component"] = "PC{}".format(pc_index)

                info_df["group"] = -1
                if pic_groups is not None:
                    for group_index, pcs_in_group in enumerate(pic_groups):
                        if "PC{}".format(pc_index) in pcs_in_group:
                            info_df["group"] = group_index
                            break

                pcs.append("PC{}".format(pc_index))
                start_n_ieqtls.append(info_df.loc["iteration0", "N"])
                end_n_ieqtls.append(info_df.loc[info_df.index[-1], "N"])

                info_df_m = info_df.melt(id_vars=["index", "covariate", "component", "group"])
                info_df_m_list.append(info_df_m)

        if len(info_df_m_list) > 1:
            info_df_m = pd.concat(info_df_m_list, axis=0)
        else:
            info_df_m = info_df_m_list[0]
        info_df_m["log10 value"] = np.log10(info_df_m["value"])

        print(pd.DataFrame({"start": start_n_ieqtls, "end": end_n_ieqtls}, index=pcs))

        print("Plotting")
        for variable in info_df_m["variable"].unique():
            print("\t{}".format(variable))

            subset_m = info_df_m.loc[info_df_m["variable"] == variable, :]
            if variable == ["N Overlap", "Overlap %"]:
                subset_m = subset_m.loc[subset_m["index"] != 1, :]

            self.lineplot(df_m=subset_m, x="index", y="value",
                          units="component", hue="group",
                          palette=self.palette,
                          xlabel="iteration", ylabel=variable,
                          filename=variable.replace(" ",
                                                    "_").lower() + "_lineplot",
                          outdir=self.plot_outdir)

            if "Likelihood" in variable:
                self.lineplot(df_m=subset_m, x="index", y="log10 value",
                              units="component", hue="group",
                              palette=self.palette,
                              xlabel="iteration", ylabel="log10 " + variable,
                              filename=variable.replace(" ",
                                                        "_").lower() + "_lineplot_log10",
                              outdir=self.plot_outdir)
        del info_df_m

        print("loading PIC data")
        pic_df_list = []
        for pc_index in np.arange(1, 26):
            fpath = os.path.join(self.indir, self.job_name + "-PC{}AsCov".format(pc_index), "components.txt.gz")
            if os.path.exists(fpath):
                pic_df = self.load_file(inpath=fpath, header=0, index_col=0).T
                pic_df.columns = ["PC{}".format(pc_index)]
                pic_df_list.append(pic_df)
        pic_df = pd.concat(pic_df_list, axis=1)
        print(pic_df)

        print("Saving data.")
        pic_df_path = os.path.join(self.data_outdir, "PICBasedOnPCX.txt.gz")
        self.save_file(df=pic_df, outpath=pic_df_path)

        print("Plotting")
        # Plot correlation_heatmap of components.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-o", self.job_name]
        self.run_command(command)

        # exit()

        # Plot correlation_heatmap of components vs Sex.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz", "-cn", "Sex", "-o", self.job_name + "_vs_Sex"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs decon.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_cell_types_DeconCell_2019-03-08.txt.gz", "-cn", "Decon-Cell cell fractions", "-o", self.job_name + "_vs_decon"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs cell fraction %.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz", "-cn", "cell fractions %", "-o", self.job_name + "_vs_CellFractionPercentages"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs RNA alignment metrics.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz", "-cn", "all STAR metrics", "-o", self.job_name + "_vs_AllSTARMetrics"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs phenotypes.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_phenotypes.txt.gz", "-cn", "phenotypes", "-o", self.job_name + "_vs_Phenotypes"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs expression correlations.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz", "-cn", "AvgExprCorrelation", "-o", self.job_name + "_vs_AvgExprCorrelation"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs SP140.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', pic_df_path, "-rn", self.job_name, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/SP140.txt.gz", "-cn", "SP140", "-o", self.job_name + "_vs_SP140"]
        self.run_command(command)

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
    def lineplot(df_m, x="x", y="y", units=None, hue=None, palette=None,
                 title="", xlabel="", ylabel="", filename="plot", outdir=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.lineplot(data=df_m,
                         x=x,
                         y=y,
                         units=units,
                         hue=hue,
                         palette=palette,
                         estimator=None,
                         legend="brief",
                         ax=ax)

        ax.set_title(title,
                     fontsize=14,
                     fontweight='bold')
        ax.set_xlabel(xlabel,
                      fontsize=10,
                      fontweight='bold')
        ax.set_ylabel(ylabel,
                      fontsize=10,
                      fontweight='bold')

        plt.tight_layout()
        outpath = "{}.png".format(filename)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)
        fig.savefig(outpath)
        plt.close()

    @staticmethod
    def run_command(command):
        print(" ".join(command))
        subprocess.call(command)

    def print_arguments(self):
        print("Arguments:")
        print("  > Input path: {}".format(self.indir))
        print("  > Job name: {}".format(self.job_name))
        print("  > PIC: {}".format(self.pic))
        print("  > Data output directory: {}".format(self.data_outdir))
        print("  > Plot output directory: {}".format(self.plot_outdir))


if __name__ == '__main__':
    m = main()
    m.start()
