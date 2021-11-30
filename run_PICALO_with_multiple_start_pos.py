#!/usr/bin/env python3

"""
File:         run_PICALO_with_multiple_start_pos.py
Created:      2021/11/30
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
from datetime import datetime
import subprocess
import argparse
import glob
import time
import os

# Third party imports.
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local application imports.

# Metadata
__program__ = "Run PICALO with Multiple Start Positions"
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
./run_PICALO_with_multiple_start_pos.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/BIOS_eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/genotype_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/expression_table.txt.gz \
    -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/first25ExpressionPCs.txt.gz \
    -tci /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/tech_covariates_with_interaction_df.txt.gz \
    -co /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/first25ExpressionPCs.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/sample_to_dataset.txt.gz \
    -maf 0.05 \
    -min_iter 50 \
    -n_components 2 \
    -o 2021-11-30-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics \
    -pp /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()

        # Safe the PICALO arguments.
        self.eqtl = getattr(arguments, 'eqtl')
        self.genotype = getattr(arguments, 'genotype')
        self.genotype_na =getattr(arguments, 'genotype_na')
        self.expression = getattr(arguments, 'expression')
        self.tech_covariate_path = getattr(arguments, 'tech_covariate')
        self.tech_covariate_with_inter_path = getattr(arguments, 'tech_covariate_with_inter')
        self.covariate = getattr(arguments, 'covariate')
        self.sample_to_dataset = getattr(arguments, 'sample_to_dataset')
        self.eqtl_alpha = getattr(arguments, 'eqtl_alpha')
        self.ieqtl_alpha = getattr(arguments, 'ieqtl_alpha')
        self.call_rate = getattr(arguments, 'call_rate')
        self.hardy_weinberg_pvalue = getattr(arguments, 'hardy_weinberg_pvalue')
        self.minor_allele_frequency = getattr(arguments, 'minor_allele_frequency')
        self.min_group_size = getattr(arguments, 'min_group_size')
        self.n_components = getattr(arguments, 'n_components')
        self.min_iter = getattr(arguments, 'min_iter')
        self.max_iter = getattr(arguments, 'max_iter')
        self.tol = getattr(arguments, 'tol')
        self.force_continue = False
        self.verbose = True
        self.picalo_outdir = getattr(arguments, 'outdir')

        # Safe the other arguments.
        self.picalo_path = getattr(arguments, 'picalo_path')
        self.print_interval = getattr(arguments, 'print_interval')
        self.sleep_time = getattr(arguments, 'sleep_time')
        self.max_end_time = int(time.time()) + ((getattr(arguments, 'max_runtime') * 60) - 5) * 60

        # Prepare an output directory.
        self.outdir = os.path.join(Path().resolve(), "run_PICALO_with_multiple_start_pos", self.picalo_outdir)
        self.covariates_outdir = os.path.join(self.outdir, "covariates")
        self.tech_covariate_with_inter_outdir = os.path.join(self.outdir, "tech_covariate_with_inter")
        for dir in [self.outdir, self.covariates_outdir, self.tech_covariate_with_inter_outdir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Define the color palette.
        self.palette = {
            -1: "#808080",
            0: "#0072B2",
            1: "#009E73",
            2: "#CC79A7",
            3: "#E69F00",
            4: "#D55E00",
            5: "#56B4E9",
            6: "#F0E442",
            7: "#000000",
            8: "#808080"
        }

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add other arguments.
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
        parser.add_argument("-na",
                            "--genotype_na",
                            type=str,
                            required=False,
                            default=-1,
                            help="The genotype value that equals a missing "
                                 "value. Default: -1.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix")
        parser.add_argument("-tc",
                            "--tech_covariate",
                            type=str,
                            default=None,
                            help="The path to the technical covariate matrix. "
                                 "Default: None.")
        parser.add_argument("-tci",
                            "--tech_covariate_with_inter",
                            type=str,
                            default=None,
                            help="The path to the technical covariate matrix"
                                 "to correct for with an interaction term. "
                                 "Default: None.")
        parser.add_argument("-co",
                            "--covariate",
                            type=str,
                            required=True,
                            help="The path to the covariate matrix")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix."
                                 "Default: None.")
        parser.add_argument("-ea",
                            "--eqtl_alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The eQTL significance cut-off. "
                                 "Default: <0.05.")
        parser.add_argument("-iea",
                            "--ieqtl_alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The interaction eQTL significance cut-off. "
                                 "Default: <0.05.")
        parser.add_argument("-cr",
                            "--call_rate",
                            type=float,
                            required=False,
                            default=0.95,
                            help="The minimal call rate of a SNP (per dataset)."
                                 "Equals to (1 - missingness). "
                                 "Default: >= 0.95.")
        parser.add_argument("-hw",
                            "--hardy_weinberg_pvalue",
                            type=float,
                            required=False,
                            default=1e-4,
                            help="The Hardy-Weinberg p-value threshold."
                                 "Default: >= 1e-4.")
        parser.add_argument("-maf",
                            "--minor_allele_frequency",
                            type=float,
                            required=False,
                            default=0.01,
                            help="The MAF threshold. Default: >0.01.")
        parser.add_argument("-mgs",
                            "--min_group_size",
                            type=int,
                            required=False,
                            default=2,
                            help="The minimal number of samples per genotype "
                                 "group. Default: >= 2.")
        parser.add_argument("-n_components",
                            type=int,
                            required=False,
                            default=10,
                            help="The number of components to extract. "
                                 "Default: 10.")
        parser.add_argument("-min_iter",
                            type=int,
                            required=False,
                            default=5,
                            help="The minimum number of optimization "
                                 "iterations per component. Default: 5.")
        parser.add_argument("-max_iter",
                            type=int,
                            required=False,
                            default=100,
                            help="The maximum number of optimization "
                                 "iterations per component. Default: 100.")
        parser.add_argument("-tol",
                            type=float,
                            required=False,
                            default=1e-3,
                            help="The convergence threshold. The optimization "
                                 "will stop when the 1 - pearson correlation"
                                 "coefficient is below this threshold. "
                                 "Default: 1e-3.")
        parser.add_argument("-force_continue",
                            action='store_true',
                            help="Force to identify more PICs even if the "
                                 "previous one did not converge."
                                 " Default: False.")
        parser.add_argument("-verbose",
                            action='store_true',
                            help="Enable verbose output. Default: False.")
        parser.add_argument("-o",
                            "--outdir",
                            type=str,
                            required=False,
                            default="output",
                            help="The name of the output folder.")
        parser.add_argument("-pp",
                            "--picalo_path",
                            type=str,
                            required=True,
                            help="The path to the picalo directory.")
        parser.add_argument("-st",
                            "--sleep_time",
                            type=int,
                            default=30,
                            help="The sleep time in seconds. Default: 30.")
        parser.add_argument("-pi",
                            "--print_interval",
                            type=int,
                            default=60,
                            help="The print interval time in seconds. "
                                 "Default: 60.")
        parser.add_argument("-mr",
                            "--max_runtime",
                            type=int,
                            default=6,
                            help="The maximum runtime in hours. Default: 6.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Creating separate covariate files.")
        # Split the covariate matrix.
        cov_df = self.load_file(self.covariate, header=0, index_col=0)
        covariates = {covariate: None for covariate in list(cov_df.index)}
        for covariate in covariates:
            outpath = os.path.join(self.covariates_outdir, "{}.txt.gz".format(covariate))
            self.save_file(df=cov_df.loc[[covariate], :], outpath=outpath)
            covariates[covariate] = outpath

        n_covariates = len(covariates.keys())
        print("Running PICALO with {} covariates as start positions".format(n_covariates))

        print("")
        tech_covariate_with_inter_path = self.tech_covariate_with_inter_path
        top_pics_per_group = []
        top_pics_per_group_path = os.path.join(self.outdir, "all_found_pics.txt.gz")
        for i in range(self.n_components):
            if i != 2:
                continue
            pic = "PIC{}".format(i+1)
            name = "{}-{}".format(self.picalo_outdir, pic)
            print("### Analyzing {} ###".format(name))

            # prep directories.
            pic_output_dir = os.path.join(self.outdir, pic)
            pic_jobs_dir = os.path.join(pic_output_dir, "jobs")
            pic_jobs_output_dir = os.path.join(pic_jobs_dir, "output")
            pic_status_dir = os.path.join(pic_output_dir, "status")
            pic_plot_dir = os.path.join(pic_output_dir, "plot")
            for dir in [pic_output_dir, pic_jobs_dir, pic_jobs_output_dir, pic_status_dir, pic_plot_dir]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            print("  Creating job files.")
            jobfile_paths = []
            job_names = []
            for covariate, covariate_path in covariates.items():
                jobfile_path, job_name = self.create_job_file(name=name,
                                                              tech_covariate_with_inter_path=tech_covariate_with_inter_path,
                                                              covariate=covariate,
                                                              covariate_path=covariate_path,
                                                              jobs_dir=pic_jobs_dir,
                                                              jobs_output_dir=pic_jobs_output_dir,
                                                              status_dir=pic_status_dir)
                jobfile_paths.append(jobfile_path)
                job_names.append(job_name)

            print("  Starting job files.")
            start_time = int(time.time())
            for jobfile_path in jobfile_paths:
                command = ['sbatch', jobfile_path]
                self.run_command(command)

            print("  Waiting for jobs to finish.")
            last_print_time = None
            while True:
                # Check how many jobs are done.
                n_completed = len(
                    glob.glob(os.path.join(pic_status_dir, "*.txt")))

                # Update user on progress.
                now_time = int(time.time())
                if last_print_time is None or (
                        now_time - last_print_time) >= self.print_interval or n_completed == n_covariates:
                    print("\t{:,}/{:,} jobs finished [{:.2f}%]".format(
                        n_completed, n_covariates,
                        (100 / n_covariates) * n_completed))
                    last_print_time = now_time

                if time.time() > self.max_end_time:
                    print("\tMax end time reached.")
                    break

                if n_completed == n_covariates:
                    rt_min, rt_sec = divmod(int(time.time()) - start_time, 60)
                    print("\t\tAll jobs are finished in {} minute(s) and "
                          "{} second(s)".format(int(rt_min),
                                                int(rt_sec)))
                    break

                time.sleep(self.sleep_time)

            # Check if we didnt exceed the time limit.
            if time.time() > self.max_end_time:
                break

            print("  Loading info files.")
            info_df_m, summary_stats_df = self.combine_picalo_info_files(job_names=job_names)
            print("Summary stats:")
            print(summary_stats_df)
            print("")

            print("  Loading PICs.")
            pics_df = self.combine_picalo_pics(job_names=job_names)
            print("PICs:")
            print(pics_df)
            print("")

            print("  Saving results.")
            self.save_file(df=summary_stats_df, outpath=os.path.join(pic_output_dir, "SummaryStats.txt.gz"))
            self.save_file(df=pics_df, outpath=os.path.join(pic_output_dir, "PICBasedOnPCX.txt.gz"))
            print("")

            print("  Selecting top PIC")
            top_covariate, _, _ = summary_stats_df.loc[summary_stats_df["start"].idxmax(), :]
            print("\tTop covariate = {}".format(top_covariate))
            top_pic = pics_df.loc[[top_covariate], :]
            top_pic.index = [pic]
            print("")

            print("  Grouping PICs.")
            pic_groups = self.group_pics(df=pics_df)
            for group_index, group_indices in pic_groups.items():
                print("\tGroup{}: {}".format(group_index, ", ".join(group_indices)))
            print("")

            print("  Selecting top PIC per group")
            for group_index, group_indices in pic_groups.items():
                group_summary_stats_df = summary_stats_df.loc[summary_stats_df["covariate"].isin(group_indices), :]
                print("\tGroup{}:\tavg. start: {:.2f}\tavg. end: {:.2f}".format(group_index, group_summary_stats_df["start"].mean(), group_summary_stats_df["end"].mean()))
                group_top_covariate, _, _ = group_summary_stats_df.loc[group_summary_stats_df["start"].idxmax(), :]
                group_top_pic = pics_df.loc[[group_top_covariate], :]
                appendix = ""
                if group_top_covariate == top_covariate:
                    appendix = "-X"
                group_top_pic.index = ["{}-{}-Group{}{}".format(pic, group_top_covariate, group_index, appendix)]
                top_pics_per_group.append(group_top_pic)

            print("\tSaving results.")
            top_pics_per_group_df = pd.concat(top_pics_per_group, axis=0)
            self.save_file(df=top_pics_per_group_df, outpath=top_pics_per_group_path)
            del top_pics_per_group_df
            print("")

            print("  Plotting.")
            self.plot_info_df_m(info_df_m=info_df_m,
                                pic_groups=pic_groups,
                                outdir=pic_plot_dir,
                                name=name)

            self.plot_pics_df(pics_df=pics_df,
                              pic_groups=pic_groups,
                              outdir=pic_plot_dir,
                              name=name)
            print("")

            print("  Adding top PIC to tech_covariate_with_inter_path")
            tech_covariate_with_inter_df = self.load_file(tech_covariate_with_inter_path, header=0, index_col=0)
            print(tech_covariate_with_inter_df)
            new_tech_covariate_with_inter_df = tech_covariate_with_inter_df.T.merge(top_pic.T, left_index=True, right_index=True).T
            print(new_tech_covariate_with_inter_df)

            print("\tSaving file.")
            new_tech_covariate_with_inter_outpath = os.path.join(self.tech_covariate_with_inter_outdir, os.path.basename(tech_covariate_with_inter_path).replace(".txt.gz", "") + "_{}.txt.gz".format(pic))
            self.save_file(df=new_tech_covariate_with_inter_df, outpath=new_tech_covariate_with_inter_outpath)

            # Overwriting -tci argument.
            tech_covariate_with_inter_path = new_tech_covariate_with_inter_outpath
            del new_tech_covariate_with_inter_df, tech_covariate_with_inter_df, new_tech_covariate_with_inter_outpath
            print("")

        print("Plotting.")
        # Plot correlation_heatmap of components.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-o", self.picalo_outdir]
        self.run_command(command)

        # Plot correlation_heatmap of components vs Sex.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz", "-cn", "Sex", "-o", self.picalo_outdir + "_vs_Sex"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs decon.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_cell_types_DeconCell_2019-03-08.txt.gz", "-cn", "Decon-Cell cell fractions", "-o", self.picalo_outdir + "_vs_decon"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs cell fraction %.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz", "-cn", "cell fractions %", "-o", self.picalo_outdir + "_vs_CellFractionPercentages"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs RNA alignment metrics.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz", "-cn", "all STAR metrics", "-o", self.picalo_outdir + "_vs_AllSTARMetrics"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs phenotypes.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_phenotypes.txt.gz", "-cn", "phenotypes", "-o", self.picalo_outdir + "_vs_Phenotypes"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs expression correlations.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz", "-cn", "AvgExprCorrelation", "-o", self.picalo_outdir + "_vs_AvgExprCorrelation"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs SP140.
        command = ['python3', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts/create_correlation_heatmap.py', '-rd', top_pics_per_group_path, "-rn", self.picalo_outdir, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/SP140.txt.gz", "-cn", "SP140", "-o", self.picalo_outdir + "_vs_SP140"]
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

    def create_job_file(self, name, tech_covariate_with_inter_path, covariate, covariate_path, jobs_dir, jobs_output_dir, status_dir):
        job_name = "{}-{}AsCov".format(name, covariate)

        lines = ["#!/bin/bash",
                 "#SBATCH --job-name={}".format(job_name),
                 "#SBATCH --output={}".format(os.path.join(jobs_output_dir, job_name + ".out")),
                 "#SBATCH --error={}".format(os.path.join(jobs_output_dir, job_name + ".out")),
                 "#SBATCH --time=05:55:00",
                 "#SBATCH --cpus-per-task=2",
                 "#SBATCH --mem=8gb",
                 "#SBATCH --nodes=1",
                 "#SBATCH --open-mode=append",
                 "#SBATCH --export=NONE",
                 "#SBATCH --get-user-env=L",
                 "",
                 "module load Python/3.7.4-GCCcore-7.3.0-bare",
                 "source $HOME/env/bin/activate",
                 "",
                 "python3 {} \\".format(os.path.join(self.picalo_path, "picalo.py")),
                 "  -eq {} \\".format(self.eqtl),
                 "  -ge {} \\".format(self.genotype),
                 "  -na {} \\".format(self.genotype_na),
                 "  -ex {} \\".format(self.expression),
                 "  -tc {} \\".format(self.tech_covariate_path),
                 "  -tci {} \\".format(tech_covariate_with_inter_path),
                 "  -co {} \\".format(covariate_path),
                 "  -std {} \\".format(self.sample_to_dataset),
                 "  -ea {} \\".format(self.eqtl_alpha),
                 "  -iea {} \\".format(self.ieqtl_alpha),
                 "  -cr {} \\".format(self.call_rate),
                 "  -hw {} \\".format(self.hardy_weinberg_pvalue),
                 "  -maf {} \\".format(self.minor_allele_frequency),
                 "  -mgs {} \\".format(self.min_group_size),
                 "  -n_components 1 \\",
                 "  -min_iter {} \\".format(self.min_iter),
                 "  -max_iter {} \\".format(self.max_iter),
                 "  -tol {} \\".format(self.tol),
                 "  -verbose \\",
                 "  -o {}".format(job_name),
                 "",
                 "echo 'completed' > {}".format(os.path.join(status_dir, job_name + ".txt")),
                 "",
                 "deactivate",
                 ""]

        jobfile_path = os.path.join(jobs_dir, job_name + ".sh")
        with open(jobfile_path, "w") as f:
            for line in lines:
                f.write(line + "\n")
        f.close()
        print("\tSaved jobfile: {}".format(os.path.basename(jobfile_path)))
        return jobfile_path, job_name

    @staticmethod
    def run_command(command):
        print(" ".join(command))
        subprocess.call(command)

    def combine_picalo_info_files(self, job_names):
        info_df_m_list = []
        summary_stats = []
        for job_name in job_names:
            fpath = os.path.join(self.picalo_path, "output", job_name, "PIC1", "info.txt.gz")
            if not os.path.exists(fpath):
                print("{} does not exist".format(fpath))
                exit()

            info_df = self.load_file(inpath=fpath, header=0, index_col=0)
            info_df["index"] = np.arange(1, (info_df.shape[0] + 1))

            summary_stats.append([info_df.loc["iteration0", "covariate"],
                                  info_df.loc["iteration0", "N"],
                                  info_df.loc[info_df.index[-1], "N"]])

            info_df_m = info_df.melt(id_vars=["index", "covariate"])
            info_df_m_list.append(info_df_m)

        # Merge info stats.
        info_df_m = pd.concat(info_df_m_list, axis=0)
        info_df_m["log10 value"] = np.log10(info_df_m["value"])

        # Construct summary stats df.
        summary_stats_df = pd.DataFrame(summary_stats,
                                        columns=["covariate", "start", "end"],
                                        index=job_names)

        return info_df_m, summary_stats_df

    def combine_picalo_pics(self, job_names):
        pic_df_list = []
        for job_name in job_names:
            fpath = os.path.join(self.picalo_path, "output", job_name, "components.txt.gz")
            covariate = job_name.split("-")[-1].replace("AsCov", "")
            if not os.path.exists(fpath):
                print("{} does not exist".format(fpath))
                exit()

            pic_df = self.load_file(inpath=fpath, header=0, index_col=0).T
            pic_df.columns = [covariate]
            pic_df_list.append(pic_df)
        pic_df = pd.concat(pic_df_list, axis=1).T

        return pic_df

    @staticmethod
    def group_pics(df):
        groups = {}
        max_group_count = 0
        for index in df.index:
            if len(groups) == 0:
                groups[max_group_count] = [index]
                max_group_count += 1
            else:
                found = False
                for group_count, group_indices in groups.items():
                    if found:
                        break
                    coefs = []
                    for group_index in group_indices:
                        coef, _ = stats.pearsonr(df.loc[group_index, :], df.loc[index, :])
                        coefs.append(np.abs(coef))

                    if np.mean(coefs) > 0.95:
                        group_indices.append(index)
                        groups[group_count] = group_indices
                        found = True

                if not found:
                    groups[max_group_count] = [index]
                    max_group_count += 1

        if list(groups.keys()) == [0]:
            groups = {-1: groups[0]}

        return groups

    def plot_info_df_m(self, info_df_m, pic_groups, outdir, name):
        info_df_m["group"] = -1
        for group_index, group_indices in pic_groups.items():
            info_df_m.loc[info_df_m["covariate"].isin(group_indices), "group"] = group_index

        for variable in info_df_m["variable"].unique():
            print("\t{}".format(variable))

            subset_m = info_df_m.loc[info_df_m["variable"] == variable, :]
            if variable == ["N Overlap", "Overlap %"]:
                subset_m = subset_m.loc[subset_m["index"] != 1, :]

            self.lineplot(df_m=subset_m, x="index", y="value",
                          units="covariate", hue="group",
                          palette=self.palette,
                          xlabel="iteration", ylabel=variable,
                          filename=name + "-" + variable.replace(" ", "_").lower() + "_lineplot",
                          outdir=outdir)

            if "Likelihood" in variable:
                self.lineplot(df_m=subset_m, x="index", y="log10 value",
                              units="covariate", hue="group",
                              palette=self.palette,
                              xlabel="iteration", ylabel="log10 " + variable,
                              filename=name + "-" + variable.replace(" ", "_").lower() + "_lineplot_log10",
                              outdir=outdir)
        del info_df_m

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

    def plot_pics_df(self, pics_df, pic_groups, outdir, name):
        annot_df = pd.DataFrame(-1, index=pics_df.index, columns=["group"])
        for group_index, group_indices in pic_groups.items():
            annot_df.loc[group_indices, "group"] = group_index

        colors = [self.palette[group] for group in annot_df["group"]]
        corr_df = self.correlate(df=pics_df)

        self.plot_clustermap(df=corr_df,
                             colors=colors,
                             outdir=outdir,
                             name=name)

    @staticmethod
    def correlate(df):
        out_df = pd.DataFrame(np.nan, index=df.index, columns=df.index)

        for i, index1 in enumerate(df.index):
            for j, index2 in enumerate(df.index):
                corr_data = df.loc[[index1], :].T.merge(df.loc[[index2], :].T, left_index=True, right_index=True)
                corr_data.dropna(inplace=True)
                coef = np.nan
                if corr_data.std(axis=0).min() > 0:
                    coef, _ = stats.pearsonr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])

                out_df.loc[index1, index2] = coef

        return out_df

    @staticmethod
    def plot_clustermap(df, colors, outdir, name):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)
        sns.set(color_codes=True)
        g = sns.clustermap(df, cmap=cmap, vmin=-1, vmax=1, center=0,
                           row_colors=colors, col_colors=colors,
                           yticklabels=True, xticklabels=True,
                           annot=df.round(2), dendrogram_ratio=(.1, .1),
                           figsize=(df.shape[0], df.shape[1]))
        plt.setp(g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=10))
        g.fig.subplots_adjust(bottom=0.05, top=0.7)
        plt.tight_layout()
        g.savefig(os.path.join(outdir, "{}_correlation_clustermap.png".format(name)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL: {}".format(self.eqtl))
        print("  > Genotype: {}".format(self.genotype))
        print("  > Genotype NA: {}".format(self.genotype_na))
        print("  > Expression: {}".format(self.expression))
        print("  > Technical covariates: {}".format(self.tech_covariate_path))
        print("  > Technical covariates with interaction: {}".format(self.tech_covariate_with_inter_path))
        print("  > Covariate: {}".format(self.covariate))
        print("  > Sample to dataset: {}".format(self.sample_to_dataset))
        print("  > eQTL alpha: <{}".format(self.eqtl_alpha))
        print("  > ieQTL alpha: <{}".format(self.ieqtl_alpha))
        print("  > Call rate: >{}".format(self.call_rate))
        print("  > Hardy-Weinberg p-value: >{}".format(self.hardy_weinberg_pvalue))
        print("  > MAF: >{}".format(self.minor_allele_frequency))
        print("  > Minimal group size: >{}".format(self.min_group_size))
        print("  > N components: {}".format(self.n_components))
        print("  > Min iterations: {}".format(self.min_iter))
        print("  > Max iterations: {}".format(self.max_iter))
        print("  > Tolerance: {}".format(self.tol))
        print("  > Force continue: {}".format(self.force_continue))
        print("  > Verbose: {}".format(self.verbose))
        print("  > PICALO output directory: {}".format(self.picalo_outdir))
        print("  > PICALO path: {}".format(self.picalo_path))
        print("  > Print interval: {} sec".format(self.sleep_time))
        print("  > Sleep time: {} sec".format(self.sleep_time))
        print("  > Max end datetime: {}".format(datetime.fromtimestamp(self.max_end_time).strftime("%d-%m-%Y, %H:%M:%S")))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()