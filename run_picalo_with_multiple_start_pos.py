#!/usr/bin/env python3

"""
File:         run_picalo_with_multiple_start_pos.py
Created:      2021/11/30
Last Changed: 2022/03/25
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
from datetime import datetime
import subprocess
import argparse
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

### BIOS ###


    
### MetaBrain ###
./run_picalo_with_multiple_start_pos.py -h
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
        self.tech_covariate = getattr(arguments, 'tech_covariate')
        self.tech_covariate_with_inter = getattr(arguments, 'tech_covariate_with_inter')
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
        self.outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        outdir = getattr(arguments, 'outdir')
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "run_PICALO_with_multiple_start_pos", self.outfolder)
        self.covariates_outdir = os.path.join(self.outdir, "covariates")
        self.tech_covariate_with_inter_outdir = os.path.join(self.outdir, "tech_covariate_with_inter")
        for dir in [self.outdir, self.covariates_outdir, self.tech_covariate_with_inter_outdir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Safe the other arguments.
        self.picalo_path = getattr(arguments, 'picalo_path')
        self.print_interval = getattr(arguments, 'print_interval')
        self.sleep_time = getattr(arguments, 'sleep_time')
        self.max_end_time = int(time.time()) + ((getattr(arguments, 'max_runtime') * 60) - 5) * 60

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
                                 "Default: <=0.05.")
        parser.add_argument("-iea",
                            "--ieqtl_alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The interaction eQTL significance cut-off. "
                                 "Default: <=0.05.")
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
        parser.add_argument("-od",
                            "--outdir",
                            type=str,
                            required=False,
                            default=None,
                            help="The name of the output path.")
        parser.add_argument("-of",
                            "--outfolder",
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
        tech_covariate_with_inter_path = self.tech_covariate_with_inter
        top_pics_per_group = []
        top_pics_per_group_path = os.path.join(self.outdir, "all_found_pics.txt.gz")
        top_pics_per_group_summary_stats = []
        previous_pics_df = None
        for i in range(self.n_components):
            pic = "PIC{}".format(i+1)
            name = "{}-{}".format(self.outfolder, pic)
            print("### Analyzing {} ###".format(name))

            # prep directories.
            pic_output_dir = os.path.join(self.outdir, pic)
            pic_jobs_dir = os.path.join(pic_output_dir, "jobs")
            pic_jobs_output_dir = os.path.join(pic_jobs_dir, "output")
            pic_plot_dir = os.path.join(pic_output_dir, "plot")
            for dir in [pic_output_dir, pic_jobs_dir, pic_jobs_output_dir, pic_plot_dir]:
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
                                                              jobs_output_dir=pic_jobs_output_dir)
                jobfile_paths.append(jobfile_path)
                job_names.append(job_name)

            print("  Starting job files.")
            start_time = int(time.time())
            for job_name, jobfile_path in zip(job_names, jobfile_paths):
                if not os.path.exists(os.path.join(self.picalo_path, "output", job_name, "SummaryStats.txt.gz")):
                    command = ['sbatch', jobfile_path]
                    self.run_command(command)

            print("  Waiting for jobs to finish.")
            last_print_time = None
            completed_jobs = set()
            while True:
                # Check how many jobs are done.
                n_completed = 0
                for job_name in job_names:
                    if os.path.exists(os.path.join(self.picalo_path, "output", job_name, "SummaryStats.txt.gz")):
                        if job_name not in completed_jobs:
                            print("\t  '{}' finished".format(job_name))
                            completed_jobs.add(job_name)

                        n_completed += 1

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

            print("  Loading interaction stats.")
            interaction_df = self.combine_interaction_stats(job_names=job_names)
            print("Interaction stats:")
            print(interaction_df)
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
                if len(group_indices) > 1:
                    group_interaction_df = interaction_df.loc[:, group_indices]
                    overlapping_ieqtls = list(group_interaction_df.loc[group_interaction_df.max(axis=1) <= 0.05, :].index)
                    print("\t\tOverlapping ieQTLs: {} [N={:,}]".format(", ".join(overlapping_ieqtls), len(overlapping_ieqtls)))
            print("")

            print("  Selecting top PIC per group")
            for group_index, group_indices in pic_groups.items():
                group_summary_stats_df = summary_stats_df.loc[summary_stats_df["covariate"].isin(group_indices), :]
                print("\tGroup{}:\tavg. start: {:.2f}\tavg. end: {:.2f}".format(group_index, group_summary_stats_df["start"].mean(), group_summary_stats_df["end"].mean()))
                group_top_covariate_index = group_summary_stats_df["start"].idxmax()
                group_top_covariate, _, _ = group_summary_stats_df.loc[group_top_covariate_index, :]
                group_top_pic = pics_df.loc[[group_top_covariate], :]
                appendix = ""
                if group_top_covariate == top_covariate:
                    appendix = "-X"
                label = "{}-{}-Group{}{}".format(pic, group_top_covariate, group_index, appendix)
                group_top_pic.index = [label]
                top_pics_per_group_summary_stats.append([label, group_summary_stats_df.loc[group_top_covariate_index, "start"], group_summary_stats_df.loc[group_top_covariate_index, "end"]])
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
            self.plot_summary_stats(summary_stats_df=summary_stats_df,
                                    pic_groups=pic_groups,
                                    outdir=pic_plot_dir,
                                    name=name)
            self.plot_pics_df(pics_df=pics_df,
                              pic_groups=pic_groups,
                              outdir=pic_plot_dir,
                              name=name)
            if previous_pics_df is not None:
                self.plot_compare_top_pics_per_group(current_pics_df=pics_df,
                                                     previous_pics_df=previous_pics_df,
                                                     outdir=pic_plot_dir,
                                                     name=name)
            previous_pics_df = pics_df
            del pics_df
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
            del new_tech_covariate_with_inter_df, tech_covariate_with_inter_df, new_tech_covariate_with_inter_outpath, summary_stats_df
            print("")

        print("  Grouping PICs.")
        top_pics_per_group_df = pd.concat(top_pics_per_group, axis=0)
        pic_groups = self.group_pics(df=top_pics_per_group_df)
        for group_index, group_indices in pic_groups.items():
            print("\tGroup{}: {}".format(group_index, ", ".join(group_indices)))
        print("")

        print("  Selecting top PIC per group")
        top_pics_per_group_summary_stats_df = pd.DataFrame(top_pics_per_group_summary_stats, columns=["label", "start", "end"])
        print(top_pics_per_group_summary_stats_df)
        top_unique_pics_per_group = []
        for group_index, group_indices in pic_groups.items():
            group_summary_stats_df = top_pics_per_group_summary_stats_df.loc[top_pics_per_group_summary_stats_df["label"].isin(group_indices), :]
            print("\tGroup{}:\tavg. start: {:.2f}\tavg. end: {:.2f}".format(group_index, group_summary_stats_df["start"].mean(), top_pics_per_group_summary_stats_df["end"].mean()))
            group_top_covariate, _, _ = group_summary_stats_df.loc[group_summary_stats_df["start"].idxmax(), :]
            group_top_pic = top_pics_per_group_df.loc[[group_top_covariate], :]
            top_unique_pics_per_group.append(group_top_pic)

        print("\tSaving results.")
        top_unique_pics_per_group_df = pd.concat(top_unique_pics_per_group, axis=0)
        top_unique_pics_per_group_path = os.path.join(self.outdir, "all_found_unique_pics.txt.gz")
        self.save_file(df=top_unique_pics_per_group_df,
                       outpath=top_unique_pics_per_group_path)
        del top_unique_pics_per_group_df
        print("")

        print("Plotting.")
        plot_scripts_dir = None
        avg_expr_corr_path = None
        rna_alignment_path = None
        cf_path = None
        phenotype_path = None
        sex_path = None
        if "BIOS" in self.outfolder:
            plot_scripts_dir = "/groups/umcg-bios/tmp01/projects/PICALO/dev/plot_scripts"
            avg_expr_corr_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz"
            rna_alignment_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz"
            cf_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz"
            phenotype_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_phenotypes.txt.gz"
            sex_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz"
        elif "MetaBrain" in self.outfolder:
            plot_scripts_dir = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/dev/plot_scripts"
            avg_expr_corr_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/MetaBrain_CorrelationsWithAverageExpression.txt.gz"
            rna_alignment_path = "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-01-31-expression-tables/2020-02-05-step6-covariate-removal/2020-02-05-step0-correlate-covariates-with-expression/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz"
            cf_path = "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-01-21-CortexEUR-cis-NegativeToZero-DatasetAndRAMCorrected/perform_deconvolution/deconvolution_table.txt.gz"
            phenotype_path = "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_metabrain_phenotype_matrix/MetaBrain_phenotypes.txt.gz"
            sex_path = "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz"
        else:
            print("Not sure if BIOS or MetaBrain")
            exit()

        # Plot correlation_heatmap of components.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-o", self.outfolder]
        self.run_command(command)

        # Plot correlation_heatmap of components vs expression correlations.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", avg_expr_corr_path, "-cn", "AvgExprCorrelation", "-o", self.outfolder + "_vs_AvgExprCorrelation"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs RNA alignment metrics.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", rna_alignment_path , "-cn", "RNAseq alignment metrics", "-o", self.outfolder + "_vs_RNASeqAlignmentMetrics"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs cell fraction %.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", cf_path, "-cn", "cell fractions", "-o", self.outfolder + "_vs_CellFractions"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs phenotypes.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", phenotype_path, "-cn", "phenotypes", "-o", self.outfolder + "_vs_Phenotypes"]
        self.run_command(command)

        # Plot correlation_heatmap of components vs sex.
        command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", sex_path, "-cn", "Sex", "-o", self.outfolder + "_vs_Sex"]
        self.run_command(command)

        if "BIOS" in self.outfolder:
            # Plot correlation_heatmap of components vs cell counts.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellCounts.txt.gz", "-cn", "cell counts", "-o", self.outfolder + "_vs_CellCounts"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs blood stats.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_BloodStats.txt.gz", "-cn", "blood stats", "-o", self.outfolder + "_vs_BloodStats"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs SP140.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/SP140.txt.gz", "-cn", "SP140", "-o", self.outfolder + "_vs_SP140"]
            self.run_command(command)

        if "MetaBrain" in self.outfolder:
            # Plot correlation_heatmap of components vs decon.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-01-21-CortexEUR-cis-NegativeToZero-DatasetAndRAMCorrected/perform_deconvolution/deconvolution_table_complete.txt.gz", "-cn", "cell fractions", "-o", self.outfolder + "_vs_CellFractionComplete"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs IHC.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/IHC_counts.txt.gz", "-cn", "IHC", "-o", self.outfolder + "_vs_IHC"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs single cell counts.
            command = ['python3', os.path.join(plot_scripts_dir, 'create_correlation_heatmap.py'), '-rd', top_unique_pics_per_group_path, "-rn", self.outfolder, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/single_cell_counts.txt.gz", "-cn", "SCC", "-o", self.outfolder + "_vs_SCC"]
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

    def create_job_file(self, name, tech_covariate_with_inter_path, covariate, covariate_path, jobs_dir, jobs_output_dir):
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
                 "  -tc {} \\".format(self.tech_covariate),
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
                info_df = pd.DataFrame([job_name.split("-")[-1].replace("AsCov", ""), 0, np.nan, np.nan, np.nan, np.nan],
                                       columns=["iteration0"],
                                       index=["covariate", "N", "N Overlap", "Overlap %", "Sum Abs Normalized Delta Log Likelihood", "Pearson r"]).T
            else:
                info_df = self.load_file(inpath=fpath, header=0, index_col=0)

            info_df["index"] = np.arange(1, (info_df.shape[0] + 1))

            summary_stats.append([info_df.loc["iteration0", "covariate"],
                                  info_df.loc["iteration0", "N"],
                                  info_df.loc[info_df.index[-1], "N"]])

            info_df_m = info_df.melt(id_vars=["index", "covariate"])
            info_df_m_list.append(info_df_m)

        # Merge info stats.
        info_df_m = pd.concat(info_df_m_list, axis=0)
        info_df_m.dropna(inplace=True)
        info_df_m["log10 value"] = np.nan
        info_df_m.loc[info_df_m["value"] > 0, "log10 value"] = np.log10(info_df_m.loc[info_df_m["value"] > 0, "value"].astype(float))

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
                continue

            pic_df = self.load_file(inpath=fpath, header=0, index_col=0).T
            pic_df.columns = [covariate]
            pic_df_list.append(pic_df)
        pic_df = pd.concat(pic_df_list, axis=1).T

        return pic_df

    def combine_interaction_stats(self, job_names):
        interaction_df_list = []
        for job_name in job_names:
            fpath = os.path.join(self.picalo_path, "output", job_name, "PIC1", "results_iteration00.txt.gz")
            covariate = job_name.split("-")[-1].replace("AsCov", "")
            if not os.path.exists(fpath):
                print("{} does not exist".format(fpath))
                continue

            interaction_df = self.load_file(inpath=fpath, header=0, index_col=None)
            interaction_df.index = interaction_df["gene"] + "_" + interaction_df["SNP"]
            interaction_df = interaction_df[["FDR"]]
            interaction_df.columns = [covariate]
            interaction_df_list.append(interaction_df)
        interaction_df = pd.concat(interaction_df_list, axis=1)

        return interaction_df

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

        palette = {key: self.palette[-1] for key in pic_groups.keys()}
        palette[-1] = self.palette[-1]
        if len(pic_groups.keys()) <= 8:
            palette = self.palette

        for variable in info_df_m["variable"].unique():
            print("\t{}".format(variable))

            subset_m = info_df_m.loc[info_df_m["variable"] == variable, :]
            if variable == ["N Overlap", "Overlap %"]:
                subset_m = subset_m.loc[subset_m["index"] != 1, :]

            self.lineplot(df_m=subset_m, x="index", y="value",
                          units="covariate", hue="group",
                          palette=palette,
                          xlabel="iteration", ylabel=variable,
                          filename=name + "-" + variable.replace(" ", "_").lower() + "_lineplot",
                          outdir=outdir)

            if "Likelihood" in variable:
                self.lineplot(df_m=subset_m, x="index", y="log10 value",
                              units="covariate", hue="group",
                              palette=palette,
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

    def plot_summary_stats(self, summary_stats_df, pic_groups, outdir, name):
        summary_stats_df["group"] = -1
        for group_index, group_indices in pic_groups.items():
            summary_stats_df.loc[summary_stats_df["covariate"].isin(group_indices), "group"] = group_index

        palette = {key: self.palette[-1] for key in pic_groups.keys()}
        palette[-1] = self.palette[-1]
        if len(pic_groups.keys()) <= 8:
            palette = self.palette

        for y in ["start", "end"]:
            print("\t{}".format(y))

            self.barplot(df=summary_stats_df,
                         x="covariate",
                         y=y,
                         hue="group",
                         palette=palette,
                         xlabel="start position",
                         ylabel="#ieQTLs (FDR<0.05)",
                         title=y,
                         filename=name + "_{}_barplot".format(y),
                         outdir=outdir
                         )

    @staticmethod
    def barplot(df, x="x", y="y", hue=None, palette=None, title="", xlabel="",
                ylabel="", filename="plot", outdir=None):
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, ax = plt.subplots()
        sns.despine(fig=fig, ax=ax)

        g = sns.barplot(x=x,
                        y=y,
                        hue=hue,
                        palette=palette,
                        dodge=False,
                        data=df)

        g.set_title(title,
                    fontsize=14,
                    fontweight='bold')
        g.set_xlabel(xlabel,
                     fontsize=10,
                     fontweight='bold')
        g.set_ylabel(ylabel,
                     fontsize=10,
                     fontweight='bold')

        g.tick_params(labelsize=12)
        g.set_xticks(range(df.shape[0]))
        g.set_xticklabels(df[x], rotation=90)

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

        colors = [self.palette[-1]] * annot_df.shape[0]
        if len(pic_groups.keys()) <= 8:
            colors = [self.palette[group] for group in annot_df["group"]]
        corr_df = self.correlate_single(df=pics_df)

        self.plot_clustermap(df=corr_df,
                             colors=colors,
                             outdir=outdir,
                             filename=name + "_correlation_clustermap")

    @staticmethod
    def correlate_single(df):
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
    def plot_clustermap(df, colors=None, outdir=None, row_cluster=True,
                        col_cluster=True, filename=""):
        cmap = sns.diverging_palette(246, 24, as_cmap=True)
        sns.set(color_codes=True)
        g = sns.clustermap(df, cmap=cmap, vmin=-1, vmax=1, center=0,
                           row_colors=colors, col_colors=colors,
                           row_cluster=row_cluster, col_cluster=col_cluster,
                           yticklabels=True, xticklabels=True,
                           annot=df.round(2), dendrogram_ratio=(.1, .1),
                           figsize=(df.shape[0], df.shape[1]))
        plt.setp(g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(),fontsize=10))
        g.fig.subplots_adjust(bottom=0.05, top=0.7)

        plt.tight_layout()
        outpath = "{}.png".format(filename)
        if outdir is not None:
            outpath = os.path.join(outdir, outpath)
        g.savefig(outpath)
        plt.close()

    def plot_compare_top_pics_per_group(self, current_pics_df, previous_pics_df, outdir, name):
        if current_pics_df.shape[0] <= 1 and previous_pics_df.shape[0] <= 1:
            return
        corr_df = self.correlate_double(df1=current_pics_df,
                                        df2=previous_pics_df)
        self.plot_clustermap(df=corr_df,
                             row_cluster=False,
                             col_cluster=False,
                             outdir=outdir,
                             filename=name + "_TopPICSPerGroupComparison")

    @staticmethod
    def correlate_double(df1, df2):
        out_df = pd.DataFrame(np.nan, index=df1.index, columns=df2.index)

        for i, index1 in enumerate(df1.index):
            for j, index2 in enumerate(df2.index):
                corr_data = df1.loc[[index1], :].T.merge(df2.loc[[index2], :].T, left_index=True, right_index=True)
                corr_data.dropna(inplace=True)
                coef = np.nan
                if corr_data.std(axis=0).min() > 0:
                    coef, _ = stats.pearsonr(corr_data.iloc[:, 1], corr_data.iloc[:, 0])

                out_df.loc[index1, index2] = coef

        return out_df

    def print_arguments(self):
        print("Arguments:")
        print("  > eQTL: {}".format(self.eqtl))
        print("  > Genotype: {}".format(self.genotype))
        print("  > Genotype NA: {}".format(self.genotype_na))
        print("  > Expression: {}".format(self.expression))
        print("  > Technical covariates: {}".format(self.tech_covariate))
        print("  > Technical covariates with interaction: {}".format(self.tech_covariate_with_inter))
        print("  > Covariate: {}".format(self.covariate))
        print("  > Sample to dataset: {}".format(self.sample_to_dataset))
        print("  > eQTL alpha: <={}".format(self.eqtl_alpha))
        print("  > ieQTL alpha: <={}".format(self.ieqtl_alpha))
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
        print("  > Output directory: {}".format(self.outdir))
        print("  > Output folder: {}".format(self.outfolder))
        print("  > PICALO path: {}".format(self.picalo_path))
        print("  > Print interval: {} sec".format(self.sleep_time))
        print("  > Sleep time: {} sec".format(self.sleep_time))
        print("  > Max end datetime: {}".format(datetime.fromtimestamp(self.max_end_time).strftime("%d-%m-%Y, %H:%M:%S")))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()