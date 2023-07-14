#!/usr/bin/env python3

"""
File:         run_PICALO_simulations.py
Created:      2023/07/14
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2022 M.Vochteloo
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
from datetime import datetime
import subprocess
import argparse
import os
import re
import time

# Third party imports.

# Local application imports.

# Metadata
__program__ = "Run PICALO Simulations"
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
./run_PICALO_simulations.py -h

### BIOS ###
./run_PICALO_simulations.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -o /groups/umcg-bios/tmp01/projects/PICALO \
    -n 1 2 \
    --dryrun
    
    
### MetaBrain ###

"""

TIME_DICT = {
    "short": "05:59:59",
    "medium": "23:59:00",
    "long": "6-23:59:00"
}


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.indir = getattr(arguments, 'indir')
        self.n_covariates = getattr(arguments, 'n_covariates')
        self.outdir = getattr(arguments, 'outdir')
        self.dryrun = getattr(arguments, 'dryrun')

        if not re.match("^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])", os.path.basename(self.indir)):
            print("Error, expected date at start of -i / --indir.")
            exit()
        self.input_date = re.search("^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])", os.path.basename(self.indir)).group(0)
        self.current_date = datetime.now().strftime("%Y-%m-%d")

        base_outdir = os.path.join(self.outdir, "run_PICALO_simulations")
        if not os.path.exists(base_outdir):
            os.makedirs(base_outdir)

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__)

        # Add other arguments.
        parser.add_argument("-i",
                            "--indir",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-n",
                            "--n_covariates",
                            nargs="+",
                            type=int,
                            default=[2],
                            help="The number of covariates to simulate. "
                                 "Default: [2].")
        parser.add_argument("-o",
                            "--outdir",
                            type=str,
                            required=True,
                            help="The path to the base output directory.")
        parser.add_argument("--dryrun",
                            action='store_true',
                            help="Add this flag to disable submitting the"
                                 "job files.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        for n_covariates in self.n_covariates:
            print("Processing N covariates = {:,}".format(n_covariates))
            covdir = os.path.join(self.outdir, "run_PICALO_simulations", "{}Covariates".format(n_covariates))
            job_dir = os.path.join(covdir, "jobs")
            jobs_outdir = os.path.join(job_dir, "output")
            for dir in [covdir, job_dir, jobs_outdir]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            print("  Submitting pre-processing script")
            fem_output_folder = "{}-first{}ExprPCForceNormalised".format(os.path.basename(self.indir).replace(self.input_date, self.current_date), n_covariates)
            output_folder = "{}-BIOS-SimulationOf-{}-first{}ExprPCForceNormalised".format(self.current_date, self.input_date, n_covariates)
            pre_jobfile, pre_logfile = self.create_jobfile(
                job_dir=job_dir,
                jobs_outdir=jobs_outdir,
                job_name="PICALO_SIMULATION_PRE_{}COVS".format(n_covariates),
                mem=16,
                preflights=["module load Python/3.7.4-GCCcore-7.3.0-bare", "source $HOME/env/bin/activate"],
                commands=["python3 {} \\".format(os.path.join(self.outdir, "fast_eqtl_mapper.py")),
                          "    -ge {} \\".format(os.path.join(self.indir, "genotype_table.txt.gz")),
                          "    -ex {} \\".format(os.path.join(self.indir, "expression_table.txt.gz")),
                          "    -co {} \\".format(os.path.join(self.indir, "first{}ExpressionPCs.txt.gz".format(n_covariates))),
                          "    -force_normalise_covariates \\",
                          "    -od {} \\".format(self.outdir),
                          "    -of {} \\".format(fem_output_folder),
                          "    -verbose",
                          "",
                          "python3 {} \\".format(os.path.join(self.outdir, "simulate_expression2.py")),
                          "    -s {} \\".format(os.path.join(self.outdir, "fast_eqtl_mapper", fem_output_folder, "eQTLSummaryStats.txt.gz")),
                          "    -od {} \\".format(self.outdir),
                          "    -of {}".format(output_folder),
                          "",
                          "python3 {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts", "plot_pca.py")),
                          "    -d {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "expression_table.txt.gz")),
                          "    -zscore \\",
                          "    -ns 10 \\",
                          "    -save \\",
                          "    -od {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts")),
                          "    -of {}-SimulatedExpression".format(output_folder),
                          "",
                          "python3 {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts", "generate_starting_vectors.py")),
                          "    -rc {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "simulated_covariates.txt.gz")),
                          "    -od {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts")),
                          "    -of {}".format(output_folder),
                          "",
                          "python3 {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts", "create_comparison_scatterplot2.py")),
                          "    -xd {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "simulated_covariates.txt.gz")),
                          "    -x_transpose \\",
                          "    -yd {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts", "generate_starting_vectors", output_folder, "starting_vectors.txt.gz")),
                          "    -y_transpose \\",
                          "    -od {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts")),
                          "    -of {}-SimulatedCovariates-vs-StartingVectors".format(output_folder),
                          "",
                          ]
            )

            pre_jobid = self.submit_job(
                jobfile=pre_jobfile,
                logfile=pre_logfile
            )
            del pre_jobfile, pre_logfile

            ####################################################################

            print("\n  Submitting PICALO all vectors script")
            pav_jobfile, pav_logfile = self.create_jobfile(
                job_dir=job_dir,
                jobs_outdir=jobs_outdir,
                job_name="PICALO_SIMULATION_PAV_{}COVS".format(n_covariates),
                cpu=2,
                mem=8,
                preflights=["module load Python/3.7.4-GCCcore-7.3.0-bare", "source $HOME/env/bin/activate"],
                commands=["python3 {} \\".format(os.path.join(self.outdir, "picalo.py")),
                          "    -eq {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "eQTLProbesProbeLevel.txt.gz")),
                          "    -ge {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "genotype_table.txt.gz")),
                          "    -ex {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "expression_table.txt.gz")),
                          "    -co {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts", "generate_starting_vectors", output_folder, "starting_vectors.txt.gz")),
                          "    -maf 0.05 \\",
                          "    -n_components 1 \\",
                          "    -o {}-AllRandomVectors-1PIC \\".format(output_folder),
                          "    -verbose"]
            )

            _ = self.submit_job(
                jobfile=pav_jobfile,
                logfile=pav_logfile,
                depend=pre_jobid
            )
            del pav_jobfile, pav_logfile

            print("\n  Submitting PICALO per rho scripts")
            for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                rho_str = str(rho).replace(".", "")
                rho_output_folder = "{}-RandomVector-R{}".format(output_folder, rho_str)
                pr_jobfile, pr_logfile = self.create_jobfile(
                    job_dir=job_dir,
                    jobs_outdir=jobs_outdir,
                    job_name="PICALO_SIMULATION_RUN_R{}_{}COVS".format(rho_str, n_covariates),
                    time="medium",
                    cpu=2,
                    mem=8,
                    preflights=["module load Python/3.7.4-GCCcore-7.3.0-bare", "source $HOME/env/bin/activate"],
                    commands=["python3 {} \\".format(os.path.join(self.outdir, "picalo.py")),
                              "    -eq {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "eQTLProbesProbeLevel.txt.gz")),
                              "    -ge {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "genotype_table.txt.gz")),
                              "    -ex {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "expression_table.txt.gz")),
                              "    -co {} \\".format(os.path.join(self.outdir, "dev", "simulation_scripts", "generate_starting_vectors", output_folder, "starting_vector_rho{}.txt.gz".format(rho_str))),
                              "    -maf 0.05 \\",
                              "    -n_components {} \\".format(n_covariates * 3),
                              "    -o {} \\".format(rho_output_folder),
                              "    -verbose"]
                )

                pr_jobid = self.submit_job(
                    jobfile=pr_jobfile,
                    logfile=pr_logfile,
                    depend=pre_jobid
                )
                del pr_jobfile, pr_logfile

                ####################################################################

                plot_commands = []
                if n_covariates > 1:
                    plot_commands = [
                        "python3 {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts", "create_comparison_scatterplot.py")),
                        "    -d {} \\".format(os.path.join(self.outdir, "output", rho_output_folder, "PICs.txt.gz")),
                        "    -transpose \\",
                        "    -n 5 \\",
                        "    -od {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts")),
                        "    -of {}-PICs".format(rho_output_folder)]
                plot_commands.extend([
                    "",
                    "python3 {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts", "create_comparison_scatterplot2.py")),
                    "    -xd {} \\".format(os.path.join(self.outdir, "simulate_expression2", output_folder, "simulated_covariates.txt.gz")),
                    "    -x_transpose \\",
                    "    -yd {} \\".format(os.path.join(self.outdir, "output", rho_output_folder, "PICs.txt.gz")),
                    "    -y_transpose \\",
                    "    -od {} \\".format(os.path.join(self.outdir, "dev", "plot_scripts")),
                    "    -of {}-SimulatedCovariates-vs-PICs".format(rho_output_folder)
                ])

                plot_jobfile, plot_logfile = self.create_jobfile(
                    job_dir=job_dir,
                    jobs_outdir=jobs_outdir,
                    job_name="PICALO_SIMULATION_PLOT_R{}_{}COVS".format(rho_str, n_covariates),
                    mem=4,
                    preflights=["module load Python/3.7.4-GCCcore-7.3.0-bare", "source $HOME/env/bin/activate"],
                    commands=plot_commands
                )

                _ = self.submit_job(
                    jobfile=plot_jobfile,
                    logfile=plot_logfile,
                    depend=pr_jobid
                )
                del plot_jobfile, plot_logfile

                print("")

    def create_jobfile(self, job_dir, jobs_outdir, job_name, commands, time="short", cpu=1,
                       mem=1, preflights=None):
        joblog_path = os.path.join(jobs_outdir, job_name + ".out")

        lines = ["#!/bin/bash",
                 "#SBATCH --job-name={}".format(job_name),
                 "#SBATCH --output={}".format(joblog_path),
                 "#SBATCH --error={}".format(joblog_path),
                 "#SBATCH --time={}".format(TIME_DICT[time]),
                 "#SBATCH --cpus-per-task={}".format(cpu),
                 "#SBATCH --mem={}gb".format(mem),
                 "#SBATCH --nodes=1",
                 "#SBATCH --open-mode=append",
                 "#SBATCH --export=NONE",
                 "#SBATCH --get-user-env=L",
                 ""]

        if preflights is not None and isinstance(preflights, list):
            for preflight in preflights:
                lines.append(preflight)

        lines.extend(commands)
        lines.extend(["", "echo 'Job finished'"])

        jobfile_path = os.path.join(job_dir, job_name + ".sh")
        self.write_lines_to_file(
            lines=lines,
            filepath=jobfile_path
        )
        return jobfile_path, joblog_path

    @staticmethod
    def write_lines_to_file(lines, filepath):
        with open(filepath, "w") as f:
            for line in lines:
                f.write(line + "\n")
        f.close()
        print("\tSaved jobfile: {}".format(os.path.basename(filepath)))

    def submit_job(self, jobfile, logfile, depend=None):
        if self.dryrun:
            return None

        if os.path.exists(logfile):
            for line in open(logfile, 'r'):
                if line == "Job finished":
                    print("\t\tSkipping '{}'".format(os.path.basename(jobfile)))
                    return None

        command = ['sbatch', jobfile]
        if depend is not None and (isinstance(depend, str) or isinstance(depend, list)):
            if isinstance(depend, str):
                depend = [depend]
            command.insert(1, '--depend=afterok:{}'.format(":".join(depend)))
            time.sleep(1)

        print("\t\t" + " ".join(command))
        sucess, output = self.run_command(command=command)
        print("\t\t{}".format(output))
        if not sucess:
            print("\t\tError, subprocess failed.")
            exit()

        return output.replace("Submitted batch job ", "").replace("\n", "")

    @staticmethod
    def run_command(command):
        success = False
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
            success = True
        except subprocess.CalledProcessError as e:
            output = e.output.decode()
        except Exception as e:
            output = str(e)
        return success, output

    def print_arguments(self):
        print("Arguments:")
        print("  > Input directory: {}".format(self.indir))
        print("    > Input date: {}".format(self.input_date))
        print("  > N covariates: {}".format(self.n_covariates))
        print("  > Current date: {}".format(self.current_date))
        print("  > Output directory: {}".format(self.outdir))
        print("  > dryrun: {}".format(self.dryrun))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()