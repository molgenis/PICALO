#!/usr/bin/env python3

"""
File:         create_PICALO_jobs.py
Created:      2021/11/23
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
import os

# Third party imports.
import numpy as np

# Local application imports.

"""
Syntax:
./create_PICALO_jobs.py
"""

# Metadata
__program__ = "Create PICALO Jobs"
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
        self.jobs_dir = os.path.join(Path().resolve(), 'create_PICALO_jobs')
        self.jobs_output_dir = os.path.join(self.jobs_dir, 'output')
        for dir in [self.jobs_dir, self.jobs_output_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def start(self):
        print("Creating job files")
        for pc_index in np.arange(1, 26):
            self.create_job_file(pc_index=pc_index)

    def create_job_file(self, pc_index):
        job_name = "2021-11-24-BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-PIC4-PC{}AsCov".format(pc_index)

        lines = ["#!/bin/bash",
                 "#SBATCH --job-name={}".format(job_name),
                 "#SBATCH --output={}".format(os.path.join(self.jobs_output_dir, job_name + ".out")),
                 "#SBATCH --error={}".format(os.path.join(self.jobs_output_dir, job_name + ".out")),
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
                 "python3 /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/picalo.py \\",
                 "  -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/BIOS_eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \\",
                 "  -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/genotype_table.txt.gz \\",
                 "  -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/expression_table.txt.gz \\",
                 "  -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/first25ExpressionPCs.txt.gz \\",
                 "  -tci /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/tech_covariates_with_interaction_df_and_PIC1_PIC2_PIC3.txt.gz \\",
                 "  -co /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/ExpressionPC{}.txt.gz \\".format(pc_index),
                 "  -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics/sample_to_dataset.txt.gz \\",
                 "  -maf 0.05 \\",
                 "  -min_iter 50 \\",
                 "  -n_components 1 \\",
                 "  -o {} \\".format(job_name),
                 "  -verbose",
                 "",
                 "deactivate",
                 ""]

        jobfile_path = os.path.join(self.jobs_dir, job_name + ".sh")
        with open(jobfile_path, "w") as f:
            for line in lines:
                f.write(line + "\n")
        f.close()
        print("\tSaved jobfile: {}".format(os.path.basename(jobfile_path)))


if __name__ == '__main__':
    m = main()
    m.start()
