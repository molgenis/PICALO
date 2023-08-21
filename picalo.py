#!/usr/bin/env python3

"""
File:         picalo.py
Created:      2020/11/16
Last Changed: 2023/08/21
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

"""
./picalo.py -h
"""

# Standard imports.
import os

# Third party imports.

# Local application imports.
from src.main import Main
from src.cmd_line_arguments import CommandLineArguments

# Metadata
__program__ = "Principal Interaction Component Analysis through Likelihood " \
              "Optimization (PICALO)"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "BSD (3-Clause)"
__version__ = 0.1
__description__ = "{} is a program developed and maintained by {}. " \
                  "This program is licensed under the {} license and is " \
                  "provided 'as-is' without any warranty or indemnification " \
                  "of any kind.".format(__program__,
                                        __author__,
                                        __license__)
if __name__ == '__main__':
    # Get the command line arguments.
    CLA = CommandLineArguments(program=__program__,
                               version=__version__,
                               description=__description__)
    EQTL_PATH = CLA.get_argument('eqtl')
    GENOTYPE_PATH = CLA.get_argument('genotype')
    GENOTYPE_NA = CLA.get_argument('genotype_na')
    EXPRESSION_PATH = CLA.get_argument('expression')
    TECH_COVARIATE_PATH = CLA.get_argument('tech_covariate')
    TECH_COVARIATE_WITH_INTERACTION_PATH = CLA.get_argument('tech_covariate_with_inter')
    COVARIATE_PATH = CLA.get_argument('covariate')
    SAMPLE_DATASET_PATH = CLA.get_argument('sample_to_dataset')
    MIN_DATASET_SIZE = CLA.get_argument('min_dataset_size')
    IEQTL_ALPHA = CLA.get_argument('ieqtl_alpha')
    CALL_RATE = CLA.get_argument('call_rate')
    HW_PVAL = CLA.get_argument('hardy_weinberg_pvalue')
    MAF = CLA.get_argument('minor_allele_frequency')
    MGS = CLA.get_argument('min_group_size')
    N_COMPONENTS = CLA.get_argument('n_components')
    MIN_ITER = CLA.get_argument('min_iter')
    MAX_ITER = CLA.get_argument('max_iter')
    TOL = CLA.get_argument('tol')
    FORCE_CONTINUE = CLA.get_argument('force_continue')
    OUTDIR = CLA.get_argument('outdir')
    VERBOSE = CLA.get_argument('verbose')

    if MAX_ITER <= MIN_ITER:
        MAX_ITER = MIN_ITER + 1

    # Define the current directory.
    CURRENT_DIR = str(os.path.dirname(os.path.abspath(__file__)))

    # Start the program.
    PROGRAM = Main(current_dir=CURRENT_DIR,
                   eqtl_path=EQTL_PATH,
                   genotype_path=GENOTYPE_PATH,
                   genotype_na=GENOTYPE_NA,
                   expression_path=EXPRESSION_PATH,
                   tech_covariate_path=TECH_COVARIATE_PATH,
                   tech_covariate_with_inter_path=TECH_COVARIATE_WITH_INTERACTION_PATH,
                   covariate_path=COVARIATE_PATH,
                   sample_dataset_path=SAMPLE_DATASET_PATH,
                   min_dataset_size=MIN_DATASET_SIZE,
                   ieqtl_alpha=IEQTL_ALPHA,
                   call_rate=CALL_RATE,
                   hw_pval=HW_PVAL,
                   maf=MAF,
                   mgs=MGS,
                   n_components=N_COMPONENTS,
                   min_iter=MIN_ITER,
                   max_iter=MAX_ITER,
                   tol=TOL,
                   force_continue=FORCE_CONTINUE,
                   outdir=OUTDIR,
                   verbose=VERBOSE
                   )
    PROGRAM.start()