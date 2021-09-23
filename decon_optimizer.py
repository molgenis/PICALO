#!/usr/bin/env python3

"""
File:         decon_optimizer.py
Created:      2020/11/16
Last Changed: 2021/09/23
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

"""
./decon_optimizer.py -h
"""

# Standard imports.

# Third party imports.

# Local application imports.
from src.main import Main
from src.cmd_line_arguments import CommandLineArguments

# Metadata
__program__ = "Deconvolution Optimizer"
__author__ = "Martijn Vochteloo"
__maintainer__ = "Martijn Vochteloo"
__email__ = "m.vochteloo@rug.nl"
__license__ = "GPLv3"
__version__ = 0.0
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
    EQTL_ALPHA = CLA.get_argument('eqtl_alpha')
    IEQTL_ALPHA = CLA.get_argument('ieqtl_alpha')
    CALL_RATE = CLA.get_argument('call_rate')
    HW_PVAL = CLA.get_argument('hardy_weinberg_pvalue')
    MAF = CLA.get_argument('minor_allele_frequency')
    MGS = CLA.get_argument('min_group_size')
    TOL = CLA.get_argument('tol')
    SLIDING_WINDOW_SIZE = CLA.get_argument('sliding_window_size')
    N_COMPONENTS = CLA.get_argument('n_components')
    MAX_ITER = CLA.get_argument('max_iter')
    VERBOSE = CLA.get_argument('verbose')
    OUTDIR = CLA.get_argument('outdir')

    # Start the program.
    PROGRAM = Main(eqtl_path=EQTL_PATH,
                   genotype_path=GENOTYPE_PATH,
                   genotype_na=GENOTYPE_NA,
                   expression_path=EXPRESSION_PATH,
                   tech_covariate_path=TECH_COVARIATE_PATH,
                   tech_covariate_with_inter_path=TECH_COVARIATE_WITH_INTERACTION_PATH,
                   covariate_path=COVARIATE_PATH,
                   sample_dataset_path=SAMPLE_DATASET_PATH,
                   eqtl_alpha=EQTL_ALPHA,
                   ieqtl_alpha=IEQTL_ALPHA,
                   call_rate=CALL_RATE,
                   hw_pval=HW_PVAL,
                   maf=MAF,
                   mgs=MGS,
                   tol=TOL,
                   sliding_window_size=SLIDING_WINDOW_SIZE,
                   n_components=N_COMPONENTS,
                   max_iter=MAX_ITER,
                   verbose=VERBOSE,
                   outdir=OUTDIR
                   )
    PROGRAM.start()