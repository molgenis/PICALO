#!/usr/bin/env python3

"""
File:         fast_eqtl_mapper.py
Created:      2023/05/19
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
import time
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.
from src.logger import Logger
from src.utilities import load_dataframe, save_dataframe
from src.statistics import inverse, fit, predict, calc_rss, fit_and_predict, calc_std, calc_p_value

# Metadata
__program__ = "Fast eQTL Mapper"
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
./fast_eqtl_mapper.py -h
    
./fast_eqtl_mapper.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLNoCovariates/simulation1/expression_table.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLNoCovariates \
    -verbose
    
./fast_eqtl_mapper.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction/simulation1/expression_table.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction/simulation1/tech_covariates_with_interaction_df.txt.gz \
    -exclude_covariate_interactions \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariateNoInteraction \
    -verbose
    
./fast_eqtl_mapper.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates/simulation1/expression_table.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates/simulation1/tech_covariates_with_interaction_df.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLOneCovariates \
    -verbose
    
./fast_eqtl_mapper.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates/simulation1/expression_table.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates/simulation1/tech_covariates_with_interaction_df.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-MaineQTLTwoCovariates \
    -verbose
    
./fast_eqtl_mapper.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate/simulation1/expression_table.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate/simulation1/PICs.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-RealInterceptAndGenotypeOneCovariate \
    -verbose
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.genotype_path = getattr(arguments, 'genotype')
        self.genotype_na = getattr(arguments, 'genotype_na')
        self.expression_path = getattr(arguments, 'expression')
        self.covariate_path = getattr(arguments, 'covariate')
        self.exclude_covariate_interactions = getattr(arguments, 'exclude_covariate_interactions')
        self.eqtl_alpha = getattr(arguments, 'eqtl_alpha')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "fast_eqtl_mapper", outfolder)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Initialize logger.
        logger = Logger(outdir=self.outdir,
                        verbose=getattr(arguments, 'verbose'),
                        clear_log=True)
        logger.print_arguments()
        self.log = logger.get_logger()

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
        parser.add_argument("-ge",
                            "--genotype",
                            type=str,
                            required=True,
                            help="The path to the genotype matrix.")
        parser.add_argument("-na",
                            "--genotype_na",
                            type=int,
                            required=False,
                            default=-1,
                            help="The genotype value that equals a missing "
                                 "value. Default: -1.")
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix.")
        parser.add_argument("-co",
                            "--covariate",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the covariate matrix (i.e. the"
                                 "matrix used as starting vector for the "
                                 "interaction term).")
        parser.add_argument("-exclude_covariate_interactions",
                            action='store_true',
                            help="Include covariate + covariate * genotype "
                                 "terms. Default: False.")
        parser.add_argument("-ea",
                            "--eqtl_alpha",
                            type=float,
                            required=False,
                            default=0.05,
                            help="The eQTL significance cut-off. "
                                 "Default: <=0.05.")
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
        parser.add_argument("-verbose",
                            action='store_true',
                            help="Enable verbose output. Default: False.")

        return parser.parse_args()

    def start(self):
        self.log.info("Starting program")
        self.print_arguments()

        ########################################################################

        self.log.info("Loading data")
        geno_df = load_dataframe(self.genotype_path, header=0, index_col=0, log=self.log)
        expr_df = load_dataframe(self.expression_path, header=0, index_col=0, log=self.log)

        # Check for nan values.
        if expr_df.isna().values.sum() > 0:
            self.log.error("\t  Expression file contains NaN values")
            exit()

        # Validate that the input data (still) matches.
        if geno_df.columns.tolist() != expr_df.columns.tolist():
                self.log.error("\tThe genotype file header does not match "
                               "the expression file header.")
                exit()

        cov_df = None
        if self.covariate_path is not None:
            cov_df = load_dataframe(self.covariate_path, header=0, index_col=0,
                                    log=self.log)
            if cov_df.isna().values.sum() > 0:
                self.log.error("\t  Covariate file contains NaN values")
                exit()
            if geno_df.columns.tolist() != cov_df.index.tolist():
                    self.log.error("\tThe genotype file header does not match "
                                   "the covariate file header.")
                    exit()
        self.log.info("")

        ########################################################################

        self.log.info("Transform to numpy matrices for speed")
        geno_m = geno_df.to_numpy(np.float64)
        expr_m = expr_df.to_numpy(np.float64)
        cov_m = None
        if cov_df is not None:
            cov_m = cov_df.to_numpy(np.float64)
        self.log.info("")

        # Fill the missing values with NaN.
        expr_m[geno_m == self.genotype_na] = np.nan
        geno_m[geno_m == self.genotype_na] = np.nan

        ########################################################################

        n_terms = 2
        n_covariates = 0
        if cov_m is not None:
            n_covariates = cov_m.shape[1]
            n_terms += n_covariates
            if not self.exclude_covariate_interactions:
                n_terms += n_covariates

        self.log.info("Mapping eQTLs")
        n_tests = geno_m.shape[0]
        ieqtl_results = np.empty((n_tests, (n_terms * 3) + 1), dtype=np.float64)
        last_print_time = None
        for eqtl_index in range(n_tests):
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 30 or (eqtl_index + 1) == n_tests:
                last_print_time = now_time
                self.log.info("\t{:,}/{:,} eQTLs analysed [{:.2f}%]".format(eqtl_index, n_tests - 1, (100 / (n_tests - 1)) * eqtl_index))

            # Get the genotype.
            genotype = geno_m[eqtl_index, :]

            # Construct the mask to remove missing values.
            mask = ~np.isnan(genotype)
            n = np.sum(mask)

            # Create the matrices. Note that only the first two columns
            # are filled in.
            X = np.empty((n, n_terms), np.float32)
            X[:, 0] = 1
            X[:, 1] = genotype[mask]

            if cov_m is not None:
                X[:, 2:(2 + n_covariates)] = cov_m[mask, :]

            if not self.exclude_covariate_interactions:
                for col_index in range(n_covariates):
                    X[:, (2 + n_covariates + col_index)] = X[:, (2 + col_index)] * X[:, 1]

            # Get the expression.
            y = expr_m[eqtl_index, mask]

            # Fit the model.
            inv_m = inverse(X)
            betas = fit(X=X, y=y, inv_m=inv_m)
            rss_alt = calc_rss(y=y, y_hat=predict(X=X, betas=betas))
            std = calc_std(rss=rss_alt, n=n, df=n_terms, inv_m=inv_m)

            # Calculate the p-values.
            ieqtl_p_values = np.empty(n_terms, np.float32)
            for col_index in range(n_terms):
                column_mask = np.ones(n_terms, bool)
                column_mask[col_index] = False

                rss_null = calc_rss(y=y, y_hat=fit_and_predict(X=X[:, column_mask], y=y))
                ieqtl_p_values[col_index] = calc_p_value(rss1=rss_null, rss2=rss_alt, df1=n_terms - 1, df2=n_terms, n=n)

            # Save results.
            ieqtl_results[eqtl_index, :] = np.hstack((np.array([n]), betas, std, ieqtl_p_values))
        self.log.info("")

        # Convert to pandas data frame.
        columns = ["intercept", "genotype"]
        if cov_m is not None:
            columns.extend(cov_df.columns)
            if not self.exclude_covariate_interactions:
                columns.extend(["{}Xgenotype".format(col) for col in cov_df.columns])

        df = pd.DataFrame(ieqtl_results, columns=["N"] +
                                                 ["beta-{}".format(col) for col in columns] +
                                                 ["std-{}".format(col) for col in columns] +
                                                 ["pvalue-{}".format(col) for col in columns])
        df.insert(0, "gene", expr_df.index.tolist())
        df.insert(0, "SNP", geno_df.index.tolist())

        # Print the number of interactions.
        self.log.info("Summarise results:")
        for col in columns:
            self.log.info("\tThe term '{}' was significant in {:,} models (FDR <{})".format(col, df.loc[df["pvalue-{}".format(col)] < 0.05, :].shape[0], self.eqtl_alpha))
        self.log.info("")

        ########################################################################

        self.log.info("Saving results")
        print(df)
        save_dataframe(df=df,
                       outpath=os.path.join(self.outdir, "eQTLSummaryStats.txt.gz"),
                       header=True,
                       index=False,
                       log=self.log)

        self.log.info("Finished")
        self.log.info("")

    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > Genotype input path: {}".format(self.genotype_path))
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Expression input path: {}".format(self.expression_path))
        self.log.info("  > Covariate input path: {}".format(self.covariate_path))
        self.log.info("  > Exclude covariate interaction: {}".format(self.exclude_covariate_interactions))
        self.log.info("  > eQTL alpha: <={}".format(self.eqtl_alpha))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()