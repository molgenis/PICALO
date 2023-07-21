#!/usr/bin/env python3

"""
File:         fast_eqtl_mapper.py
Created:      2023/05/19
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import time
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy.special import ndtri
from statsmodels.regression.linear_model import OLS

# Local application imports.
from src.logger import Logger
from src.utilities import load_dataframe, save_dataframe
from src.statistics import inverse, fit, predict, calc_rss, fit_and_predict, calc_std, calc_p_value, calc_residuals, calc_pearsonr_vector

# Metadata
__program__ = "Fast eQTL Mapper"
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
./fast_eqtl_mapper.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.genotype_path = getattr(arguments, 'genotype')
        self.genotype_na = getattr(arguments, 'genotype_na')
        self.filter_variants = getattr(arguments, 'filter_variants')
        self.call_rate = getattr(arguments, 'call_rate')
        self.hw_pval = getattr(arguments, 'hardy_weinberg_pvalue')
        self.maf = getattr(arguments, 'minor_allele_frequency')
        self.mgs = getattr(arguments, 'min_group_size')
        self.expression_path = getattr(arguments, 'expression')
        self.covariate_path = getattr(arguments, 'covariate')
        self.force_normalise_covariates = getattr(arguments, 'force_normalise_covariates')
        self.exclude_covariate_interactions = getattr(arguments, 'exclude_covariate_interactions')
        self.ols = getattr(arguments, 'ols')
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
        parser.add_argument("-filter_variants",
                            action='store_true',
                            help="Filter the genotype variants. Default: False.")
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
        parser.add_argument("-force_normalise_covariates",
                            action='store_true',
                            help="Force normalise the covariates. "
                                 "Default: False.")
        parser.add_argument("-exclude_covariate_interactions",
                            action='store_true',
                            help="Include covariate + covariate * genotype "
                                 "terms. Default: False.")
        parser.add_argument("-ols",
                            action='store_true',
                            help="Use OLS from statsmodels instead of matrix"
                                 "inverse. Default: False.")
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
            self.log.info("\tLoading covariate matrix.")
            cov_df = load_dataframe(self.covariate_path, header=0, index_col=0,
                                    log=self.log)

            if (cov_df.shape[0] != geno_df.shape[1]) and (cov_df.shape[1] == geno_df.shape[1]):
                cov_df = cov_df.T

            print(cov_df)

            if self.force_normalise_covariates:
                self.log.info("\t  Force normalise covariate matrix.")
                cov_df = ndtri((cov_df.rank(axis=0, ascending=True) - 0.5) / cov_df.shape[0])

            print(geno_df)

            if cov_df.isna().values.sum() > 0:
                self.log.error("\t  Covariate file contains NaN values")
                exit()
            if geno_df.columns.tolist() != cov_df.index.tolist():
                    self.log.error("\t  The genotype file header does not match "
                                   "the covariate file header.")
                    exit()
        self.log.info("")

        self.log.info("Calculating genotype stats")
        geno_stats_df = self.calculate_genotype_stats(df=geno_df)
        save_dataframe(df=geno_stats_df,
                       outpath=os.path.join(self.outdir, "GenotypeStats.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)
        geno_stats_df.reset_index(drop=True, inplace=True)

        if self.filter_variants:
            self.log.info("Filtering variants.")
            cr_keep_mask = (geno_stats_df.loc[:, "CR"] >= self.call_rate).to_numpy(dtype=bool)
            n_keep_mask = (geno_stats_df.loc[:, "N"] >= 6).to_numpy(dtype=bool)
            mgs_keep_mask = (geno_stats_df.loc[:, "min GS"] >= self.mgs).to_numpy(dtype=bool)
            hwpval_keep_mask = (geno_stats_df.loc[:, "HW pval"] >= self.hw_pval).to_numpy(dtype=bool)
            maf_keep_mask = (geno_stats_df.loc[:, "MAF"] > self.maf).to_numpy(dtype=bool)
            combined_keep_mask = cr_keep_mask & n_keep_mask & mgs_keep_mask & hwpval_keep_mask & maf_keep_mask
            geno_n_skipped = np.size(combined_keep_mask) - np.sum(combined_keep_mask)
            if geno_n_skipped > 0:
                self.log.warning("\t  {:,} eQTL(s) failed the call rate threshold".format(np.size(cr_keep_mask) - np.sum(cr_keep_mask)))
                self.log.warning("\t  {:,} eQTL(s) failed the sample size threshold".format(np.size(n_keep_mask) - np.sum(n_keep_mask)))
                self.log.warning("\t  {:,} eQTL(s) failed the min. genotype group size threshold".format(np.size(mgs_keep_mask) - np.sum(mgs_keep_mask)))
                self.log.warning("\t  {:,} eQTL(s) failed the Hardy-Weinberg p-value threshold".format(np.size(hwpval_keep_mask) - np.sum(hwpval_keep_mask)))
                self.log.warning("\t  {:,} eQTL(s) failed the MAF threshold".format(np.size(maf_keep_mask) - np.sum(maf_keep_mask)))
                self.log.warning("\t  ----------------------------------------")
                self.log.warning("\t  {:,} eQTL(s) are discarded in total".format(geno_n_skipped))

            geno_stats_df = geno_stats_df.loc[combined_keep_mask, :]
            geno_stats_df.reset_index(drop=True, inplace=True)
            geno_df = geno_df.loc[combined_keep_mask, :]
            expr_df = expr_df.loc[combined_keep_mask, :]

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
        ieqtl_results = np.empty((n_tests, (n_terms * 3) + 4), dtype=np.float64)
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

            # Save results.
            if self.ols:
                ieqtl_results[eqtl_index, :] = self.ols_model(y=y,
                                                              X=X,
                                                              n=n)
            else:
                ieqtl_results[eqtl_index, :] = self.matrix_model(y=y,
                                                                 X=X,
                                                                 n=n,
                                                                 n_terms=n_terms)

        self.log.info("")

        # Convert to pandas data frame.
        columns = ["intercept", "genotype"]
        if cov_m is not None:
            columns.extend(cov_df.columns)
            if not self.exclude_covariate_interactions:
                columns.extend(["{}Xgenotype".format(col) for col in cov_df.columns])

        df = pd.DataFrame(ieqtl_results, columns=["N"] +
                                                 ["beta-{}".format(col) for col in columns] +
                                                 ["beta-noise"] +
                                                 ["std-{}".format(col) for col in columns] +
                                                 ["std-noise"] +
                                                 ["pvalue-{}".format(col) for col in columns] +
                                                 ["r-squared"])
        df = pd.concat([df, geno_stats_df[["CR", "HW pval", "MAF"]]], axis=1)
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

    def calculate_genotype_stats(self, df):
        rounded_m = df.to_numpy(dtype=np.float64)
        rounded_m = np.rint(rounded_m)

        # Calculate the total samples that are not NaN.
        nan = np.sum(rounded_m == self.genotype_na, axis=1)
        n = rounded_m.shape[1] - nan

        # Calculate the call rate
        cr = n / rounded_m.shape[1]

        # Count the genotypes.
        zero_a = np.sum(rounded_m == 0, axis=1)
        one_a = np.sum(rounded_m == 1, axis=1)
        two_a = np.sum(rounded_m == 2, axis=1)

        # Calculate the smallest genotype group size.
        sgz = np.minimum.reduce([zero_a, one_a, two_a])

        # Calculate the Hardy-Weinberg p-value.
        hwe_pvalues_a = self.calc_hwe_pvalue(obs_hets=one_a, obs_hom1=zero_a, obs_hom2=two_a)

        # Count the alleles.
        allele1_a = (zero_a * 2) + one_a
        allele2_a = (two_a * 2) + one_a

        # Calculate the MAF.
        maf = np.minimum(allele1_a, allele2_a) / (allele1_a + allele2_a)

        # Determine which allele is the minor allele.
        allele_m = np.column_stack((allele1_a, allele2_a))
        ma = np.argmin(allele_m, axis=1) * 2

        # Construct output data frame.
        output_df = pd.DataFrame({"N": n,
                                  "CR": cr,
                                  "NaN": nan,
                                  "0": zero_a,
                                  "1": one_a,
                                  "2": two_a,
                                  "min GS": sgz,
                                  "HW pval": hwe_pvalues_a,
                                  "allele1": allele1_a,
                                  "allele2": allele2_a,
                                  "MA": ma,
                                  "MAF": maf
                                  }, index=df.index)
        del rounded_m, allele_m

        return output_df


    @staticmethod
    def calc_hwe_pvalue(obs_hets, obs_hom1, obs_hom2):
        """
        exact SNP test of Hardy-Weinberg Equilibrium as described in Wigginton,
        JE, Cutler, DJ, and Abecasis, GR (2005) A Note on Exact Tests of
        Hardy-Weinberg Equilibrium. AJHG 76: 887-893

        Adapted by M.Vochteloo to work on matrices.
        """
        if not 'int' in str(obs_hets.dtype) or not 'int' in str(obs_hets.dtype) or not 'int' in str(obs_hets.dtype):
            obs_hets = np.rint(obs_hets)
            obs_hom1 = np.rint(obs_hom1)
            obs_hom2 = np.rint(obs_hom2)

        # Force homc to be the max and homr to be the min observed genotype.
        obs_homc = np.maximum(obs_hom1, obs_hom2)
        obs_homr = np.minimum(obs_hom1, obs_hom2)

        # Calculate some other stats we need.
        rare_copies = 2 * obs_homr + obs_hets
        l_genotypes = obs_hets + obs_homc + obs_homr
        n = np.size(obs_hets)

        # Get the distribution midpoint.
        mid = np.rint(rare_copies * (2 * l_genotypes - rare_copies) / (2 * l_genotypes)).astype(np.int)
        mid[mid % 2 != rare_copies % 2] += 1

        # Calculate the start points for the evaluation.
        curr_homr = (rare_copies - mid) / 2
        curr_homc = l_genotypes - mid - curr_homr

        # Calculate the left side.
        left_steps = np.floor(mid / 2).astype(int)
        max_left_steps = np.max(left_steps)
        left_het_probs = np.zeros((n, max_left_steps + 1), dtype=np.float64)
        left_het_probs[:, 0] = 1
        for i in np.arange(0, max_left_steps, 1, dtype=np.float64):
            prob = left_het_probs[:, int(i)] * (mid - (i * 2)) * ((mid - (i * 2)) - 1.0) / (4.0 * (curr_homr + i + 1.0) * (curr_homc + i + 1.0))
            prob[mid - (i * 2) <= 0] = 0
            left_het_probs[:, int(i) + 1] = prob

        # Calculate the right side.
        right_steps = np.floor((rare_copies - mid) / 2).astype(int)
        max_right_steps = np.max(right_steps)
        right_het_probs = np.zeros((n, max_right_steps + 1), dtype=np.float64)
        right_het_probs[:, 0] = 1
        for i in np.arange(0, max_right_steps, 1, dtype=np.float64):
            prob = right_het_probs[:, int(i)] * 4.0 * (curr_homr - i) * (curr_homc - i) / (((i * 2) + mid + 2.0) * ((i * 2) + mid + 1.0))
            prob[(i * 2) + mid >= rare_copies] = 0
            right_het_probs[:, int(i) + 1] = prob

        # Combine the sides.
        het_probs = np.hstack((np.flip(left_het_probs, axis=1), right_het_probs[:, 1:]))

        # Normalize.
        sum = np.sum(het_probs, axis=1)
        het_probs = het_probs / sum[:, np.newaxis]

        # Replace values higher then probability of obs_hets with 0.
        threshold_col_a = (max_left_steps - left_steps) + np.floor(obs_hets / 2).astype(int)
        threshold = np.array([het_probs[i, threshold_col] for i, threshold_col in enumerate(threshold_col_a)])
        het_probs[het_probs > threshold[:, np.newaxis]] = 0

        # Calculate the p-values.
        p_hwe = np.sum(het_probs, axis=1)
        p_hwe[p_hwe > 1] = 1

        return p_hwe

    @staticmethod
    def matrix_model(y, X, n, n_terms):
        # Fit the model.
        inv_m = inverse(X)
        betas = fit(X=X, y=y, inv_m=inv_m)
        y_hat = predict(X=X, betas=betas)
        rss_alt = calc_rss(y=y, y_hat=y_hat)
        std = calc_std(rss=rss_alt, n=n, df=n_terms, inv_m=inv_m)

        # Calculate residuals.
        res = calc_residuals(y=y, y_hat=y_hat)

        # Calculate R2.
        pearsonr = calc_pearsonr_vector(x=y, y=y_hat)
        r_squared = pearsonr * pearsonr

        # Calculate the p-values.
        p_values = np.empty(n_terms, np.float32)
        for col_index in range(n_terms):
            column_mask = np.ones(n_terms, bool)
            column_mask[col_index] = False

            rss_null = calc_rss(y=y,
                                y_hat=fit_and_predict(X=X[:, column_mask],
                                                      y=y))
            p_values[col_index] = calc_p_value(rss1=rss_null,
                                               rss2=rss_alt,
                                               df1=n_terms - 1,
                                               df2=n_terms,
                                               n=n)

        return np.hstack((np.array([n]),
                          betas,
                          np.mean(res),
                          std,
                          np.std(res),
                          p_values,
                          np.array([r_squared])))

    @staticmethod
    def ols_model(y, X, n):
        model = OLS(y, X).fit()
        return np.hstack((np.array([n]),
                          model.params,
                          np.mean(model.resid),
                          model.bse,
                          np.std(model.resid),
                          model.pvalues,
                          np.array([model.rsquared])))


    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > Genotype input path: {}".format(self.genotype_path))
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Filter variants: {}".format(self.filter_variants))
        if self.filter_variants:
            self.log.info("  > SNP call rate: >{}".format(self.call_rate))
            self.log.info("  > Hardy-Weinberg p-value: >={}".format(self.hw_pval))
            self.log.info("  > MAF: >{}".format(self.maf))
            self.log.info("  > Minimal group size: >={}".format(self.mgs))
        self.log.info("  > Expression input path: {}".format(self.expression_path))
        self.log.info("  > Covariate input path: {}".format(self.covariate_path))
        self.log.info("  > Force normalise covariate: {}".format(self.force_normalise_covariates))
        self.log.info("  > Exclude covariate interaction: {}".format(self.exclude_covariate_interactions))
        self.log.info("  > OLS: {}".format(self.ols))
        self.log.info("  > eQTL alpha: <={}".format(self.eqtl_alpha))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()