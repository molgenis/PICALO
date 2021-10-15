"""
File:         main.py
Created:      2020/11/16
Last Changed: 2021/10/15
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
import pandas as pd
import numpy as np

# Local application imports.
from src.logger import Logger
from src.force_normaliser import ForceNormaliser
from src.objects.data import Data
from src.inter_optimizer import InteractionOptimizer
from src.statistics import remove_covariates, fit_and_predict
from src.utilities import save_dataframe


class Main:
    def __init__(self, eqtl_path, genotype_path, genotype_na, expression_path,
                 tech_covariate_path, tech_covariate_with_inter_path,
                 covariate_path, sample_dataset_path,
                 eqtl_alpha, ieqtl_alpha, call_rate, hw_pval, maf, mgs, tol,
                 sliding_window_size, n_components,  max_iter, verbose, outdir):
        # Safe arguments.
        self.genotype_na = genotype_na
        self.eqtl_alpha = eqtl_alpha
        self.ieqtl_alpha = ieqtl_alpha
        self.call_rate = call_rate
        self.hw_pval = hw_pval
        self.maf = maf
        self.mgs = mgs
        self.tol = tol
        self.sliding_window_size = sliding_window_size
        self.n_components = n_components
        self.max_iter = max_iter

        # Other global variables.
        self.min_dataset_sample_size = 30

        # Define the current directory.
        current_dir = str(Path(__file__).parent.parent)

        # Prepare an output directory.
        self.outdir = os.path.join(current_dir, "output", outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Initialize logger.
        logger = Logger(outdir=self.outdir,
                        verbose=verbose,
                        clear_log=True)
        logger.print_arguments()
        self.log = logger.get_logger()

        # Initialize data object.
        self.data = Data(eqtl_path=eqtl_path,
                         genotype_path=genotype_path,
                         expression_path=expression_path,
                         tech_covariate_path=tech_covariate_path,
                         tech_covariate_with_inter_path=tech_covariate_with_inter_path,
                         covariate_path=covariate_path,
                         sample_dataset_path=sample_dataset_path,
                         log=self.log)
        self.data.print_arguments()

    def start(self):
        self.log.info("Starting program")
        self.print_arguments()

        ########################################################################

        self.log.info("Loading eQTL data and filter on FDR values of the "
                      "main eQTL effect")
        eqtl_df = self.data.get_eqtl_df()
        eqtl_fdr_keep_mask = (eqtl_df["FDR"] < self.eqtl_alpha).to_numpy(dtype=bool)
        eqtl_signif_df = eqtl_df.loc[eqtl_fdr_keep_mask, :]
        eqtl_signif_df.reset_index(drop=True, inplace=True)

        eqtl_fdr_n_skipped = np.size(eqtl_fdr_keep_mask) - np.sum(eqtl_fdr_keep_mask)
        if eqtl_fdr_n_skipped > 0:
            self.log.warning("\t{:,} eQTLs have been skipped due to "
                             "FDR cut-off".format(eqtl_fdr_n_skipped))
        self.log.info("")

        ########################################################################

        self.log.info("Loading genotype data and dataset info")
        skiprows = None
        if eqtl_fdr_n_skipped > 0:
            skiprows = [x+1 for x in eqtl_df.index[~eqtl_fdr_keep_mask]]
        geno_df = self.data.get_geno_df(skiprows=skiprows, nrows=max(eqtl_signif_df.index)+1)
        std_df = self.data.get_std_df()

        # Validate that the input data matches.
        self.validate_data(std_df=std_df,
                           geno_df=geno_df)

        self.log.info("\tChecking dataset sample sizes")
        # Check if each dataset has the minimal number of samples.
        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        max_dataset_length = np.max([len(str(dataset[0])) for dataset in dataset_sample_counts])
        for dataset, sample_size in dataset_sample_counts:
            self.log.info("\t  {:{}s}  {:,} samples".format(dataset, max_dataset_length, sample_size))
        if dataset_sample_counts[-1][1] < self.min_dataset_sample_size:
            self.log.warning("\t\tOne or more datasets have a smaller sample "
                             "size than recommended. Consider excluded these")
        self.log.info("")

        # Construct dataset df.
        dataset_df = self.construct_dataset_df(std_df=std_df)

        self.log.info("\tCalculating genotype call rate per dataset")
        geno_df, call_rate_df = self.calculate_call_rate(geno_df=geno_df,
                                                         dataset_df=dataset_df)
        call_rate_n_skipped = (call_rate_df.min(axis=1) < self.call_rate).sum()
        if call_rate_n_skipped > 0:
            self.log.warning("\t  {:,} eQTLs have had dataset(s) filled with NaN "
                             "values due to call rate threshold ".format(call_rate_n_skipped))

        save_dataframe(df=call_rate_df,
                       outpath=os.path.join(self.outdir, "call_rate.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)
        self.log.info("")

        self.log.info("\tCalculating genotype stats for inclusing criteria")
        geno_stats_df = self.calculate_genotype_stats(df=geno_df)

        # Checking which eQTLs pass the requirements
        n_keep_mask = (geno_stats_df.loc[:, "N"] >= 6).to_numpy()
        mgs_keep_mask = (geno_stats_df.loc[:, "min GS"] >= self.mgs).to_numpy()
        hwpval_keep_mask = (geno_stats_df.loc[:, "HW pval"] >= self.hw_pval).to_numpy()
        maf_keep_mask = (geno_stats_df.loc[:, "MAF"] > self.maf).to_numpy()
        combined_keep_mask = n_keep_mask & mgs_keep_mask & hwpval_keep_mask & maf_keep_mask
        geno_n_skipped = np.size(combined_keep_mask) - np.sum(combined_keep_mask)
        if geno_n_skipped > 0:
            self.log.warning("\t  {:,} eQTL(s) failed the sample size threshold".format(np.size(n_keep_mask) - np.sum(n_keep_mask)))
            self.log.warning("\t  {:,} eQTL(s) failed the min. genotype group size threshold".format(np.size(mgs_keep_mask) - np.sum(mgs_keep_mask)))
            self.log.warning("\t  {:,} eQTL(s) failed the Hardy-Weinberg p-value threshold".format(np.size(hwpval_keep_mask) - np.sum(hwpval_keep_mask)))
            self.log.warning("\t  {:,} eQTL(s) failed the MAF threshold".format(np.size(maf_keep_mask) - np.sum(maf_keep_mask)))
            self.log.warning("\t  ----------------------------------------")
            self.log.warning("\t  {:,} eQTL(s) are discarded in total".format(geno_n_skipped))

        save_dataframe(df=geno_stats_df,
                       outpath=os.path.join(self.outdir, "genotype_stats.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)
        self.log.info("")

        # Select rows that meet requirements.
        eqtl_signif_df = eqtl_signif_df.loc[combined_keep_mask, :]
        geno_df = geno_df.loc[combined_keep_mask, :]

        # Combine the skip masks.
        keep_mask = np.copy(eqtl_fdr_keep_mask)
        keep_mask[eqtl_fdr_keep_mask] = combined_keep_mask

        del call_rate_df, geno_stats_df, eqtl_fdr_keep_mask, n_keep_mask, mgs_keep_mask, hwpval_keep_mask, maf_keep_mask, combined_keep_mask

        ########################################################################

        self.log.info("Loading other data")
        self.log.info("\tIncluded {:,} eQTLs".format(np.sum(keep_mask)))
        skiprows = None
        if (eqtl_fdr_n_skipped + geno_n_skipped) > 0:
            skiprows = [x+1 for x in eqtl_df.index[~keep_mask]]
        expr_df = self.data.get_expr_df(skiprows=skiprows, nrows=max(eqtl_signif_df.index)+1)
        covs_df = self.data.get_covs_df()

        # Check for nan values.
        if covs_df.isna().values.sum() > 0:
            self.log.error("\t  Covariate file contains nan values")
            exit()

        # Transpose if need be.
        if covs_df.shape[0] == geno_df.shape[1]:
            self.log.warning("\t  Transposing covariate matrix")
            covs_df = covs_df.T

        covariates = covs_df.index.tolist()
        self.log.info("\t  Covariates: {}".format(", ".join(covariates)))

        # Validate that the input data (still) matches.
        self.validate_data(std_df=std_df,
                           eqtl_df=eqtl_signif_df,
                           geno_df=geno_df,
                           expr_df=expr_df,
                           covs_df=covs_df)
        samples = std_df.iloc[:, 0].to_numpy(object)
        self.log.info("")

        ########################################################################

        self.log.info("Transform to numpy matrices for speed")
        eqtl_m = eqtl_signif_df[["SNPName", "ProbeName"]].to_numpy(object)
        geno_m = geno_df.to_numpy(np.float64)
        expr_m = expr_df.to_numpy(np.float64)
        dataset_m = dataset_df.to_numpy(np.uint8)
        covs_m = covs_df.to_numpy(np.float64)
        self.log.info("")
        del eqtl_df, geno_df, expr_df, dataset_df, covs_df

        ########################################################################

        self.log.info("Loading technical covariates")
        tcov_df = self.data.get_tcov_df()
        get_tcov_inter_df = self.data.get_tcov_inter_df()

        tcov_m, tcov_labels = self.load_tech_cov(df=tcov_df, name="tech. cov. without interaction", std_df=std_df)
        tcov_inter_m, tcov_inter_labels = self.load_tech_cov(df=get_tcov_inter_df, name="tech. cov. with interaction", std_df=std_df)
        self.log.info("")

        # Create the correction matrices.
        corr_m = np.copy(dataset_m[:, :(dataset_m.shape[1] - 1)])
        corr_inter_m = np.copy(dataset_m[:, :(dataset_m.shape[1] - 1)])

        if tcov_m is not None:
            corr_m = np.hstack((corr_m, tcov_m))

        if tcov_inter_m is not None:
            corr_m = np.hstack((corr_m, tcov_inter_m))
            corr_inter_m = np.hstack((corr_inter_m, tcov_inter_m))

        ########################################################################

        self.log.info("Force normalising covariate matrix")
        # TODO uncomment this?
        # covs_m = np.vstack((covs_m, tcov_m.T[:10, :]))
        # covariates = covariates + tcov_labels[:10]
        covs_m = tcov_m.T[:10, :]
        covariates = tcov_labels[:10]
        fn = ForceNormaliser(dataset_m=dataset_m, samples=samples, log=self.log)
        covs_m = fn.process(covs_m)
        self.log.info("")
        del fn

        ########################################################################

        self.log.info("Starting identifying interaction components")

        io = InteractionOptimizer(covariates=covariates,
                                  dataset_m=dataset_m,
                                  samples=samples,
                                  genotype_na=self.genotype_na,
                                  ieqtl_alpha=self.ieqtl_alpha,
                                  max_iter=self.max_iter,
                                  tol=self.tol,
                                  sliding_window_size=self.sliding_window_size,
                                  log=self.log)

        pic_m = np.empty((self.n_components, np.size(samples)), dtype=np.float64)
        n_components_performed = 0
        pic_a = None
        stop = False
        for comp_count in range(self.n_components):
            if stop:
                self.log.warning("Last component did not converge, stop "
                                 "further identification of components")
                break

            self.log.info("\tIdentifying PIC {}".format(comp_count + 1))

            # Prepare component output directory.
            comp_outdir = os.path.join(self.outdir, "PIC{}".format(comp_count + 1))
            if not os.path.exists(comp_outdir):
                os.makedirs(comp_outdir)

            # Add component to the base matrix.
            if pic_a is not None:
                if corr_m is not None:
                    corr_m = np.hstack((corr_m, pic_a[:, np.newaxis]))
                else:
                    corr_m = pic_a[:, np.newaxis]

                if corr_inter_m is not None:
                    corr_inter_m = np.hstack((corr_inter_m, pic_a[:, np.newaxis]))
                else:
                    corr_inter_m = pic_a[:, np.newaxis]

            component_path = os.path.join(comp_outdir, "component.npy")
            if os.path.exists(component_path):
                with open(component_path, 'rb') as f:
                    pic_a = np.load(f)
                f.close()
                pic_m[comp_count, :] = pic_a
            else:
                self.log.info("\t  Optimizing interaction component")

                # Remove tech. covs. + components from expression matrix.
                self.log.info("\t  Correcting expression matrix")
                comp_expr_m = remove_covariates(y_m=expr_m,
                                                X_m=corr_m,
                                                X_inter_m=corr_inter_m,
                                                inter_m=geno_m,
                                                log=self.log)

                # Optimize the cell fractions in X iterations.
                pic_a, stop = io.process(eqtl_m=eqtl_m,
                                         geno_m=geno_m,
                                         expr_m=comp_expr_m,
                                         covs_m=covs_m,
                                         corr_m=corr_m,
                                         corr_inter_m=corr_inter_m,
                                         outdir=comp_outdir)

                # Save.
                pic_m[comp_count, :] = pic_a

                with open(component_path, 'wb') as f:
                    np.save(f, pic_a)
                f.close()

                # Increment counter.
                n_components_performed += 1

        components_df = pd.DataFrame(pic_m[:n_components_performed, :],
                                     index=["PIC{}".format(i+1) for i in range(n_components_performed)],
                                     columns=samples)

        save_dataframe(df=components_df,
                       outpath=os.path.join(self.outdir, "components.txt.gz"),
                       header=True,
                       index=True)

        # ########################################################################
        # Model the cell fractions as a linear combination of components.
        # TODO not working, fix this.
        #
        # self.log.info("Modelling optimized covariates")
        # optimized_m = self.model_optimized_covs(covs_m=covs_m, components_m=pic_m)
        #
        # optimized_df = pd.DataFrame(optimized_m,
        #                             index=samples,
        #                             columns=covariates)
        # save_dataframe(df=optimized_df,
        #                outpath=os.path.join(self.outdir,
        #                                     "optimized_covariates.txt.gz"),
        #                header=True,
        #                index=True)
        # self.log.info("")
        #
        # ########################################################################

        self.log.info("Finished")
        self.log.info("")

    def validate_data(self, std_df, eqtl_df=None, geno_df=None,
                      expr_df=None, covs_df=None, tcovs_df=None):

        # Check the samples.
        samples = std_df.iloc[:, 0].values.tolist()
        if geno_df is not None and geno_df.columns.tolist() != samples:
                self.log.error("\tThe genotype file header does not match "
                               "the sample-to-dataset link file")
                exit()

        if expr_df is not None and expr_df.columns.tolist() != samples:
                self.log.error("\tThe expression file header does not match "
                               "the sample-to-dataset link file")
                exit()

        if covs_df is not None and covs_df.columns.tolist() != samples:
                self.log.error("\tThe covariates file header does not match "
                               "the sample-to-dataset link file")
                exit()

        if tcovs_df is not None and tcovs_df.index.tolist() != samples:
                self.log.error("\tThe technical covariates file indices does "
                               "not match the sample-to-dataset link file")
                exit()

        # Check the eQTLs.
        if eqtl_df is not None:
            snp_reference = eqtl_df["SNPName"].values.tolist()
            probe_reference = eqtl_df["ProbeName"].values.tolist()

            if geno_df is not None and geno_df.index.tolist() != snp_reference:
                self.log.error("The genotype file indices do not match the "
                               "eQTL file")
                exit()

            if expr_df is not None and expr_df.index.tolist() != probe_reference:
                self.log.error("The expression file indices do not match the "
                               "eQTL file")
                exit()

    def calculate_call_rate(self, geno_df, dataset_df):
        # Calculate the fraction of NaNs per dataset.
        call_rate_df = pd.DataFrame(np.nan, index=geno_df.index, columns=["{} CR".format(dataset) for dataset in dataset_df.columns])
        for dataset, sample_mask in dataset_df.T.iterrows():
            call_rate_s = (geno_df.loc[:, sample_mask.astype(bool)] != self.genotype_na).astype(int).sum(axis=1) / sample_mask.sum()
            call_rate_df.loc[:, "{} CR".format(dataset)] = call_rate_s

            # If the call rate is too high, replace all genotypes of that
            # dataset with missing.
            row_mask = call_rate_s < self.call_rate
            geno_df.loc[row_mask, sample_mask.astype(bool)] = self.genotype_na

        return geno_df, call_rate_df

    def calculate_genotype_stats(self, df):
        rounded_m = df.to_numpy(dtype=np.float64)
        rounded_m = np.rint(rounded_m)

        # Calculate the total samples that are not NaN.
        nan = np.sum(rounded_m == self.genotype_na, axis=1)
        n = rounded_m.shape[1] - nan

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
                                  "NaN": nan,
                                  "0": zero_a,
                                  "1": one_a,
                                  "2": two_a,
                                  "min GS": sgz,
                                  "HW pval": hwe_pvalues_a,
                                  "allele 1": allele1_a,
                                  "allele 2": allele2_a,
                                  "MA": ma,
                                  "MAF": maf,
                                  })
        del rounded_m, allele_m

        return output_df

    @staticmethod
    def calc_hwe_pvalue(obs_hets, obs_hom1, obs_hom2):
        """
        exact SNP test of Hardy-Weinberg Equilibrium as described in Wigginton,
        JE, Cutler, DJ, and Abecasis, GR (2005) A Note on Exact Tests of
        Hardy-Weinberg Equilibrium. AJHG 76: 887-893
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
    def construct_dataset_df(std_df):
        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]

        dataset_df = pd.DataFrame(0, index=std_df.iloc[:, 0], columns=datasets)
        for dataset in datasets:
            dataset_df.loc[(std_df.iloc[:, 1] == dataset).values, dataset] = 1
        dataset_df.index.name = "-"

        return dataset_df

    def load_tech_cov(self, df, name, std_df):
        if df is None:
            return None, []

        n_samples = std_df.shape[0]

        self.log.info("\tWorking on technical covariates matrix matrix '{}'".format(name))

        # Check for nan values.
        if df.isna().values.sum() > 0:
            self.log.error("\t  Matrix contains nan values")
            exit()

        # Put the samples on the rows.
        if df.shape[1] == n_samples:
            self.log.warning("\t  Transposing matrix")
            df = df.T

        # Check if valid.
        self.validate_data(std_df=std_df,
                           tcovs_df=df)

        # Check for variables with zero std.
        variance_mask = df.std(axis=0) != 0
        n_zero_variance = variance_mask.shape[0] - variance_mask.sum()
        if n_zero_variance > 0:
            self.log.warning("\t  Dropping {} rows with 0 variance".format(
                n_zero_variance))
            df = df.loc[:, variance_mask]

        # Convert to numpy.
        m = df.to_numpy(np.float64)
        columns = df.columns.tolist()
        del df

        covariates = columns
        self.log.info("\t  Technical covariates [{}]: {}".format(len(covariates), ", ".join(covariates)))

        return m, covariates

    @staticmethod
    def model_optimized_covs(covs_m, components_m):
        """
        TODO not working, fix this.
        """
        X = np.hstack((np.ones((components_m.shape[1], 1)), components_m.T))

        # Model the cell fractions as linear combinations of components.
        optimized_m = np.empty_like(covs_m, dtype=np.float64)
        for cov_index in range(covs_m.shape[1]):
            optimized_m[cov_index, :] = fit_and_predict(X=X, y=covs_m[cov_index, :])

        return optimized_m

    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > eQTL alpha: <{}".format(self.eqtl_alpha))
        self.log.info("  > SNP call rate: >{}".format(self.call_rate))
        self.log.info("  > Hardy-Weinberg p-value: >{}".format(self.hw_pval))
        self.log.info("  > MAF: >{}".format(self.maf))
        self.log.info("  > Minimal group size: >{}".format(self.mgs))
        self.log.info("  > ieQTL alpha: <{}".format(self.ieqtl_alpha))
        self.log.info("  > N components: {}".format(self.n_components))
        self.log.info("  > Max iterations: {}".format(self.max_iter))
        self.log.info("  > Tolerance: {}".format(self.tol))
        self.log.info("  > Sliding window size: {}".format(self.sliding_window_size))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")




