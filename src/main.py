"""
File:         main.py
Created:      2020/11/16
Last Changed: 2021/08/30
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
from src.interaction_optimizer import InteractionOptimizer
from src.statistics import remove_covariates, fit_and_predict
from src.utilities import save_dataframe


class Main:
    def __init__(self, eqtl_path, genotype_path, genotype_na, expression_path,
                 tech_covariate_path, tech_covariate_with_inter_path,
                 covariate_path, sample_dataset_path, eqtl_alpha, ieqtl_alpha, maf,
                 mgs, n_components, n_iterations, outdir):
        # Safe arguments.
        self.genotype_na = genotype_na
        self.eqtl_alpha = eqtl_alpha
        self.ieqtl_alpha = ieqtl_alpha
        self.maf = maf
        self.mgs = mgs
        self.n_components = n_components
        self.n_iterations = n_iterations

        # Define the current directory.
        current_dir = str(Path(__file__).parent.parent)

        # Prepare an output directory.
        self.outdir = os.path.join(current_dir, "output", outdir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Initialize logger.
        logger = Logger(outdir=self.outdir, clear_log=True)
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
        self.log.info("Starting program.")
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
            self.log.warning("\t{} eQTLs have been skipped due to "
                             "FDR cut-off.".format(eqtl_fdr_n_skipped))
        self.log.info("")

        ########################################################################

        self.log.info("Loading genotype data")
        skiprows = None
        if eqtl_fdr_n_skipped > 0:
            skiprows = [x+1 for x in eqtl_df.index[~eqtl_fdr_keep_mask]]
        geno_df = self.data.get_geno_df(skiprows=skiprows, nrows=max(eqtl_signif_df.index)+1)

        self.log.info("Checking MAf and SNP group sizes of genotype matrix.")
        eqtl_stats = geno_df.apply(self.calculate_eqtl_stats, 1)
        eqtl_stats.reset_index(drop=True, inplace=True)
        eqtl_stats["FDR"] = eqtl_signif_df["FDR"]

        # Checking which eQTLs pass the requirements
        geno_keep_mask = ((eqtl_stats.loc[:, "MAF"] > self.maf) | (eqtl_stats.loc[:, "MGS"] > self.mgs)).to_numpy(dtype=bool)
        geno_n_skipped = np.size(geno_keep_mask) - np.sum(geno_keep_mask)
        if geno_n_skipped > 0:
            self.log.warning("\t{} eQTLs have been skipped due to "
                             "FDR cut-off.".format(geno_n_skipped))

        # Select rows that meet requirements.
        eqtl_signif_df = eqtl_signif_df.loc[geno_keep_mask, :]
        geno_df = geno_df.loc[geno_keep_mask, :]

        # Combine the skip masks.
        keep_mask = np.copy(eqtl_fdr_keep_mask)
        keep_mask[eqtl_fdr_keep_mask] = geno_keep_mask
        self.log.info("")

        del eqtl_stats, eqtl_fdr_keep_mask, geno_keep_mask

        ########################################################################

        self.log.info("Loading other data")
        self.log.info("\tIncluded {} eQTLs.".format(np.sum(keep_mask)))
        skiprows = None
        if (eqtl_fdr_n_skipped + geno_n_skipped) > 0:
            skiprows = [x+1 for x in eqtl_df.index[~keep_mask]]
        expr_df = self.data.get_expr_df(skiprows=skiprows, nrows=max(eqtl_signif_df.index)+1)
        dataset_df = self.data.get_dataset_df()
        datasets = dataset_df.columns.tolist()
        self.log.info("\t  Datasets: {}".format(", ".join(datasets)))
        covs_df = self.data.get_covs_df()

        # Check for nan values.
        if covs_df.isna().values.sum() > 0:
            print("\t  Covariate file contains nan values.")
            exit()

        # Transpose if need be.
        if covs_df.shape[0] == geno_df.shape[1] and covs_df.shape[0] == expr_df.shape[1]:
            self.log.warning("\t  Transposing covariate matrix.")
            covs_df = covs_df.T

        covariates = covs_df.index.tolist()
        self.log.info("\t  Covariates: {}".format(", ".join(covariates)))
        self.log.info("")

        ########################################################################

        self.log.info("Validating input.")
        self.validate_data(eqtl_df=eqtl_signif_df,
                           geno_df=geno_df,
                           expr_df=expr_df,
                           dataset_df=dataset_df,
                           covs_df=covs_df)
        samples = geno_df.columns.to_numpy()
        self.log.info("")

        ########################################################################

        self.log.info("Transform to numpy matrices for speed.")
        eqtl_m = eqtl_signif_df[["SNPName", "ProbeName"]].to_numpy(object)
        geno_m = geno_df.to_numpy(np.float64)
        expr_m = expr_df.to_numpy(np.float64)
        dataset_m = dataset_df.to_numpy(np.uint8)
        covs_m = covs_df.to_numpy(np.float64)
        self.log.info("")
        del eqtl_df, geno_df, expr_df, dataset_df, covs_df

        ########################################################################

        self.log.info("Loading technical covariates.")
        tcov_df = self.data.get_tcov_df()
        get_tcov_inter_df = self.data.get_tcov_inter_df()

        tcov_m, tcov_labels = self.load_tech_cov(df=tcov_df, name="tech. cov. without interaction", samples=samples)
        tcov_inter_m, tcov_inter_labels = self.load_tech_cov(df=get_tcov_inter_df, name="tech. cov. with interaction", samples=samples)
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

        self.log.info("Force normalising covariate matrix.")
        # TODO uncomment this?
        # covs_m = np.vstack((covs_m, tcov_m.T[:10, :]))
        # covariates = covariates + tcov_labels[:10]
        # covs_m = tcov_m.T[:10, :]
        # covariates = tcov_labels[:10]
        fn = ForceNormaliser(dataset_m=dataset_m, samples=samples, log=self.log)
        covs_m = fn.process(covs_m)
        self.log.info("")
        del fn

        ########################################################################

        self.log.info("Starting identifying interaction components.")

        io = InteractionOptimizer(n_iterations=self.n_iterations,
                                  covariates=covariates,
                                  dataset_m=dataset_m,
                                  samples=samples,
                                  genotype_na=self.genotype_na,
                                  ieqtl_alpha=self.ieqtl_alpha,
                                  log=self.log)

        pic_m = np.empty((self.n_components, len(samples)), dtype=np.float64)
        pic_a = None
        for comp_count in range(self.n_components):
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
                                                inter_m=geno_m)

                # Optimize the cell fractions in X iterations.
                pic_a = io.process(eqtl_m=eqtl_m,
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

        components_df = pd.DataFrame(pic_m,
                                     index=["PIC{}".format(i+1) for i in range(self.n_components)],
                                     columns=samples)

        save_dataframe(df=components_df,
                       outpath=os.path.join(self.outdir, "components_df.txt.gz"),
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

        self.log.info("Finished.")
        self.log.info("")

    def validate_data(self, eqtl_df, geno_df, expr_df, dataset_df, covs_df):
        snp_reference = eqtl_df["SNPName"].copy()
        snp_reference.rename("-", inplace=True)

        probe_reference = eqtl_df["ProbeName"].copy()
        probe_reference.rename("-", inplace=True)

        if not pd.Series(geno_df.index,
                         index=snp_reference.index,
                         name="-").equals(snp_reference):
            self.log.error("The genotype file indices do not match the "
                           "eQTL file.")
            exit()

        if not pd.Series(expr_df.index,
                         index=probe_reference.index,
                         name="-").equals(probe_reference):
            self.log.error("The expression file indices do not match the "
                           "eQTL file.")
            exit()

        if not geno_df.columns.equals(expr_df.columns):
            self.log.error("The genotype file header does not match the "
                           "expression file header.")
            exit()

        if not geno_df.columns.equals(dataset_df.index):
            self.log.error("The genotype file header does not match the "
                           "dataset file index.")
            exit()

        if not geno_df.columns.equals(covs_df.columns):
            self.log.error("The genotype file header does not match the "
                           "covariates file header.")
            exit()

    def calculate_eqtl_stats(self, genotype):
        group_sizes = self.calculate_group_sizes(genotype)
        maf = self.calculate_maf(group_sizes)
        smallest_group_size = min(group_sizes)
        n = sum(group_sizes)

        return pd.Series([maf, smallest_group_size, n], index=["MAF", "MGS", "N"])

    @staticmethod
    def calculate_group_sizes(genotype):
        x = genotype.copy()
        x = np.rint(x)
        unique, counts = np.unique(x, return_counts=True)
        group_sizes = dict(zip(unique, counts))
        return np.array([group_sizes.get(0, 0), group_sizes.get(1, 0), group_sizes.get(2, 0)], dtype=np.float64)

    @staticmethod
    def calculate_maf(group_sizes):
        allele1 = group_sizes[0] * 2 + group_sizes[1]
        allele2 = group_sizes[2] * 2 + group_sizes[1]
        return min(allele1, allele2) / (allele1 + allele2)

    def load_tech_cov(self, df, name, samples):
        if df is None:
            return None, []

        n_samples = len(samples)

        self.log.info("\tWorking on technical covariates matrix matrix '{}'.".format(name))

        # Check for nan values.
        if df.isna().values.sum() > 0:
            self.log.error("\t  Matrix contains nan values.")
            exit()

        # Put the samples on the rows.
        if df.shape[1] == n_samples:
            self.log.warning("\t  Transposing matrix.")
            df = df.T

        # Check if valid.
        if not (df.index.to_numpy() == samples).all():
            self.log.error("\t  The header does not match the expression file header.")
            exit()

        # Check for variables with zero std.
        variance_mask = df.std(axis=0) != 0
        n_zero_variance = variance_mask.shape[0] - variance_mask.sum()
        if n_zero_variance > 0:
            self.log.warning("\t  Dropping {} rows with 0 variance .".format(
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
        self.log.info("  > MAF: >{}".format(self.maf))
        self.log.info("  > Minimal group size: >{}".format(self.mgs))
        self.log.info("  > ieQTL alpha: <{}".format(self.ieqtl_alpha))
        self.log.info("  > N components: {}".format(self.n_components))
        self.log.info("  > N iterations: {}".format(self.n_iterations))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")




