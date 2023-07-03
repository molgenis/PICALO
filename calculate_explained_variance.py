#!/usr/bin/env python3

"""
File:         calculate_explained_variance.py
Created:      2022/01/18
Last Changed: 2022/07/25
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import time
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

# Local application imports.
from src.cmd_line_arguments import CommandLineArguments
from src.logger import Logger
from src.objects.data import Data
from src.utilities import save_dataframe

# Metadata
__program__ = "Calculate Explained Variance"
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
./calculate_explained_variance.py -h

### BIOS ###

./calculate_explained_variance.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tci /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_and_first31ExpressionPCs_df.txt.gz \
    -co /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -maf 0.05 \
    -o 2022-07-25-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-Patrick \
    -verbose
    
./calculate_explained_variance.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tc /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first40ExpressionPCs.txt.gz \
    -tci /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_df.txt.gz \
    -co /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first31ExpressionPCs.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -maf 0.05 \
    -o 2022-07-25-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PCsAsCov \
    -verbose
    
### MetaBrain ###

./calculate_explained_variance.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first80ExpressionPCs.txt.gz \
    -tci /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_df.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -maf 0.05 \
    -o 2022-07-25-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PICsAsCov \
    -verbose
    
./calculate_explained_variance.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -tc /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first80ExpressionPCs.txt.gz \
    -tci /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/tech_covariates_with_interaction_df.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first21ExpressionPCs.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -maf 0.05 \
    -o 2022-07-25-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-PCsAsCov \
    -verbose

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        cla = CommandLineArguments(program=__program__,
                                   version=__version__,
                                   description=__description__)
        self.genotype_na = cla.get_argument('genotype_na')
        self.min_dataset_sample_size = cla.get_argument('min_dataset_size')
        self.eqtl_alpha = cla.get_argument('eqtl_alpha')
        self.ieqtl_alpha = cla.get_argument('ieqtl_alpha')
        self.call_rate = cla.get_argument('call_rate')
        self.hw_pval = cla.get_argument('hardy_weinberg_pvalue')
        self.maf = cla.get_argument('minor_allele_frequency')
        self.mgs = cla.get_argument('min_group_size')

        # Define the current directory.
        current_dir = str(os.path.dirname(os.path.abspath(__file__)))

        # Prepare an output directory.
        self.outdir = os.path.join(current_dir, "calculate_explained_variance", cla.get_argument('outdir'))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Initialize logger.
        logger = Logger(outdir=self.outdir,
                        verbose=cla.get_argument('verbose'),
                        clear_log=True)
        logger.print_arguments()
        self.log = logger.get_logger()

        # Initialize data object.
        self.data = Data(eqtl_path=cla.get_argument('eqtl'),
                         genotype_path=cla.get_argument('genotype'),
                         expression_path=cla.get_argument('expression'),
                         tech_covariate_path=cla.get_argument('tech_covariate'),
                         tech_covariate_with_inter_path=cla.get_argument('tech_covariate_with_inter'),
                         covariate_path=cla.get_argument('covariate'),
                         sample_dataset_path=cla.get_argument('sample_to_dataset'),
                         log=self.log)
        self.data.print_arguments()

    def start(self):
        self.log.info("Starting program")
        self.print_arguments()

        ########################################################################

        self.log.info("Loading eQTL data and filter on FDR values of the "
                      "main eQTL effect")
        eqtl_df = self.data.get_eqtl_df()
        eqtl_fdr_keep_mask = (eqtl_df["FDR"] <= self.eqtl_alpha).to_numpy(dtype=bool)
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

        if std_df is not None:
            # Validate that the input data matches.
            self.validate_data(std_df=std_df,
                               geno_df=geno_df)
        else:
            # Create sample-to-dataset file with all the samples having the
            # same dataset.
            std_df = pd.DataFrame({"sample": geno_df.columns, "dataset": "None"})

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
            self.log.warning("\t  {:,} eQTLs have had dataset(s) filled with "
                             "NaN values due to call rate "
                             "threshold ".format(call_rate_n_skipped))

        save_dataframe(df=call_rate_df,
                       outpath=os.path.join(self.outdir, "call_rate.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)
        self.log.info("")

        self.log.info("\tCalculating genotype stats for inclusing criteria")
        cr_keep_mask = ~(geno_df == self.genotype_na).all(axis=1).to_numpy(dtype=bool)
        geno_stats_df = pd.DataFrame(np.nan, index=geno_df.index, columns=["N", "NaN", "0", "1", "2", "min GS", "HW pval", "allele1", "allele2", "MA", "MAF"])
        geno_stats_df["N"] = 0
        geno_stats_df["NaN"] = geno_df.shape[1]
        geno_stats_df.loc[cr_keep_mask, :] = self.calculate_genotype_stats(df=geno_df.loc[cr_keep_mask, :])

        # Checking which eQTLs pass the requirements
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

        # Select rows that meet requirements.
        eqtl_signif_df = eqtl_signif_df.loc[combined_keep_mask, :]
        geno_df = geno_df.loc[combined_keep_mask, :]

        # Combine the skip masks.
        keep_mask = np.copy(eqtl_fdr_keep_mask)
        keep_mask[eqtl_fdr_keep_mask] = combined_keep_mask

        # Add mask to genotype stats data frame.
        geno_stats_df["mask"] = 0
        geno_stats_df.loc[keep_mask, "mask"] = 1

        save_dataframe(df=geno_stats_df,
                       outpath=os.path.join(self.outdir, "genotype_stats.txt.gz"),
                       header=True,
                       index=True,
                       log=self.log)
        self.log.info("")

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
        if geno_df.isna().values.sum() > 0:
            self.log.error("\t  Genotype file contains NaN values")
            exit()
        if expr_df.isna().values.sum() > 0:
            self.log.error("\t  Expression file contains NaN values")
            exit()
        if covs_df.isna().values.sum() > 0:
            self.log.error("\t  Covariate file contains NaN values")
            exit()

        # Transpose if need be.
        if covs_df.shape[0] != geno_df.shape[0]:
            self.log.warning("\t  Transposing covariate matrix")
            covs_df = covs_df.T

        covariates = covs_df.columns.tolist()
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

        self.log.info("Loading technical covariates")
        tcov_df = self.load_tech_cov(df=self.data.get_tcov_df(),
                                     name="tech. cov. without interaction",
                                     std_df=std_df)
        tcov_inter_df = self.load_tech_cov(df=self.data.get_tcov_inter_df(),
                                           name="tech. cov. with interaction",
                                           std_df=std_df)

        ########################################################################

        self.log.info("Merging matrices.")
        default_matrix = pd.concat([x for x in [pd.DataFrame({"intercept": 1}, index=samples),
                                                dataset_df,
                                                tcov_df,
                                                tcov_inter_df,
                                                 covs_df] if x is not None],
                                   axis=1)

        interaction_matrix = None
        interaction_matrix_dfs = [x for x in [dataset_df, tcov_inter_df, covs_df] if x is not None]
        if len(interaction_matrix_dfs) > 0:
            interaction_matrix = pd.concat(
                [x for x in [dataset_df,
                             tcov_inter_df,
                             covs_df] if x is not None],
                axis=1)

        ########################################################################

        self.log.info("Calculating squared-residuals")
        n_eqtls = geno_df.shape[0]
        summary_results = []
        # params_results = []
        # bse_results = []
        last_print_time = None
        for eqtl_index in range(n_eqtls):
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 30 or (eqtl_index + 1) == n_eqtls:
                last_print_time = now_time
                self.log.info("\t{:,}/{:,} eQTLs analysed [{:.2f}%]".format(eqtl_index, n_eqtls - 1, (100 / (n_eqtls - 1)) * eqtl_index))

            # Copy the matrix.
            X = default_matrix.copy()

            # Add the genotype and expression term.
            X["genotype"] = geno_df.iloc[eqtl_index, :]

            # Drop the Nan values.
            X = X.loc[X["genotype"] != -1, :]

            # Add the covariates with interaction term.
            if interaction_matrix is not None:
                inter_X = interaction_matrix.multiply(X["genotype"], axis=0)
                inter_X.columns = ["{}xGenotype".format(col) for col in inter_X.columns]
                X = pd.merge(X, inter_X, left_index=True, right_index=True)

            y = expr_df.iloc[eqtl_index, :].loc[X.index]

            # Calculate the R^2 for the full model.
            # base_model = OLS(y, X).fit()
            # base_rsquared = base_model.rsquared
            pc_interaction_columns = [col for col in X.columns if col.startswith("Comp") and col.endswith("xGenotype")]
            pic_interaction_columns = [col for col in X.columns if col.startswith("PIC") and col.endswith("xGenotype")]
            no_inter_model = OLS(y, X.loc[:, [col for col in X.columns if col not in pc_interaction_columns and col not in pic_interaction_columns]]).fit()

            # params = base_model.params.to_frame()
            # params.columns = [eqtl_index]
            # params.index = ["{} coef".format(value) for value in params.index]
            #
            # bse = base_model.bse.to_frame()
            # bse.columns = [eqtl_index]
            # bse.index = ["{} std error".format(value) for value in bse.index]

            # Calculate the R^2 for the alternative models.
            # genotype_model = OLS(y, X.loc[:, [col for col in X.columns if col != "genotype"]]).fit().rsquared
            # context_model = OLS(y, X.loc[:, [col for col in X.columns if col not in covariates]]).fit().rsquared
            # context_interaction_model = OLS(y, X.loc[:, [col for col in X.columns if col not in ["{}xGenotype".format(cov) for cov in covariates]]]).fit().rsquared
            with_pc_inter_model = OLS(y, X.loc[:, [col for col in X.columns if col not in pic_interaction_columns]]).fit()
            with_pic_inter_model = OLS(y, X.loc[:, [col for col in X.columns if col not in pc_interaction_columns]]).fit()

            # Save results.
            # summary_results.append([X.shape[0], base_rsquared, base_rsquared - genotype_model, base_rsquared - context_model, base_rsquared - context_interaction_model])
            # params_results.append(params)
            # bse_results.append(bse)
            summary_results.append([X.shape[0], no_inter_model.rsquared, with_pc_inter_model.rsquared, with_pic_inter_model.rsquared])

        ########################################################################

        self.log.info("Saving results")
        # Merging data.
        annot_df = eqtl_signif_df[["SNPName", "ProbeName"]].copy()
        annot_df.reset_index(drop=True, inplace=True)
        df = pd.DataFrame(summary_results, columns=["n", "noInter r-squared", "PCInter r-squared", "PICInter r-squared"])
        print(annot_df)
        print(df)
        df = pd.concat([annot_df, df], axis=1)

        # df = pd.DataFrame(summary_results, columns=["n", "full r-squared", "genotype r-squared", "context r-squared", "context interaction r-squared"])
        # params_results = pd.concat(params_results, axis=1)
        # bse_results = pd.concat(bse_results, axis=1)
        # print(annot_df)
        # print(df)
        # print(params_results.T)
        # print(bse_results.T)
        # df = pd.concat([annot_df, df, params_results.T, bse_results.T], axis=1)

        # Save results.
        save_dataframe(df=df,
                       outpath=os.path.join(self.outdir, "results.txt.gz"),
                       header=True,
                       index=False,
                       log=self.log)

        ########################################################################

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

        if covs_df is not None and covs_df.index.tolist() != samples:
                self.log.error("\tThe covariates file index does not match "
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
            call_rate_s = (geno_df.loc[:, sample_mask.to_numpy(dtype=bool)] != self.genotype_na).astype(int).sum(axis=1) / np.sum(sample_mask)
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
            return None

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

        self.log.info("\t  Technical covariates [{}]: {}".format(df.shape[1], ", ".join(df.columns)))

        return df

    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Minimal dataset size: >={}".format(self.genotype_na))
        self.log.info("  > eQTL alpha: <={}".format(self.eqtl_alpha))
        self.log.info("  > SNP call rate: >{}".format(self.call_rate))
        self.log.info("  > Hardy-Weinberg p-value: >{}".format(self.hw_pval))
        self.log.info("  > MAF: >{}".format(self.maf))
        self.log.info("  > Minimal group size: >={}".format(self.mgs))
        self.log.info("  > ieQTL alpha: <={}".format(self.ieqtl_alpha))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()