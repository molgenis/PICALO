#!/usr/bin/env python3

"""
File:         simulate_expression.py
Created:      2023/06/07
Last Changed: 2023/07/13
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import argparse
import os

# Third party imports.
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import betainc
from sklearn.decomposition import PCA

# Local application imports.
from src.logger import Logger
from src.statistics import inverse, fit, predict, calc_pearsonr_vector

# Metadata
__program__ = "Simulate Expression"
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
./simulate_expression.py -h

### MetaBrain 2 covariates ###

./simulate_expression.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/eQTLSummaryStats.txt.gz \
    -use_real_distributions \
    -kc 0 \
    -hc 2 \
    -of 2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised
    
### BIOS 2 covariates ###
    
./simulate_expression.py \
    -eq /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -d /groups/umcg-bios/tmp01/projects/PICALO/fast_eqtl_mapper/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/eQTLSummaryStats.txt.gz \
    -use_real_distributions \
    -kc 0 \
    -hc 2 \
    -of 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl = getattr(arguments, 'eqtl')
        self.genotype = getattr(arguments, 'genotype')
        self.genotype_na = getattr(arguments, 'genotype_na')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.distributions = getattr(arguments, 'distributions')
        self.use_real_distributions = getattr(arguments, 'use_real_distributions')
        self.n_individuals = getattr(arguments, 'individuals')
        self.resample_individuals = getattr(arguments, 'resample_individuals')
        self.n_experiments = getattr(arguments, 'experiments')
        self.n_eqtls = getattr(arguments, 'eqtls')
        self.n_known_covariates = getattr(arguments, 'n_known_covariates')
        self.n_hidden_covariates = getattr(arguments, 'n_hidden_covariates')
        self.n_starting_vectors = getattr(arguments, 'n_starting_vectors')
        self.exclude_covariate_interactions = getattr(arguments, 'exclude_covariate_interactions')
        self.resample_covariates = getattr(arguments, 'resample_covariates')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        self.n_covariates = self.n_known_covariates + self.n_hidden_covariates
        self.n_interaction_covariates = self.n_covariates
        if self.exclude_covariate_interactions:
            self.n_interaction_covariates = 0
        self.n_terms = 2 + self.n_covariates + self.n_interaction_covariates + 1

        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "simulate_expression", outfolder)
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
        parser.add_argument("-eq",
                            "--eqtl",
                            type=str,
                            required=True,
                            help="The path to the eqtl matrix.")
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
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-d",
                            "--distributions",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the distribution matrix.")
        parser.add_argument("-use_real_distributions",
                            action='store_true',
                            help="Use the real distributions. Default: False.")
        parser.add_argument("-in",
                            "--individuals",
                            type=int,
                            required=False,
                            default=None,
                            help="The number of individuals. Default: all.")
        parser.add_argument("-resample_individuals",
                            action='store_true',
                            help="Resample the individuals for each "
                                 "experiment. Default: False.")
        parser.add_argument("-ex",
                            "--experiments",
                            type=int,
                            required=False,
                            default=1,
                            help="The number of experiments. Default: 1.")
        parser.add_argument("-ne",
                            "--eqtls",
                            type=int,
                            required=False,
                            default=None,
                            help="The number of eQTLs. Default: all.")
        parser.add_argument("-kc",
                            "--n_known_covariates",
                            type=int,
                            required=False,
                            default=2,
                            help="The number of known covariates. Default: 2.")
        parser.add_argument("-hc",
                            "--n_hidden_covariates",
                            type=int,
                            required=False,
                            default=3,
                            help="The number of hidden covariates. Default: 3.")
        parser.add_argument("-nc",
                            "--n_starting_vectors",
                            type=int,
                            required=False,
                            default=25,
                            help="The number of starting vectors. Default: 25.")
        parser.add_argument("-exclude_covariate_interactions",
                            action='store_true',
                            help="Include covariate + covariate * genotype "
                                 "terms. Default: False.")
        parser.add_argument("-resample_covariates",
                            action='store_true',
                            help="Resample the covariates for each "
                                 "experiment. Default: False.")
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

        self.log.info("Loading input data")
        eqtl_df = self.load_file(inpath=self.eqtl, header=0, index_col=None,
                                 nrows=self.n_eqtls)
        print(eqtl_df)

        geno_df = self.load_file(inpath=self.genotype, header=0, index_col=0,
                                 nrows=self.n_eqtls)
        geno_df[geno_df == self.genotype_na] = np.nan
        print(geno_df)

        n_eqtls = geno_df.shape[0]
        n_individuals = geno_df.shape[1]
        if self.n_individuals is None:
            self.n_individuals = n_individuals
        if self.n_eqtls is None:
            self.n_eqtls = n_eqtls

        self.log.info("\tLoaded {:,} eQTLs".format(n_eqtls))
        self.log.info("\tLoaded {:,} individuals".format(n_individuals))
        if eqtl_df["SNPName"].tolist() != geno_df.index.tolist():
            self.log.error("\tError, eQTL file and genotype file don't match.")
            exit()

        eqtl_names = (eqtl_df["ProbeName"] + "_" + eqtl_df["SNPName"]).tolist()
        gene_names = eqtl_df["ProbeName"].tolist()

        ########################################################################

        dataset_df = pd.DataFrame({"sample": geno_df.columns, "dataset": "None"})
        if self.std_path is not None:
            dataset_df = self.load_file(self.std_path, header=0, index_col=None)
            if dataset_df.iloc[:, 0].values.tolist() != geno_df.columns.tolist():
                print("\tError, Dataset does not match genotype.")
                exit()

        self.log.info("Simulating genotypes")
        geno_df = self.simulate_genotypes(df=geno_df, dataset_df=dataset_df)
        print(geno_df)

        ########################################################################

        self.log.info("Defining distributions to sample from")
        dist_beta_m = np.zeros((self.n_eqtls, self.n_terms), dtype=np.float64)
        dist_std_m = np.ones((self.n_eqtls, self.n_terms), dtype=np.float64)
        if self.use_real_distributions:
            self.log.info("\tUsing real distributions")
            distribution_df = self.load_file(inpath=self.distributions,
                                             header=0,
                                             index_col=0,
                                             nrows=self.n_eqtls)
            beta_df = distribution_df.loc[:, [col for col in distribution_df.columns if "beta-" in col]].copy()
            dist_beta_m = beta_df.to_numpy()

            std_df = distribution_df.loc[:, [col for col in distribution_df.columns if "std-" in col]].copy()
            dist_std_m = std_df.to_numpy()
            del distribution_df, beta_df, std_df

        self.log.info("Beta:")
        print(pd.DataFrame(dist_beta_m))
        self.log.info("Std:")
        print(pd.DataFrame(dist_std_m))

        ########################################################################

        self.log.info("Selecting individuals")
        individuals_mask = np.ones(n_individuals, dtype=np.bool)
        if self.n_individuals is not None and self.n_individuals != n_individuals:
            if self.n_individuals > n_individuals:
                self.log.error("\tError, more samples requested than available.")
                exit()
            individuals_mask[np.random.choice(n_individuals,
                                              size=n_individuals - self.n_individuals,
                                              replace=False)] = 0

        ########################################################################

        cov_m = None
        if self.n_covariates > 0:
            self.log.info("Generate covariate matrix")
            # Each entry is drawn from N(0,1).
            cov_m = np.random.normal(0, 1, size=(self.n_individuals, self.n_covariates))

        # Generate the model columns.
        model_columns = ["intercept", "genotype"] + \
                        ["known_covariate{}".format(i) for i in range(self.n_known_covariates)] + \
                        ["hidden_covariate{}".format(i) for i in range(self.n_hidden_covariates)]
        if not self.exclude_covariate_interactions:
            model_columns += ["known_covariate_interaction{}".format(i) for i in range(self.n_known_covariates)] + \
                             ["hidden_covariate_interaction{}".format(i) for i in range(self.n_hidden_covariates)]
        model_columns += ["noise"]

        ########################################################################

        # Iterate through the experiments.
        self.log.info("Generate experiments")
        for simulation_id in range(1, self.n_experiments + 1):
            self.log.info("  Simulation {}".format(simulation_id))
            exp_outdir = os.path.join(self.outdir, "simulation{}".format(simulation_id))
            if not os.path.exists(exp_outdir):
                os.makedirs(exp_outdir)

            # Resample individuals if need be.
            if self.n_individuals is not None and self.n_individuals != n_individuals and self.resample_individuals:
                self.log.info("\tResampling individuals")
                individuals_mask = np.ones(n_individuals, dtype=np.bool)
                individuals_mask[np.random.choice(n_individuals,
                                                  size=n_individuals - self.n_individuals,
                                                  replace=False)] = False
            sample_names = geno_df.columns[individuals_mask].tolist()

            expr_m = np.empty((self.n_eqtls, self.n_individuals), dtype=np.float64)
            beta_m = np.empty((self.n_eqtls, self.n_terms), dtype=np.float64)
            std_m = np.empty((self.n_eqtls, self.n_terms), dtype=np.float64)
            r_squared_m = np.empty((self.n_eqtls, 1), dtype=np.float64)

            self.log.info("\tLoop over eQTLs")
            for eqtl_index in range(self.n_eqtls):
                if (eqtl_index == 0) or (eqtl_index % 250 == 0):
                    print("\tprocessed {} eQTLs".format(eqtl_index))

                # Generate the empty model matrix.
                model_m = np.empty(shape=(self.n_individuals, self.n_terms))

                # Fill in the intercept.
                model_m[:, 0] = 1

                # Fill in the real genotype data.
                model_m[:, 1] = geno_df.iloc[eqtl_index, individuals_mask].copy().to_numpy()

                if self.n_covariates > 0:
                    # # Resample the covariate matrix if need be. Each entry is drawn
                    # # from N(0,1).
                    # if self.resample_covariates:
                    #     self.log.info("\tResampling covariates")
                    #     cov_m = np.random.normal(0, 1, size=(self.n_individuals, self.n_covariates))

                    # Fill in the simulated covariate data.
                    model_m[:, 2:(self.n_covariates + 2)] = cov_m

                    if not self.exclude_covariate_interactions:
                        # Calculate and fill the interaction data.
                        model_m[:, (self.n_covariates + 2):(self.n_terms - 1)] = model_m[:, 2:(self.n_covariates + 2)] * model_m[:, [1]]

                # Fill in the noise matrix. Each entry is drawn from N(0,1).
                model_m[:, self.n_terms - 1] = np.random.normal(0, 1, size=(self.n_individuals))

                if eqtl_index < 2:
                    self.log.info("\tExample model")
                    print(pd.DataFrame(model_m, index=sample_names, columns=model_columns))

        ########################################################################

                # Generate the model beta's.
                weights_m = np.empty(shape=(self.n_individuals, self.n_terms))
                for term_index in range(self.n_terms):
                    weights_m[:, term_index] = np.random.normal(dist_beta_m[eqtl_index, term_index], dist_std_m[eqtl_index, term_index], size=(self.n_individuals,))

                # if eqtl_index < 2:
                #     self.log.info("\tIntended distributions")
                #     print(pd.DataFrame({"loc": dist_beta_m[eqtl_index, :], "scale": dist_std_m[eqtl_index, :]}, index=model_columns))
                #     self.log.info("\tExample weights")
                #     print(pd.DataFrame(weights_m, index=sample_names, columns=model_columns))
                #     self.log.info("\tReal distributions")
                #     print(pd.DataFrame({"loc": weights_m.mean(axis=0), "scale": weights_m.std(axis=0)}, index=model_columns))

        ########################################################################

                # Generate the expression matrix.
                weighted_model_m = model_m * weights_m
                expr_a = np.sum(weighted_model_m, axis=1)

                # if eqtl_index < 2:
                #     self.log.info("\tWeighted model distributions")
                #     print(pd.DataFrame({"loc": weighted_model_m.mean(axis=0), "scale": weighted_model_m.std(axis=0)}, index=model_columns))
                #     self.log.info("\tExample expression")
                #     print(pd.Series(expr_a, index=sample_names))

                # Save.
                expr_m[eqtl_index, :] = expr_a
                beta_m[eqtl_index, :] = weights_m.mean(axis=0)
                std_m[eqtl_index, :] = weights_m.std(axis=0)

                # OLS.
                inv_m = inverse(model_m)
                betas = fit(X=model_m, y=expr_a, inv_m=inv_m)
                y_hat = predict(X=model_m, betas=betas)
                pearsonr = calc_pearsonr_vector(x=expr_a, y=y_hat)
                r_squared = pearsonr * pearsonr
                r_squared_m[eqtl_index, 0] = r_squared

        ########################################################################

            self.log.info("\tGenerating starting vectors")
            zscores = (expr_m - np.mean(expr_m, axis=0)) / np.std(expr_m, axis=0)
            pca = PCA(n_components=self.n_starting_vectors)
            pca.fit(zscores)
            expr_pcs_df = pd.DataFrame(pca.components_)
            expr_pcs_df.index = ["PC{}".format(i + 1) for i, _ in enumerate(expr_pcs_df.index)]
            expr_pcs_df.columns = sample_names
            print(expr_pcs_df)
            explained_var_df = pd.DataFrame(pca.explained_variance_ratio_ * 100, index=["PC{}".format(i + 1) for i in range(self.n_starting_vectors)], columns=["ExplainedVariance"])
            print(explained_var_df)

            if self.n_hidden_covariates > 0:
                self.log.info("\tChecking correlation between hidden covariates and PCs")
                coef_m, pvalue_m = self.corrcoef(m1=cov_m[:, self.n_known_covariates:self.n_covariates],
                                                 m2=np.transpose(pca.components_))
                coef_df = pd.DataFrame(coef_m,
                                       index=model_columns[(self.n_known_covariates + 2):(self.n_covariates + 2)],
                                       columns=expr_pcs_df.index)
                pvalue_df = pd.DataFrame(pvalue_m,
                                         index=model_columns[(self.n_known_covariates + 2):(self.n_covariates + 2)],
                                         columns=expr_pcs_df.index)
                corr_out_df = self.merge_coef_and_p(coef_df=coef_df, pvalue_df=pvalue_df)
                print(corr_out_df)
                self.save_file(df=corr_out_df,
                               outpath=os.path.join(exp_outdir, "hidden_covariate_PC_correlations.txt.gz"))
                del coef_m, pvalue_m, coef_df, pvalue_df, corr_out_df

            self.log.info("\t Generating random starting vectors")
            random_start_m = np.random.normal(0, 1, size=(self.n_starting_vectors, self.n_individuals))
            coef_m, pvalue_m = self.corrcoef(
                m1=cov_m[:, self.n_known_covariates:self.n_covariates],
                m2=np.transpose(random_start_m)
            )
            coef_df = pd.DataFrame(coef_m,
                                   index=model_columns[(self.n_known_covariates + 2):(self.n_covariates + 2)],
                                   columns=["Random{}".format(i + 1) for i in range(self.n_starting_vectors)])
            pvalue_df = pd.DataFrame(pvalue_m,
                                     index=model_columns[(self.n_known_covariates + 2):(self.n_covariates + 2)],
                                     columns=["Random{}".format(i + 1) for i in range(self.n_starting_vectors)])
            corr_out_df = self.merge_coef_and_p(coef_df=coef_df,
                                                pvalue_df=pvalue_df)
            print(corr_out_df)
            self.save_file(df=corr_out_df,
                           outpath=os.path.join(exp_outdir, "hidden_covariate_random_correlations.txt.gz"))

        ########################################################################

            self.log.info("\tSaving data")

            # Save.
            self.save_file(df=geno_df, outpath=os.path.join(exp_outdir, "genotype_table.txt.gz"))
            if self.n_known_covariates > 0:
                self.save_file(df=pd.DataFrame(cov_m[:, :self.n_known_covariates], index=sample_names, columns=model_columns[2:(self.n_known_covariates + 2)]),
                               outpath=os.path.join(exp_outdir, "tech_covariates_with_interaction_df.txt.gz"))
            if self.n_hidden_covariates > 0:
                self.save_file(df=pd.DataFrame(cov_m[:, self.n_known_covariates:self.n_covariates], index=sample_names, columns=model_columns[(self.n_known_covariates + 2):(self.n_covariates + 2)]),
                           outpath=os.path.join(exp_outdir, "PICs.txt.gz"))
            self.save_file(df=pd.DataFrame(beta_m, index=eqtl_names, columns=model_columns),
                           outpath=os.path.join(exp_outdir, "model_betas.txt.gz"))
            self.save_file(df=pd.DataFrame(std_m, index=eqtl_names, columns=model_columns),
                           outpath=os.path.join(exp_outdir, "model_std.txt.gz"))
            self.save_file(df=pd.DataFrame(expr_m, index=gene_names, columns=sample_names),
                           outpath=os.path.join(exp_outdir, "expression_table.txt.gz"))
            self.save_file(df=pd.DataFrame(r_squared_m, index=eqtl_names, columns=["rsquared"]),
                           outpath=os.path.join(exp_outdir, "model_rsquared.txt.gz"))
            self.save_file(df=expr_pcs_df,
                           outpath=os.path.join(exp_outdir, "first{}ExpressionPCs.txt.gz".format(self.n_starting_vectors)))
            self.save_file(df=explained_var_df,
                           outpath=os.path.join(exp_outdir, "first{}ExpressionPCsExplainedVariance.txt.gz".format(self.n_starting_vectors)))
            self.save_file(df=pd.DataFrame(random_start_m, index=["Random{}".format(i + 1) for i in range(self.n_starting_vectors)], columns=sample_names),
                           outpath=os.path.join(exp_outdir, "{}RandomVectors.txt.gz".format(self.n_starting_vectors)))

            self.log.info("")

        self.log.info("Finished")
        self.log.info("")

    def load_file(self, inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        self.log.info("\tLoaded dataframe: {} "
                      "with shape: {}".format(os.path.basename(inpath),
                                              df.shape))
        return df

    @staticmethod
    def simulate_genotypes(df, dataset_df):
        rounded_m = df.to_numpy(dtype=np.float64)
        rounded_m = np.rint(rounded_m)

        n_rows = rounded_m.shape[0]

        # Count the genotypes.
        zero_a = np.sum(rounded_m == 0, axis=1)
        one_a = np.sum(rounded_m == 1, axis=1)
        two_a = np.sum(rounded_m == 2, axis=1)

        counts = np.column_stack((zero_a, one_a, two_a))
        fractions = counts / np.sum(counts, axis=1)[:, None]
        # print(pd.DataFrame(fractions))

        simulated_geno_m = np.empty_like(rounded_m)
        for dataset in dataset_df.iloc[:, 1].unique():
            dataset_mask = (dataset_df.iloc[:, 1] == dataset).to_numpy()

            # Calculate the rounded counts for each genotype.
            # Based on https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal/51451847#51451847
            # adapted to work on matrices.
            N = np.sum(dataset_mask)
            xs = fractions * N
            Rs = np.floor(xs).astype(int)
            K = N - np.sum(Rs, axis=1)
            fs = xs - Rs
            indices = (-fs).argsort()
            ys = Rs
            for i in range(np.max(K)):
                mask = i < K
                ys[mask, indices[mask, i]] += 1

            # Create the genotype matrix.
            dataset_geno_m = np.empty((n_rows, N), dtype=np.float64)
            for i in range(n_rows):
                dataset_geno_m[i, :] = np.array([0] * ys[i, 0] + [1] * ys[i, 1] + [2] * ys[i, 2])
            np.random.shuffle(dataset_geno_m.T)

            # Save.
            simulated_geno_m[:, dataset_mask] = dataset_geno_m
            del dataset_geno_m

        return pd.DataFrame(simulated_geno_m, index=df.index, columns=df.columns)

    @staticmethod
    def corrcoef(m1, m2):
        """
        Pearson correlation over the columns.

        https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
        """
        m1_dev = m1 - np.mean(m1, axis=0)
        m2_dev = m2 - np.mean(m2, axis=0)

        m1_rss = np.sum(m1_dev * m1_dev, axis=0)
        m2_rss = np.sum(m2_dev * m2_dev, axis=0)

        r = np.empty((m1_dev.shape[1], m2_dev.shape[1]), dtype=np.float64)
        for i in range(m1_dev.shape[1]):
            for j in range(m2_dev.shape[1]):
                r[i, j] = np.sum(m1_dev[:, i] * m2_dev[:, j]) / np.sqrt(
                    m1_rss[i] * m2_rss[j])

        rf = r.flatten()
        df = m1.shape[0] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = betainc(0.5 * df, 0.5, df / (df + ts))
        p = pf.reshape(m1.shape[1], m2.shape[1])
        return r, p

    @staticmethod
    def merge_coef_and_p(coef_df, pvalue_df):
        coef_df["index"] = coef_df.index
        corr_dfm = coef_df.melt(id_vars=["index"])
        corr_dfm.columns = ["hidden_covariate", "PC", "coef"]
        corr_dfm["abs coef"] = corr_dfm["coef"].abs()
        pvalue_df["index"] = pvalue_df.index
        pvalue_dfm = pvalue_df.melt(id_vars=["index"])
        pvalue_dfm.columns = ["hidden_covariate", "PC", "pvalue"]

        corr_out_df = corr_dfm.merge(pvalue_dfm, on=["hidden_covariate", "PC"])
        corr_out_df.sort_values(by="abs coef", ascending=False, inplace=True)
        corr_out_df.drop(["abs coef"], axis=1, inplace=True)
        return corr_out_df

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

    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > eQTL data: {}".format(self.eqtl))
        self.log.info("  > Genotype data: {}".format(self.genotype))
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Sample-to-dataset data: {}".format(self.std_path))
        self.log.info("  > Distributions: {}".format(self.distributions))
        self.log.info("  > Use real distributions: {}".format(self.use_real_distributions))
        self.log.info("  > Individuals: {}".format(self.n_individuals))
        self.log.info("  > Resample individuals: {}".format(self.resample_individuals))
        self.log.info("  > N experiments: {}".format(self.n_experiments))
        self.log.info("  > N eQTLs: {}".format(self.n_eqtls))
        self.log.info("  > N known covariates: {}".format(self.n_known_covariates))
        self.log.info("  > N hidden covariates: {}".format(self.n_hidden_covariates))
        self.log.info("  > N starting vectors: {}".format(self.n_starting_vectors))
        self.log.info("  > Exclude covariate interactions: {}".format(self.exclude_covariate_interactions))
        self.log.info("  > Resample covariates: {}".format(self.resample_covariates))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()