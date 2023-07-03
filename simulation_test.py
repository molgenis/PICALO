#!/usr/bin/env python3

"""
File:         simulation_test.py
Created:      2023/06/27
Last Changed:
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
import numpy as np
import pandas as pd
from scipy.special import ndtri
from statsmodels.regression.linear_model import OLS

# Local application imports.
from src.logger import Logger
from src.utilities import load_dataframe

# Metadata
__program__ = "Simulation Test"
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
./simulation_test.py -h
   
./simulation_test.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table.txt.gz \
    -co /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/first1ExpressionPCs.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -verbose

./simulation_test.py \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/simulate_expression/2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first1ExpressionPCDistributions/simulation1/expression_table.txt.gz \
    -od /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
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
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "simulation_test", outfolder)
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
                            help="The path to the covariate matrix.")
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
        geno_df = load_dataframe(self.genotype_path, header=0, index_col=0, log=self.log, nrows=1).T
        geno_df.columns = ["genotype"]
        expr_df = load_dataframe(self.expression_path, header=0, index_col=0, log=self.log, nrows=1).T
        expr_df.columns = ["expression"]


        model_df = expr_df.merge(geno_df, left_index=True, right_index=True)

        if self.covariate_path is not None:
            cov_df = load_dataframe(self.covariate_path, header=0, index_col=0, log=self.log, nrows=1).T
            cov_df.columns = ["covariate"]
            cov_df["covariate"] = ndtri((cov_df["covariate"].rank(ascending=True) - 0.5) / cov_df.shape[0])
            model_df = model_df.merge(cov_df, left_index=True, right_index=True)

        model_df = model_df.loc[model_df["genotype"] != self.genotype_na, :]
        n_samples = model_df.shape[0]

        model_df.insert(1, "intercept", 1)
        if self.covariate_path is None:
            model_df["covariate"] = np.random.normal(0, 1, size=(n_samples, ))
        model_df["interaction"] = model_df["genotype"] * model_df["covariate"]
        print("Model:")
        print(model_df)
        model_params = pd.concat([model_df.mean(axis=0), model_df.std(axis=0)], axis=1)
        model_params.columns = ["loc", "scale"]
        print(model_params)
        print("")

        print("Fit OLS model")
        ols_model = OLS(model_df["expression"], model_df[["intercept", "genotype", "covariate", "interaction"]]).fit()
        print(ols_model.summary())
        ols_model_residuals = ols_model.resid
        ols_model_params = pd.concat([ols_model.params, ols_model.bse], axis=1)
        ols_model_params.columns = ["loc", "scale"]
        ols_model_params.loc["error", "loc"] = np.mean(ols_model_residuals)
        ols_model_params.loc["error", "scale"] = np.std(ols_model_residuals)
        print(ols_model_params)
        print("")

        weights_df = pd.DataFrame({
            "intercept": np.random.normal(ols_model_params.loc["intercept", "loc"], ols_model_params.loc["intercept", "scale"], size=(n_samples, )),
            "genotype": np.random.normal(ols_model_params.loc["genotype", "loc"], ols_model_params.loc["genotype", "scale"], size=(n_samples, )),
            "covariate": np.random.normal(ols_model_params.loc["covariate", "loc"], ols_model_params.loc["covariate", "scale"], size=(n_samples, )),
            "interaction": np.random.normal(ols_model_params.loc["interaction", "loc"], ols_model_params.loc["interaction", "scale"], size=(n_samples, )),
            "error": np.random.normal(ols_model_params.loc["error", "loc"], ols_model_params.loc["error", "scale"], size=(n_samples,))
        }, index=model_df.index)
        print("Weights:")
        print(weights_df)
        weights_params = pd.concat([weights_df.mean(axis=0), weights_df.std(axis=0)], axis=1)
        weights_params.columns = ["loc", "scale"]
        print(weights_params)
        print("")

        print("Adjusted model:")
        adjusted_model = model_df.iloc[:, 1:].copy()
        adjusted_model["covariate"] = np.random.normal(0, 1, size=(n_samples,))
        adjusted_model["interaction"] = adjusted_model["covariate"] * adjusted_model["genotype"]
        adjusted_model["error"] = 1
        print(adjusted_model)
        print("")

        multiplied_df = adjusted_model * weights_df
        print("Multiplied:")
        print(multiplied_df)
        multiplied_params = pd.concat([multiplied_df.mean(axis=0), multiplied_df.std(axis=0)], axis=1)
        multiplied_params.columns = ["loc", "scale"]
        print("")

        expr_a = np.sum(multiplied_df, axis=1)
        # expr_a = (expr_a - np.mean(expr_a)) * (model_params.loc["expression", "scale"] / np.std(expr_a)) + model_params.loc["expression", "loc"]
        print("Expression:")
        print(expr_a)
        print("scale: {:.6f}\tloc: {:.6f}".format(np.mean(expr_a), np.std(expr_a)))
        print("")

        print("Fit OLS model")
        simulated_model = OLS(expr_a, adjusted_model[["intercept", "genotype", "covariate", "interaction"]]).fit()
        print(simulated_model.summary())
        simulated_model_params = pd.concat([simulated_model.params, simulated_model.bse], axis=1)
        simulated_model_params.columns = ["loc", "scale"]
        simulated_model_params.loc["error", "loc"] = np.mean(ols_model_residuals)
        simulated_model_params.loc["error", "scale"] = np.std(ols_model_residuals)
        # print(simulated_model_params)
        print("")

        # print("Harm-Jan test")
        # print(multiplied_df)
        # multiplied_df["interaction"] = np.random.normal(0, 1, size=(n_samples, )) * np.random.normal(ols_model_params.loc["interaction", "loc"], ols_model_params.loc["interaction", "scale"], size=(n_samples, ))
        # print(multiplied_df)
        # print("")
        #
        # expr_a = np.sum(multiplied_df, axis=1) + np.random.normal(0, 1, size=(n_samples, ))
        # print("Expression:")
        # print(expr_a)
        # print("")
        #
        # print("Fit OLS model")
        # simulated_model = OLS(expr_a, model_df[["intercept", "genotype", "covariate", "interaction"]]).fit()
        # print(simulated_model.summary())
        # simulated_model_params = pd.concat([simulated_model.params, simulated_model.bse], axis=1)
        # simulated_model_params.columns = ["loc", "scale"]
        # # print(simulated_model_params)
        # print("")


    def print_arguments(self):
        self.log.info("Arguments:")
        self.log.info("  > Genotype input path: {}".format(self.genotype_path))
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Expression input path: {}".format(self.expression_path))
        self.log.info("  > Covariate input path: {}".format(self.covariate_path))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()