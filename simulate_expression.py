#!/usr/bin/env python3

"""
File:         simulate_expression.py
Created:      2023/06/07
Last Changed: 2023/06/08
Author:       M.Vochteloo, based on work by Zhou et al. Genome Biology (2022).

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
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.
from src.logger import Logger

# Metadata
__program__ = "Simulate Expression"
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
./simulate_expression.py -h

# Real eQTL beta.
./simulate_expression.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -al /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_alleles_table.txt.gz \
    -use_real_genotype_beta \
    -resample_covariates \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
    
# All simulated data.
./simulate_expression.py \
    -eq /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz \
    -ge /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/genotype_table.txt.gz \
    -resample_covariates \
    -of 2023-06-08-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.eqtl = getattr(arguments, 'eqtl')
        self.genotype = getattr(arguments, 'genotype')
        self.genotype_na = getattr(arguments, 'genotype_na')
        self.alleles = getattr(arguments, 'alleles')
        self.use_real_genotype_beta = getattr(arguments, 'use_real_genotype_beta')
        self.n_individuals = getattr(arguments, 'individuals')
        self.resample_individuals = getattr(arguments, 'resample_individuals')
        self.n_experiments = getattr(arguments, 'experiments')
        self.n_eqtls = getattr(arguments, 'eqtls')
        self.n_known_covariates = getattr(arguments, 'known_covariates')
        self.n_hidden_covariates = getattr(arguments, 'hidden_covariates')
        self.resample_covariates = getattr(arguments, 'resample_covariates')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        if self.use_real_genotype_beta and self.alleles is None:
            self.log.error("\tError, -al / --alleles required for using"
                           "real eQTL beta's.")
            exit()

        # Set variables.
        self.n_covariates = self.n_known_covariates + self.n_hidden_covariates
        self.n_terms = (self.n_covariates * 2) + 2
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
                            required=False,
                            default=None,
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
        parser.add_argument("-al",
                            "--alleles",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the alleles matrix")
        parser.add_argument("-use_real_genotype_beta",
                            action='store_true',
                            help="Use the real genotype beta. Default: False.")
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
        parser.add_argument("-k1",
                            "--known_covariates",
                            type=int,
                            required=False,
                            default=2,
                            help="The number of known covariates. Default: 2.")
        parser.add_argument("-k2",
                            "--hidden_covariates",
                            type=int,
                            required=False,
                            default=3,
                            help="The number of hidden covariates. Default: 3.")
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

        # Generating genotype beta.
        geno_beta_a = np.random.normal(0, 0.013, size=(self.n_eqtls, ))
        if self.use_real_genotype_beta:
            self.log.info("Using real genotype beta's")
            alleles_df = self.load_file(inpath=self.alleles, header=0, index_col=0,
                                        nrows=self.n_eqtls)
            print(alleles_df)

            alleles_df["AlleleAssessed"] = alleles_df["Alleles"].str.split("/", n=1, expand=True)[0]
            eqtl_beta_flip = pd.DataFrame({"ref_aa": eqtl_df["AlleleAssessed"].tolist(),
                                           "alt_aa": alleles_df["AlleleAssessed"].tolist()})
            eqtl_beta_flip["flip"] = (eqtl_beta_flip["ref_aa"] == eqtl_beta_flip["alt_aa"]).map({True: 1, False: -1})
            geno_beta_a = eqtl_df["Meta-Beta (SE)"].str.split(" ", n=1, expand=True)[0].astype(float).to_numpy()
            geno_beta_a *= eqtl_beta_flip["flip"].to_numpy()
            del alleles_df, eqtl_beta_flip
        del eqtl_df

        self.log.info("Selecting individuals")
        individuals_mask = np.ones(n_individuals, dtype=np.bool)
        if self.n_individuals is not None and self.n_individuals != n_individuals:
            if self.n_individuals > n_individuals:
                self.log.error("\tError, more samples requested than available.")
                exit()
            individuals_mask[np.random.choice(n_individuals,
                                              size=n_individuals - self.n_individuals,
                                              replace=False)] = 0

        self.log.info("Generate covariate matrix")
        # Each entry is drawn from N(0,1).
        cov_m = np.random.normal(0, 1, size=(self.n_individuals, self.n_covariates))

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

            self.log.info("\tConstructing model matrix")
            # Generate the empty model matrix.
            model_m = np.empty(shape=(self.n_eqtls, self.n_individuals, self.n_terms))

            # Fill in the real genotype data.
            model_m[:, :, 0] = geno_df.iloc[:, individuals_mask].copy().to_numpy()
            sample_names = geno_df.columns[individuals_mask].tolist()

            # Resample the covariate matrix if need be. Each entry is drawn
            # from N(0,1).
            if self.resample_covariates:
                self.log.info("\tResampling covariates")
                cov_m = np.random.normal(0, 1, size=(self.n_individuals, self.n_covariates))

            # Fill in the simulated covariate data.
            model_m[:, :, 1:self.n_covariates + 1] = np.repeat(cov_m[np.newaxis, :, :], self.n_eqtls, axis=0)

            # Calculate and fill the interaction data.
            model_m[:, :, self.n_covariates + 1:self.n_terms - 1] = model_m[:, :, 1:self.n_covariates + 1] * model_m[:, :, [0]]

            # Generate the noise matrix. Each entry is drawn from N(0,1).
            noise_m = np.random.normal(0, 1, size=(self.n_eqtls, self.n_individuals))
            # Fill in the noise matrix.
            model_m[:, :, self.n_terms - 1] = noise_m

        ########################################################################

            self.log.info("\tGenerating model beta's")
            # Generate the model beta's.
            beta_m = np.empty(shape=(self.n_eqtls, self.n_terms))
            # Fill in the real genotype beta's.
            beta_m[:, 0] = geno_beta_a
            for i in range(self.n_covariates):
                beta_m[:, i + 1] = np.random.normal(0, 0.042, size=(self.n_eqtls, ))
            for i in range(self.n_covariates):
                beta_m[:, i + self.n_covariates + 1] = np.random.normal(0, 0.056, size=(self.n_eqtls, ))
            beta_m[:, self.n_terms - 1] = 1

            self.log.info("\tCalculating simulated expression")
            # Generate the expression matrix.
            expr_m = np.einsum('ijk,ik->ij', model_m, beta_m)

        ########################################################################

            self.log.info("\tSaving data")

            # Generate the model columns.
            model_columns = ["genotype"] + ["known_covariate{}".format(i) for i in range(self.n_known_covariates)] + ["hidden_covariate{}".format(i) for i in range(self.n_hidden_covariates)] + ["known_covariate_interaction{}".format(i) for i in range(self.n_known_covariates)] + ["hidden_covariate_interaction{}".format(i) for i in range(self.n_hidden_covariates)] + ["noise"]

            # Save the expression matrix.
            self.save_file(df=pd.DataFrame(cov_m[:, :self.n_known_covariates], index=sample_names, columns=model_columns[:self.n_known_covariates]),
                           outpath=os.path.join(exp_outdir, "tech_covariates_with_interaction_df.txt.gz"))
            self.save_file(df=pd.DataFrame(cov_m[:, self.n_known_covariates:self.n_covariates], index=sample_names, columns=model_columns[self.n_known_covariates:self.n_covariates]),
                           outpath=os.path.join(exp_outdir, "PICs.txt.gz"))
            self.save_file(df=pd.DataFrame(noise_m, index=eqtl_names, columns=sample_names),
                           outpath=os.path.join(exp_outdir, "noise.txt.gz"))
            self.save_file(df=pd.DataFrame(beta_m, index=eqtl_names, columns=model_columns),
                           outpath=os.path.join(exp_outdir, "model_betas.txt.gz"))
            self.save_file(df=pd.DataFrame(expr_m, index=gene_names, columns=sample_names),
                           outpath=os.path.join(exp_outdir, "expression_table.txt.gz"))

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
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        print(outpath)
        print(df)
        return
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
        self.log.info("  > Genotype data: {}".format(self.genotype))
        self.log.info("  > Genotype NA value: {}".format(self.genotype_na))
        self.log.info("  > Alleles data: {}".format(self.alleles))
        self.log.info("  > Use real genotype beta: {}".format(self.use_real_genotype_beta))
        self.log.info("  > Individuals: {}".format(self.n_individuals))
        self.log.info("  > Resample individuals: {}".format(self.resample_individuals))
        self.log.info("  > N experiments: {}".format(self.n_experiments))
        self.log.info("  > N eQTLs: {}".format(self.n_eqtls))
        self.log.info("  > N known covariates: {}".format(self.n_known_covariates))
        self.log.info("  > N hidden covariates: {}".format(self.n_hidden_covariates))
        self.log.info("  > Resample covariates: {}".format(self.resample_covariates))
        self.log.info("  > Output directory: {}".format(self.outdir))
        self.log.info("")


if __name__ == '__main__':
    m = main()
    m.start()