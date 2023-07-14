#!/usr/bin/env python3

"""
File:         simulate_expression2.py
Created:      2023/07/13
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
import pandas as pd
import numpy as np

# Local application imports.

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
./simulate_expression2.py -h

### MetaBrain 2 covariates ###

./simulate_expression2.py \
    -s /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/fast_eqtl_mapper/2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/eQTLSummaryStats.txt.gz \
    -of 2023-07-13-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised
    
### BIOS 2 covariates ###
    
./simulate_expression2.py \
    -s /groups/umcg-bios/tmp01/projects/PICALO/fast_eqtl_mapper/2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised/eQTLSummaryStats.txt.gz \
    -of 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA-first2ExprPCForceNormalised

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.stats_path = getattr(arguments, 'summary_stats')
        self.low_memory = getattr(arguments, 'low_memory')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "simulate_expression2", outfolder)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        parser.add_argument("-s",
                            "--summary_stats",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the ieQTL summary statistics.")
        parser.add_argument("-low_memory",
                            action='store_true',
                            help="Enable low memory mode. Default: False.")
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

        return parser.parse_args()

    def start(self):
        print("Starting program")
        self.print_arguments()

        print("Loading input data")
        stats_df = self.load_file(inpath=self.stats_path, header=0, index_col=None)
        # print(stats_df)

        # Calculate the total number of probes and samples
        n_probes = stats_df.shape[0]
        n_samples = int(stats_df.loc[0, "N"] / stats_df.loc[0, "CR"])
        print("\tN probes: {:,}".format(n_probes))
        print("\tN samples: {:,}".format(n_samples))

        # Determine the beta and std columns.
        beta_columns = [col for col in stats_df.columns if col.startswith("beta-")]
        std_columns = [col for col in stats_df.columns if col.startswith("std-")]

        # Determine the number of hidden covariates
        n_terms = len(beta_columns)
        n_covariates = int((n_terms - 3) / 2)
        print("\tN covariates: {:,}".format(n_covariates))

        # Generate labels.
        probe_labels = ["Probe_{}".format(i) for i in range(n_probes)]
        variant_labels = ["SNP_{}".format(i) for i in range(n_probes)]
        sample_labels = ["Sample_{}".format(i) for i in range(n_samples)]
        covariate_labels = ["Context_{}".format(i) for i in range(n_covariates)]

        print("Construct eQTL data")
        eqtl_df = pd.DataFrame({
            "SNPName": variant_labels,
            "ProbeName": probe_labels,
            "FDR": 0
        })
        print(eqtl_df)

        print("Simulate covariate data")
        # Simulate N normally distributed hidden covariates.
        cov_m = np.random.normal(0, 1, size=(n_samples, n_covariates))
        # print(pd.DataFrame(cov_m, index=sample_labels, columns=covariate_labels))

        print("Simulate genotype data")
        geno_m = np.zeros((n_probes, n_samples), dtype=np.uint8)
        maf_m = np.repeat(stats_df["MAF"].to_numpy()[:, np.newaxis], n_samples, axis=1)
        for i in range(2):
            geno_m += (np.random.uniform(0, 1, size=(n_probes, n_samples)) < maf_m).astype(np.uint8)
        # print(pd.DataFrame(geno_m, index=variant_labels, columns=sample_labels))

        print("Simulate expression data")
        random_m = np.random.normal(0, 1, size=(n_probes, n_samples, n_terms))
        expr_m = None
        if self.low_memory:
            # This implementation is kinda slow but does not require to expand the matrix into a third dimension therefore using less memory.
            beta_m = stats_df.loc[:, beta_columns].to_numpy()
            std_m = stats_df.loc[:, std_columns].to_numpy()
            expr_m = np.zeros((n_probes, n_samples), dtype=np.float64)
            for probe_id in range(n_probes):
                for sample_id in range(n_samples):
                    expr = random_m[probe_id, sample_id, 0] * std_m[probe_id, 0] + beta_m[probe_id, 0]
                    expr += (random_m[probe_id, sample_id, 1] * std_m[probe_id, 1] + beta_m[probe_id, 1]) * geno_m[probe_id, sample_id]
                    for covariate_id in range(n_covariates):
                        expr += (random_m[probe_id, sample_id, 2 + covariate_id] * std_m[probe_id, 2 + covariate_id] + beta_m[probe_id, 2 + covariate_id]) * cov_m[sample_id, covariate_id]
                        expr += (random_m[probe_id, sample_id, 2 + n_covariates + covariate_id] * std_m[probe_id, 2 + n_covariates + covariate_id] + beta_m[probe_id, 2 + n_covariates + covariate_id]) * geno_m[probe_id, sample_id] * cov_m[sample_id, covariate_id]
                    expr += (random_m[probe_id, sample_id, 2 + (n_covariates * 2)] * std_m[probe_id, 2 + (n_covariates * 2)] + beta_m[probe_id, 2 + (n_covariates * 2)])

                    expr_m[probe_id, sample_id] = expr
        else:
            # This code gives the same results but is waaaay faster but uses for memory. I needed ~10Gb RAM for ~13k eQTLs and ~3k samples to run this code.
            beta_m = np.repeat(stats_df.loc[:, beta_columns].to_numpy()[:, np.newaxis, :], n_samples, axis=1)
            std_m = np.repeat(stats_df.loc[:, std_columns].to_numpy()[:, np.newaxis, :], n_samples, axis=1)

            # Construct the ieQTL model matrix.
            model_m = np.ones((n_probes, n_samples, n_terms), dtype=np.float64)  # initialise with 1's for the intercept and noise
            model_m[:, :, 1] = geno_m  # genotype term
            model_m[:, :, 2:(2 + n_covariates)] = np.repeat(cov_m[np.newaxis, :, :], n_probes, axis=0)  # simulated covariates
            model_m[:, :, (2 + n_covariates):(2 + (n_covariates * 2))] = model_m[:, :, [1]] * model_m[:, :, 2:(2 + n_covariates)]  # interactions

            # Calculate the expression
            expr_m = np.sum((random_m * std_m + beta_m) * model_m, axis=2)
        print(pd.DataFrame(expr_m, index=probe_labels, columns=sample_labels))

        print("Save data")
        self.save_file(df=eqtl_df,
                       outpath=os.path.join(self.outdir, "eQTLProbesProbeLevel.txt.gz"))
        self.save_file(df=pd.DataFrame(cov_m, index=sample_labels, columns=covariate_labels).T,
                       outpath=os.path.join(self.outdir, "simulated_covariates.txt.gz"))
        self.save_file(df=pd.DataFrame(geno_m, index=variant_labels, columns=sample_labels),
                       outpath=os.path.join(self.outdir, "genotype_table.txt.gz"))
        self.save_file(df=pd.DataFrame(expr_m, index=probe_labels, columns=sample_labels),
                       outpath=os.path.join(self.outdir, "expression_table.txt.gz"))

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

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
        print("Arguments:")
        print("  > Summary statistics: {}".format(self.stats_path))
        print("  > Low memory: {}".format(self.low_memory))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()