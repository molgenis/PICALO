#!/usr/bin/env python3

"""
File:         split_picalo_files.py
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
import math
import os

# Third party imports.
import numpy as np
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Split PICALO files"
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
./split_picalo_files.py -h


### BIOS ###
./split_picalo_files.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -of 2023-07-13-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -split_eqtls \
    -split_samples
    
### MetaBrain ###

./split_picalo_files.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/ \
    -of 2023-07-17-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -split_samples
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_dir = getattr(arguments, 'input_dir')
        self.split_eqtls = getattr(arguments, 'split_eqtls')
        self.split_samples = getattr(arguments, 'split_samples')
        self.step = 250
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        if outdir is None:
            outdir = str(os.path.dirname(os.path.abspath(__file__)))
        self.outdir = os.path.join(outdir, "split_picalo_files", outfolder)
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
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--input_dir",
                            type=str,
                            required=True,
                            help="")
        parser.add_argument("-split_eqtls",
                            action='store_true',
                            help="")
        parser.add_argument("-split_samples",
                            action='store_true',
                            help="")
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
        self.print_arguments()

        print("Processing data files.")
        std_df = self.load_file(os.path.join(self.input_dir, "sample_to_dataset.txt.gz"), header=0, index_col=None)
        dataset_df = self.load_file(os.path.join(self.input_dir, "datasets_table.txt.gz"), header=0, index_col=0)
        eqtl_df = self.load_file(os.path.join(self.input_dir, "eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"), header=0, index_col=None)
        geno_df = self.load_file(os.path.join(self.input_dir, "genotype_table.txt.gz"), header=0, index_col=0)
        allele_df = self.load_file(os.path.join(self.input_dir, "genotype_alleles_table.txt.gz"), header=0, index_col=0)
        expr_df = self.load_file(os.path.join(self.input_dir, "expression_table.txt.gz"), header=0, index_col=0)
        pcpc_df = self.load_file(os.path.join(self.input_dir, "first100ExpressionPCs.txt.gz"), header=0, index_col=0)
        sex_df = self.load_file(os.path.join(self.input_dir, "sex_table.txt.gz"), header=0, index_col=0)
        mds_df = self.load_file(os.path.join(self.input_dir, "mds_table.txt.gz"), header=0, index_col=0)
        correction_df = self.load_file(os.path.join(self.input_dir, "tech_covariates_with_interaction_df.txt.gz"), header=0, index_col=0)

        if self.split_eqtls:
            print("Splitting data files in odd and even chromosomes.")
            even_mask = (eqtl_df["SNPChr"] % 2 == 0).to_numpy()
            odd_mask = (eqtl_df["SNPChr"] % 2 != 0).to_numpy()
            print("\tFound {:,} eQTLs on the even chromosomes [{:.2f}%].".format(np.sum(even_mask), (100 / eqtl_df.shape[0]) * np.sum(even_mask)))
            print("\tFound {:,} eQTLs on the odd chromosomes [{:.2f}%].".format(np.sum(odd_mask), (100 / eqtl_df.shape[0]) * np.sum(odd_mask)))
            print("")

            for mask, label in ((even_mask, "even"), (odd_mask, "odd")):
                print("\tProcessing {}".format(label))
                subset_outdir = os.path.join(self.outdir, label)
                if not os.path.exists(subset_outdir):
                    os.makedirs(subset_outdir)

                self.save_file(df=std_df, outpath=os.path.join(subset_outdir, "sample_to_dataset.txt.gz"), index=False)
                self.save_file(df=dataset_df, outpath=os.path.join(subset_outdir, "datasets_table.txt.gz"))
                self.save_file(df=eqtl_df.loc[mask, :], outpath=os.path.join(subset_outdir, "eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"), index=False)
                self.save_file(df=geno_df.loc[mask, :], outpath=os.path.join(subset_outdir, "genotype_table.txt.gz"))
                self.save_file(df=allele_df.loc[mask, :], outpath=os.path.join(subset_outdir, "genotype_alleles_table.txt.gz"))
                self.save_file(df=expr_df.loc[mask, :], outpath=os.path.join(subset_outdir, "expression_table.txt.gz"))
                for n_pcs in range(5, 105, 5):
                    if n_pcs <= pcpc_df.shape[0]:
                        self.save_file(df=pcpc_df.iloc[:n_pcs, :], outpath=os.path.join(subset_outdir, "first{}ExpressionPCs.txt.gz".format(n_pcs)))
                self.save_file(df=sex_df, outpath=os.path.join(subset_outdir, "sex_table.txt.gz"))
                self.save_file(df=mds_df, outpath=os.path.join(subset_outdir, "mds_table.txt.gz"))
                self.save_file(df=correction_df, outpath=os.path.join(subset_outdir, "tech_covariates_with_interaction_df.txt.gz"))
                print("")

        if self.split_samples:
            print("Splitting data files in different number of samples.")
            dss_df = dataset_df.sum(axis=0).to_frame()
            dss_df.columns = ["N"]
            dss_df["frac"] = dss_df / dss_df.sum(axis=0)
            dss_df["subset"] = np.nan
            dss_df.sort_values(by="N", ascending=False, inplace=True)
            print(dss_df)
            print("")

            min_n = math.ceil(30 * dataset_df.shape[1] / self.step) * self.step
            max_n = math.ceil(dataset_df.shape[0] / self.step) * self.step

            for n_samples in range(min_n, max_n, self.step):
                print("\tProcessing {:,} samples".format(n_samples))
                subset_outdir = os.path.join(self.outdir, "{}Samples".format(n_samples))
                if not os.path.exists(subset_outdir):
                    os.makedirs(subset_outdir)

                # Calculate the proportion number of samples to select per dataset.
                dss_df_copy = dss_df.copy()
                dss_df_copy["subset"] = self.round_series_retain_integer_sum(xs=(dss_df_copy["frac"] * n_samples).to_numpy(), N=n_samples)
                if dss_df_copy["subset"].min() < 30:
                    mask = dss_df_copy["subset"] < 30
                    dss_df_copy.loc[mask, "subset"] = 30
                    n_remaining = n_samples - dss_df_copy.loc[mask, "subset"].sum()
                    dss_df_copy.loc[~mask, "frac"] = dss_df_copy.loc[~mask, "N"] / dss_df_copy.loc[~mask, "N"].sum()
                    dss_df_copy.loc[~mask, "subset"] = self.round_series_retain_integer_sum((dss_df_copy.loc[~mask, "frac"] * n_remaining).to_numpy(), N=n_remaining)

                # Pick random samples.
                mask = np.zeros(dataset_df.shape[0])
                for dataset, row in dss_df_copy.iterrows():
                    dataset_mask = dataset_df.loc[:, dataset].to_numpy()
                    select_mask = np.zeros((np.sum(dataset_mask)))
                    select_mask[:int(row["subset"])] = 1
                    np.random.shuffle(select_mask)
                    mask[dataset_mask.astype(bool)] = select_mask
                mask = mask.astype(bool)

                # Check samples.
                print(dataset_df.loc[mask, :].sum(axis=0))
                print("")

                # Save output.
                self.save_file(df=std_df.loc[mask, :], outpath=os.path.join(subset_outdir, "sample_to_dataset.txt.gz"), index=False)
                self.save_file(df=dataset_df.loc[mask, :], outpath=os.path.join(subset_outdir, "datasets_table.txt.gz"))
                self.save_file(df=eqtl_df, outpath=os.path.join(subset_outdir, "eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"), index=False)
                self.save_file(df=geno_df.loc[:, mask], outpath=os.path.join(subset_outdir, "genotype_table.txt.gz"))
                self.save_file(df=allele_df, outpath=os.path.join(subset_outdir, "genotype_alleles_table.txt.gz"))
                self.save_file(df=expr_df.loc[:, mask], outpath=os.path.join(subset_outdir, "expression_table.txt.gz"))
                for n_pcs in range(5, 105, 5):
                    if n_pcs <= pcpc_df.shape[0]:
                        self.save_file(df=pcpc_df.iloc[:n_pcs, :].loc[:, mask], outpath=os.path.join(subset_outdir, "first{}ExpressionPCs.txt.gz".format(n_pcs)))
                self.save_file(df=sex_df.loc[:, mask], outpath=os.path.join(subset_outdir, "sex_table.txt.gz"))
                self.save_file(df=mds_df.loc[:, mask], outpath=os.path.join(subset_outdir, "mds_table.txt.gz"))
                self.save_file(df=correction_df.loc[:, mask], outpath=os.path.join(subset_outdir, "tech_covariates_with_interaction_df.txt.gz"))
                print("")

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
    def round_series_retain_integer_sum(xs, N):
        """
        https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal/51451847#51451847
        """
        Rs = np.floor(xs).astype(int)
        K = N - np.sum(Rs)
        assert K == round(K)
        fs = xs - Rs
        indices = (-fs).argsort()
        ys = Rs
        for i in range(np.max(K)):
            ys[indices[i]] += 1
        return ys

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
        print("  > Input directory: {}".format(self.input_dir))
        print("  > Split eqtls: {}".format(self.split_eqtls))
        print("  > Split samples: {}".format(self.split_samples))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
