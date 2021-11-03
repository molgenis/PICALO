#!/usr/bin/env python3

"""
File:         pre_process_bios_gte_and_std.py
Created:      2021/10/28
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
import glob
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Pre-Process BIOS GTE and STD"
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
./pre_process_bios_gte_and_std.py -i /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -e /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -o ../BIOS_GTESubset

./pre_process_bios_gte_and_std.py -i /groups/umcg-bios/tmp01/projects/decon_optimizer/data/datasets_biosdata -gte /groups/umcg-bios/prm03/projects/BIOS_EGCUT_for_eQTLGen/BIOS_EGCUT/eqtlpipeline_bios_egcut_backup010517/GTE_LLDEEP_and_BIOS_last_related_removed_110417.txt -e /groups/umcg-bios/tmp01/projects/BIOS_for_eQTLGenII/data/BIOS_EGCUT_for_eQTLGen/BIOS_only/eqtlpipeline_lld_backup150317/1-normalise/normalise/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.CPM.Log2Transformed.ProbesCentered.SamplesZTransformed.txt -se ../data/BIOS-allchr-mds-BIOS-GTESubset-VariantSubsetFilter_outliers.txt.gz -o ../BIOS_GTESubset_noOutlier
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_directory = getattr(arguments, 'input')
        self.gte_path = getattr(arguments, 'gene_to_expression')
        self.expression_path = getattr(arguments, 'expression')
        self.se_path = getattr(arguments, 'sample_exclude')
        self.outdir = getattr(arguments, 'output')

    @staticmethod
    def create_argument_parser():
        parser = argparse.ArgumentParser(prog=__program__,
                                         description=__description__,
                                         )

        # Add optional arguments.
        parser.add_argument("-v",
                            "--version",
                            action="version",
                            version="{} {}".format(__program__,
                                                   __version__),
                            help="show program's version number and exit.")
        parser.add_argument("-i",
                            "--input",
                            type=str,
                            required=True,
                            help="The path to the input directory.")
        parser.add_argument("-gte",
                            "--gene_to_expression",
                            type=str,
                            required=True,
                            help="The path to the gene-to-expression coupling "
                                 "table.")
        parser.add_argument("-e",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix.")
        parser.add_argument("-se",
                            "--sample_exclude",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample exclude file.")
        parser.add_argument("-o",
                            "--output",
                            type=str,
                            required=True,
                            help="The path to the output directory.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading GTE data")
        gte_df = self.load_file(self.gte_path, index_col=None)
        gte_df.columns = ["genotype_id", "rnaseq_id"]
        gte_dict = dict(zip(gte_df["genotype_id"], gte_df["rnaseq_id"]))
        print(gte_df)

        ########################################################################

        print("Loading dosage data")

        # Find the datasets folders (i.e. subdirectories).
        datasets = [name for name in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, name))]

        # Loop over datasets
        dosage_dfs = []
        gtd_data = []
        for dataset in datasets:
            filepaths = []
            for filepath in glob.glob(os.path.join(self.input_directory, dataset, "output.dosages.*")):
                filepaths.append(filepath)

            if len(filepaths) != 1:
                print("Unexpected number of output.dosage.* files.")
                exit()

            dosage_df = self.load_file(filepaths[0], nrows=None)
            dosage_dfs.append(dosage_df)

            for sample in dosage_df.columns:
                gtd_data.append([sample, dataset])

        # Merge the data frames.
        comb_dosage_df = pd.concat(dosage_dfs, axis=1)

        # Fille with -1 als missing.
        comb_dosage_df.fillna(-1, inplace=True)
        print(comb_dosage_df)

        ########################################################################

        # Add the genotype to dataset info to the gte file.
        gtd_df = pd.DataFrame(gtd_data, columns=["genotype_id", "dataset"])
        gte_df = gte_df.merge(gtd_df, on="genotype_id", how="left")
        print(gte_df)
        print(gte_df["dataset"].value_counts())

        del gtd_df

        ########################################################################

        # Load the expression file.
        expr_df = self.load_file(self.expression_path, nrows=1)

        # Get the ids of the data that we have.
        genotype_ids = set(comb_dosage_df.columns)
        rnaseq_ids = set(expr_df.columns)

        # Check per GTE match if we have both the genotype sample in the
        # dosage matrix and the expression sample in the expression matrix.
        mask = np.zeros(gte_df.shape[0], dtype=bool)
        genotype_missing = 0
        expression_missing = 0
        for i, (_, (genotype_id, rnaseq_id, _)) in enumerate(gte_df.iterrows()):
            if genotype_id not in genotype_ids:
                genotype_missing += 1
            if rnaseq_id not in rnaseq_ids:
                expression_missing += 1

            if genotype_id in genotype_ids and rnaseq_id in rnaseq_ids:
                mask[i] = True

        n_missing = np.size(mask) - np.sum(mask)
        if n_missing > 0:
            print("Warning, for {} samples I can't find both expression"
                  "and genotype data".format(n_missing))
            print(gte_df.loc[~mask, :])
            print("\tGenotype missing: {}".format(genotype_missing))
            print("\tExpression missing: {}".format(expression_missing))

            # Remove missing from gte file.
            gte_df = gte_df.loc[mask, :]

        ########################################################################

        # Remove other samples.
        if self.se_path is not None:
            se_df = self.load_file(self.se_path, header=None, index_col=None)
            exclude_samples = list(se_df.iloc[:, 0].values)

            # Remove samples from gte file.
            mask = np.ones(gte_df.shape[0], dtype=bool)
            for i, (_, (genotype_id, rnaseq_id, _)) in enumerate(gte_df.iterrows()):
                if genotype_id in exclude_samples or rnaseq_id in exclude_samples:
                    mask[i] = False
            print("\tSample exclude path is given, removing {} samples".format(np.size(mask) - np.sum(mask)))
            gte_df = gte_df.loc[mask, :]

        ########################################################################

        # Define our sample list.
        genotype_ids = list(gte_df["genotype_id"])

        # Subset the genotype data.
        comb_dosage_df = comb_dosage_df.loc[:, genotype_ids]

        # Translate our genotype_id in the comb_dosage_df matrix to rnaseq_id.
        comb_dosage_df.columns = [gte_dict[genotype_id] for genotype_id in comb_dosage_df.columns]

        del gte_dict

        ########################################################################

        print("Saving files.")

        # Genotype dosage file.
        self.save_file(df=comb_dosage_df, outpath=os.path.join(self.outdir, "GenotypeMatrix.txt.gz"))

        # Gene-to-expression file.
        self.save_file(df=gte_df, outpath=os.path.join(self.outdir, "BIOS_GenotypeToExpression.txt.gz"), index=False)

        # Sample-to-dataset file.
        std_df = gte_df.loc[:, ["rnaseq_id", "dataset"]]
        std_df.columns = ["sample", "dataset"]
        self.save_file(df=std_df, outpath=os.path.join(self.outdir, "BIOS_SampleToDataset.txt.gz"), index=False)

        # Family-genotype file (for MDS analyses).
        gte_fid_df = gte_df.loc[:, ["genotype_id"]].copy()
        gte_fid_df.insert(0, "family_id", 0)
        self.save_file(df=gte_fid_df, outpath=os.path.join(self.outdir, "GTE-BIOS-all-fid.txt"), header=False, index=False)

    @staticmethod
    def load_file(inpath, header=0, index_col=0, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        if inpath.endswith(".pkl"):
            df = pd.read_pickle(inpath)
        else:
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
        print("  > Input directory: {}".format(self.input_directory))
        print("  > GTE path: {}".format(self.gte_path))
        print("  > Expression path: {}".format(self.expression_path))
        print("  > Sample exclude path: {}".format(self.se_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
