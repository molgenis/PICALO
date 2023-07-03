#!/usr/bin/env python3

"""
File:         export_masked_matrices.py
Created:      2022/07/11
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
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Export Masked Matrices"
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
./export_masked_matrices.py -h

./export_masked_matrices.py \
    -ex /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/pre_process_expression_matrix/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_WithUncenteredPCA/data/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -pic /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz
    
./export_masked_matrices.py \
    -ex /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/expression_table_CovariatesRemovedOLS.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/sample_to_dataset.txt.gz \
    -pic /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA/PICs.txt.gz
    
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.expr_path = getattr(arguments, 'expression')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.pic_path = getattr(arguments, 'pic_loadings')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'export_masked_matrices')
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
        parser.add_argument("-ex",
                            "--expression",
                            type=str,
                            required=True,
                            help="The path to the expression matrix.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=True,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-pic",
                            "--pic_loadings",
                            type=str,
                            required=True,
                            help="The path to the PICALO PIC loadings matrix.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("### Step1 ###")
        print("Loading data")
        expr_df = self.load_file(self.expr_path, header=0, index_col=0)
        std_df = self.load_file(self.std_path, header=0, index_col=None)
        pics_df = self.load_file(self.pic_path, header=0, index_col=0)

        print(expr_df)
        print(std_df)
        print(pics_df)

        print("### Step2 ###")
        print("Validating data")
        samples = std_df.iloc[:, 0].values.tolist()
        datasets = std_df.iloc[:, 1].unique().tolist()

        if expr_df is not None and expr_df.columns.tolist() != samples:
            print("\tThe expression file header does not match "
                  "the sample-to-dataset link file")
            exit()

        if pics_df is not None and pics_df.columns.tolist() != samples:
            print("\tThe PIC loadings file header does not match "
                  "the sample-to-dataset link file")
            exit()

        print("### Step3 ###")
        print("Creating mask")
        n_samples = std_df.shape[0]
        n_datasets = len(datasets)
        sample_trans_dict = {sample: "sample{:0{}d}".format(i, len(str(n_samples))) for i, sample in enumerate(samples)}
        dataset_trans_dict = {dataset: "dataset{:0{}d}".format(i, len(str(n_datasets))) for i, dataset in enumerate(datasets)}

        self.save_dict(dict=sample_trans_dict, order=samples, outpath=os.path.join(self.outdir, "sample_translate.txt.gz"))
        self.save_dict(dict=dataset_trans_dict, order=datasets, outpath=os.path.join(self.outdir, "dataset_translate.txt.gz"))

        print("### Step3 ###")
        print("Masking matrices")
        masked_expr_df = expr_df.copy()
        masked_expr_df.columns = [sample_trans_dict[sample] for sample in masked_expr_df.columns]
        print(masked_expr_df)

        masked_std_df = std_df.copy()
        masked_std_df["sample"] = masked_std_df["sample"].map(sample_trans_dict)
        masked_std_df["dataset"] = masked_std_df["dataset"].map(dataset_trans_dict)
        print(masked_std_df)

        masked_pics_df = pics_df.copy()
        masked_pics_df.columns = [sample_trans_dict[sample] for sample in pics_df.columns]
        print(masked_pics_df)

        print("### Step4 ###")
        print("Saving matrices")
        self.save_file(df=masked_expr_df, outpath=os.path.join(self.outdir, os.path.basename(self.expr_path).replace(".txt.gz", "_masked.txt.gz")))
        self.save_file(df=masked_std_df, outpath=os.path.join(self.outdir, os.path.basename(self.std_path).replace(".txt.gz", "_masked.txt.gz")), index=False)
        self.save_file(df=masked_pics_df, outpath=os.path.join(self.outdir, os.path.basename(self.pic_path).replace(".txt.gz", "_masked.txt.gz")))

    @staticmethod
    def load_file(inpath, header=None, index_col=None, sep="\t", low_memory=True,
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

    def save_dict(self, dict, order, outpath):
        data = []
        for value in order:
            data.append([value, dict[value]])

        self.save_file(df=pd.DataFrame(data, columns=["key", "value"]),
                       outpath=outpath,
                       index=False)

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
        print("  > Expression path: {}".format(self.expr_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > PICs path: {}".format(self.pic_path))
        print("  > Output directory: {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
