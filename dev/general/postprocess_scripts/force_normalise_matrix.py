#!/usr/bin/env python3

"""
File:         force_normalise_matrix.py
Created:      2021/11/03
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
import os

# Third party imports.
import numpy as np
import pandas as pd
from scipy.special import ndtri

# Local application imports.

# Metadata
__program__ = "Force normalise matrix"
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
./force_normalise_matrix.py -h

### MetaBrain ###

./force_normalise_matrix.py \
    -d /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/2021-12-07-CortexEUR-cis-ProbesWithZeroVarianceRemoved/perform_deconvolution/deconvolution_table.txt.gz \
    -transpose \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -o 2021-12-10-CellFractions_MetaBrain_CortexEUR_ExonTPM_Log2_NoDev_MajorCellTypes

./force_normalise_matrix.py \
    -d /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/pre_process_expression_matrix/MetaBrain_CortexEUR_NoENA_NoMDSOutlier_NoRNAseqAlignmentMetrics/data/MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.txt.gz \
    -std /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/MetaBrain-CortexEUR-cis-NoENA-NoMDSOutlier-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -o 2021-12-10-MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS

### BIOS ###

./force_normalise_matrix.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages.txt.gz \
    -transpose \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -o 2021-12-10-CellFractionPercentages_BIOS

./force_normalise_matrix.py \
    -d /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_expression_matrix/BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics/data/gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.txt.gz \
    -std /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/BIOS-BIOS-cis-NoRNAPhenoNA-NoSexNA-NoMixups-NoMDSOutlier-NoRNAseqAlignmentMetrics-GT1AvgExprFilter-PrimaryeQTLs/sample_to_dataset.txt.gz \
    -o 2021-12-10-gene_read_counts_BIOS_and_LLD_passQC.tsv.SampleSelection.ProbesWithZeroVarianceRemoved.TMM.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.transpose = getattr(arguments, 'transpose')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.out_filename = getattr(arguments, 'outfile')

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'force_normalise_matrix')
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
        parser.add_argument("-d",
                            "--data",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-transpose",
                            action='store_true',
                            help="Combine the created files with force."
                                 " Default: False.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-o",
                            "--outfile",
                            type=str,
                            default="output",
                            help="The name of the outfile. Default: output.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data.")
        data_df = self.load_file(self.data_path, header=0, index_col=0, nrows=None)
        std_df = self.load_file(self.std_path, header=0, index_col=None)

        print("Pre-process")
        if self.transpose:
            data_df = data_df.T

        print("Subset samples")
        samples = [x for x in list(std_df.iloc[:, 0].values) if x in set(data_df.columns)]
        data_df = data_df.loc[:, samples]
        std_df = std_df.loc[std_df.iloc[:, 0].isin(samples), :]
        print(data_df)
        print(std_df)

        print("Construct dataset matrix")
        dataset_df = self.construct_dataset_df(std_df=std_df)
        dataset_m = dataset_df.to_numpy(bool)
        del std_df

        print("Force normalise")
        normal_data_df = self.force_normalise(data_df=data_df,
                                              dataset_m=dataset_m)

        if self.transpose:
            normal_data_df = normal_data_df.T

        print("Save file")
        self.save_file(df=normal_data_df,
                       outpath=os.path.join(self.outdir,  "{}_ForceNormalised.txt.gz".format(self.out_filename))
                       )

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
    def construct_dataset_df(std_df):
        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]

        dataset_df = pd.DataFrame(0, index=std_df.iloc[:, 0], columns=datasets)
        for dataset in datasets:
            dataset_df.loc[(std_df.iloc[:, 1] == dataset).values, dataset] = 1
        dataset_df.index.name = "-"

        return dataset_df

    @staticmethod
    def force_normalise(data_df, dataset_m):
        normal_df = pd.DataFrame(np.nan, index=data_df.index, columns=data_df.columns)
        for cohort_index in range(dataset_m.shape[1]):
            mask = dataset_m[:, cohort_index]
            if np.sum(mask) > 0:
                normal_df.loc[:, mask] = ndtri((data_df.loc[:, mask].rank(axis=1, ascending=True) - 0.5) / np.sum(mask))

        return normal_df

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
        print("  > Data path: {}".format(self.data_path))
        print("  > Transpose: {}".format(self.transpose))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Output filename: {}".format(self.out_filename))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
