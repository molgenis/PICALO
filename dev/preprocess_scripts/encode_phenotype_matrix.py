#!/usr/bin/env python3

"""
File:         encode_phenotype_matrix.py
Created:      2021/10/13
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
from pathlib import Path
import argparse
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Encode Matrix"
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
./encode_phenotype_matrix.py -filepath /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-02-03-phenotype-table/2020-03-09.brain.phenotypes.txt -header 0 -index_col 4 -low_memory -std /groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/matrix_preparation/CortexEUR-cis-NoCovCorrected-NoENA-NoGVEX/combine_gte_files/SampleToDataset.txt.gz
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.filepath = getattr(arguments, 'filepath')
        self.header = getattr(arguments, 'header')
        self.index_col = getattr(arguments, 'index_col')
        self.low_memory = getattr(arguments, 'low_memory')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.min_samples = getattr(arguments, 'min_samples')

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'encode_phenotype_matrix')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, os.path.basename(self.filepath))

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
        parser.add_argument("-filepath",
                            type=str,
                            required=True,
                            help="The path to the data matrix.")
        parser.add_argument("-header",
                            type=int,
                            default=None,
                            help="Row number(s) to use as the column names, "
                                 "and the start of the data.")
        parser.add_argument("-index_col",
                            type=int,
                            default=None,
                            help="Column(s) to use as the row labels of the "
                                 "DataFrame, either given as string name or "
                                 "column index.")
        parser.add_argument("-low_memory",
                            action='store_false',
                            help="Internally process the file in chunks, "
                                 "resulting in lower memory use while "
                                 "parsing, but possibly mixed type inference.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-min",
                            "--min_samples",
                            type=int,
                            default=20,
                            help="The minimal number of samples for a "
                                 "phenotype.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading data")
        df = pd.read_csv(self.filepath,
                         sep="\t",
                         header=self.header,
                         index_col=self.index_col,
                         low_memory=self.low_memory)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(self.filepath),
                                      df.shape))

        print(df)
        print(df.dtypes)

        if self.std_path is not None:
            print("Select samples")
            std_df = pd.read_csv(self.std_path,
                                 sep="\t",
                                 header=0,
                                 index_col=None)
            df = df.loc[std_df.iloc[:, 0], :]
        print("N-samples: {}".format(df.shape[0]))

        print("Encoding data")
        encoded_dfs = []
        for index, row in df.T.iterrows():
            if index in ["SpecificBrainRegion", "BroadBrainRegion", "cohort", "MetaCohort", "Diagnosis", "BrodmannArea", "library_selection", "Comorbidities", "Reported_Genomic_Mutations", "OtherDiagnosis", "libraryPrep", "TissueState", "RnaSequencingPlatform", "Detailed.Diagnosis", "Agonal.State", "predicted.brain.region"]:
                encoded_df = self.to_dummies(index=index, row=row)
                encoded_dfs.append(encoded_df)
            elif index in ["apoe_genotype", "PMI_(in_hours)", "RIN", "educ", "cts_mmse30", "braaksc", "ceradsc", "cogdx", "CDR", "NP.1", "PlaqueMean", "RNA_isolation_28S_18S_ratio", "RNA_isolation_TotalYield_ug", "RNA_isolation_260_280_ratio", "Total_DNA_ug", "Brain_Weight_(in_grams)", "Height_(Inches)", "Weight_(pounds)", "DNA_isolation_Total_Yield_ug", "DNA_isolation_260_280_ratio", "DNA_isolation_260_230_ratio", "Disease_Duration_(Onset_to_Tracheostomy_or_Death)_in_Months", "C9_repeat_size", "ATXN2_repeat_size", "YearAutopsy", "AgeOnset", "Lifetime_Antipsychotics", "IQ", "ADI.R.A..cut.off.10.", "ADI.R.C..cut.off.3.", "ADI.R.D..cut.off.1."]:
                encoded_df = row.astype(np.float64).to_frame()
                encoded_df.columns = [index.split("_(")[0]]
                encoded_dfs.append(encoded_df)
            elif index in ["Gender", "Family_History_of_ALS/FTD?", "has_C9orf27_repeat_expansion", "has_ATXN2_repeat_expansion", "Modifier_of_ALS_Spectrum_MND_-_Family_History_of_ALS/FTD?", "Modifier_of_ALS_Spectrum_MND_-_FTD?", "Modifier_of_ALS_Spectrum_MND_-_Dementia?", "ERCC_Added", "Smoker", "Seizures", "Pyschiatric.Medications", "sex.by.expression"]:
                codes, _ = pd.factorize(row)
                encoded_df = pd.Series(codes, index=row.index).to_frame()
                encoded_df.columns = [index]
                encoded_df.replace(-1, np.nan, inplace=True)
                encoded_dfs.append(encoded_df)
            elif index in ["Age", "AgeDeath", "AgeAtDiagnosis"]:
                encoded_df = row.to_frame()
                encoded_df.columns = [index]
                encoded_df.loc[~encoded_df[index].isnull(), index] = encoded_df.loc[~encoded_df[index].isnull(), index].map(lambda x: x.rstrip('+').replace(' or Older', '')).astype(np.float64)
                encoded_dfs.append(encoded_df)
            elif index in ["Antipsychotics", "Antidepressants", "Cotinine", "Nicotine"]:
                encoded_df = pd.DataFrame(np.nan, index=row.index, columns=[index])
                encoded_df.loc[row == "Negative", index] = 0
                encoded_df.loc[row == "Positive", index] = 1
                encoded_dfs.append(encoded_df)
            elif index in ["Hemisphere"]:
                encoded_df = pd.DataFrame(np.nan, index=row.index, columns=[index])
                encoded_df.loc[(row == "left") | (row == "Left"), index] = 0
                encoded_df.loc[(row == "right") | (row == "Right"), index] = 1
                encoded_dfs.append(encoded_df)
            elif index in ["Site_of_Motor_Onset", "Site_of_Motor_Onset_Detail"]:
                encoded_df = self.to_dummies(index=index, row=row)
                encoded_df = encoded_df.loc[:, [x for x in encoded_df.columns if x.split("-")[1] not in ["Not Applicable", "Unknown"]]]
                encoded_dfs.append(encoded_df)
            elif index in ["EtohResults"]:
                encoded_df = row.to_frame()
                encoded_df.columns = [index]
                encoded_df.loc[encoded_df[index] == "Not Tested", index] = np.nan
                encoded_df[index] = encoded_df[index].astype(np.float64)
                encoded_dfs.append(encoded_df)
            elif index in ["ADI.R.B..NV..cut.off.7.", "ADI.R.B..V..cut.off.8."]:
                encoded_df = row.to_frame()
                encoded_df.columns = [index]
                encoded_df.loc[encoded_df[index] == "NT", index] = np.nan
                encoded_df[index] = encoded_df[index].astype(np.float64)
                encoded_dfs.append(encoded_df)
            elif index in ["Seizure_notes"]:
                encoded_df = pd.DataFrame(np.nan, index=row.index, columns=["Epilepsy"])
                encoded_df.loc[row == "Epilepsy", index] = 1
                encoded_dfs.append(encoded_df)
        encoded_df = pd.concat(encoded_dfs, axis=1)

        print("Printing data")
        mask = []
        for index, row in encoded_df.T.iterrows():
            value_counts = row.value_counts()

            prefix = ""
            if len(value_counts.index) <= 2 and value_counts.min() < self.min_samples:
                mask.append(False)
                prefix = "[EXCLUDED] "
            else:
                mask.append(True)

            values_str = ', '.join(["{} [N={:,}]".format(name, value) for name, value in value_counts.iteritems()])
            print("{}{}: {}".format(prefix, index, values_str))

        print("Removing columns with too little samples or that are duplicated.")
        encoded_df = encoded_df.loc[:, mask]
        n_sample = encoded_df.shape[0] - encoded_df.isnull().sum(axis=0)
        encoded_df = encoded_df.loc[:, n_sample >= self.min_samples]
        encoded_df = encoded_df.T.drop_duplicates().T
        print(encoded_df)

        print("Saving data.")
        compression = 'infer'
        if self.outpath.endswith('.gz'):
            compression = 'gzip'

        encoded_df.to_csv(self.outpath,
                          sep='\t',
                          header=True,
                          index=True,
                          compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(self.outpath),
                                      encoded_df.shape))

    def to_dummies(self, index, row):
        dummies = pd.get_dummies(row)
        dummies.columns = ["{}-{}".format(index, colname) for colname in dummies.columns]

        return dummies

    def print_arguments(self):
        print("Arguments:")
        print("  > Filepath: {}".format(self.filepath))
        print("  > Header: {}".format(self.header))
        print("  > Index_col: {}".format(self.index_col))
        print("  > Low_memory: {}".format(self.low_memory))
        print("  > Output file: {}".format(self.outpath))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Min samples: {}".format(self.min_samples))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
