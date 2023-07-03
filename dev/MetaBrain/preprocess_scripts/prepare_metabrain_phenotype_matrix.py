#!/usr/bin/env python3

"""
File:         prepare_metabrain_phenotype_matrix.py
Created:      2021/10/13
Last Changed: 2021/12/06
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
import os

# Third party imports.
import pandas as pd
import numpy as np

# Local application imports.

# Metadata
__program__ = "Prepaere MetaBrain Phenotype Matrix"
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
./prepare_metabrain_phenotype_matrix.py -h
"""


class main():
    def __init__(self):
        self.pheno_path = "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-02-03-phenotype-table/2020-03-09.brain.phenotypes.txt"

        # Set variables.
        self.outdir = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'prepare_metabrain_phenotype_matrix')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start(self):
        df = self.load_file(self.pheno_path, index_col=4, header=0, low_memory=False)
        df.index.name = None
        print(df)

        encoded_dfs = []
        for index, row in df.T.iterrows():
            if index in ["SpecificBrainRegion", "BroadBrainRegion", "cohort", "MetaCohort", "Diagnosis", "BrodmannArea", "library_selection", "Comorbidities", "Reported_Genomic_Mutations", "OtherDiagnosis", "libraryPrep", "TissueState", "RnaSequencingPlatform", "Detailed.Diagnosis", "Agonal.State", "predicted.brain.region", "EtohResults"]:
                encoded_df = self.to_dummies(index=index, row=row)
                encoded_dfs.append(encoded_df)
            elif index in ["apoe_genotype", "PMI_(in_hours)", "RIN", "educ", "cts_mmse30", "braaksc", "ceradsc", "cogdx", "CDR", "NP.1", "PlaqueMean", "RNA_isolation_28S_18S_ratio", "RNA_isolation_TotalYield_ug", "RNA_isolation_260_280_ratio", "Total_DNA_ug", "Brain_Weight_(in_grams)", "Height_(Inches)", "Weight_(pounds)", "DNA_isolation_Total_Yield_ug", "DNA_isolation_260_280_ratio", "DNA_isolation_260_230_ratio", "Disease_Duration_(Onset_to_Tracheostomy_or_Death)_in_Months", "C9_repeat_size", "ATXN2_repeat_size", "YearAutopsy", "AgeOnset", "Lifetime_Antipsychotics", "IQ", "ADI.R.A..cut.off.10.", "ADI.R.C..cut.off.3.", "ADI.R.D..cut.off.1."]:
                encoded_df = row.astype(np.float64).to_frame()
                encoded_df.columns = [index.split("_(")[0]]
                encoded_dfs.append(encoded_df)
            elif index in ["Gender", "Family_History_of_ALS/FTD?", "has_C9orf27_repeat_expansion", "has_ATXN2_repeat_expansion", "Modifier_of_ALS_Spectrum_MND_-_Family_History_of_ALS/FTD?", "Modifier_of_ALS_Spectrum_MND_-_FTD?", "Modifier_of_ALS_Spectrum_MND_-_Dementia?", "ERCC_Added", "Smoker", "Seizures", "Pyschiatric.Medications"]:
                codes, _ = pd.factorize(row)
                encoded_df = pd.Series(codes, index=row.index).to_frame()
                encoded_df.columns = [index]
                encoded_df.replace(-1, np.nan, inplace=True)
                encoded_dfs.append(encoded_df)
            elif index in ["sex.by.expression"]:
                encoded_df = pd.DataFrame(np.nan, index=row.index, columns=[index])
                encoded_df.loc[row == "F", index] = 0
                encoded_df.loc[row == "M", index] = 1
                encoded_dfs.append(encoded_df)
            elif index in ["Age", "AgeDeath", "AgeAtDiagnosis"]:
                encoded_df = row.to_frame()
                encoded_df.columns = [index]
                encoded_df.loc[~encoded_df[index].isnull(), index] = encoded_df.loc[~encoded_df[index].isnull(), index].map(lambda x: x.rstrip('+').replace(' or Older', '').replace('?', '').replace('Unknown', '').replace('Not Applicable', ''))
                encoded_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
                encoded_df.loc[:, index] = encoded_df.loc[:, index].astype(np.float64)
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
            if len(value_counts.index) <= 2 and value_counts.min() < 30:
                mask.append(False)
                prefix = "[EXCLUDED] "
            else:
                mask.append(True)

            values_str = ', '.join(["{} [N={:,}]".format(name, value) for name, value in value_counts.iteritems()])
            print("{}{}: {}".format(prefix, index, values_str))

        print("Removing columns with too little samples or that are duplicated.")
        encoded_df = encoded_df.loc[:, mask]
        n_sample = encoded_df.shape[0] - encoded_df.isnull().sum(axis=0)
        encoded_df = encoded_df.loc[:, n_sample >= 30]
        encoded_df = encoded_df.T.drop_duplicates().T
        print(encoded_df)

        print("Saving data")
        self.save_file(df=encoded_df, outpath=os.path.join(self.outdir, "MetaBrain_phenotypes.txt.gz"))

        sex_df = encoded_df.loc[:, ["sex.by.expression"]]
        sex_df.dropna(inplace=True)
        self.save_file(df=sex_df, outpath=os.path.join(self.outdir, "MetaBrain_sex.txt.gz"))
        del sex_df, encoded_df

    @staticmethod
    def load_file(inpath, header, index_col, sep="\t", low_memory=True,
                  nrows=None, skiprows=None):
        df = pd.read_csv(inpath, sep=sep, header=header, index_col=index_col,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows)
        print("\tLoaded dataframe: {} "
              "with shape: {}".format(os.path.basename(inpath),
                                      df.shape))
        return df

    def to_dummies(self, index, row):
        dummies = pd.get_dummies(row)
        dummies.columns = ["{}-{}".format(index, colname) for colname in dummies.columns]

        return dummies

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


if __name__ == '__main__':
    m = main()
    m.start()
