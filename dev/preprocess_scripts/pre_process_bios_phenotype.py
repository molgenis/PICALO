#!/usr/bin/env python3

"""
File:         pre_process_bios_phenotype.py
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
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Pre-Process BIOS Phenotype"
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
./pre_process_bios_phenotype.py -h
"""


class main():
    def __init__(self):
        self.input_dir = "/groups/umcg-bios/tmp04/projects/phenotypes"
        self.gte_file = "../data/BIOS_GenotypeToExpression.txt.gz"
        self.datasets = {'LL', 'NTR', 'RS', 'PAN', 'LLS', 'CODAM'}

    def start(self):
        gte_df = self.load_file(self.gte_file, sep="\t", header=0, index_col=None)
        gte_df["major_dataset"] = gte_df["dataset"].map({"RS": "RS",
                                                         "PAN": "PAN",
                                                         "LL": "LL",
                                                         "CODAM": "CODAM",
                                                         "LLS_660Q": "LLS",
                                                         "NTR_GONL": "NTR",
                                                         "LLS_OmniExpr": "LLS",
                                                         "NTR_AFFY": "NTR",
                                                         "GONL": "GONL"})
        print(gte_df)

        trans_dict = {}
        for dataset in gte_df["major_dataset"].unique():
            dataset_trans_dict = {}
            gtd_subset_df = gte_df.loc[gte_df["major_dataset"] == dataset, :]
            for genotype_id in gtd_subset_df["genotype_id"]:
                if "_" in genotype_id:
                    dataset_trans_dict[genotype_id.split("_")[-1]] = genotype_id
            trans_dict[dataset] = dataset_trans_dict

        gte_genotype_ids = set(gte_df["genotype_id"])

        bios_pheno_df = self.load_file(os.path.join(self.input_dir, "BIOS_phenotypes_25072017.txt"), sep="\t", header=0, index_col=None)
        print(bios_pheno_df)

        genotype_ids = []
        for id in bios_pheno_df["ids"]:
            splitted_id = id.split("-")
            dataset = splitted_id[0]
            genotype_id = "-".join(splitted_id[1:])

            if dataset == "LL":
                genotype_id = "1_" + genotype_id
            elif dataset in ["NTR", "RS", "CODAM"]:
                if "-" in genotype_id:
                    genotype_id = genotype_id.split("-")[-1]
                if genotype_id in trans_dict[dataset]:
                    genotype_id = trans_dict[dataset][genotype_id]
            elif dataset in ["PAN", "LLS"]:
                genotype_id = genotype_id + "_" + genotype_id

            genotype_ids.append(genotype_id)

        bios_pheno_df["genotype_id"] = genotype_ids
        bios_pheno_df["hasData"] = "TRUE"

        merge_df = gte_df.merge(bios_pheno_df.loc[:, ["genotype_id", "hasData"]], on="genotype_id", how="left")
        merge_df.fillna("FALSE", inplace=True)
        print(merge_df)
        print(merge_df["hasData"].value_counts())
        print(merge_df["dataset"].value_counts())
        print(merge_df.loc[merge_df["hasData"] == "FALSE", :])
        print(merge_df.loc[merge_df["hasData"] == "FALSE", "dataset"].value_counts())


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

if __name__ == '__main__':
    m = main()
    m.start()
