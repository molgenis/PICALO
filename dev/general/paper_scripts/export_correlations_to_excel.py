#!/usr/bin/env python3

"""
File:         export_correlations_to_excel.py
Created:      2022/07/04
Last Changed:
Author:       M.Vochteloo

Copyright (C) 2020 University Medical Center Groningen.

A copy of the BSD 3-Clause "New" or "Revised" License can be found in the
LICENSE file in the root directory of this source tree.
"""

# Standard imports.
from __future__ import print_function
from pathlib import Path
import argparse
import math
import os

# Third party imports.
import numpy as np
import pandas as pd
import xlsxwriter
from scipy.special import betainc

# Local application imports.

"""
Syntax:
./export_correlation_to_excel.py -h

### BIOS ###

./export_correlations_to_excel.py \
    -i /groups/umcg-bios/tmp01/projects/PICALO/output/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -pf /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -ep /groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/pre_process_expression_matrix/2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_WithUncenteredPCA \
    -d blood \
    -p /groups/umcg-bios/tmp01/projects/PICALO/data/BIOSColorPalette.json \
    -o 2022-03-24-BIOS_NoRNAPhenoNA_NoSexNA_NoMixups_NoMDSOutlier_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
    
### MetaBrain ###
    
./export_correlations_to_excel.py \
    -i /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/output/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -pf /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_picalo_files/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA \
    -ep /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/pre_process_expression_matrix/2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_WithUncenteredPCA \
    -d brain \
    -p /groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrainColorPalette.json \
    -o 2022-03-24-MetaBrain_CortexEUR_NoENA_NoRNAseqAlignmentMetrics_GT1AvgExprFilter_PrimaryeQTLs_UncenteredPCA
"""


# Metadata
__program__ = "Export Correlations to Excel"
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


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_data_path = getattr(arguments, 'input_data')
        self.pf_path = getattr(arguments, 'picalo_files')
        self.expression_preprocessing_path = getattr(arguments, 'expression_preprocessing_dir')
        self.dataset = getattr(arguments, 'dataset')
        self.palette_path = getattr(arguments, 'palette')
        self.outname = getattr(arguments, 'outname')
        self.extensions = getattr(arguments, 'extensions')

        # Set variables.
        outdir = os.path.join(str(Path(__file__).parent.parent), 'export_correlations_to_excel')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.outpath = os.path.join(outdir, self.outname + ".xlsx")

        self.data_files = {}
        if self.dataset == "blood":
            self.data_files = {
                "AvgExprCorrelation": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/BIOS_CorrelationsWithAverageExpression.txt.gz",
                "Datasets": os.path.join(self.pf_path, "datasets_table.txt.gz"),
                "RNASeqAlignmentMetrics": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_RNA_AlignmentMetrics.txt.gz",
                "Sex": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz",
                "GenotypeMDS": os.path.join(self.pf_path, "mds_table.txt.gz"),
                "CellFractionPercentages": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellFractionPercentages_forPlotting.txt.gz",
                "CellCounts": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_CellCounts.txt.gz",
                "BloodStats": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_BloodStats.txt.gz",
                "Decon-cell": "/groups/umcg-bios/tmp01/projects/PICALO/data/BIOS_cell_types_DeconCell_2019-03-08.txt.gz",
                "Phenotypes": "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_phenotypes.txt.gz",
            }
        elif self.dataset == "brain":
            self.data_files = {
                "AvgExprCorrelation": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/MetaBrain_CorrelationsWithAverageExpression.txt.gz",
                "Datasets": os.path.join(self.pf_path, "datasets_table.txt.gz"),
                "RNASeqAlignmentMetrics": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz",
                "Sex": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_metabrain_phenotype_matrix/MetaBrain_sex.txt.gz",
                "GenotypeMDS": os.path.join(self.pf_path, "mds_table.txt.gz"),
                "CellFractionFull": "/groups/umcg-biogen/prm03/projects/2022-DeKleinEtAl/output/2020-10-12-deconvolution/deconvolution/matrix_preparation/2022-01-21-CortexEUR-cis-NegativeToZero-DatasetAndRAMCorrected/perform_deconvolution/deconvolution_table_complete.txt.gz",
                "CellFraction": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/data/MetaBrain_CellFractionPercentages_forPlotting.txt.gz",
                "IHCCounts": "/groups/umcg-biogen/prm03/projects/2022-DeKleinEtAl/output/2020-10-12-deconvolution/deconvolution/data/AMP-AD/IHC_counts.txt.gz",
                "SNCounts": "/groups/umcg-biogen/prm03/projects/2022-DeKleinEtAl/output/2020-10-12-deconvolution/deconvolution/data/AMP-AD/single_cell_counts.txt.gz",
                "Phenotypes": "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_metabrain_phenotype_matrix/MetaBrain_phenotypes.txt.gz",
            }
        else:
            print("Error")
            exit()

        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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
        parser.add_argument("-id",
                            "--input_data",
                            type=str,
                            required=True,
                            help="The path to PICALO results.")
        parser.add_argument("-pf",
                            "--picalo_files",
                            type=str,
                            required=True,
                            help="The path to the picalo files.")
        parser.add_argument("-ep",
                            "--expression_preprocessing_dir",
                            type=str,
                            required=True,
                            help="The path to the expression preprocessing data.")
        parser.add_argument("-d",
                            "--dataset",
                            type=str,
                            required=True,
                            choices=["blood", "brain"],
                            help="The dataset.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=True,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-o",
                            "--outname",
                            type=str,
                            required=True,
                            help="The name of the output files.")
        parser.add_argument("-e",
                            "--extensions",
                            type=str,
                            nargs="+",
                            default=["png"],
                            choices=["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"],
                            help="The output file format(s), default: ['png']")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        print("Loading PICs")
        col_df = self.load_file(os.path.join(self.input_data_path, "PICs.txt.gz"), header=0, index_col=0)
        col_df = col_df._get_numeric_data()

        if col_df.shape[1] > col_df.shape[0]:
            col_df = col_df.T

        col_df = col_df.loc[:, col_df.std(axis=0) != 0]

        col_labels = col_df.columns.tolist()
        col_n_na = col_df.isna().sum(axis=0).values.tolist()
        col_indices = ["{} [N={:,}]".format(label, col_df.shape[0] - n_na) for label, n_na in zip(col_labels, col_n_na)]

        with pd.ExcelWriter(self.outpath, engine='xlsxwriter') as writer:
            for sheet_name, filepath in self.data_files.items():
                print("Processing {}.".format(sheet_name))
                row_df = self.load_file(filepath, header=0, index_col=0)
                row_df = row_df._get_numeric_data()

                if row_df.shape[1] > row_df.shape[0]:
                    row_df = row_df.T

                row_df = row_df.loc[:, row_df.std(axis=0) != 0]

                row_labels = row_df.columns.tolist()
                row_n_na = row_df.isna().sum(axis=0).values.tolist()
                row_indices = ["{} [N={:,}]".format(label, row_df.shape[0] - n_na) for label, n_na in zip(row_labels, row_n_na)]

                print("\tGetting overlap.")
                # Make sure order is the same.
                samples = set(col_df.index.tolist()).intersection(set(row_df.index.tolist()))
                print("\tN = {}".format(len(samples)))
                if len(samples) == 0:
                    print("No data overlapping.")
                    continue
                row_m = row_df.loc[samples, :].to_numpy()
                col_m = col_df.loc[samples, :].to_numpy()

                print(row_df.loc[samples, :])
                print(col_df.loc[samples, :])

                print("Correlating.")
                corr_m, _ = self.corrcoef(m1=row_m, m2=col_m)
                corr_df = pd.DataFrame(corr_m, index=row_indices, columns=col_indices)
                print(corr_df)

                corr_df.to_excel(writer, sheet_name=sheet_name, na_rep="NA", index=True)
                print("Saving sheet '{}' with shape {}".format(sheet_name, corr_df.shape))
                print("")

                worksheet = writer.sheets[sheet_name]
                worksheet.conditional_format(1, 1, corr_df.shape[0], corr_df.shape[1],
                                            {"type": "3_color_scale",
                                             'min_type': 'num',
                                             "min_value": -1,
                                             'mid_type': 'num',
                                             "mid_value": 0,
                                             'max_type': 'num',
                                             "max_value": 1,
                                             "min_color": "#437bb2",
                                             "mid_color": "#FFFFFF",
                                             "max_color": "#BA5B39"})

    @staticmethod
    def load_file(path, sep="\t", header=0, index_col=0, nrows=None):
        df = pd.read_csv(path, sep=sep, header=header, index_col=index_col,
                         nrows=nrows)
        print("Loaded dataframe: {} "
              "with shape: {}".format(os.path.basename(path),
                                      df.shape))
        return df

    @staticmethod
    def corrcoef(m1, m2):
        """
        Pearson correlation over the columns.

        https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
        """
        m1_dev = m1 - np.nanmean(m1, axis=0)
        m2_dev = m2 - np.nanmean(m2, axis=0)

        m1_rss = np.nansum(m1_dev * m1_dev, axis=0)
        m2_rss = np.nansum(m2_dev * m2_dev, axis=0)

        r = np.empty((m1_dev.shape[1], m2_dev.shape[1]), dtype=np.float64)
        for i in range(m1_dev.shape[1]):
            for j in range(m2_dev.shape[1]):
                mask = np.logical_or(np.isnan(m1_dev[:, i]), np.isnan(m2_dev[:, j]))
                r[i, j] = np.sum(m1_dev[~mask, i] * m2_dev[~mask, j]) / np.sqrt(m1_rss[i] * m2_rss[j])

        rf = r.flatten()
        df = m1.shape[0] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = betainc(0.5 * df, 0.5, df / (df + ts))
        p = pf.reshape(m1.shape[1], m2.shape[1])
        return r, p

    def print_arguments(self):
        print("Arguments:")
        print("  > Input data path: {}".format(self.input_data_path))
        print("  > Picalo files path: {}".format(self.pf_path))
        print("  > Expression pre-processing data path: {}".format(self.expression_preprocessing_path))
        print("  > Dataset: {}".format(self.dataset))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Outname: {}".format(self.outname))
        print("  > Output filename: {}".format(self.outpath))
        print("  > Extensions: {}".format(self.extensions))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
