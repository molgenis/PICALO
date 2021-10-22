#!/usr/bin/env python3

"""
File:         visualise_results.py
Created:      2021/05/06
Last Changed: 2021/10/21
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
import subprocess
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Visualise Results"
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
./visualise_results.py -h

"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_data_path = getattr(arguments, 'input_data')
        self.palette_path = getattr(arguments, 'palette')
        self.expression_preprocessing_path = getattr(arguments, 'expression_preprocessing_dir')
        self.matrix_preparation_path = getattr(arguments, 'matrix_preparation_dir')
        self.outname = getattr(arguments, 'outname')

        # Set variables.
        self.outdir = os.path.join(str(Path(__file__).parent.parent), 'plot')
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
        parser.add_argument("-id",
                            "--input_data",
                            type=str,
                            required=True,
                            help="The path to PICA results.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
        parser.add_argument("-ep",
                            "--expression_preprocessing_dir",
                            type=str,
                            required=True,
                            help="The path to the expression preprocessing data.")
        parser.add_argument("-mp",
                            "--matrix_preparation_dir",
                            type=str,
                            required=True,
                            help="The path to the matrix preparation data.")
        parser.add_argument("-o",
                            "--outname",
                            type=str,
                            required=True,
                            help="The name of the output files.")

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Plot overview lineplot.
        command = ['python3', 'overview_lineplot.py', '-i', self.input_data_path, '-p', self.palette_path]
        self.run_command(command)

        # Plot eQTL upsetplot.
        command = ['python3', 'create_upsetplot.py', '-i', self.input_data_path, '-e', os.path.join(self.matrix_preparation_path, "combine_eqtlprobes", "eQTLprobes_combined.txt.gz"), '-p', self.palette_path]
        self.run_command(command)

        # Plot interaction venn diagram.
        command = ['python3', 'interaction_overview_plot.py', '-i', self.input_data_path, '-p', self.palette_path]
        self.run_command(command)

        # Plot #ieQTLs per sample boxplot.
        command = ['python3', 'no_ieqtls_per_sample_plot.py', '-i', self.input_data_path, '-p', self.palette_path]
        self.run_command(command)

        for i in range(1, 11):
            comp_iterations_path = os.path.join(self.input_data_path, "PIC{}".format(i), "iteration.txt.gz")

            if os.path.exists(comp_iterations_path):
                # Plot scatterplot.
                command = ['python3', 'create_scatterplot.py', '-d', comp_iterations_path,
                           "-hr", "0", "-ic", "0", "-a", "1", "-std", os.path.join(self.matrix_preparation_path, "combine_gte_files", "SampleToDataset.txt.gz"), '-p', self.palette_path, "-o", self.outname + "_comp{}".format(i)]
                self.run_command(command)

        # Create components_df if not exists.
        components_path = os.path.join(self.input_data_path, "components.txt.gz")
        if not os.path.exists(components_path):
            data = []
            columns = []
            for i in range(1, 11):
                pic = "PIC{}".format(i)
                comp_iterations_path = os.path.join(self.input_data_path, pic, "iteration.txt.gz")
                if os.path.exists(comp_iterations_path):
                    df = self.load_file(comp_iterations_path, header=0, index_col=0)
                    last_iter = df.iloc[[df.shape[0] - 1], :].T
                    data.append(last_iter)
                    columns.append(pic)

            if len(data) > 0:
                components_df = pd.concat(data, axis=1)
                components_df.columns = columns
                self.save_file(components_df, outpath=components_path, header=True, index=True)

        # Plot comparison scatterplot.
        command = ['python3', 'create_comparison_scatterplot.py', '-d', components_path,
                   "-std", os.path.join(self.matrix_preparation_path, "combine_gte_files", "SampleToDataset.txt.gz"), '-p', self.palette_path, '-o', self.outname]
        self.run_command(command)

        if os.path.exists(components_path):
            # Plot correlation_heatmap of components.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-o", self.outname]
            self.run_command(command)

            # Plot correlation_heatmap of components vs tech. covs.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.matrix_preparation_path, "create_correction_matrix", "technical_covariates_table.txt.gz"), "-cn", "tech. covs.", "-o", self.outname + "_vs_techcovs"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs MDS.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.matrix_preparation_path, "create_correction_matrix", "mds_covariates_table.txt.gz"), "-cn", "tech. covs.", "-o", self.outname + "_vs_MDS"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs decon.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.matrix_preparation_path, "perform_deconvolution", "deconvolution_table.txt.gz"), "-cn", "cell fractions", "-o", self.outname + "_vs_decon"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs PCA without cov correction.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.expression_preprocessing_path, 'data', 'MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.PCAOverSamplesEigenvectors.txt.gz'), "-cn", "PCA before cov. corr.", "-o", self.outname + "_vs_pca_before_corr"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs PCA with cov correction.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.expression_preprocessing_path, "data", "MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz"), "-cn", "PCA", "-o", self.outname + "_vs_pca"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs IHC.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/IHC_counts.txt.gz", "-cn", "IHC", "-o", self.outname + "_vs_IHC"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs single cell counts.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/single_cell_counts.txt.gz", "-cn", "IHC", "-o", self.outname + "_vs_SCC"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs MetaBrain phenotype.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2020-11-10-PICA/preprocess_scripts/encode_phenotype_matrix/2020-03-09.brain.phenotypes.txt", "-cn", "MetaBrain Phenotype", "-o", self.outname + "_vs_MetaBrainPhenotypes"]
            self.run_command(command)

            # # Plot correlation_heatmap of components vs MetaBrain PCs.
            # command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-01-31-expression-tables/2020-02-05-step6-covariate-removal/2020-02-06-step4-remove-residual-covariates/pc1_10_residual.txt", "-cn", "MetaBrain PCA", "-o", self.outname + "_vs_MetaBrainPCs"]
            # self.run_command(command)

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
    def save_file(df, outpath, header=True, index=False, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    @staticmethod
    def run_command(command):
        print(" ".join(command))
        subprocess.call(command)

    def print_arguments(self):
        print("Arguments:")
        print("  > Input data path: {}".format(self.input_data_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Expression pre-processing data path: {}".format(self.expression_preprocessing_path))
        print("  > Matrix preparation data path: {}".format(self.matrix_preparation_path))
        print("  > Outname {}".format(self.outname))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
