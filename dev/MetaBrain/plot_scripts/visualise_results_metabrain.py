#!/usr/bin/env python3

"""
File:         visualise_results_metabrain.py
Created:      2021/05/06
Last Changed: 2021/12/06
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
import glob
import os

# Third party imports.
import pandas as pd

# Local application imports.

# Metadata
__program__ = "Visualise Results MetaBrain"
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
./visualise_results_metabrain.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.input_data_path = getattr(arguments, 'input_data')
        self.pf_path = getattr(arguments, 'picalo_files')
        self.expression_preprocessing_path = getattr(arguments, 'expression_preprocessing_dir')
        self.palette_path = getattr(arguments, 'palette')
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

        return parser.parse_args()

    def start(self):
        self.print_arguments()

        # Plot overview lineplot.
        command = ['python3', 'overview_lineplot.py', '-i', self.input_data_path, '-p', self.palette_path, "-o", self.outname]
        self.run_command(command)

        # Plot covariate selection overview lineplot.
        command = ['python3', 'covariate_selection_lineplot.py', '-i', self.input_data_path, '-p', self.palette_path, "-o", self.outname]
        self.run_command(command)

        # Plot genotype stats.
        command = ['python3', 'create_histplot.py', '-d', os.path.join(self.input_data_path, "genotype_stats.txt.gz"), "-o", self.outname + "_GenotypeStats"]
        self.run_command(command)

        # Plot eQTL upsetplot.
        command = ['python3', 'create_upsetplot.py', '-i', self.input_data_path, '-e', os.path.join(self.pf_path, "BIOS_eQTLProbesFDR0.05-ProbeLevel-Available.txt.gz"), '-p', self.palette_path, '-o', self.outname]
        self.run_command(command)

        # Plot interaction overview plot.
        command = ['python3', 'interaction_overview_plot.py', '-i', self.input_data_path, '-p', self.palette_path, '-o', self.outname]
        self.run_command(command)

        # Plot #ieQTLs per sample boxplot.
        command = ['python3', 'no_ieqtls_per_sample_plot.py', '-i', self.input_data_path, '-p', self.palette_path, '-o', self.outname]
        self.run_command(command)

        pics = []
        last_iter_fpaths = []
        for i in range(1, 50):
            pic = "PIC{}".format(i)
            comp_iterations_path = os.path.join(self.input_data_path, pic, "iteration.txt.gz")

            if os.path.exists(comp_iterations_path):
                pics.append(pic)
                fpaths = glob.glob(os.path.join(self.input_data_path, pic, "results_iteration*"))
                fpaths.sort()
                last_iter_fpaths.append(fpaths[-1])

                # Plot scatterplot.
                command = ['python3', 'create_scatterplot.py', '-d', comp_iterations_path,
                           "-hr", "0", "-ic", "0", "-a", "1", "-std", os.path.join(self.pf_path, "sample_to_dataset.txt.gz"), '-p', self.palette_path, "-o", self.outname + "_PIC{}".format(i)]
                self.run_command(command)

                # # Plot correlation_heatmap of iterations.
                # command = ['python3', 'create_correlation_heatmap.py', '-rd', comp_iterations_path, "-rn", self.outname, "-o", self.outname + "_{}".format(pic)]
                # self.run_command(command)

        # Compare iterative t-values .
        command = ['python3', 'compare_tvalues.py', '-d'] + last_iter_fpaths + ['-n'] + pics + ['-o', self.outname + '_IterativeTValuesOverview']
        self.run_command(command)

        # Create components_df if not exists.
        components_path = os.path.join(self.input_data_path, "components.txt.gz")
        if not os.path.exists(components_path):
            print("Components file does not exists, loading iteration files")
            data = []
            columns = []
            for i in range(1, 25):
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
                self.save_file(components_df.T, outpath=components_path, header=True, index=True)

        # Plot comparison to expression mean correlation.
        for pic in pics:
            command = ['python3', 'create_regplot.py', '-xd', components_path,
                       "-xi", pic, '-yd', '/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/correlate_samples_with_avg_gene_expression/MetaBrain_CorrelationsWithAverageExpression.txt.gz', '-y_transpose', '-yi', 'AvgExprCorrelation', '-std', os.path.join(self.pf_path, "sample_to_dataset.txt.gz"), '-p', self.palette_path, '-o', self.outname + "_{}_vs_AvgExprCorrelation".format(pic)]
            self.run_command(command)

        # Check for which PICs we have the interaction stats.
        pics = []
        pic_interactions_fpaths = []
        for i in range(1, 25):
            pic = "PIC{}".format(i)
            fpath = os.path.join(self.input_data_path, "PIC_interactions", "{}.txt.gz".format(pic))
            if os.path.exists(fpath):
                pics.append(pic)
                pic_interactions_fpaths.append(fpath)

        if len(pic_interactions_fpaths) > 0:
            # Compare t-values.
            command = ['python3', 'compare_tvalues.py', '-d'] + pic_interactions_fpaths + ['-n'] + pics + ['-o', '{}_TValuesOverview'.format(self.outname)]
            self.run_command(command)

        # Plot comparison scatterplot.
        command = ['python3', 'create_comparison_scatterplot.py', '-d', components_path,
                   '-std', os.path.join(self.pf_path, "sample_to_dataset.txt.gz"), '-p', self.palette_path, '-o', self.outname + "_ColoredByDataset"]
        self.run_command(command)

        # Plot comparison scatterplot.
        command = ['python3', 'create_comparison_scatterplot.py', '-d', components_path,
                   '-std', "/groups/umcg-biogen/tmp01/output/2020-11-10-PICALO/preprocess_scripts/prepare_metabrain_phenotype_matrix/MetaBrain_sex.txt.gz", '-p', self.palette_path, '-o', self.outname + "_ColoredBySex"]
        self.run_command(command)

        if os.path.exists(components_path):
            # Plot correlation_heatmap of components.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-o", self.outname]
            self.run_command(command)

            # Plot correlation_heatmap of components vs datasets.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.pf_path, "datasets_table.txt.gz"), "-cn", "phenotypes", "-o", self.outname + "_vs_Datasets"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs RNA alignment metrics.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-01-31-expression-tables/2020-02-05-step6-covariate-removal/2020-02-05-step0-correlate-covariates-with-expression/2020-02-05-freeze2dot1.TMM.Covariates.withBrainRegion-noncategorical-variable.txt.gz", "-cn", "RNAseq alignment metrics", "-o", self.outname + "_vs_IncludedSTARMetrics"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs Sex.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-bios/tmp01/projects/PICALO/preprocess_scripts/prepare_bios_phenotype_matrix/BIOS_sex.txt.gz", "-cn", "Sex", "-o", self.outname + "_vs_Sex"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs MDS.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.pf_path, "mds_table.txt.gz"), "-cn", "MDS", "-o", self.outname + "_vs_MDS"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs PCA without cov correction.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.expression_preprocessing_path, 'data', 'MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.PCAOverSamplesEigenvectors.txt.g'), "-cn", "PCA before cov. corr.", "-o", self.outname + "_vs_PCABeforeCorrection"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs PCA with cov correction.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", os.path.join(self.expression_preprocessing_path, "data", "MetaBrain.allCohorts.2020-02-16.TMM.freeze2dot1.SampleSelection.SampleSelection.ProbesWithZeroVarianceRemoved.Log2Transformed.ProbesCentered.SamplesZTransformed.CovariatesRemovedOLS.PCAOverSamplesEigenvectors.txt.gz"), "-cn", "PCA after cov. corr.", "-o", self.outname + "_vs_PCAAfterCorrection"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs decon.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/partial_deconvolution/PSYCHENCODE_PROFILE_METABRAIN_AND_PSYCHENCODE_EXON_TPM_LOG2_NODEV/IHC_0CPM_LOG2/deconvolution.txt.gz", "-cn", "NNLS cell fractions", "-o", self.outname + "_vs_decon"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs IHC.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/IHC_counts.txt.gz", "-cn", "IHC", "-o", self.outname + "_vs_decon"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs single cell counts.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "/groups/umcg-biogen/tmp01/output/2019-11-06-FreezeTwoDotOne/2020-10-12-deconvolution/deconvolution/data/AMP-AD/single_cell_counts.txt.gz", "-cn", "SCC", "-o", self.outname + "_vs_SCC"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs MetaBrain phenotype.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "", "-cn", "phenotypes", "-o", self.outname + "_vs_Phenotypes"]
            self.run_command(command)

            # Plot correlation_heatmap of components vs expression correlations.
            command = ['python3', 'create_correlation_heatmap.py', '-rd', components_path, "-rn", self.outname, "-cd", "", "-cn", "AvgExprCorrelation", "-o", self.outname + "_vs_AvgExprCorrelation"]
            self.run_command(command)

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
        print("  > Picalo files path: {}".format(self.pf_path))
        print("  > Expression pre-processing data path: {}".format(self.expression_preprocessing_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Outname {}".format(self.outname))
        print("  > Output directory {}".format(self.outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
