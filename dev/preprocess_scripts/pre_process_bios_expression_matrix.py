#!/usr/bin/env python3

"""
File:         pre_process_bios_expression_matrix.py
Created:      2021/10/22
Last Changed: 2021/11/04
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
from functools import reduce
import json
import argparse
import time
import glob
import os

# Third party imports.
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local application imports.

# Metadata
__program__ = "Pre-process BIOS Expression Matrix"
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
./pre_process_bios_expression_matrix.py -h
"""


class main():
    def __init__(self):
        # Get the command line arguments.
        arguments = self.create_argument_parser()
        self.data_path = getattr(arguments, 'data')
        self.tcov_path = getattr(arguments, 'technical_covariates')
        self.mds_path = getattr(arguments, 'mds')
        self.std_path = getattr(arguments, 'sample_to_dataset')
        self.palette_path = getattr(arguments, 'palette')
        outdir = getattr(arguments, 'outdir')
        outfolder = getattr(arguments, 'outfolder')

        # Set variables.
        if outdir is None:
            outdir = str(Path(__file__).parent.parent)
        self.plot_outdir = os.path.join(outdir, 'pre_process_bios_expression_matrix', outfolder, 'plot')
        self.file_outdir = os.path.join(outdir, 'pre_process_bios_expression_matrix', outfolder, 'data')
        for outdir in [self.plot_outdir, self.file_outdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # Loading palette.
        self.palette = None
        if self.palette_path is not None:
            with open(self.palette_path) as f:
                self.palette = json.load(f)
            f.close()

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
        parser.add_argument("-t",
                            "--technical_covariates",
                            type=str,
                            required=True,
                            help="The path to the technical covariates matrix.")
        parser.add_argument("-m",
                            "--mds",
                            type=str,
                            required=True,
                            help="The path to the mds matrix.")
        parser.add_argument("-std",
                            "--sample_to_dataset",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to the sample-dataset link matrix.")
        parser.add_argument("-p",
                            "--palette",
                            type=str,
                            required=False,
                            default=None,
                            help="The path to a json file with the"
                                 "dataset to color combinations.")
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

        # Construct the output filename.
        filename = os.path.basename(self.data_path).replace(".gz", "").replace(".txt", "")

        # Load sample-dataset file.
        print("Loading sample-to-dataset.")
        std_df = self.load_file(self.std_path, header=0, index_col=None)

        # Pre-process data.
        print("Pre-processing samples-to-dataset.")
        samples = std_df.iloc[:, 0].values.tolist()
        sample_to_dataset = dict(zip(std_df.iloc[:, 0], std_df.iloc[:, 1]))

        dataset_sample_counts = list(zip(*np.unique(std_df.iloc[:, 1], return_counts=True)))
        dataset_sample_counts.sort(key=lambda x: -x[1])
        datasets = [csc[0] for csc in dataset_sample_counts]
        print("\tDatasets: {} [N = {}]".format(", ".join(datasets), len(datasets)))

        dataset_s = std_df.copy()
        dataset_s.set_index(std_df.columns[0], inplace=True)
        dataset_df = pd.get_dummies(dataset_s, prefix="", prefix_sep="")
        dataset_df = dataset_df.loc[:, datasets]

        # Load data.
        print("Loading expression data.")
        df = self.load_file(self.data_path, header=0, index_col=0, nrows=100)
        print(df)

        print("Step 1: reorder samples.")
        df = df.loc[:, samples]

        print("Step 2: PCA analysis.")
        self.pca(df=df,
                 filename=filename,
                 sample_to_dataset=sample_to_dataset,
                 plot_appendix="_1")

        print("Step 3: Construct technical covariate matrix.")
        tcov_df = self.load_file(self.tcov_path, header=0, index_col=0)
        mds_df = self.load_file(self.mds_path, header=0, index_col=0)
        correction_df = self.prepare_correction_matrix(tcov_df=tcov_df.loc[samples, :],
                                                       mds_df=mds_df.loc[samples, :],
                                                       dataset_df=dataset_df)

        print("\tSaving file.")
        self.save_file(df=correction_df, outpath=os.path.join(self.file_outdir, "correction_matrix.txt.gz"))

        print("Step 4: remove technical covariates OLS.")
        corrected_df = self.calculate_residuals(df=df, correction_df=correction_df)

        print("\tSaving file.")
        self.save_file(df=corrected_df, outpath=os.path.join(self.file_outdir, "{}.CovariatesRemovedOLS.txt.gz".format(filename)))

        print("Step 5: PCA analysis.")
        self.pca(df=corrected_df,
                 filename=filename,
                 sample_to_dataset=sample_to_dataset,
                 file_appendix="CovariatesRemovedOLS",
                 plot_appendix="_2_CovariatesRemovedOLS")

        del correction_df, corrected_df

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
    def save_file(df, outpath, header=True, index=True, sep="\t"):
        compression = 'infer'
        if outpath.endswith('.gz'):
            compression = 'gzip'

        df.to_csv(outpath, sep=sep, index=index, header=header,
                  compression=compression)
        print("\tSaved dataframe: {} "
              "with shape: {}".format(os.path.basename(outpath),
                                      df.shape))

    @staticmethod
    def reverse_dict(dict):
        out_dict = {}
        seen_keys = set()
        for key, value in dict.items():
            if key in seen_keys:
                print("Key {} has muiltiple values.".format(key))
            seen_keys.add(key)

            if value in out_dict.keys():
                keys = out_dict[value]
                keys.append(key)
                out_dict[value] = keys
            else:
                out_dict[value] = [key]

        return out_dict

    def prepare_correction_matrix(self, tcov_df, mds_df, dataset_df):
        df = tcov_df.copy()

        # Remove columns without variance.
        df = df.loc[:, df.std(axis=0) != 0]

        # Remove dataset columns.
        dataset_mask = np.array([set(df[col].unique()) == {0, 1} for col in df.columns])
        df = df.loc[:, ~dataset_mask]

        # filter the technical covariates on VIF.
        df = self.remove_multicollinearity(df)

        # Merge the tcov_df with the mds_df.
        correction_df = df.merge(mds_df, left_index=True, right_index=True)

        # merge the tcov_df with the dataset_df but exclude the dataset with
        # the highest number of samples.
        correction_df = correction_df.merge(dataset_df.iloc[:, 1:], left_index=True, right_index=True)
        correction_df.insert(0, "INTERCEPT", 1)
        correction_df.index.name = "-"

        return correction_df

    def remove_multicollinearity(self, df, threshold=0.9999):
        indices = np.arange(df.shape[1])
        max_r2 = np.inf
        while len(indices) > 1 and max_r2 > threshold:
            r2 = np.array([self.calc_ols_rsquared(df.iloc[:, indices], ix) for ix in range(len(indices))])
            max_r2 = max(r2)

            if max_r2 > threshold:
                max_index = np.where(r2 == max_r2)[0][0]
                indices = np.delete(indices, max_index)

        return df.iloc[:, indices]

    @staticmethod
    def calc_ols_rsquared(df, idx):
        return OLS(df.iloc[:, idx], df.loc[:, np.arange(df.shape[1]) != idx]).fit().rsquared

    @staticmethod
    def calculate_residuals(df, correction_df):
        corrected_m = np.empty_like(df, dtype=np.float64)
        last_print_time = None
        n_tests = df.shape[0]
        for i in range(n_tests):
            now_time = int(time.time())
            if last_print_time is None or (now_time - last_print_time) >= 10 or (i + 1) == n_tests:
                last_print_time = now_time
                print("\t{}/{} ieQTLs analysed [{:.2f}%]".format((i + 1), n_tests, (100 / n_tests) * (i + 1)))

            ols = OLS(df.iloc[i, :], correction_df)
            results = ols.fit()
            # print(results.summary())
            corrected_m[i, :] = results.resid

        return pd.DataFrame(corrected_m, index=df.index, columns=df.columns)

    def pca(self, df, filename, sample_to_dataset, file_appendix="", plot_appendix=""):
        # samples should be on the columns and genes on the rows.
        zscores = (df - df.mean(axis=0)) / df.std(axis=0)
        pca = PCA(n_components=25)
        pca.fit(zscores)
        expl_variance = {"PC{}".format(i+1): pca.explained_variance_ratio_[i] * 100 for i in range(25)}
        components_df = pd.DataFrame(pca.components_)
        components_df.index = ["Comp{}".format(i + 1) for i, _ in enumerate(components_df.index)]
        components_df.columns = df.columns

        print("\tSaving file.")
        self.save_file(df=components_df, outpath=os.path.join(self.file_outdir, "{}.{}.PCAOverSamplesEigenvectors.txt.gz".format(filename, file_appendix)))
        self.save_file(df=components_df.iloc[:10, :], outpath=os.path.join(self.file_outdir, "{}.{}.PCAOverSamplesEigenvectors.First10PCs.txt.gz".format(filename, file_appendix)))

        print("\tPlotting PCA")
        plot_df = components_df.T
        plot_df["hue"] = plot_df.index.map(sample_to_dataset)
        self.plot(df=plot_df,
                  x="Comp1",
                  y="Comp2",
                  hue="hue",
                  palette=self.palette,
                  xlabel="PC1 [{:.2f}%]".format(expl_variance["PC1"]),
                  ylabel="PC2 [{:.2f}%]".format(expl_variance["PC2"]),
                  title="PCA - eigenvectors",
                  filename="eigenvectors_plot{}".format(plot_appendix))

    def plot(self, df, x="x", y="y", hue=None, palette=None, xlabel=None,
             ylabel=None, title="", filename="PCA_plot"):
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y

        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set_style("ticks")
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [0.9, 0.1]})
        sns.despine(fig=fig, ax=ax1)
        ax2.axis('off')

        sns.scatterplot(x=x,
                        y=y,
                        hue=hue,
                        data=df,
                        s=100,
                        linewidth=0,
                        legend=None,
                        palette=palette,
                        ax=ax1)

        ax1.set_title(title,
                      fontsize=20,
                      fontweight='bold')
        ax1.set_ylabel(ylabel,
                       fontsize=14,
                       fontweight='bold')
        ax1.set_xlabel(xlabel,
                       fontsize=14,
                       fontweight='bold')

        if palette is not None:
            handles = []
            for label, color in palette.items():
                if label in df[hue].values.tolist():
                    handles.append(mpatches.Patch(color=color, label=label))
            ax2.legend(handles=handles, loc="center")

        #fig.savefig(os.path.join(self.plot_outdir, "{}.pdf".format(filename)))
        fig.savefig(os.path.join(self.plot_outdir, "{}.png".format(filename)))
        plt.close()

    def print_arguments(self):
        print("Arguments:")
        print("  > Data: {}".format(self.data_path))
        print("  > Technical covariates: {}".format(self.tcov_path))
        print("  > MDS: {}".format(self.mds_path))
        print("  > Sample-to-dataset path: {}".format(self.std_path))
        print("  > Palette path: {}".format(self.palette_path))
        print("  > Plot output directory: {}".format(self.plot_outdir))
        print("  > File output directory: {}".format(self.file_outdir))
        print("")


if __name__ == '__main__':
    m = main()
    m.start()
